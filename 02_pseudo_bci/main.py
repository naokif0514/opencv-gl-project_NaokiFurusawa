"""
main.py
-------
擬似 BCI デコーダ体験のエントリーポイント。実際の BCI の
    「運動（手の動き）→ 脳信号（ニューロンの発火）→ デコーダ → 機械（カーソル）を動かす」
という流れを、脳波計なしで再現する。

  1. hand_tracker.py  : カメラ（別スレッド）で手の位置と速度を求める。テストモードではマウスを使う
  2. neuron_model.py  : その速度から、運動野ニューロン集団のスパイクを 20 ms ごとに生成する
  3. decoder.py       : スパイク数だけから速度を推定し、カーソルを動かす。精度（相関・RMSE）も計算
                        ポピュレーションベクトル法（PV）と線形デコーダ（LIN）の 2 つを常に同時に動かす
     calibration.py   : 線形デコーダを学習するためのキャリブレーション課題（ターゲットを手で追う）
     robot_arm.py     : カーソルを手先の目標として、2 関節アームの角度を逆運動学で求める
  4. renderer.py      : 3D のニューロン、手とカーソルの比較、アーム、ラスタープロット、HUD を描く

シミュレーションは描画のフレームレートと関係なく、実時間に合わせて 20 ms 刻みで進める
（描画が 1 フレーム 40 ms かかったら、そのフレームで 2 ステップ進める）。

使い方:
    cd 02_pseudo_bci
    ../.venv/bin/python main.py              # カメラで手を追跡
    ../.venv/bin/python main.py --test       # マウスで操作（テストモード）
    ../.venv/bin/python main.py --camera 1   # 別のカメラを使う
    ../.venv/bin/python main.py --experiment [--trials 20] [--only neurons,noise,dropout,assumptions]
                                             # 画面なしの自動実験モード（experiment.py）。results/ に保存
"""

import argparse
import os
import sys
import time

# Wayland 環境（Ubuntu の標準）でも安定して動くよう、ウィンドウは X11 互換モード（XWayland）で開く
#   QT_QPA_PLATFORM        : OpenCV のウィンドウ（Qt）用
#   PYGLFW_LIBRARY_VARIANT : glfw 用（Wayland 版は GNOME の設定と合わず異常終了することがある）
#   PYOPENGL_PLATFORM      : PyOpenGL 用（X11 版 glfw に合わせて GLX を使う）
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
if "--experiment" in sys.argv:
    # 実験モードは小さな行列計算を大量にくり返すので、数値計算ライブラリのマルチスレッドは
    # かえって遅くなる（スレッドの奪い合い）。numpy を読み込む前に 1 スレッドにしておく
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2           # noqa: E402
import glfw          # noqa: E402
import numpy as np   # noqa: E402

import config        # noqa: E402
from calibration import CalibrationTask                         # noqa: E402
from decoder import (DecodingMetrics, LinearDecoder,             # noqa: E402
                     PopulationVectorDecoder, RateEstimator)
from hand_tracker import HandTracker, MouseHand                # noqa: E402
from neuron_model import NOISE_TYPES, NeuronPopulation         # noqa: E402
from robot_arm import TwoLinkArm                               # noqa: E402
from renderer import PANEL, PANEL_X, PANEL_Y, Renderer          # noqa: E402
from session_logger import SessionLogger                        # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_WINDOW = "Hand Tracking Debug"
MESSAGE_SEC = 6.0      # 画面に出すお知らせ（キャリブレーションの結果など）の表示時間 [秒]


class App:
    def __init__(self, args):
        self.rng = np.random.default_rng()

        # ---- カメラ（見つからなければテストモード）----
        self.tracker = None
        if not args.test:
            tr = HandTracker(args.camera)
            if tr.open():
                self.tracker = tr
                tr.start()
                print(f"[main] カメラ {args.camera} を開きました")
            else:
                print(f"[main] カメラ {args.camera} を開けませんでした → テストモード（マウス）で起動します",
                      file=sys.stderr)
        self.test_mode = self.tracker is None
        self.mouse = MouseHand()
        self.debug_on = self.tracker is not None
        if self.tracker is not None:
            self.tracker.debug_enabled = self.debug_on

        # ---- 描画 ----
        self.renderer = Renderer()
        glfw.set_key_callback(self.renderer.window, self.on_key)

        # ---- シミュレーションとデコーダ ----
        self.n = config.N_NEURONS
        self.noise = config.NOISE_STD
        self.noise_type = config.NOISE_TYPE
        self.primary = "PV"          # カーソル・アームを動かすデコーダ（C キーで切り替え）
        self.compare = False         # 比較モード（V キー）：2 つのカーソルを同時に表示
        self.arm_on = True           # ロボットアームの表示（A キー）
        self.arm = TwoLinkArm()
        self.calib = CalibrationTask()
        self.metrics = {"PV": DecodingMetrics(), "LIN": DecodingMetrics()}
        self.message, self.message_until = "", 0.0
        self.build_population()
        self.logger = SessionLogger(BASE_DIR)
        self.sim_time = 0.0

    def build_population(self):
        """
        ニューロン集団とデコーダを（作り）直す。
        ニューロンが入れかわるので、線形デコーダの学習結果は使えなくなる（もう一度 L でキャリブレーション）
        """
        self.pop = NeuronPopulation(self.n, self.rng)
        self.pop.noise_std = self.noise
        self.pop.noise_type = self.noise_type
        self.rates = RateEstimator(self.n)
        self.decoders = {"PV": PopulationVectorDecoder(self.pop), "LIN": LinearDecoder(self.n)}
        self.dropout_since_calib = 0
        if self.calib.active:
            self.calib.cancel()
        if self.primary == "LIN":
            self.primary = "PV"
        for m in self.metrics.values():
            m.reset()
        self.renderer.set_population(self.pop, self.rng)
        print(f"[main] ニューロン {self.n} 個で再生成しました")

    def notify(self, text):
        print(f"[main] {text}")
        self.message, self.message_until = text, time.monotonic() + MESSAGE_SEC

    # --------------------------------------------------------
    # キー操作
    # --------------------------------------------------------
    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_UP:
            self.n = min(config.N_MAX, self.n + config.N_STEP)
            self.build_population()
        elif key == glfw.KEY_DOWN:
            self.n = max(config.N_MIN, self.n - config.N_STEP)
            self.build_population()
        elif key == glfw.KEY_N:
            self.noise = min(config.NOISE_MAX, self.noise + config.NOISE_STEP)
            self.pop.noise_std = self.noise
        elif key == glfw.KEY_M:
            self.noise = max(0.0, self.noise - config.NOISE_STEP)
            self.pop.noise_std = self.noise
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3) and action == glfw.PRESS:
            self.noise_type = NOISE_TYPES[key - glfw.KEY_1]
            self.pop.noise_type = self.noise_type
            self.notify(f"noise type: {self.noise_type}")
        elif key == glfw.KEY_K and action == glfw.PRESS:
            k = self.pop.dropout()
            if self.decoders["LIN"].ready:
                self.dropout_since_calib += k
            self.notify(f"dropout: {k} neurons lost ({self.pop.n_alive} / {self.pop.n} alive)")
        elif key == glfw.KEY_L and action == glfw.PRESS:
            if self.calib.active:
                self.calib.cancel()
                self.notify("calibration cancelled")
            else:
                self.calib.start()
                self.message = ""
                print("[main] キャリブレーション開始：マゼンタのターゲットを手で追いかけてください")
        elif key == glfw.KEY_C and action == glfw.PRESS:
            if self.primary == "PV":
                if self.decoders["LIN"].ready:
                    self.primary = "LIN"
                else:
                    self.notify("LIN is not calibrated yet - press L first")
            else:
                self.primary = "PV"
        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.compare = not self.compare
        elif key == glfw.KEY_A and action == glfw.PRESS:
            self.arm_on = not self.arm_on
        elif key == glfw.KEY_T and action == glfw.PRESS:
            if self.tracker is None:
                print("[main] カメラがないのでテストモードのままです")
            else:
                self.test_mode = not self.test_mode
                print(f"[main] {'テストモード（マウス）' if self.test_mode else 'カメラモード'}")
        elif key == glfw.KEY_D and action == glfw.PRESS:
            self.toggle_debug()
        elif key == glfw.KEY_R and action == glfw.PRESS:
            # ニューロンを作り直し（脱落も元に戻る。線形デコーダの学習結果も消える）、カーソル・評価・表示をリセット
            self.build_population()
            self.renderer.reset_display()

    def toggle_debug(self):
        if self.tracker is None:
            print("[main] カメラがないのでデバッグウィンドウは使えません")
            return
        self.debug_on = not self.debug_on
        self.tracker.debug_enabled = self.debug_on
        if not self.debug_on:
            cv2.destroyWindow(DEBUG_WINDOW)
            cv2.waitKey(1)

    # --------------------------------------------------------
    # 入力（手 or マウス）
    # --------------------------------------------------------
    def read_hand(self, now):
        if self.test_mode:
            win = self.renderer.window
            mx, my = glfw.get_cursor_pos(win)
            ww, wh = glfw.get_window_size(win)
            inside = glfw.get_window_attrib(win, glfw.HOVERED) == 1
            # マウスの位置を右の 2D 平面の座標に合わせる（平面の上でマウスを動かすと白いリングが重なる）
            x = (mx * config.WINDOW_WIDTH / ww - PANEL_X) / PANEL
            y = 1.0 - (my * config.WINDOW_HEIGHT / wh - PANEL_Y) / PANEL
            return self.mouse.update(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0), inside, now)
        return self.tracker.get_state()

    # --------------------------------------------------------
    # シミュレーション 1 ステップ（20 ms）
    # --------------------------------------------------------
    def sim_step(self, hand):
        true_v = hand.vel
        counts = self.pop.step(true_v)             # 運動 → ニューロンの発火
        rate_hat = self.rates.step(counts)         # スパイク数 → 発火率の推定（100 ms の窓）
        lin = self.decoders["LIN"]
        for name, dec in self.decoders.items():    # 発火 → 速度の推定（本当の速度は使わない）
            dec.step(rate_hat)
            if name == "PV" or lin.ready:
                self.metrics[name].add(true_v, dec.v_hat)   # 評価のためだけに本当の速度と比べる

        # キャリブレーション中は (発火率, 本当の速度) を記録し、終わったら線形デコーダを学習する
        if self.calib.step(rate_hat, true_v, hand.detected):
            self.finish_calibration()

        self.renderer.on_spikes(counts, self.pop.alive, rate_hat)
        self.sim_time += config.DT
        prim = self.decoders[self.primary]
        self.logger.record(self.sim_time, "test" if self.test_mode else "camera", hand, true_v,
                           prim.v_hat, prim.cursor, self.pop.n, self.pop.n_alive, self.noise, counts.sum(),
                           primary=self.primary, noise_type=self.noise_type, calibrating=self.calib.active,
                           pv=self.decoders["PV"], lin=lin)

    def finish_calibration(self):
        if self.calib.valid_sec < config.CALIB_MIN_SEC:
            self.notify(f"calibration failed: hand detected only {self.calib.valid_sec:.1f} s")
            return
        X, Y = self.calib.data()
        info = self.decoders["LIN"].fit(X, Y, n_alive=self.pop.n_alive)
        self.dropout_since_calib = 0
        self.metrics["LIN"].reset()
        self.primary = "LIN"
        self.compare = True        # 学習直後は比較モードにして、PV と並べて見せる
        self.notify(f"LIN calibrated: {info['samples']} samples, train R2 = {info['r2']:.2f}")

    def lin_status(self):
        """HUD に出す線形デコーダの状態（文字と色）"""
        lin = self.decoders["LIN"]
        if self.calib.active:
            return "LIN: calibrating...", (255, 120, 220)
        if not lin.ready:
            return "LIN: not calibrated (press L)", (150, 160, 185)
        if self.dropout_since_calib > 0:
            return (f"LIN: stale ({self.dropout_since_calib} lost since calib, L = recal)", (255, 160, 110))
        return f"LIN: calibrated with {lin.fit_info['n_alive']} neurons (R2 {lin.fit_info['r2']:.2f})", \
            (150, 220, 255)

    def update_debug_window(self):
        if not self.debug_on or self.tracker is None:
            return
        frame = self.tracker.pop_debug_frame()
        if frame is not None:
            cv2.imshow(DEBUG_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            glfw.set_window_should_close(self.renderer.window, True)
        elif key in (ord("d"), ord("D")):
            self.toggle_debug()

    # --------------------------------------------------------
    # メインループ
    # --------------------------------------------------------
    def run(self):
        prev = time.monotonic()
        acc = 0.0           # まだ進めていないシミュレーション時間
        fps = 0.0
        last_metric = 0.0
        try:
            while not self.renderer.should_close():
                now = time.monotonic()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.95 * fps + 0.05 / dt if fps > 0 else 1.0 / dt

                glfw.poll_events()
                hand = self.read_hand(now)

                # 実時間に合わせて 20 ms 刻みでシミュレーションを進める
                acc += dt
                steps = 0
                while acc >= config.DT and steps < config.MAX_STEPS_PER_FRAME:
                    self.sim_step(hand)
                    acc -= config.DT
                    steps += 1
                if steps == config.MAX_STEPS_PER_FRAME:
                    acc = 0.0      # 大きく遅れたときは追いつくのをあきらめる

                if now - last_metric > 0.2:
                    for m in self.metrics.values():
                        m.compute()
                    last_metric = now

                if self.test_mode:
                    mode = "TEST MODE (mouse = hand)" + ("" if self.tracker else " - no camera")
                    mode_color = (255, 200, 110)
                else:
                    mode, mode_color = "CAMERA MODE (hand tracking)", (140, 230, 160)

                # 表示するカーソル：ふだんは使っているデコーダだけ、比較モードでは両方（学習済みなら）
                shown = [self.primary]
                if self.compare:
                    shown = ["PV"] + (["LIN"] if self.decoders["LIN"].ready else [])
                decoders = [(name, self.decoders[name].cursor, self.decoders[name].v_hat) for name in shown]

                # ロボットアーム：使っているデコーダのカーソルを手先の目標にして逆運動学で角度を求める
                arm_target = self.decoders[self.primary].cursor
                arm_pose = self.arm.solve(arm_target) if self.arm_on else None

                calib = None
                if self.calib.active:
                    calib = {"target": self.calib.target, "progress": self.calib.progress,
                             "remaining": self.calib.remaining, "recording": self.calib.recording,
                             "valid_sec": self.calib.valid_sec}
                lin_text, lin_color = self.lin_status()

                self.renderer.render({
                    "hand": hand, "true_v": hand.vel, "decoders": decoders, "primary": self.primary,
                    "alive": self.pop.alive, "n": self.pop.n, "n_alive": self.pop.n_alive,
                    "noise": self.noise, "noise_type": self.noise_type,
                    "metrics": {k: (m.corr, m.rmse) for k, m in self.metrics.items()},
                    "lin_status": lin_text, "lin_status_color": lin_color,
                    "arm": arm_pose, "arm_target": arm_target, "calib": calib,
                    "message": self.message if now < self.message_until else "",
                    "mode": mode, "mode_color": mode_color, "fps": fps,
                }, dt)
                self.update_debug_window()
        finally:
            self.shutdown()

    def shutdown(self):
        if self.tracker is not None:
            self.tracker.stop()
        cv2.destroyAllWindows()
        self.renderer.close()
        path = self.logger.save()
        print(f"[main] ログを保存しました: {path}")


def main():
    parser = argparse.ArgumentParser(description="擬似 BCI デコーダ体験")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="カメラ番号")
    parser.add_argument("--test", action="store_true", help="カメラを使わず、マウスを手の代わりにする")
    parser.add_argument("--experiment", action="store_true",
                        help="画面なしの自動実験モード（残りの引数は experiment.py に渡す。例：--trials 20）")
    args, rest = parser.parse_known_args()
    if args.experiment:
        import experiment
        experiment.main(rest)
        return
    if rest:
        parser.error(f"不明な引数: {' '.join(rest)}")
    App(args).run()


if __name__ == "__main__":
    main()
