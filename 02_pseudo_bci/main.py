"""
main.py
-------
擬似 BCI デコーダ体験のエントリーポイント。実際の BCI の
    「運動（手の動き）→ 脳信号（ニューロンの発火）→ デコーダ → 機械（カーソル）を動かす」
という流れを、脳波計なしで再現する。

  1. hand_tracker.py  : カメラ（別スレッド）で手の位置と速度を求める。テストモードではマウスを使う
  2. neuron_model.py  : その速度から、運動野ニューロン集団のスパイクを 20 ms ごとに生成する
  3. decoder.py       : スパイク数だけから速度を推定し、カーソルを動かす。精度（相関・RMSE）も計算
  4. renderer.py      : 3D のニューロン、手とカーソルの比較、ラスタープロット、HUD を描く

シミュレーションは描画のフレームレートと関係なく、実時間に合わせて 20 ms 刻みで進める
（描画が 1 フレーム 40 ms かかったら、そのフレームで 2 ステップ進める）。

使い方:
    cd 02_pseudo_bci
    ../.venv/bin/python main.py              # カメラで手を追跡
    ../.venv/bin/python main.py --test       # マウスで操作（テストモード）
    ../.venv/bin/python main.py --camera 1   # 別のカメラを使う
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

import cv2           # noqa: E402
import glfw          # noqa: E402
import numpy as np   # noqa: E402

import config        # noqa: E402
from decoder import DecodingMetrics, PopulationVectorDecoder   # noqa: E402
from hand_tracker import HandTracker, MouseHand                # noqa: E402
from neuron_model import NeuronPopulation                      # noqa: E402
from renderer import PANEL, PANEL_X, PANEL_Y, Renderer          # noqa: E402
from session_logger import SessionLogger                        # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_WINDOW = "Hand Tracking Debug"


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
        self.build_population()
        self.metrics = DecodingMetrics()
        self.logger = SessionLogger(BASE_DIR)
        self.sim_time = 0.0

    def build_population(self):
        """ニューロン集団とデコーダを（作り）直す"""
        self.pop = NeuronPopulation(self.n, self.rng)
        self.pop.noise_std = self.noise
        self.decoder = PopulationVectorDecoder(self.pop)
        self.renderer.set_population(self.pop, self.rng)
        print(f"[main] ニューロン {self.n} 個で再生成しました")

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
            self.metrics.reset()
        elif key == glfw.KEY_DOWN:
            self.n = max(config.N_MIN, self.n - config.N_STEP)
            self.build_population()
            self.metrics.reset()
        elif key == glfw.KEY_N:
            self.noise = min(config.NOISE_MAX, self.noise + config.NOISE_STEP)
            self.pop.noise_std = self.noise
        elif key == glfw.KEY_M:
            self.noise = max(0.0, self.noise - config.NOISE_STEP)
            self.pop.noise_std = self.noise
        elif key == glfw.KEY_K and action == glfw.PRESS:
            k = self.pop.dropout()
            print(f"[main] {k} 個のニューロンが脱落しました（残り {self.pop.n_alive} / {self.pop.n}）")
        elif key == glfw.KEY_T and action == glfw.PRESS:
            if self.tracker is None:
                print("[main] カメラがないのでテストモードのままです")
            else:
                self.test_mode = not self.test_mode
                print(f"[main] {'テストモード（マウス）' if self.test_mode else 'カメラモード'}")
        elif key == glfw.KEY_D and action == glfw.PRESS:
            self.toggle_debug()
        elif key == glfw.KEY_R and action == glfw.PRESS:
            # ニューロンを作り直し（脱落も元に戻る）、カーソル・評価・表示をリセット
            self.build_population()
            self.metrics.reset()
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
        v_hat = self.decoder.step(counts)          # 発火 → 速度の推定（本当の速度は使わない）
        self.metrics.add(true_v, v_hat)            # 評価のためだけに本当の速度と比べる
        self.renderer.on_spikes(counts, self.pop.alive, self.decoder.rate_hat)
        self.sim_time += config.DT
        self.logger.record(self.sim_time, "test" if self.test_mode else "camera", hand, true_v, v_hat,
                           self.decoder.cursor, self.pop.n, self.pop.n_alive, self.noise, counts.sum())

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
                    self.metrics.compute()
                    last_metric = now

                if self.test_mode:
                    mode = "TEST MODE (mouse = hand)" + ("" if self.tracker else " - no camera")
                    mode_color = (255, 200, 110)
                else:
                    mode, mode_color = "CAMERA MODE (hand tracking)", (140, 230, 160)

                self.renderer.render({
                    "hand": hand, "true_v": hand.vel, "v_hat": self.decoder.v_hat,
                    "cursor": self.decoder.cursor, "alive": self.pop.alive,
                    "n": self.pop.n, "n_alive": self.pop.n_alive, "noise": self.noise,
                    "corr": self.metrics.corr, "rmse": self.metrics.rmse,
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
    args = parser.parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
