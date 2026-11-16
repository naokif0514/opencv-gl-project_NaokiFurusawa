"""
main.py
-------
BCI オペレーター（ゲーム本体）。実際の BCI 研究の「ハイブリッド BCI」をもとにしたゲーム。
  ・視線 → 行き先やつかむ物を選ぶ（前頭眼野・視覚野）                       … gaze.py
  ・手の動き → 擬似ニューロンの発火 → デコード → ロボットアーム（運動野）       … cortex_sim.py / robot.py
  ・手を握る/開く → 把持ニューロンの発火 → デコード → グリッパー（運動野・体性感覚野）
  ・背景はカメラ映像（AR）。現実の色の物がゲーム内のアイテムになる             … vision.py / items.py
  ・横に本物の脳の 3D モデルを表示し、今使っている領域が光る                   … brain_renderer.py

■ 画面（状態）の流れ
  title（タイトル。脳は pial）
   → connect（脳と接続中：pial → inflated へふくらむ）
   → gaze_calib（視線同期：5 点。テストモードでは省略）
   → stage_intro（ステージの説明）
   → hand_calib（同期フェーズ：手のキャリブレーション。速度と把持のデコーダを学習）
   → play（ミッション。mission.py）→ stage_clear（スコアとデコード精度の推移グラフ）
   → 次のステージの stage_intro … → result（全ステージの結果）→ title

使い方:
    cd 03_bci_game
    ../.venv/bin/python main.py            # カメラ（視線 + 手 + 色の物）
    ../.venv/bin/python main.py --test     # カメラなし：マウスの位置 = 視線、マウスの動き = 手、クリック = 握る
    ../.venv/bin/python main.py --stage 3  # ステージ 3 から始める（確認用）
"""

import argparse
import os
import sys
import time

# Wayland 環境（Ubuntu の標準）でも安定して動くよう、ウィンドウは X11 互換モード（XWayland）で開く
# （02_pseudo_bci/main.py と同じ設定）
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import cv2           # noqa: E402
import glfw          # noqa: E402
import numpy as np   # noqa: E402
from OpenGL import GL as gl   # noqa: E402

import config        # noqa: E402
from brain_model import BrainModel                                  # noqa: E402
from brain_renderer import BrainView                                # noqa: E402
from cortex_sim import CortexSim                                    # noqa: E402
from game_log import GameLogger                                     # noqa: E402
from game_view import GameView                                      # noqa: E402
from gaze import DwellSelector, GazeTracker                         # noqa: E402
from hand_calib import HandCalibration                              # noqa: E402
from hand_tracker import MouseHand                                  # noqa: E402
from items import Goal, RealItemTracker, VirtualItems               # noqa: E402
from layout import AH, AW, AX, AY, H, W, calib_points, world_radius, world_to_screen   # noqa: E402
from mission import STAGES, GripSwitch, Mission, gripper_tip        # noqa: E402
from robot import Robot                                             # noqa: E402
from vision import VisionThread                                     # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_WINDOW = "Camera Debug"
CONNECT_SEC = config.MORPH_SEC + 0.6
GOAL_POS = np.array([0.84, 0.15])


class Game:
    def __init__(self, args):
        self.rng = np.random.default_rng()
        self.first_stage = max(1, min(len(STAGES), args.stage)) - 1

        # ---- カメラ（見つからなければテストモード）----
        self.vision = None
        if not args.test:
            v = VisionThread(args.camera)
            if v.open():
                self.vision = v
                v.start()
                print(f"[main] カメラ {args.camera} を開きました")
            else:
                print("[main] カメラを開けませんでした → テストモード（マウス）で起動します", file=sys.stderr)
        self.test_mode = self.vision is None
        self.debug_on = False

        self._init_window()
        print("[main] 脳モデルを読み込みます")
        self.model = BrainModel()
        self.brain = BrainView(self.model)
        self.brain.set_surface("pial", animate=False)
        self.view = GameView(self)
        self.logger = GameLogger(BASE_DIR)

        self.mouse_hand = MouseHand()
        self.mouse_down = False
        self.gaze = GazeTracker(calib_points())
        self.dwell = DwellSelector()
        self.show_gaze = True
        self.robot = Robot()
        self.real_items = RealItemTracker()
        self.virtual = VirtualItems(self.rng)
        self.use_virtual = self.test_mode
        self.extra_items = []         # スキャンしたデジタルの荷物・ゴールに置いた荷物
        self.hand_calib = HandCalibration()
        self.grip_switch = GripSwitch()
        self.message, self.message_until = "", 0.0
        self.banner, self.banner_until = "", 0.0
        self.results = []
        self.mission = None
        self.stage_idx = self.first_stage
        self.recalibrating = False
        self.resume_play = False
        self.selected = None
        self.last_g = 0.0
        self.hand = None
        self.events = []              # このフレームで起きたこと（ログ用）
        self.build_cortex(STAGES[self.stage_idx])
        self.set_state("title")

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("glfw の初期化に失敗しました")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, config.MSAA_SAMPLES)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)
        self.window = glfw.create_window(W, H, config.WINDOW_TITLE, None, None)
        if not self.window:
            raise RuntimeError("ウィンドウを作成できませんでした")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        if config.MSAA_SAMPLES > 0:
            gl.glEnable(gl.GL_MULTISAMPLE)
        print(f"[main] {gl.glGetString(gl.GL_RENDERER).decode()}")
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)

    # --------------------------------------------------------
    # ステージ・状態
    # --------------------------------------------------------
    @property
    def stage(self):
        return STAGES[self.stage_idx]

    def build_cortex(self, st):
        """ステージのニューロン数・ノイズで運動野を作り直す（デコーダは未学習に戻る）"""
        self.cortex = CortexSim(st.n_neurons, self.rng, st.noise, st.noise_type)
        self.brain.set_neurons(self.cortex.n_all, self.cortex.colors, self.rng)

    def set_state(self, s):
        self.state, self.state_t = s, 0.0
        self.events.append(f"state:{s}")
        if s == "title":
            self.brain.set_surface("pial")
            self.brain.camera.sway = None
            self.brain.camera.goal_dist = 3.3
        elif s == "connect":
            self.brain.set_surface("infl")
            self.brain.camera.yaw, self.brain.camera.pitch, self.brain.camera.goal_dist = -100.0, 30.0, 3.7
            self.brain.camera.sway = (-105.0, 25.0)       # ゲーム中は左半球の外側面を見ながら ±25° 揺れる
        elif s == "gaze_calib":
            self.gaze.calib.start()
            self.dwell.reset()
        elif s == "stage_intro":
            self.build_cortex(self.stage)
            self.robot.reset()
            self.selected = None
            self.extra_items = []
            self.virtual.spawn(avoid=[self.robot.pos, GOAL_POS])
            self.grip_switch = GripSwitch()
        elif s == "hand_calib":
            self.hand_calib.start()
        elif s == "play":
            if self.mission is None or self.mission.stage is not self.stage:
                self.mission = Mission(self.stage)
            self.next_acc_log = 0.0

    def start_game(self):
        self.results = []
        self.stage_idx = self.first_stage
        self.mission = None
        self.set_state("connect")

    def notify(self, text, sec=3.5):
        print(f"[main] {text}")
        self.message, self.message_until = text, time.monotonic() + sec

    @property
    def goal(self):
        return Goal(GOAL_POS, self.stage.goal_radius)

    # --------------------------------------------------------
    # キー・マウス
    # --------------------------------------------------------
    def on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        go = key in (glfw.KEY_SPACE, glfw.KEY_ENTER)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_D and self.vision is not None:
            self.debug_on = not self.debug_on
            self.vision.debug_enabled = self.debug_on
            if not self.debug_on:
                cv2.destroyWindow(DEBUG_WINDOW)
                cv2.waitKey(1)
        elif key == glfw.KEY_T and self.vision is not None and self.state == "title":
            self.test_mode = not self.test_mode
            self.use_virtual = self.test_mode or self.use_virtual
        elif self.state == "title" and go:
            self.start_game()
        elif self.state == "stage_intro" and go:
            self.set_state("hand_calib")
        elif self.state == "stage_clear" and go:
            self.next_stage()
        elif self.state == "result" and go:
            self.set_state("title")
        elif key == glfw.KEY_G:
            self.show_gaze = not self.show_gaze
        elif key == glfw.KEY_C and self.state == "play" and not self.test_mode:
            self.resume_play = True                      # 視線同期をやり直したら play に戻る
            self.set_state("gaze_calib")
        elif key == glfw.KEY_L and self.state == "play" and not self.recalibrating:
            self.recalibrating = True                   # 再同期（制限時間は進み続ける）
            self.hand_calib.start()
            self.events.append("recalibrate_start")
            self.notify("再同期を開始：ターゲットを手で追いかける")
        elif key == glfw.KEY_V and self.state == "play":
            self.use_virtual = not self.use_virtual
            self.selected = None
            self.notify("仮想アイテム" if self.use_virtual else "現実の色の物をアイテムにする")
        elif key == glfw.KEY_P and self.vision is not None and not self.test_mode:
            self.n_registered = getattr(self, "n_registered", 0) + 1
            name = f"color{(self.n_registered - 1) % 3 + 1}"
            self.vision.register_color(name)
            self.pending_register = name
        elif key == glfw.KEY_R and self.state == "play":
            self.robot.reset()
            self.selected = None

    def on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_down = action == glfw.PRESS
            if action == glfw.PRESS and self.state == "title" and self.view.start_button_hit(self.mouse_layout()):
                self.start_game()

    def mouse_layout(self):
        mx, my = glfw.get_cursor_pos(self.window)
        ww, wh = glfw.get_window_size(self.window)
        return np.array([mx * W / ww, my * H / wh])

    # --------------------------------------------------------
    # 入力
    # --------------------------------------------------------
    def read_inputs(self, now, dt):
        """(手の状態, 本当の握り具合 g, 注視点, 色の塊, 視線の特徴量)"""
        if self.test_mode:
            m = self.mouse_layout()
            inside = AX <= m[0] <= AX + AW and AY <= m[1] <= AY + AH
            hand = self.mouse_hand.update((m[0] - AX) / AW, 1.0 - (m[1] - AY) / AH, inside, now)
            g = 1.0 if self.mouse_down else 0.0            # クリック = 握る
            gaze = self.gaze.update(None, dt, mouse=m) if inside else None
            return hand, g, gaze, [], None
        vs = self.vision.get_state()
        if vs.hand_open is not None:
            self.last_g = 1.0 - vs.hand_open             # 手が一瞬見えなくなっても、握り具合は直前の値を保つ
        gaze = self.gaze.update(vs.gaze_feat, dt, feat_time=vs.gaze_time) if self.gaze.calibrated else None
        return vs.hand, self.last_g, gaze, vs.blobs, vs.gaze_feat

    def items(self, blobs, now, dt):
        base = self.virtual.items if self.use_virtual else self.real_items.update(blobs, now, dt)
        return base + self.extra_items

    # --------------------------------------------------------
    # メインループ
    # --------------------------------------------------------
    def run(self):
        prev = time.monotonic()
        start = prev
        acc, fps = 0.0, 0.0
        try:
            while not glfw.window_should_close(self.window):
                now = time.monotonic()
                dt = min(now - prev, 0.1)
                prev = now
                if dt > 0:
                    fps = 0.95 * fps + 0.05 / dt if fps > 0 else 1.0 / dt
                glfw.poll_events()
                self.state_t += dt
                self.check_color_register()

                hand, g, gaze, blobs, feat = self.read_inputs(now, dt)
                self.hand = hand
                items = self.items(blobs, now, dt)

                # ---- 運動野のシミュレーション（20 ms 刻み）。同期フェーズのデータもここで集める ----
                acc += dt
                steps = 0
                while acc >= config.DT and steps < config.MAX_STEPS_PER_FRAME:
                    self.cortex.step(hand.vel, g, hand.detected)
                    if self.hand_calib.active:
                        if self.hand_calib.step(self.cortex.rate_hat, self.cortex.grip_rate_hat, hand.vel, g,
                                                hand.detected):
                            self.finish_hand_calib()
                    acc -= config.DT
                    steps += 1
                if steps == config.MAX_STEPS_PER_FRAME:
                    acc = 0.0
                self.cortex.decay(dt)

                selecting = self.update_state(dt, gaze, feat, items)
                self.robot.update(dt)

                # ---- 脳の光 ----
                #   視線で選ぶ → 前頭眼野・視覚野、手を動かす → 運動野、握る → 運動野 + 体性感覚野
                motor = max(self.cortex.motor_drive, self.cortex.grip_drive)
                self.brain.update_glow({"fef": selecting, "visual": selecting, "motor": motor, "hand": motor,
                                        "somato": self.cortex.grip_drive}, dt)

                self.view.render(now - start, dt, gaze, items, fps)
                self.log_frame(now - start, gaze, g)
                self.update_debug_window()
        finally:
            if self.vision is not None:
                self.vision.stop()
            cv2.destroyAllWindows()
            glfw.terminate()
            path = self.logger.save()
            if path:
                print(f"[main] プレイログを保存しました: {path}")

    def update_state(self, dt, gaze, feat, items):
        """状態ごとの処理。視線で選んでいる強さ（脳の前頭眼野・視覚野の光）を返す"""
        s = self.state
        if s == "connect" and self.state_t >= CONNECT_SEC:
            self.set_state("stage_intro" if self.test_mode else "gaze_calib")
        elif s == "gaze_calib":
            if self.test_mode:
                self.gaze.calib.cancel()
                self.after_gaze_calib()
            elif self.gaze.calib.step(feat, dt):
                self.gaze.reset_filters()
                self.gaze.reset_offset()
                self.notify(f"視線同期 完了（ずれ 約 {self.gaze.calib.residual:.0f} 画素）")
                self.after_gaze_calib()
            return 1.0
        elif s == "stage_intro" and self.state_t >= config.STAGE_INTRO_SEC:
            self.set_state("hand_calib")
        elif s == "play":
            return self.update_play(dt, gaze, items)
        return 0.0

    def after_gaze_calib(self):
        if self.resume_play:
            self.resume_play = False
            self.state, self.state_t = "play", 0.0
        else:
            self.set_state("stage_intro")

    def finish_hand_calib(self):
        hc = self.hand_calib
        if not hc.enough:
            msg = "同期に失敗：手が見えていないか、握る・開くができていません"
            self.events.append("hand_calib_failed")
            if self.state == "hand_calib":
                self.notify(msg + " → もう一度", 4.0)
                hc.start()
            else:
                self.notify(msg, 4.0)
                self.recalibrating = False
            return
        info = self.cortex.fit(*hc.data())
        self.events.append(f"hand_calib_ok:r2={info['r2']:.2f}")
        self.notify(f"同期完了（ニューロン {self.cortex.n_alive} 個、学習データでの R² = {info['r2']:.2f}）")
        self.grip_switch = GripSwitch()
        self.recalibrating = False
        if self.state == "hand_calib":
            self.set_state("play")

    # --------------------------------------------------------
    def update_play(self, dt, gaze, items):
        m = self.mission
        m.tick(dt)
        st = self.stage

        # ---- 電極劣化イベント（ステージ 3）----
        if st.degrade_at is not None and not m.degraded and m.play_time >= st.degrade_at:
            m.degraded = True
            k = self.cortex.dropout(st.degrade_fraction, config.DEGRADE_GRIP_FRACTION)
            m.log(f"degrade:{k}")
            self.events.append(f"degrade:{k}")
            self.banner, self.banner_until = "electrode", time.monotonic() + 5.0

        # ---- デコード精度の推移を記録 ----
        if m.play_time >= self.next_acc_log:
            self.cortex.metrics.compute()
            m.result.accuracy.append((m.play_time, self.cortex.metrics.corr, self.cortex.metrics.rmse))
            self.next_acc_log = m.play_time + config.ACCURACY_LOG_SEC

        selecting = 0.0
        if not self.recalibrating:
            # ---- 視線で選ぶ：持っていないときはアイテムだけ、持っているときはゴールだけを選べる ----
            #      （アームを動かしている間に視線が別の物に留まって、運ぶ途中で行き先が変わらないように）
            if m.held is None:
                targets = [(it.id, world_to_screen(it.pos), world_radius(it.radius)) for it in items
                           if m.selectable(it)]
            else:
                targets = [("goal", world_to_screen(GOAL_POS), world_radius(st.goal_radius))]
            chosen = self.dwell.update(gaze, targets, dt, exclude=self.selected)
            if chosen is not None:
                self.select(chosen, items)
            if self.dwell.focus is not None:
                selecting = 0.4 + 0.6 * self.dwell.progress

            # ---- 手 → アーム（デコードした速度）、握る → グリッパー（デコードした把持）----
            if not self.robot.moving:
                self.robot.drive_arm(self.cortex.v_hat * config.ARM_VEL_GAIN, dt)
            ev = self.grip_switch.update(self.cortex.grip.value)
            if ev is not None:
                self.on_grip_event(ev, items)
        else:
            self.robot.drive_arm(np.zeros(2), dt)

        if m.held is not None:                     # つかんでいる荷物はグリッパーといっしょに動く
            m.held.pos = gripper_tip(self.robot)

        if m.done:
            self.recalibrating = False
            self.hand_calib.cancel()
            self.results.append(m.finish())
            self.events.append(f"stage_end:score={m.result.score}")
            self.set_state("stage_clear")
        return selecting

    def on_grip_event(self, ev, items):
        m = self.mission
        self.robot.set_grip(ev == "close")
        tip = gripper_tip(self.robot)
        self.events.append(f"grip_{ev}")
        if ev == "close":
            res = m.on_close(tip, items)
            if res == "miss":
                self.notify("つかみ損ね（失敗 +1）")
            elif res is not None:
                if res[1] is not None:                     # 現実の物をスキャンして作った荷物
                    self.extra_items.append(res[1])
                    self.notify("スキャンしてつかんだ！ ゴールを見つめて移動")
                else:
                    self.notify("つかんだ！ ゴールを見つめて移動")
                self.selected = None
            return
        res = m.on_open(tip, self.goal)
        if res is None:
            return
        kind, it, pts, d = res
        if kind == "deliver":
            self.notify(f"成功！ +{pts}（ゴールの中心から {world_radius(d):.0f} 画素）")
        else:
            self.notify("ゴールの外に落とした（失敗 +1）")
        self.selected = None

    def select(self, chosen, items):
        """視線で対象が決定されたとき：ずれを補正し、ロボットの台座をその近くへ移動させる"""
        pos = GOAL_POS if chosen == "goal" else next((i.pos for i in items if i.id == chosen), None)
        if pos is None:
            return
        if not self.test_mode:
            self.gaze.correct(world_to_screen(pos))
        self.selected = chosen
        self.robot.approach(pos)
        self.robot.offset = None
        self.events.append(f"lock:{chosen}")
        self.notify("ゴールをロックオン → 移動" if chosen == "goal" else "アイテムをロックオン → 移動")

    def next_stage(self):
        if self.stage_idx + 1 >= len(STAGES):
            self.set_state("result")
            return
        self.stage_idx += 1
        self.mission = None
        self.set_state("stage_intro")

    # --------------------------------------------------------
    def check_color_register(self):
        name = getattr(self, "pending_register", None)
        if name is None or self.vision is None:
            return
        res = getattr(self.vision, "last_register", None)
        if res is None or res[0] != name:
            return
        self.pending_register = None
        if res[1] is None:
            self.notify("色を登録できませんでした：中央の枠に色のはっきりした物を入れてください")
        else:
            config.ITEM_COLORS[name] = res[1]
            self.use_virtual = False
            self.notify(f"色を登録しました（{name}）")

    def log_frame(self, t, gaze, g):
        m = self.mission if self.state == "play" else None
        h = self.hand
        self.logger.record(
            time_sec=t, state=self.state + ("+recal" if self.recalibrating else ""), stage=self.stage.number,
            mode="test" if self.test_mode else "camera",
            gaze_x=None if gaze is None else float(gaze[0]), gaze_y=None if gaze is None else float(gaze[1]),
            dwell_target=self.dwell.focus, dwell_progress=self.dwell.progress,
            hand_detected=int(h.detected), true_vx=float(h.vel[0]), true_vy=float(h.vel[1]),
            decoded_vx=float(self.cortex.v_hat[0]), decoded_vy=float(self.cortex.v_hat[1]),
            decoder="LIN" if self.cortex.calibrated else "PV", true_grip=float(g),
            decoded_grip=float(self.cortex.grip.value), gripper_closed=int(self.grip_switch.closed),
            holding=int(m is not None and m.held is not None),
            robot_x=float(self.robot.pos[0]), robot_y=float(self.robot.pos[1]),
            n_alive=self.cortex.n_alive, n_neurons=self.cortex.n_all,
            score=None if m is None else m.result.score, time_left=None if m is None else m.time_left,
            event=";".join(self.events))
        self.events = []

    def update_debug_window(self):
        if not self.debug_on or self.vision is None:
            return
        frame = self.vision.pop_debug_frame()
        if frame is not None:
            cv2.imshow(DEBUG_WINDOW, frame)
        cv2.waitKey(1)


def main():
    parser = argparse.ArgumentParser(description="BCI オペレーター")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--test", action="store_true", help="カメラを使わない（マウスで視線・手・握るの代わり）")
    parser.add_argument("--stage", type=int, default=1, help="このステージから始める（1〜4）")
    Game(parser.parse_args()).run()


if __name__ == "__main__":
    main()
