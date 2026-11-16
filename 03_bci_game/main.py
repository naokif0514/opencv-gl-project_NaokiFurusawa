"""
main.py
-------
BCI オペレーター（ゲーム本体）。実際の BCI 研究の「ハイブリッド BCI」をもとにしたゲーム。
制限時間内に、机の上の物をロボットでどれだけ集められるかで勝負する（1 ステージ制）。
  ・視線 → 集める物・カゴを選ぶ                                       … gaze.py
  ・手の動き → 擬似ニューロンの発火 → デコード → ロボットアーム         … cortex_sim.py / robot.py
  ・手を握る/開く → 把持ニューロンの発火 → デコード → グリッパー
  ・現実の色の物をカメラに見せる → ボーナスアイテムが降ってくる         … vision.py / items.py / field.py
  ・BCI らしさはイベントとして：最初の同期フェーズ、プレイ中の電極劣化（L で再同期するかは自分で判断）
  ・脳はタイトル（pial → inflated の「接続中」）、結果画面の振り返り、B キーの「裏側」で見られる

■ 画面（状態）の流れ
  title → connect（脳と接続中）→ gaze_calib（視線同期。テストモードでは省略）→ hand_calib（同期フェーズ）
  → countdown → play（GAME_TIME 秒）→ result（スコア・ハイスコア・プレイ中の脳の活動）→ title

使い方:
    cd 03_bci_game
    ../.venv/bin/python main.py            # カメラ（視線 + 右手 + 色の物）
    ../.venv/bin/python main.py --test     # カメラなし：マウスの位置 = 視線、マウスの動き = 手、クリック = 握る
    ../.venv/bin/python main.py --light    # 軽量モード（config.apply_light_mode）
    ../.venv/bin/python main.py --window   # ディスプレイいっぱいにせず、1280×720 のウィンドウで
F11 で全画面 ⇔ ウィンドウを切りかえる。
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
import highscore     # noqa: E402
from brain_model import BrainModel                                  # noqa: E402
from brain_record import BrainRecord                                # noqa: E402
from brain_renderer import BrainView                                # noqa: E402
from cortex_sim import CortexSim                                    # noqa: E402
from field import Field                                             # noqa: E402
from game_log import GameLogger                                     # noqa: E402
from game_view import GameView                                      # noqa: E402
from gaze import DwellSelector, GazeTracker                         # noqa: E402
from hand_calib import HandCalibration                              # noqa: E402
from hand_tracker import MouseHand                                  # noqa: E402
from items import RealItemTracker                                   # noqa: E402
from layout import H, W, calib_points                               # noqa: E402
from mission import GripSwitch, Mission, gripper_tip                # noqa: E402
from robot import Robot                                             # noqa: E402
from scene3d import Scene3D                                         # noqa: E402
from vision import VisionThread                                     # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_WINDOW = "Camera Debug"
CONNECT_SEC = config.MORPH_SEC + 0.8
GOAL_POS = np.array(config.GOAL_POS)
SCENE_RECT = (0, 0, W, H)          # 3D の場面は画面いっぱい（論理座標）


class Game:
    def __init__(self, args):
        self.rng = np.random.default_rng()

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
        self.scene = Scene3D(self.rng)
        self.view = GameView(self)
        self.logger = GameLogger(BASE_DIR)

        self.mouse_hand = MouseHand()
        self.mouse_down = False
        self.gaze = GazeTracker(calib_points())
        self.dwell = DwellSelector()
        self.show_gaze = True
        self.show_backstage = False      # B キー：脳とラスター（「裏側」）を重ねて表示
        self.robot = Robot()
        self.field = Field(self.rng)
        self.real_items = RealItemTracker()
        self.hand_calib = HandCalibration()
        self.grip_switch = GripSwitch()
        self.message, self.message_until = "", 0.0
        self.flash_until = 0.0           # 電極劣化の赤いフラッシュ
        self.popups = []                 # (文字, ワールド座標, 出た時刻)：得点の表示
        self.mission = None
        self.record = BrainRecord()
        self.final = None                # 結果画面に出すもの
        self.recalibrating = False
        self.resume_play = False
        self.selected = None
        self.lock_time = -1e9          # 最後に行き先を決めた時刻（time.monotonic）
        self.last_g = 0.0
        self.hand = None
        self.events = []
        self._prev_score = 0
        self.build_cortex()
        self.set_state("title")

    def _init_window(self):
        """
        ウィンドウを作る。既定ではディスプレイいっぱい（最大化）。中身は決まった解像度のオフスクリーンに描いて
        ウィンドウの大きさへ拡大するので（render_target.py）、ウィンドウの大きさは自由に変えられる
        """
        if not glfw.init():
            raise RuntimeError("glfw の初期化に失敗しました")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_TRUE)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.fullscreen = config.WINDOW_MODE == "fullscreen"
        if self.fullscreen:
            self.window = glfw.create_window(mode.size.width, mode.size.height, config.WINDOW_TITLE, monitor, None)
        else:
            if config.WINDOW_MODE == "maximized":
                glfw.window_hint(glfw.MAXIMIZED, gl.GL_TRUE)
            self.window = glfw.create_window(W, H, config.WINDOW_TITLE, None, None)
        if not self.window:
            raise RuntimeError("ウィンドウを作成できませんでした")
        if config.WINDOW_MODE == "maximized":
            glfw.maximize_window(self.window)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        print(f"[main] {gl.glGetString(gl.GL_RENDERER).decode()}  描画の解像度 "
              f"{round(W * config.RENDER_SCALE)}×{round(H * config.RENDER_SCALE)}"
              + ("（軽量モード）" if config.LIGHT_MODE else ""))
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)

    # --------------------------------------------------------
    # 状態
    # --------------------------------------------------------
    def build_cortex(self):
        """運動野の擬似ニューロンを作り直す（ニューロン数・ノイズは固定。デコーダは未学習に戻る）"""
        self.cortex = CortexSim(config.GAME_N_NEURONS, self.rng, config.GAME_NOISE, config.GAME_NOISE_TYPE)
        self.brain.set_neurons(self.cortex.n_all, self.cortex.colors, self.rng)

    def set_state(self, s):
        self.state, self.state_t = s, 0.0
        self.events.append(f"state:{s}")
        cam = self.brain.camera
        if s == "title":
            self.brain.set_surface("pial")
            cam.sway, cam.goal_dist = None, 3.3
        elif s == "connect":
            self.brain.set_surface("infl")
        elif s == "gaze_calib":
            self.gaze.calib.start()
            self.dwell.reset()
        elif s == "hand_calib":
            self.hand_calib.start()
        elif s == "play":
            self.mission = Mission(self.rng)
            self.record = BrainRecord()
            self.grip_switch = GripSwitch()
            self._prev_score = 0
            cam.yaw, cam.pitch, cam.goal_dist = -100.0, 30.0, 3.7
            cam.sway = (-105.0, 25.0)            # 「裏側」の脳は、左半球の外側面を見ながら ±25° 揺れる
            print(f"[main] 電極劣化の予定（プレイ開始からの秒）: {[round(t) for t in self.mission.degrade_times]}")
        elif s == "result":
            self.finish_game()

    def start_game(self):
        self.build_cortex()
        self.robot.reset()
        self.field.reset(self.robot.pos)
        self.selected = None
        self.lock_time = -1e9
        self.mission = None
        self.popups = []
        self.recalibrating = False
        self.set_state("connect")

    def notify(self, text, sec=3.0):
        print(f"[main] {text}")
        self.message, self.message_until = text, time.monotonic() + sec

    # --------------------------------------------------------
    # キー・マウス
    # --------------------------------------------------------
    def on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        go = key in (glfw.KEY_SPACE, glfw.KEY_ENTER)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_F11:
            self.toggle_fullscreen()
        elif key == glfw.KEY_D and self.vision is not None:
            self.debug_on = not self.debug_on
            self.vision.debug_enabled = self.debug_on
            if not self.debug_on:
                cv2.destroyWindow(DEBUG_WINDOW)
                cv2.waitKey(1)
        elif key == glfw.KEY_T and self.vision is not None and self.state == "title":
            self.test_mode = not self.test_mode
        elif self.state == "title" and go:
            self.start_game()
        elif self.state == "result" and go:
            self.set_state("title")
        elif key == glfw.KEY_B:
            self.show_backstage = not self.show_backstage
        elif key == glfw.KEY_G:
            self.show_gaze = not self.show_gaze
        elif key == glfw.KEY_C and self.state == "play" and not self.test_mode and not self.recalibrating:
            self.resume_play = True                      # 視線同期をやり直したら play に戻る（時間は進む）
            self.state, self.state_t = "gaze_calib", 0.0
            self.gaze.calib.start()
        elif key == glfw.KEY_L and self.state == "play" and not self.recalibrating:
            self.recalibrating = True                   # 再同期（制限時間は進み続ける）
            self.hand_calib.start()
            self.record.mark(self.mission.play_time, "recal_start")
            self.mission.log("recalibrate_start")
            self.events.append("recalibrate_start")
            self.notify("再同期：ターゲットを手で追いかける")
        elif key == glfw.KEY_P and self.vision is not None and not self.test_mode:
            self.n_registered = getattr(self, "n_registered", 0) + 1
            name = f"color{(self.n_registered - 1) % 3 + 1}"
            self.vision.register_color(name)
            self.pending_register = name
        elif key == glfw.KEY_R and self.state == "play":
            self.robot.reset()
            self.selected = None

    def toggle_fullscreen(self):
        """F11：全画面 ⇔ ディスプレイいっぱいのウィンドウ"""
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            glfw.set_window_monitor(self.window, monitor, 0, 0, mode.size.width, mode.size.height,
                                    mode.refresh_rate)
        else:
            glfw.set_window_monitor(self.window, None, 0, 0, W, H, 0)
            glfw.maximize_window(self.window)

    def on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_down = action == glfw.PRESS
            if action == glfw.PRESS and self.state == "title" and self.view.start_button_hit(self.mouse_layout()):
                self.start_game()

    def mouse_layout(self):
        """マウスの位置（論理座標 1280×720。拡大と黒い帯を戻したもの）"""
        return self.view.window_to_layout(*glfw.get_cursor_pos(self.window))

    # --------------------------------------------------------
    # 入力
    # --------------------------------------------------------
    def read_inputs(self, now, dt):
        """(手の状態, 本当の握り具合 g, 注視点, 色の塊, 視線の特徴量)"""
        if self.test_mode:
            m = self.mouse_layout()
            inside = 0 <= m[0] <= W and 0 <= m[1] <= H
            hand = self.mouse_hand.update(m[0] / W, 1.0 - m[1] / H, inside, now)
            g = 1.0 if self.mouse_down else 0.0            # クリック = 握る
            gaze = self.gaze.update(None, dt, mouse=m) if inside else None
            return hand, g, gaze, [], None
        vs = self.vision.get_state()
        if vs.hand_open is not None:
            self.last_g = 1.0 - vs.hand_open             # 手が一瞬見えなくなっても、握り具合は直前の値を保つ
        gaze = self.gaze.update(vs.gaze_feat, dt, feat_time=vs.gaze_time) if self.gaze.calibrated else None
        return vs.hand, self.last_g, gaze, vs.blobs, vs.gaze_feat

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

                selecting = self.update_state(dt, gaze, feat, blobs, now)
                self.robot.update(dt)

                # ---- 脳の光（「裏側」と結果画面の振り返り用）----
                #   視線で選ぶ → 前頭眼野・視覚野、手を動かす → 運動野、握る → 運動野 + 体性感覚野
                if self.state != "result":
                    motor = max(self.cortex.motor_drive, self.cortex.grip_drive)
                    self.brain.update_glow({"fef": selecting, "visual": selecting, "motor": motor, "hand": motor,
                                            "somato": self.cortex.grip_drive}, dt)
                if self.state == "play":
                    self.record.update(dt, self.brain.glow, self.cortex.metrics, self.mission.play_time)

                self.view.render(now - start, dt, gaze, fps)
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

    def update_state(self, dt, gaze, feat, blobs, now):
        """状態ごとの処理。視線で選んでいる強さ（脳の前頭眼野・視覚野の光）を返す"""
        s = self.state
        if s == "connect" and self.state_t >= CONNECT_SEC:
            self.set_state("hand_calib" if self.test_mode else "gaze_calib")
        elif s == "gaze_calib":
            if self.resume_play and self.mission is not None:
                self.mission.tick(dt)                    # プレイ中の視線同期のやり直しも、時間は進む
            if self.test_mode:
                self.gaze.calib.cancel()
                self.after_gaze_calib()
            elif self.gaze.calib.step(feat, dt):
                self.gaze.reset_filters()
                self.gaze.reset_offset()
                self.notify(f"視線同期 完了（ずれ 約 {self.gaze.calib.residual:.0f} 画素）")
                self.events.append(f"gaze_calib_ok:err_px={self.gaze.calib.residual:.0f}")
                self.after_gaze_calib()
            return 1.0
        elif s == "countdown" and self.state_t >= config.COUNTDOWN_SEC:
            self.set_state("play")
        elif s == "play":
            return self.update_play(dt, gaze, blobs, now)
        return 0.0

    def after_gaze_calib(self):
        if self.resume_play:
            self.resume_play = False
            self.state, self.state_t = "play", 0.0
        else:
            self.set_state("hand_calib")

    def finish_hand_calib(self):
        hc = self.hand_calib
        if not hc.enough:
            if hc.valid_sec < config.HC_MIN_SEC:
                msg = "同期に失敗：手がカメラに映っていない時間が長すぎます"
            else:
                msg = "同期に失敗：「握る！」「開く！」の合図に合わせて、しっかり握る・開くをしてください"
            self.events.append("hand_calib_failed")
            if self.state == "hand_calib":
                self.notify(msg + " → もう一度", 4.0)
                hc.start()
            else:
                self.notify(msg, 4.0)
                self.recalibrating = False
                self.record.mark(self.mission.play_time, "recal_end")
            return
        info = self.cortex.fit(*hc.data())
        lv = self.cortex.grip.levels
        self.events.append(f"hand_calib_ok:r2={info['r2']:.2f}:grip_sep={hc.grip_separation:.2f}"
                           + (f":grip_levels={lv[0]:.2f}/{lv[1]:.2f}" if lv else ""))
        self.grip_switch = GripSwitch()
        if self.recalibrating:
            self.recalibrating = False
            self.record.mark(self.mission.play_time, "recal_end")
            self.mission.log("recalibrate_ok")
            self.notify("再同期 完了：精度が戻った")
        elif self.state == "hand_calib":
            self.notify("同期 完了")
            self.set_state("countdown")

    # --------------------------------------------------------
    def update_play(self, dt, gaze, blobs, now):
        m = self.mission
        m.tick(dt)

        # ---- 電極劣化イベント ----
        if m.degrade_due():
            k = self.cortex.dropout(config.DEGRADE_FRACTION, config.DEGRADE_GRIP_FRACTION)
            m.log(f"degrade:{k}")
            self.record.mark(m.play_time, "degrade")
            self.events.append(f"degrade:{k}")
            self.flash_until = time.monotonic() + config.DEGRADE_FLASH_SEC
            self.scene.shake()
            self.notify("電極劣化！ アームが流されるようになった（L で再同期）", 5.0)

        # ---- ボーナス：現実の色の物を BONUS_SHOW_SEC 秒見せると降ってくる ----
        if not self.test_mode:
            for r in self.real_items.update(blobs, now, dt):
                if not r.used and now - r.first_seen >= config.BONUS_SHOW_SEC and m.bonus_allowed():
                    r.used = True
                    self.field.spawn("bonus", avoid=[self.robot.pos])
                    m.on_bonus()
                    self.events.append(f"bonus_drop:{r.color}")
                    self.notify("ボーナスが降ってきた！（高得点）")

        self.field.update(dt, avoid=[self.robot.pos])

        selecting = 0.0
        if not self.recalibrating:
            # ---- 視線で選ぶ：持っていないときは物だけ、持っているときはカゴだけ ----
            targets = []
            if m.held is None:
                for it in self.field.selectable():
                    p = self.scene.project(it.pos, it.radius, SCENE_RECT)
                    if p is not None:
                        targets.append((it.id, p, self.scene.screen_radius(it.pos, 0.0, it.radius, SCENE_RECT)))
            else:
                p = self.scene.project(GOAL_POS, 0.03, SCENE_RECT)
                if p is not None:
                    targets.append(("goal", p, self.scene.screen_radius(GOAL_POS, 0.0, config.GOAL_RADIUS, SCENE_RECT)))
            # 行き先を決めたあと、まだつかんでいない物から別の物へ乗り換えるには長く見つめる必要がある。
            # 決めた直後（LOCK_COOLDOWN 秒）は次の選択を受け付けない（視線の揺れで行き先がすぐ変わらないように）
            switching = m.held is None and isinstance(self.selected, int)
            dwell_sec = config.SWITCH_DWELL_SEC if switching else config.DWELL_SEC
            if now - self.lock_time >= config.LOCK_COOLDOWN:
                chosen = self.dwell.update(gaze, targets, dt, current=self.selected, dwell_sec=dwell_sec)
            else:
                chosen = None
                self.dwell.reset()
            if chosen is not None:
                self.select(chosen)
            if self.dwell.focus is not None:
                selecting = 0.4 + 0.6 * self.dwell.progress

            # ---- 手 → アーム（デコードした速度）、握る → グリッパー（デコードした把持）----
            if not self.robot.moving:
                self.robot.drive_arm(self.cortex.v_hat * config.ARM_VEL_GAIN, dt)
            gd = self.cortex.grip
            ev = self.grip_switch.update(gd.value, gd.close_th, gd.open_th)
            if ev is not None:
                self.on_grip_event(ev)
        else:
            self.robot.drive_arm(np.zeros(2), dt)

        if m.held is not None:                     # つかんでいる物はグリッパーといっしょに動く
            m.held.pos = gripper_tip(self.robot)

        if m.done:
            self.recalibrating = False
            self.hand_calib.cancel()
            self.set_state("result")
        return selecting

    def on_grip_event(self, ev):
        m = self.mission
        self.robot.set_grip(ev == "close")
        tip = gripper_tip(self.robot)
        self.events.append(f"grip_{ev}")
        if ev == "close":
            res = m.on_close(tip, self.field.items)
            if res == "miss":
                self.notify(f"つかみ損ね（コンボ ×{m.combo_mult:.1f} に下がった）")
            elif res is not None:
                self.events.append(f"grasp_ok:{res.type}")
                self.record.snapshot_raster(self.cortex.raster)
                self.notify(f"{res.spec['name']}をつかんだ！ カゴを見つめて運ぶ")
                self.selected = None
            return
        res = m.on_open(tip)
        if res is None:
            return
        kind, it, pts = res
        if kind == "deliver":
            self.field.remove(it)
            self.field.schedule_refill()
            self.popups.append((f"+{pts}", GOAL_POS.copy(), time.monotonic()))
            self.events.append(f"deliver_ok:{it.type}:+{pts}")
            self.notify(f"{it.spec['name']} +{pts}" + (f"（コンボ ×{m.combo_mult:.1f}）" if m.combo > 0 else ""))
        else:
            self.events.append(f"drop:{it.type}")
            self.notify(f"カゴの外に落とした（床に残っている。コンボ ×{m.combo_mult:.1f}）")
        self.selected = None

    def select(self, chosen):
        """視線で対象が決定されたとき：ずれを補正し、ロボットの台座をその近くへ移動させる"""
        if chosen == "goal":
            pos, h = GOAL_POS, 0.03
        else:
            it = next((i for i in self.field.items if i.id == chosen), None)
            if it is None:
                return
            pos, h = it.pos, it.radius
        if not self.test_mode:
            p = self.scene.project(pos, h, SCENE_RECT)
            if p is not None:
                self.gaze.correct(p)
        self.selected = chosen
        self.lock_time = time.monotonic()
        self.robot.approach(pos)
        self.robot.offset = None
        self.events.append(f"lock:{chosen}")

    def finish_game(self):
        """ゲーム終了：ハイスコアに加え、結果画面の振り返りを用意する"""
        m = self.mission
        if self.record.raster is None:
            self.record.snapshot_raster(self.cortex.raster)
        scores, rank = highscore.add(m.result.score, m.result.collected)
        self.final = {"result": m.result, "scores": scores, "rank": rank, "record": self.record,
                      "usage": self.record.usage()}
        self.events.append(f"game_end:score={m.result.score}:collected={m.result.collected}")
        print(f"[main] 終了：スコア {m.result.score}、集めた数 {m.result.collected}、失敗 {m.result.failures}、"
              f"順位 {None if rank is None else rank + 1}")
        # 結果画面の脳：プレイ中にどれだけ使ったかで光らせる
        self.brain.glow[:] = 0
        self.brain.update_glow(self.final["usage"], 10.0)

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
            self.notify("色を登録できませんでした：カメラの中央に色のはっきりした物を入れてください")
        else:
            config.ITEM_COLORS[name] = res[1]
            self.notify(f"色を登録しました（{name}）。この色の物を見せるとボーナスが降ってくる")

    def log_frame(self, t, gaze, g):
        m = self.mission if self.state == "play" else None
        h = self.hand
        held = None if m is None else m.held
        delta = None
        if m is not None:
            delta = m.result.score - self._prev_score
            self._prev_score = m.result.score
        self.logger.record(
            time_sec=t, state=self.state + ("+recal" if self.recalibrating else ""),
            mode="test" if self.test_mode else "camera",
            gaze_x=None if gaze is None else float(gaze[0]), gaze_y=None if gaze is None else float(gaze[1]),
            dwell_target=self.dwell.focus, dwell_progress=self.dwell.progress,
            hand_detected=int(h.detected), true_vx=float(h.vel[0]), true_vy=float(h.vel[1]),
            decoded_vx=float(self.cortex.v_hat[0]), decoded_vy=float(self.cortex.v_hat[1]),
            decoder="LIN" if self.cortex.calibrated else "PV", true_grip=float(g),
            decoded_grip=float(self.cortex.grip.value), gripper_closed=int(self.grip_switch.closed),
            holding=int(held is not None), holding_type=None if held is None else held.type,
            robot_x=float(self.robot.pos[0]), robot_y=float(self.robot.pos[1]),
            n_alive=self.cortex.n_alive, n_neurons=self.cortex.n_all,
            degrade_count=None if m is None else m.degrade_count,
            collected=None if m is None else m.result.collected, score=None if m is None else m.result.score,
            score_delta=delta, combo=None if m is None else m.combo_mult,
            time_left=None if m is None else m.time_left, event=";".join(self.events))
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
    parser.add_argument("--light", action="store_true", help="軽量モード（重い PC 向け）")
    parser.add_argument("--window", action="store_true", help="1280×720 のウィンドウで開く")
    args = parser.parse_args()
    if args.light or config.LIGHT_MODE:
        config.apply_light_mode()
    if args.window:
        config.WINDOW_MODE = "window"
    Game(args).run()


if __name__ == "__main__":
    main()
