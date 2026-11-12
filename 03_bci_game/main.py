"""
main.py
-------
BCI オペレーター（ゲーム本体）。実際の BCI 研究の「ハイブリッド BCI」をもとにしたゲーム。
  ・視線 → 行き先やつかむ物を選ぶ（前頭眼野・視覚野）              … gaze.py
  ・手の動き → 擬似ニューロンの発火 → デコード → アーム（運動野）    … cortex_sim.py（段階 3 でアームを操作）
  ・背景はカメラ映像（AR）。現実の色の物がゲーム内のアイテムになる    … vision.py / items.py
  ・横に本物の脳の 3D モデルを表示し、今使っている領域が光る          … brain_renderer.py

【段階 2 の時点の流れ】
  1. 脳と接続中：脳が pial から inflated へふくらむ（約 2 秒）
  2. 視線同期：AR 画面の 5 点（四隅 + 中央）を順に見て、視線のキャリブレーションをする
  3. 操作：アイテムやゴールを約 1 秒見つめると選ばれ、ロボットの台座がその近くへ移動する

使い方:
    cd 03_bci_game
    ../.venv/bin/python main.py            # カメラ（視線 + 手 + 色の物）
    ../.venv/bin/python main.py --test     # カメラなし：マウスの位置 = 視線、マウスの動き = 手
キー：G 注視点の表示、C 視線同期をやり直す、V 現実の色の物 ⇔ 仮想アイテム、R ロボットを初期位置へ、
      T テストモード、D デバッグ用カメラ窓、ESC 終了
"""

import argparse
import math
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
from brain_model import WARNING_TEXT, WARNING_TEXT_EN, BrainModel   # noqa: E402
from brain_renderer import REGION_COLORS, BrainView                 # noqa: E402
from cortex_sim import CortexSim                                    # noqa: E402
from gaze import DwellSelector, GazeTracker                         # noqa: E402
from gfx2d import Batch2D, TexturedQuad, rgb255                     # noqa: E402
from hand_tracker import MouseHand                                  # noqa: E402
from items import Goal, RealItemTracker, VirtualItems               # noqa: E402
from jtext import JText                                             # noqa: E402
from robot import Robot                                             # noqa: E402
from robot_view import RobotView                                    # noqa: E402
from vision import VisionThread                                     # noqa: E402

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
AX, AY, AW, AH = config.AR_RECT
BX, BY, BW, BH = config.BRAIN_RECT
PANEL_X = BX
DEBUG_WINDOW = "Camera Debug"
CONNECT_SEC = config.MORPH_SEC + 0.6


def world_to_screen(p):
    return np.array([AX + p[0] / config.WORLD_W * AW, AY + (1.0 - p[1] / config.WORLD_H) * AH])


def screen_to_world(q):
    return np.array([(q[0] - AX) / AW * config.WORLD_W, (1.0 - (q[1] - AY) / AH) * config.WORLD_H])


def calib_points():
    """視線同期の 5 点：AR 画面の四隅（少し内側）と中央"""
    m = config.GAZE_CALIB_INSET
    xs, ys = (AX + AW * m, AX + AW * (1 - m)), (AY + AH * m, AY + AH * (1 - m))
    return [(xs[0], ys[0]), (xs[1], ys[0]), (AX + AW / 2, AY + AH / 2), (xs[0], ys[1]), (xs[1], ys[1])]


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
        self.brain.camera.yaw, self.brain.camera.pitch, self.brain.camera.goal_dist = -100.0, 30.0, 3.7
        self.brain.camera.sway = (-105.0, 25.0)      # ゲーム中は左半球の外側面を見ながら ±25° 揺れる
        self.robot_view = RobotView()
        self.batch = Batch2D()
        self.ar_quad = TexturedQuad(gl.GL_LINEAR)
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.overlay_img = np.zeros((H, W, 4), dtype=np.uint8)
        self._last_overlay = -1.0
        self._ar_frame_id = -1
        self.jtext = JText()

        self.cortex = CortexSim(config.N_NEURONS, self.rng)
        self.brain.set_population(self.cortex.pop, self.rng)
        self.mouse_hand = MouseHand()
        self.gaze = GazeTracker(calib_points())
        self.dwell = DwellSelector()
        self.show_gaze = True
        self.robot = Robot()
        self.real_items = RealItemTracker()
        self.virtual = VirtualItems(self.rng)
        self.use_virtual = self.test_mode
        self.goal = Goal(np.array([0.84, 0.16]), 0.06)
        self.virtual.spawn(avoid=[config.ROBOT_START, self.goal.pos])
        self.selected = None          # 選ばれた対象の id（"goal" またはアイテムの id）
        self.message, self.message_until = "", 0.0

        self.state = "connect"        # connect → gaze_calib → play
        self.state_t = 0.0
        self.brain.set_surface("infl")   # 脳と接続中：pial → inflated へふくらむ

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

    # --------------------------------------------------------
    # 状態
    # --------------------------------------------------------
    def set_state(self, s):
        self.state, self.state_t = s, 0.0
        if s == "gaze_calib":
            self.gaze.calib.start()
            self.dwell.reset()

    def notify(self, text, sec=4.0):
        print(f"[main] {text}")
        self.message, self.message_until = text, time.monotonic() + sec

    def on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_G:
            self.show_gaze = not self.show_gaze
        elif key == glfw.KEY_C and self.state != "connect" and not self.test_mode:
            self.set_state("gaze_calib")
        elif key == glfw.KEY_V:
            self.use_virtual = not self.use_virtual
            if self.use_virtual:
                self.virtual.spawn(avoid=[self.robot.pos, self.goal.pos])
            self.selected = None
            self.notify("仮想アイテム" if self.use_virtual else "現実の色の物をアイテムにする")
        elif key == glfw.KEY_R:
            self.robot.reset()
            self.selected = None
        elif key == glfw.KEY_T and self.vision is not None:
            self.test_mode = not self.test_mode
            self.notify("テストモード（マウス）" if self.test_mode else "カメラモード")
            if not self.test_mode and not self.gaze.calibrated and self.state == "play":
                self.set_state("gaze_calib")
        elif key == glfw.KEY_D and self.vision is not None:
            self.debug_on = not self.debug_on
            self.vision.debug_enabled = self.debug_on
            if not self.debug_on:
                cv2.destroyWindow(DEBUG_WINDOW)
                cv2.waitKey(1)

    # --------------------------------------------------------
    # 入力
    # --------------------------------------------------------
    def mouse_layout(self):
        mx, my = glfw.get_cursor_pos(self.window)
        ww, wh = glfw.get_window_size(self.window)
        return np.array([mx * W / ww, my * H / wh])

    def read_inputs(self, now, dt):
        """(手の状態, 手の開き具合, 注視点, 色の塊のリスト)"""
        if self.test_mode:
            m = self.mouse_layout()
            inside = AX <= m[0] <= AX + AW and AY <= m[1] <= AY + AH
            hand = self.mouse_hand.update((m[0] - AX) / AW, 1.0 - (m[1] - AY) / AH, inside, now)
            gaze = self.gaze.update(None, dt, mouse=m) if inside else None
            return hand, None, gaze, []
        vs = self.vision.get_state()
        gaze = self.gaze.update(vs.gaze_feat, dt) if self.gaze.calibrated else None
        return vs.hand, vs.hand_open, gaze, vs.blobs

    def current_items(self, blobs, now, dt):
        if self.use_virtual:
            return self.virtual.items
        return self.real_items.update(blobs, now, dt)

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

                feat = None if self.test_mode else self.vision.get_state().gaze_feat
                hand, hand_open, gaze, blobs = self.read_inputs(now, dt)
                items = self.current_items(blobs, now, dt)

                # ---- 運動野のシミュレーション（20 ms 刻み）----
                acc += dt
                steps = 0
                while acc >= config.DT and steps < config.MAX_STEPS_PER_FRAME:
                    self.cortex.step(hand.vel)
                    acc -= config.DT
                    steps += 1
                if steps == config.MAX_STEPS_PER_FRAME:
                    acc = 0.0
                self.cortex.decay(dt)

                # ---- 状態ごとの処理 ----
                selecting = 0.0
                if self.state == "connect":
                    if self.state_t >= CONNECT_SEC:
                        self.set_state("play" if self.test_mode else "gaze_calib")
                elif self.state == "gaze_calib":
                    selecting = 1.0
                    if self.test_mode:
                        self.gaze.calib.cancel()
                        self.set_state("play")
                    elif self.gaze.calib.step(feat, dt):
                        self.gaze.filter.reset()
                        self.notify(f"視線同期 完了（ずれ 約 {self.gaze.calib.residual:.0f} 画素）")
                        self.set_state("play")
                elif self.state == "play":
                    targets = [("goal", world_to_screen(self.goal.pos), self.goal.radius / config.WORLD_W * AW)]
                    targets += [(it.id, world_to_screen(it.pos), it.radius / config.WORLD_W * AW) for it in items]
                    chosen = self.dwell.update(gaze, targets, dt, exclude=self.selected)
                    if chosen is not None:
                        self.select(chosen, items)
                    if self.dwell.focus is not None:
                        selecting = 0.4 + 0.6 * self.dwell.progress
                self.robot.update(dt)

                # ---- 脳の光：視線で選ぶ → 前頭眼野・視覚野、手を動かす → 運動野 ----
                motor = self.cortex.motor_drive
                self.brain.update_glow({"fef": selecting, "visual": selecting, "motor": motor, "hand": motor}, dt)

                self.render(now - start, dt, hand, hand_open, gaze, items, fps)
                self.update_debug_window()
        finally:
            if self.vision is not None:
                self.vision.stop()
            cv2.destroyAllWindows()
            glfw.terminate()

    def select(self, chosen, items):
        """視線で対象が決定されたとき：ロボットの台座をその近くへ移動させる"""
        if chosen == "goal":
            self.selected = "goal"
            self.robot.approach(self.goal.pos)
            self.notify("ゴールをロックオン → 移動")
            return
        it = next((i for i in items if i.id == chosen), None)
        if it is not None:
            self.selected = it.id
            self.robot.approach(it.pos)
            self.notify(f"アイテム（{it.color}）をロックオン → 移動")

    def update_debug_window(self):
        if not self.debug_on or self.vision is None:
            return
        frame = self.vision.pop_debug_frame()
        if frame is not None:
            cv2.imshow(DEBUG_WINDOW, frame)
        cv2.waitKey(1)

    # --------------------------------------------------------
    # 描画
    # --------------------------------------------------------
    def render(self, t, dt, hand, hand_open, gaze, items, fps):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w == 0 or fb_h == 0:
            return
        sx, sy = fb_w / W, fb_h / H
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.025, 0.03, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        b = self.batch

        # ---- AR の背景（カメラ映像）----
        if not self.test_mode:
            fid, frame = self.vision.get_frame()
            if frame is not None:
                if fid != self._ar_frame_id:
                    self.ar_quad.upload(frame)
                    self._ar_frame_id = fid
                self.ar_quad.draw(AX, AY, AW, AH)
            b.rect(AX, AY, AW, AH, (0.0, 0.0, 0.03, 1.0 - config.AR_DIM))
        else:
            b.rect(AX, AY, AW, AH, (0.05, 0.06, 0.09, 1.0))
            for k in range(1, 8):
                b.line(AX + AW * k / 8, AY, AX + AW * k / 8, AY + AH, 1, (0.12, 0.14, 0.2, 1.0))
            for k in range(1, 6):
                b.line(AX, AY + AH * k / 6, AX + AW, AY + AH * k / 6, 1, (0.12, 0.14, 0.2, 1.0))

        # ---- 地面の上のもの（ゴール、ロボットの影）----
        gp = world_to_screen(self.goal.pos)
        gr = self.goal.radius / config.WORLD_W * AW
        b.circle(*gp, gr, (0.2, 1.0, 0.5, 0.12))
        b.ring(*gp, gr, 3, (0.3, 1.0, 0.55, 0.9))
        b.line(gp[0] - gr * 0.5, gp[1], gp[0] + gr * 0.5, gp[1], 2, (0.3, 1.0, 0.55, 0.7))
        b.line(gp[0], gp[1] - gr * 0.5, gp[0], gp[1] + gr * 0.5, 2, (0.3, 1.0, 0.55, 0.7))
        rp = world_to_screen(self.robot.pos)
        b.circle(rp[0] + 6, rp[1] + 8, 52, (0, 0, 0, 0.35), 24)
        if self.robot.goal is not None:      # 目的地への点線
            q = world_to_screen(self.robot.goal)
            for k in range(0, 12, 2):
                b.line(*(rp + (q - rp) * k / 12), *(rp + (q - rp) * (k + 1) / 12), 2, (0.4, 0.85, 1.0, 0.6))
        b.flush()

        # ---- ロボットと仮想アイテム（3D）----
        self.robot_view.draw(self.robot, items, (int(AX * sx), int((H - AY - AH) * sy), int(AW * sx), int(AH * sy)))
        gl.glViewport(0, 0, fb_w, fb_h)

        # ---- 上に重ねるもの：アイテムの枠、ドウェルのリング、注視点、視線同期の点 ----
        for it in items:
            p = world_to_screen(it.pos)
            r = it.radius / config.WORLD_W * AW + 8
            c = it.rgb
            if it.kind == "real":
                b.ring(*p, r, 2.5, (*c, 0.95))
                for k in range(4):                      # 照準のような 4 本の線
                    a = k * math.pi / 2 + t * 0.8
                    b.line(p[0] + math.cos(a) * r, p[1] + math.sin(a) * r,
                           p[0] + math.cos(a) * (r + 10), p[1] + math.sin(a) * (r + 10), 2, (*c, 0.95))
            if self.selected == it.id:
                b.ring(*p, r + 14, 2, (1.0, 1.0, 1.0, 0.8))
        if self.selected == "goal":
            b.ring(*gp, gr + 12, 2, (1.0, 1.0, 1.0, 0.8))
        if self.state == "play" and self.dwell.focus is not None and self.dwell.focus_pos is not None:
            p = self.dwell.focus_pos
            self._progress_ring(p, 40, self.dwell.progress, (1.0, 0.85, 0.3))
        if self.state == "gaze_calib":
            self._draw_calib(t)
        if self.show_gaze and gaze is not None and self.state != "gaze_calib":
            b.circle(*gaze, 6, (1.0, 1.0, 1.0, 0.85), 12)
            b.ring(*gaze, 11, 1.5, (0.2, 0.9, 1.0, 0.8), 20)
        b.rect(AX - 1, AY - 1, AW + 2, 1, (0.3, 0.4, 0.55, 1))
        b.rect(AX - 1, AY + AH, AW + 2, 1, (0.3, 0.4, 0.55, 1))
        b.rect(AX - 1, AY, 1, AH, (0.3, 0.4, 0.55, 1))
        b.rect(AX + AW, AY, 1, AH, (0.3, 0.4, 0.55, 1))
        rx, ry, rw, rh = config.RASTER_RECT
        b.rect(rx - 1, ry - 1, rw + 2, rh + 2, (0.25, 0.3, 0.4, 1.0))
        b.rect(BX, BY, BW, BH, (0.035, 0.04, 0.065, 1.0))
        b.flush()

        # ---- 脳 ----
        self.brain.draw((int(BX * sx), int((H - BY - BH) * sy), int(BW * sx), int(BH * sy)), t, dt,
                        self.cortex.flash, self.cortex.activity, self.cortex.pop.alive)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glEnable(gl.GL_BLEND)

        # ---- ラスター ----
        self.raster_quad.upload(self.cortex.raster)
        self.raster_quad.draw(rx, ry, rw, rh)

        # ---- 文字 ----
        if t - self._last_overlay >= config.OVERLAY_INTERVAL:
            self._draw_overlay(t, hand, hand_open, gaze, items, fps)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        glfw.swap_buffers(self.window)

    def _progress_ring(self, p, r, progress, color):
        """見つめている間にたまっていくリング（progress 0〜1）"""
        b = self.batch
        b.ring(*p, r, 3, (*color, 0.25), 48)
        seg = max(1, int(48 * progress))
        for k in range(seg):
            a0 = -math.pi / 2 + 2 * math.pi * k / 48
            a1 = -math.pi / 2 + 2 * math.pi * (k + 1) / 48
            b.line(p[0] + r * math.cos(a0), p[1] + r * math.sin(a0),
                   p[0] + r * math.cos(a1), p[1] + r * math.sin(a1), 5, (*color, 0.95))

    def _draw_calib(self, t):
        """視線同期：今の点を大きく光らせ、集めている間はリングがたまる"""
        b = self.batch
        b.rect(AX, AY, AW, AH, (0.0, 0.02, 0.05, 0.55))
        cal = self.gaze.calib
        for i, p in enumerate(cal.points):
            done = i < cal.index
            b.circle(*p, 5, (0.3, 1.0, 0.6, 0.9) if done else (0.5, 0.55, 0.65, 0.5), 12)
        p = cal.current_point
        if p is None:
            return
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        shrink = 1.0 - min(1.0, cal.t / config.GAZE_CALIB_SETTLE)
        b.ring(*p, 16 + 40 * shrink, 2, (0.4, 0.9, 1.0, 0.6 + 0.3 * pulse), 40)
        b.circle(*p, 7, (1.0, 1.0, 1.0, 1.0), 16)
        if cal.collecting:
            self._progress_ring(p, 26, (cal.t - config.GAZE_CALIB_SETTLE) / config.GAZE_CALIB_COLLECT,
                                (0.4, 0.9, 1.0))

    @staticmethod
    def _text(img, text, x, y, scale=0.45, color=(225, 230, 240), thick=1):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), thick, cv2.LINE_AA)

    def _jt(self, img, ja, en, x, y, size=18, color=(255, 255, 255), anchor="left"):
        """日本語が書ければ日本語、書けなければ英語（cv2）で書く"""
        if self.jtext.available:
            self.jtext.draw(img, ja, x, y, size, color, anchor)
        else:
            if anchor == "center":
                (tw, _), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)
                x -= tw / 2
            self._text(img, en, x, y + size * 0.8, size / 30, color)

    def _draw_overlay(self, t, hand, hand_open, gaze, items, fps):
        img = self.overlay_img
        img[:] = 0
        T = self._text
        dim = (150, 160, 185)

        # ---- 状態ごとの大きな表示 ----
        cx, cy = AX + AW / 2, AY + AH / 2
        if self.state == "connect":
            k = min(1.0, self.state_t / config.MORPH_SEC)
            cv2.rectangle(img, (int(cx - 230), int(cy - 50)), (int(cx + 230), int(cy + 50)), (6, 10, 24, 210), -1)
            self._jt(img, "脳と接続中…", "LINKING TO THE BRAIN...", cx, cy - 36, 32, (130, 220, 255), "center")
            cv2.rectangle(img, (int(cx - 180), int(cy + 22)), (int(cx + 180), int(cy + 30)), (60, 70, 90, 255), -1)
            cv2.rectangle(img, (int(cx - 180), int(cy + 22)), (int(cx - 180 + 360 * k), int(cy + 30)),
                          (130, 220, 255, 255), -1)
        elif self.state == "gaze_calib":
            cal = self.gaze.calib
            cv2.rectangle(img, (int(cx - 260), int(AY + 18)), (int(cx + 260), int(AY + 92)), (6, 10, 24, 210), -1)
            self._jt(img, f"視線同期  {min(cal.index + 1, 5)} / 5", f"GAZE SYNC  {min(cal.index + 1, 5)} / 5",
                     cx, AY + 24, 28, (130, 220, 255), "center")
            face = self.vision.get_state().face_detected if self.vision else False
            self._jt(img, "光る点を見つめてください（頭は楽に）" if face else "顔が見つかりません：カメラの正面へ",
                     "Look at the glowing dot" if face else "Face not found: face the camera",
                     cx, AY + 62, 18, (220, 225, 235) if face else (255, 140, 120), "center")

        # ---- 対象の名前 ----
        for it in items:
            p = world_to_screen(it.pos)
            r = it.radius / config.WORLD_W * AW + 22
            T(img, f"{it.color} {'(real)' if it.kind == 'real' else ''}", p[0] + r * 0.7, p[1] - r * 0.7, 0.45,
              rgb255(it.rgb))
        gp = world_to_screen(self.goal.pos)
        T(img, "GOAL", gp[0] - 20, gp[1] + self.goal.radius / config.WORLD_W * AW + 18, 0.5, (90, 255, 140))

        if self.message and time.monotonic() < self.message_until:
            self._jt(img, self.message, self.message, AX + AW / 2, AY + AH - 40, 20, (255, 255, 255), "center")

        # ---- 脳の凡例 ----
        T(img, "BRAIN (left hemisphere = right hand)", BX + 8, BY + 18, 0.42, dim)
        y = BY + BH - 52
        for name, label in (("motor", "motor (hand)"), ("somato", "somatosensory"), ("fef", "frontal eye field"),
                            ("visual", "visual")):
            from brain_model import REGION_ID
            g = self.brain.glow[REGION_ID[name]]
            c = rgb255(REGION_COLORS[name])
            x = BX + 10 + (0 if name in ("motor", "fef") else 190)
            yy = y + (0 if name in ("motor", "somato") else 22)
            cv2.rectangle(img, (x, yy - 10), (x + 12, yy + 2), (*c, 255) if g > 0.3 else (70, 75, 90, 255), -1)
            T(img, label, x + 18, yy, 0.42, c if g > 0.3 else dim)
        if self.model.simplified:
            cv2.rectangle(img, (BX, BY + 26), (BX + BW, BY + 50), (90, 30, 10, 220), -1)
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", BX + 6, BY + 28, 15, (255, 210, 120))

        # ---- HUD（右下）----
        x, y = PANEL_X, BY + BH + 30
        T(img, "BCI OPERATOR", x, y, 0.7, (255, 255, 255), 2)
        T(img, "stage 2 preview: gaze + AR + robot", x, y + 22, 0.42, dim)
        mode = "TEST MODE (mouse = gaze & hand)" if self.test_mode else "CAMERA MODE"
        T(img, mode, x, y + 48, 0.48, (255, 200, 110) if self.test_mode else (140, 230, 160))
        vs = None if self.test_mode else self.vision.get_state()
        if vs is not None:
            T(img, f"camera {vs.fps:4.1f} fps  (face {vs.face_fps:4.1f} / hand {vs.hand_fps:4.1f})", x, y + 72, 0.45)
            gz = (f"gaze: calibrated (err ~{self.gaze.calib.residual:.0f} px)" if self.gaze.calibrated
                  else "gaze: not calibrated")
            T(img, gz if vs.face_detected else "gaze: face NOT FOUND", x, y + 94, 0.45,
              (140, 230, 160) if vs.face_detected and self.gaze.calibrated else (255, 140, 120))
            ho = "-" if hand_open is None else f"{hand_open:.2f}"
            T(img, f"hand: {'detected' if hand.detected else 'NOT FOUND'}   open {ho}", x, y + 116, 0.45,
              (140, 230, 160) if hand.detected else (255, 140, 120))
        else:
            T(img, "gaze = mouse position, hand = mouse motion", x, y + 72, 0.45)
        T(img, f"items: {len(items)} ({'virtual' if self.use_virtual else 'real color objects'})", x, y + 138, 0.45)
        st = "moving" if self.robot.moving else "idle"
        T(img, f"robot: {st}   motor drive {self.cortex.motor_drive:.2f}", x, y + 160, 0.45)
        T(img, f"render {fps:4.1f} fps", x, y + 182, 0.42, dim)
        T(img, "look ~1 s at an item / GOAL to move the robot", x, y + 212, 0.42, (220, 225, 235))
        T(img, "G gaze dot  C gaze sync  V virtual/real items", x, y + 236, 0.4, dim)
        T(img, "R reset robot  T test mode  D debug  ESC quit", x, y + 256, 0.4, dim)

        rx, ry, rw, rh = config.RASTER_RECT
        T(img, "M1 hand-knob spikes", rx + 4, ry + 14, 0.38, (200, 205, 220))


def main():
    parser = argparse.ArgumentParser(description="BCI オペレーター")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--test", action="store_true", help="カメラを使わない（マウスで視線と手の代わり）")
    Game(parser.parse_args()).run()


if __name__ == "__main__":
    main()
