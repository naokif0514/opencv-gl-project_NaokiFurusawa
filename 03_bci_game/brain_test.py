"""
brain_test.py
-------------
段階 1 の確認用テスト画面：本物の脳モデルと、その上の擬似ニューロンを確かめる。

  ・脳（fsaverage5）を回転表示し、キーで各領域（運動野・体性感覚野・前頭眼野・視覚野）を光らせる
  ・02 と同じように、手（カメラ）またはマウスを動かすと、左半球の手の領域に置いた擬似ニューロンが発火して光り、
    運動野も手の動きの強さに応じて光る

使い方:
    cd 03_bci_game
    ../.venv/bin/python brain_test.py          # カメラで手を追跡（なければマウス）
    ../.venv/bin/python brain_test.py --test   # マウスを手の代わりにする

キー：1 運動野 / 2 体性感覚野 / 3 前頭眼野 / 4 視覚野（押すたびに点灯・消灯）、0 すべて消す、
      I pial ⇔ inflated（変形アニメーション）、Z 手の領域に寄る・戻る、↑/↓ ニューロン数、N/M ノイズ、K 脱落、T テストモード、D デバッグ窓、ESC 終了
マウス：脳の上でドラッグ = 回転、ホイール = 拡大縮小。テストモードでは右下の「hand pad」の上でマウスを動かす
"""

import argparse
import math
import os
import sys
import time

# Wayland 環境（Ubuntu の標準）でも安定して動くよう、ウィンドウは X11 互換モード（XWayland）で開く
# （02_pseudo_bci/main.py と同じ設定）
#   QT_QPA_PLATFORM        : OpenCV のウィンドウ（Qt）用
#   PYGLFW_LIBRARY_VARIANT : glfw 用（Wayland 版は GNOME の設定と合わず異常終了することがある）
#   PYOPENGL_PLATFORM      : PyOpenGL 用（X11 版 glfw に合わせて GLX を使う）
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import cv2           # noqa: E402
import glfw          # noqa: E402
import numpy as np   # noqa: E402
from OpenGL import GL as gl   # noqa: E402

import config        # noqa: E402
from brain_model import REGION_ID, WARNING_TEXT, WARNING_TEXT_EN, BrainModel, vertex_normals   # noqa: E402
from brain_renderer import REGION_COLORS, BrainView            # noqa: E402
from decoder import PopulationVectorDecoder, RateEstimator     # noqa: E402
from gfx2d import Batch2D, TexturedQuad, rgb255                # noqa: E402
from gl_utils import hue_to_rgb                                # noqa: E402
from hand_tracker import HandTracker, MouseHand                # noqa: E402
from jtext import JText                                        # noqa: E402
from neuron_model import NeuronPopulation                      # noqa: E402

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
BRAIN_VP = (0, 0, 860, 560)                  # 脳の表示 (x, y, 幅, 高さ)、左上原点
PANEL_X = 880                                # 右のパネル
PAD = (PANEL_X + 150, 330, 220, 220)         # テストモードの hand pad (x, y, 幅, 高さ)
COMPASS = (PANEL_X + 55, 390, 42)
WHEEL = (PANEL_X + 55, 525, 32)
RASTER = (60, 592, W - 80, 100)
DEBUG_WINDOW = "Hand Tracking Debug"

# キー → 領域（手動で光らせる）
REGION_KEYS = {glfw.KEY_1: "motor", glfw.KEY_2: "somato", glfw.KEY_3: "fef", glfw.KEY_4: "visual"}
REGION_LABELS = {"motor": "motor cortex (M1)", "hand": "hand knob", "somato": "somatosensory (S1)",
                 "fef": "frontal eye field", "visual": "visual cortex"}


class BrainTest:
    def __init__(self, args):
        self.rng = np.random.default_rng()
        print("[brain_test] 脳モデルを読み込みます")
        self.model = BrainModel()

        # ---- カメラ（見つからなければテストモード）----
        self.tracker = None
        if not args.test:
            tr = HandTracker(args.camera)
            if tr.open():
                self.tracker = tr
                tr.start()
            else:
                print("[brain_test] カメラを開けませんでした → テストモード（マウス）", file=sys.stderr)
        self.test_mode = self.tracker is None
        self.mouse = MouseHand()
        self.debug_on = self.tracker is not None
        if self.tracker is not None:
            self.tracker.debug_enabled = True

        # ---- ウィンドウ ----
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
        print(f"[brain_test] {gl.glGetString(gl.GL_RENDERER).decode()}")
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)

        self.brain = BrainView(self.model)
        self.batch = Batch2D()
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.overlay_img = np.zeros((H, W, 4), dtype=np.uint8)
        self._last_overlay = -1.0
        self.jtext = JText()
        self._region_anchor()

        self.manual = set()           # キーで点灯させている領域
        self.n = config.N_NEURONS
        self.noise = config.NOISE_STD
        self.build_population()

    # --------------------------------------------------------
    def _region_anchor(self):
        """
        各領域の代表点と向き（ラベルを書く位置）。左半球の領域の重心を、pial と inflated の両方で求めておき、
        描くときに今の変形の割合で混ぜる（変形中もラベルが表面に乗ったまま動く）
        """
        h = self.model.hemis["left"]
        self.anchors = {}
        for name in REGION_COLORS:
            idx = self.model.region_vertices("left", name)
            pair = []
            for surf in ("pial", "infl"):
                v = self.brain.surface_coords("left", surf)
                n = self.brain.dir_to_gl(vertex_normals(v, h["faces"])[idx].mean(axis=0))
                pair.append((self.brain.to_gl(v[idx].mean(axis=0)), n / np.linalg.norm(n)))
            self.anchors[name] = pair

    def build_population(self):
        self.pop = NeuronPopulation(self.n, self.rng)
        self.pop.noise_std = self.noise
        self.rates = RateEstimator(self.n)
        self.pv = PopulationVectorDecoder(self.pop)
        self.brain.set_population(self.pop, self.rng)
        self.flash = np.zeros(self.n)
        self.activity = np.zeros(self.n)
        self.motor_drive = 0.0
        bins = int(round(config.RASTER_SEC / config.DT))
        self.raster_bg = np.array([14, 16, 26, 255], dtype=np.uint8)
        self.raster = np.empty((self.n, bins, 4), dtype=np.uint8)
        self.raster[:] = self.raster_bg
        self.raster_colors = (hue_to_rgb(self.pop.pd_angle / (2 * np.pi))[::-1] * 255).astype(np.uint8)
        print(f"[brain_test] ニューロン {self.n} 個を左半球の手の領域に配置しました")

    # --------------------------------------------------------
    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key in REGION_KEYS and action == glfw.PRESS:
            self.manual ^= {REGION_KEYS[key]}
        elif key == glfw.KEY_0:
            self.manual.clear()
        elif key == glfw.KEY_Z and action == glfw.PRESS:
            self.brain.toggle_focus()
        elif key == glfw.KEY_I and action == glfw.PRESS:
            self.brain.toggle_surface()
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
        elif key == glfw.KEY_K and action == glfw.PRESS:
            self.pop.dropout()
        elif key == glfw.KEY_T and action == glfw.PRESS and self.tracker is not None:
            self.test_mode = not self.test_mode
        elif key == glfw.KEY_D and action == glfw.PRESS and self.tracker is not None:
            self.debug_on = not self.debug_on
            self.tracker.debug_enabled = self.debug_on
            if not self.debug_on:
                cv2.destroyWindow(DEBUG_WINDOW)
                cv2.waitKey(1)

    def _win_to_layout(self, x, y):
        ww, wh = glfw.get_window_size(self.window)
        return x * W / ww, y * H / wh

    def on_mouse_button(self, window, button, action, mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = self._win_to_layout(*glfw.get_cursor_pos(window))
        bx, by, bw, bh = BRAIN_VP
        if action == glfw.PRESS and bx <= x < bx + bw and by <= y < by + bh:
            self.brain.camera.mouse_down(x, y)
        elif action == glfw.RELEASE:
            self.brain.camera.mouse_up()

    def on_cursor(self, window, x, y):
        self.brain.camera.mouse_move(*self._win_to_layout(x, y))

    def on_scroll(self, window, dx, dy):
        self.brain.camera.scroll(dy)

    # --------------------------------------------------------
    def read_hand(self, now):
        if self.test_mode:
            mx, my = self._win_to_layout(*glfw.get_cursor_pos(self.window))
            px, py, pw, ph = PAD
            inside = px <= mx <= px + pw and py <= my <= py + ph
            return self.mouse.update((mx - px) / pw, 1.0 - (my - py) / ph, inside, now)
        return self.tracker.get_state()

    def sim_step(self, hand):
        counts = self.pop.step(hand.vel)
        rate_hat = self.rates.step(counts)
        self.pv.step(rate_hat)
        fired = counts > 0
        self.flash[fired] = 1.0
        act = np.clip((rate_hat - config.BASE_RATE) / config.MODULATION, 0.0, 1.0)
        self.activity += 0.25 * (act - self.activity)
        # 運動野の光：手の領域のニューロン集団がどれだけベースラインより発火しているか
        self.motor_drive = float(np.clip(config.MOTOR_GLOW_GAIN * act[self.pop.alive].mean(), 0, 1)) \
            if self.pop.n_alive else 0.0
        # ラスター（02 と同じ：1 列左へずらし、右端に今回のスパイクを書く）
        self.raster[:, :-1] = self.raster[:, 1:]
        col = np.empty((self.n, 4), dtype=np.uint8)
        col[:] = self.raster_bg
        col[(~self.pop.alive)[::-1]] = (34, 34, 38, 255)
        rows = fired[::-1]
        col[rows, :3] = self.raster_colors[rows]
        self.raster[:, -1] = col

    # --------------------------------------------------------
    def run(self):
        prev = time.monotonic()
        start = prev
        acc, fps = 0.0, 0.0
        try:
            while not glfw.window_should_close(self.window):
                now = time.monotonic()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.95 * fps + 0.05 / dt if fps > 0 else 1.0 / dt
                glfw.poll_events()
                hand = self.read_hand(now)
                acc += dt
                steps = 0
                while acc >= config.DT and steps < config.MAX_STEPS_PER_FRAME:
                    self.sim_step(hand)
                    acc -= config.DT
                    steps += 1
                if steps == config.MAX_STEPS_PER_FRAME:
                    acc = 0.0
                self.flash *= math.exp(-dt / config.SPIKE_FLASH_TAU)

                targets = {name: 1.0 for name in self.manual}
                motor = max(targets.get("motor", 0.0), self.motor_drive)
                targets["motor"] = motor
                targets["hand"] = max(motor, self.motor_drive)
                self.brain.update_glow(targets, dt)
                self.render(now - start, dt, hand, fps)
                self.update_debug_window()
        finally:
            if self.tracker is not None:
                self.tracker.stop()
            cv2.destroyAllWindows()
            glfw.terminate()

    def update_debug_window(self):
        if not self.debug_on or self.tracker is None:
            return
        frame = self.tracker.pop_debug_frame()
        if frame is not None:
            cv2.imshow(DEBUG_WINDOW, frame)
        cv2.waitKey(1)

    # --------------------------------------------------------
    def render(self, t, dt, hand, fps):
        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w == 0 or fb_h == 0:
            return
        sx, sy = fb_w / W, fb_h / H
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.025, 0.03, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        bx, by, bw, bh = BRAIN_VP
        vp = (int(bx * sx), int((H - by - bh) * sy), int(bw * sx), int(bh * sy))
        self.brain.draw(vp, t, dt, self.flash, self.activity, self.pop.alive)

        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        b = self.batch
        b.line(PANEL_X - 10, 10, PANEL_X - 10, 570, 1, (0.25, 0.3, 0.4, 1.0))
        rx, ry, rw, rh = RASTER
        b.rect(rx - 2, ry - 2, rw + 4, rh + 4, (0.25, 0.3, 0.4, 1.0))
        # hand pad（テストモード）
        px, py, pw, ph = PAD
        b.rect(px, py, pw, ph, (0.06, 0.07, 0.11, 1.0) if self.test_mode else (0.04, 0.045, 0.06, 1.0))
        if hand.pos is not None:
            hx, hy = px + hand.pos[0] * pw, py + (1 - hand.pos[1]) * ph
            if 0 <= hand.pos[0] <= 1 and 0 <= hand.pos[1] <= 1:
                b.ring(hx, hy, 10, 2.5, (1, 1, 1, 0.7))
        # 速度（白 = 本当、黄 = PV 法で推定）
        cx, cy, r = COMPASS
        b.circle(cx, cy, r, (0.06, 0.07, 0.11, 1.0))
        b.ring(cx, cy, r, 1.5, (0.3, 0.35, 0.45, 1.0))
        for v, color, w in ((hand.vel, (1, 1, 1, 0.45), 7), (self.pv.v_hat, (1.0, 0.85, 0.2, 1.0), 2.5)):
            vx, vy = np.clip(v, -1.2, 1.2)
            b.arrow(cx, cy, cx + vx * r / 1.2, cy - vy * r / 1.2, w, color)
        # PD の色の凡例
        cx, cy, r = WHEEL
        for k in range(36):
            a0, a1 = 2 * np.pi * k / 36, 2 * np.pi * (k + 1) / 36
            c = hue_to_rgb((a0 + a1) / 2 / (2 * np.pi))
            p0 = (cx + r * math.cos(a0), cy - r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy - r * math.sin(a1))
            q0 = (cx + 0.6 * r * math.cos(a0), cy - 0.6 * r * math.sin(a0))
            q1 = (cx + 0.6 * r * math.cos(a1), cy - 0.6 * r * math.sin(a1))
            b._tri([p0, p1, q1, p0, q1, q0], (*c, 1.0))
        b.flush()

        self.raster_quad.upload(self.raster)
        self.raster_quad.draw(rx, ry, rw, rh)

        if t - self._last_overlay >= config.OVERLAY_INTERVAL:
            self._draw_overlay(hand, fps, vp)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        glfw.swap_buffers(self.window)

    @staticmethod
    def _text(img, text, x, y, scale=0.45, color=(225, 230, 240), thick=1):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), thick, cv2.LINE_AA)

    def _draw_overlay(self, hand, fps, vp):
        img = self.overlay_img
        img[:] = 0
        T = self._text
        dim = (150, 160, 185)

        # ---- 光っている領域のラベル（脳の上。こちらを向いている領域だけ）----
        bx, by, bw, bh = BRAIN_VP
        eye = self.brain.camera.eye
        for name, ((p0, n0), (p1, n1)) in self.anchors.items():
            p, n = self.brain.mix(p0, p1), self.brain.mix(n0, n1)
            g = self.brain.glow[REGION_ID[name]]
            if g < 0.3 or name == "hand":
                continue
            if np.dot(n, eye - p) <= 0:
                continue
            q = self.brain.project(p, (0, 0, bw, bh))
            if q is None:
                continue
            x, y = bx + q[0], by + q[1]
            c = rgb255(REGION_COLORS[name])
            cv2.circle(img, (int(x), int(y)), 3, (*c, 255), -1, cv2.LINE_AA)
            label = REGION_LABELS[name]
            # 運動野と体性感覚野は隣り合っているので、運動野（前側）のラベルは点の左、ほかは右に書く
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            front = name in ("motor", "fef")
            lx = x - tw - 8 if front else x + 8
            ly = y + 16 if name == "motor" else y - 6
            cv2.rectangle(img, (int(lx) - 3, int(ly) - 15), (int(lx + tw) + 3, int(ly) + 5), (8, 10, 20, 170), -1)
            T(img, label, lx, ly, 0.5, c, 1)

        # ---- 右のパネル ----
        x = PANEL_X
        T(img, "BRAIN TEST (stage 1)", x, 32, 0.6, (255, 255, 255))
        surf = "pial" if self.brain.morph_goal < 0.5 else "inflated"
        if self.brain.morphing:
            surf = f"-> {surf} {self.brain.morph:.0%}" if self.brain.morph_goal > 0.5 else \
                f"-> {surf} {1 - self.brain.morph:.0%}"
        model = "simplified model" if self.model.simplified else "fsaverage5+Destrieux"
        T(img, f"{model}  surface: {surf}", x, 54, 0.42, dim)
        mode = "TEST MODE (mouse on hand pad)" if self.test_mode else "CAMERA MODE (hand tracking)"
        T(img, mode, x, 80, 0.45, (255, 200, 110) if self.test_mode else (140, 230, 160))
        T(img, "hand: detected" if hand.detected else "hand: NOT FOUND (v = 0)", x, 102, 0.45,
          (140, 230, 160) if hand.detected else (255, 120, 120))
        T(img, f"neurons: {self.pop.n_alive} / {self.pop.n}  (left hand knob)", x, 124)
        T(img, f"noise: {self.noise:.0f} Hz   motor drive: {self.motor_drive:.2f}", x, 146)

        y = 176
        T(img, "regions (key = toggle)", x, y, 0.45, dim)
        for key, name in (("1", "motor"), ("2", "somato"), ("3", "fef"), ("4", "visual")):
            y += 22
            g = self.brain.glow[REGION_ID[name]]
            c = rgb255(REGION_COLORS[name])
            on = name in self.manual
            T(img, f"[{key}] {REGION_LABELS[name]}", x + 6, y, 0.48, c if (on or g > 0.3) else dim)
            cv2.rectangle(img, (x + 250, y - 11), (x + 250 + int(120 * g), y - 1), (*c, 255), -1)
            cv2.rectangle(img, (x + 250, y - 11), (x + 370, y - 1), (90, 100, 120, 255), 1)
        T(img, "0 off  I pial/inflated  Z zoom hand  drag/wheel view", x, y + 26, 0.4, dim)
        T(img, "UP/DOWN neurons  N/M noise  K dropout  T test  D debug", x, y + 46, 0.4, dim)

        T(img, "velocity", COMPASS[0] - 30, COMPASS[1] - COMPASS[2] - 10, 0.42, dim)
        T(img, "white true / yellow PV", COMPASS[0] - 50, COMPASS[1] + COMPASS[2] + 18, 0.38, dim)
        T(img, "PD color", WHEEL[0] - 32, WHEEL[1] - WHEEL[2] - 10, 0.42, dim)
        T(img, "hand pad (test mode)", PAD[0], PAD[1] - 8, 0.42, dim)
        T(img, f"render {fps:4.1f} fps   camera {hand.fps:4.1f} fps", 14, BRAIN_VP[3] - 8, 0.4, dim)

        rx, ry, rw, rh = RASTER
        T(img, f"Spike raster (last {config.RASTER_SEC:.0f} s, sorted by preferred direction)", rx, ry - 8, 0.42, dim)
        T(img, f"#{self.n - 1}", 10, ry + 12, 0.4, dim)
        T(img, "#0", 10, ry + rh, 0.4, dim)
        T(img, "Left hemisphere: neurons on the hand knob of M1 (right hand -> left hemisphere)",
          14, 30, 0.45, (200, 205, 220))
        if self.model.simplified:      # 標準脳データがないときの警告
            cv2.rectangle(img, (10, 44), (520, 76), (90, 30, 10, 220), -1)
            if self.jtext.available:
                self.jtext.draw(img, "！ " + WARNING_TEXT, 20, 48, 20, (255, 210, 120))
            else:
                T(img, "! " + WARNING_TEXT_EN, 20, 66, 0.48, (255, 210, 120))


def main():
    parser = argparse.ArgumentParser(description="BCI オペレーター：脳モデルのテスト画面（段階 1）")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--test", action="store_true", help="カメラを使わず、マウスを手の代わりにする")
    BrainTest(parser.parse_args()).run()


if __name__ == "__main__":
    main()
