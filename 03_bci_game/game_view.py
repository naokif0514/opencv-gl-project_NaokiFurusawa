"""
game_view.py
------------
BCI オペレーターの描画（main.py の Game の状態を読んで描くだけ。ゲームの進行は変えない）。

  ・タイトル画面：本物の脳（pial）を大きく回転表示し、タイトルと遊び方
  ・ゲーム画面：中央に AR（カメラ映像 + ロボット + アイテム）、右に脳（光る領域）と HUD、下に小さなラスター
  ・重ねて表示する画面：視線同期、ステージの説明、同期フェーズ、電極劣化の警告、ステージクリア（スコアと
    デコード精度の推移グラフ）、結果（全ステージのスコアと精度の推移）

描く順番：AR の背景（2D）→ ロボット・仮想アイテム（3D）→ 枠・リング・注視点（2D）→ 脳（3D）→ ラスター → 文字。
文字（英数字は cv2.putText、日本語は jtext）は 1 枚の透明な画像に 0.1 秒ごとに描き直して重ねる。
"""

import math
import time

import cv2
import glfw
import numpy as np
from OpenGL import GL as gl

import config
import hand_calib as HC
from brain_model import REGION_ID, WARNING_TEXT
from brain_renderer import REGION_COLORS
from gfx2d import Batch2D, TexturedQuad, rgb255
from jtext import JText
from layout import AH, AW, AX, AY, BH, BW, BX, BY, H, W, hand_to_screen, world_radius, world_to_screen
from mission import STAGES, gripper_tip
from robot_view import RobotView

# グラフの系列の色（暗い背景用。ステージ 1〜4、デコード精度の 2 指標）
SERIES = [(0.22, 0.53, 0.90), (0.85, 0.35, 0.15), (0.10, 0.62, 0.44), (0.79, 0.52, 0.0)]
INK = (225, 230, 240)
DIM = (150, 160, 185)
GOOD = (140, 230, 160)
BAD = (255, 140, 120)
ACCENT = (130, 220, 255)
TITLE_BRAIN = (320, 70, 640, 470)          # タイトル画面の脳の表示 (x, y, 幅, 高さ)
START_BUTTON = (W / 2 - 140, 600, 280, 56)


class GameView:
    def __init__(self, game):
        self.g = game
        self.robot_view = RobotView()
        self.batch = Batch2D()
        self.ar_quad = TexturedQuad(gl.GL_LINEAR)
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.overlay_img = np.zeros((H, W, 4), dtype=np.uint8)
        self._last_overlay = -1.0
        self._ar_frame_id = -1
        self._last_state = None
        self.jtext = JText()

    def start_button_hit(self, p):
        x, y, w, h = START_BUTTON
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    # ========================================================
    def render(self, t, dt, gaze, items, fps):
        g = self.g
        win = g.window
        fb_w, fb_h = glfw.get_framebuffer_size(win)
        if fb_w == 0 or fb_h == 0:
            return
        self.sx, self.sy = fb_w / W, fb_h / H
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.025, 0.03, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        if g.state == "title":
            self._render_title(t, dt)
        else:
            self._render_game(t, dt, gaze, items)

        if t - self._last_overlay >= config.OVERLAY_INTERVAL or g.state != self._last_state:
            self._last_state = g.state           # 画面が切りかわったら、すぐに文字を描き直す
            self.overlay_img[:] = 0
            if g.state == "title":
                self._text_title(t)
            else:
                self._text_game(t, gaze, items, fps)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glEnable(gl.GL_BLEND)
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        glfw.swap_buffers(win)

    def _vp(self, x, y, w, h):
        return (int(x * self.sx), int((H - y - h) * self.sy), int(w * self.sx), int(h * self.sy))

    def _brain(self, rect, t, dt):
        c = self.g.cortex
        self.g.brain.draw(self._vp(*rect), t, dt, c.flash, c.activity, c.alive)
        fb_w, fb_h = glfw.get_framebuffer_size(self.g.window)
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glEnable(gl.GL_BLEND)

    # ========================================================
    # タイトル画面
    # ========================================================
    def _render_title(self, t, dt):
        b = self.batch
        b.rect(0, 0, W, H, (0.02, 0.025, 0.045, 1.0))
        for k in range(24):                              # 背景の薄い同心円（スキャナーのような演出）
            r = 120 + k * 28 + (t * 20) % 28
            b.ring(W / 2, 300, r, 1, (0.2, 0.5, 0.8, max(0.0, 0.10 - k * 0.004)), 64)
        x, y, w, h = START_BUTTON
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        b.rect(x, y, w, h, (0.08, 0.25, 0.4, 0.8 + 0.2 * pulse))
        b.rect(x, y, w, 2, (0.5, 0.85, 1.0, 1))
        b.rect(x, y + h - 2, w, 2, (0.5, 0.85, 1.0, 1))
        b.flush()
        self._brain(TITLE_BRAIN, t, dt)

    def _text_title(self, t):
        img = self.overlay_img
        g = self.g
        self._jt(img, "BCI オペレーター", "BCI OPERATOR", W / 2, 18, 46, (235, 245, 255), "center")
        self._jt(img, "視線で選び、脳の信号でロボットを動かせ", "Select with your eyes, move the robot with your brain",
                 W / 2, 82, 20, ACCENT, "center")
        self._jt(img, "スタート（SPACE / クリック）", "START (SPACE / click)", W / 2, START_BUTTON[1] + 14, 22,
                 (255, 255, 255), "center")
        lines = [
            ("ハイブリッド BCI：", "Hybrid BCI:"),
            ("・視線 → 行き先・つかむ物を選ぶ（前頭眼野・視覚野）", "- gaze -> choose target (FEF / visual)"),
            ("・手の動き → 運動野のニューロン → アーム", "- hand -> motor cortex neurons -> arm"),
            ("・手を握る / 開く → 把持ニューロン → グリッパー", "- grip -> grasp neurons -> gripper"),
        ]
        for i, (ja, en) in enumerate(lines):
            self._jt(img, ja, en, 30, 470 + i * 26, 17, INK if i else ACCENT)
        mode = "テストモード：マウス（位置 = 視線、動き = 手、クリック = 握る）" if g.test_mode else "カメラモード（顔と右手をカメラに）"
        self._jt(img, mode, "TEST MODE (mouse)" if g.test_mode else "CAMERA MODE", W - 30, 470, 17,
                 (255, 200, 110) if g.test_mode else GOOD, "right")
        if g.vision is not None:
            self._jt(img, "T：カメラ / テストモードの切り替え", "T: camera / test mode", W - 30, 496, 15, DIM, "right")
        self._jt(img, "ESC：終了", "ESC: quit", W - 30, 520, 15, DIM, "right")
        self._jt(img, "標準脳 fsaverage5（FreeSurfer）+ Destrieux アトラス", "fsaverage5 + Destrieux atlas",
                 W / 2, TITLE_BRAIN[1] + TITLE_BRAIN[3] - 6, 13, DIM, "center")
        if g.model.simplified:
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", W / 2, 120, 17, (255, 210, 120), "center")

    # ========================================================
    # ゲーム画面
    # ========================================================
    def _render_game(self, t, dt, gaze, items):
        g = self.g
        b = self.batch

        # ---- AR の背景（カメラ映像）----
        if not g.test_mode:
            fid, frame = g.vision.get_frame()
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
        if g.state == "play" and not g.test_mode and not g.use_virtual:
            self._register_box()

        # ---- ゴール、ロボットの影と行き先 ----
        goal = g.goal
        gp, gr = world_to_screen(goal.pos), world_radius(goal.radius)
        b.circle(*gp, gr, (0.2, 1.0, 0.5, 0.15))
        b.ring(*gp, gr, 3, (0.3, 1.0, 0.55, 0.95))
        b.circle(*gp, 4, (0.3, 1.0, 0.55, 0.9), 12)
        rp = world_to_screen(g.robot.pos)
        b.circle(rp[0] + 6, rp[1] + 8, 52, (0, 0, 0, 0.35), 24)
        if g.robot.goal is not None:
            q = world_to_screen(g.robot.goal)
            for k in range(0, 12, 2):
                b.line(*(rp + (q - rp) * k / 12), *(rp + (q - rp) * (k + 1) / 12), 2, (0.4, 0.85, 1.0, 0.6))
        b.flush()

        # ---- ロボットと仮想アイテム（3D）----
        self.robot_view.draw(g.robot, items, self._vp(AX, AY, AW, AH))
        fb_w, fb_h = glfw.get_framebuffer_size(g.window)
        gl.glViewport(0, 0, fb_w, fb_h)

        # ---- アイテムの枠、選択、ドウェルのリング、グリッパーの先 ----
        m = g.mission if g.state == "play" else None
        for it in items:
            p = world_to_screen(it.pos)
            r = world_radius(it.radius) + 8
            c = it.rgb
            if it.kind == "real":
                ok = m is None or m.selectable(it)
                b.ring(*p, r, 2.5, (*c, 0.95 if ok else 0.3))
                for k in range(4):
                    a = k * math.pi / 2 + t * 0.8
                    b.line(p[0] + math.cos(a) * r, p[1] + math.sin(a) * r,
                           p[0] + math.cos(a) * (r + 10), p[1] + math.sin(a) * (r + 10), 2, (*c, 0.95 if ok else 0.3))
            if g.selected == it.id:
                b.ring(*p, r + 14, 2, (1.0, 1.0, 1.0, 0.8))
        if g.selected == "goal":
            b.ring(*gp, gr + 12, 2, (1.0, 1.0, 1.0, 0.8))
        if g.state == "play":
            tip = world_to_screen(gripper_tip(g.robot))
            b.ring(*tip, 7, 1.5, (1.0, 0.9, 0.4, 0.8), 16)
            if g.dwell.focus is not None and g.dwell.focus_pos is not None:
                self._progress_ring(g.dwell.focus_pos, 40, g.dwell.progress, (1.0, 0.85, 0.3))
        if g.state == "gaze_calib":
            self._draw_gaze_calib(t)
        if g.hand_calib.active:
            self._draw_hand_calib(t)
        if g.show_gaze and gaze is not None and g.state in ("play", "stage_intro"):
            b.circle(*gaze, 6, (1.0, 1.0, 1.0, 0.85), 12)
            b.ring(*gaze, 11, 1.5, (0.2, 0.9, 1.0, 0.8), 20)
        if g.state in ("stage_intro", "stage_clear", "result", "connect"):
            b.rect(AX, AY, AW, AH, (0.0, 0.01, 0.03, 0.72))
        if g.state == "stage_clear":
            self._graph_stage(g.results[-1])
        elif g.state == "result":
            self._graph_result()
        for (x, y, w, h) in ((AX - 1, AY - 1, AW + 2, 1), (AX - 1, AY + AH, AW + 2, 1), (AX - 1, AY, 1, AH),
                             (AX + AW, AY, 1, AH)):
            b.rect(x, y, w, h, (0.3, 0.4, 0.55, 1))
        rx, ry, rw, rh = config.RASTER_RECT
        b.rect(rx - 1, ry - 1, rw + 2, rh + 2, (0.25, 0.3, 0.4, 1.0))
        b.rect(BX, BY, BW, BH, (0.035, 0.04, 0.065, 1.0))
        self._grip_meter()
        if g.banner == "electrode" and time.monotonic() < g.banner_until:
            a = 0.5 + 0.5 * math.sin(t * 10)
            b.rect(AX, AY + AH / 2 - 50, AW, 100, (0.5, 0.05, 0.02, 0.55 + 0.25 * a))
        b.flush()

        self._brain(config.BRAIN_RECT, t, dt)
        self.raster_quad.upload(g.cortex.raster)
        self.raster_quad.draw(rx, ry, rw, rh)

    # --------------------------------------------------------
    def _register_box(self):
        b = self.batch
        x0, y0, x1, y1 = config.COLOR_REGISTER_BOX
        p0, p1 = (AX + x0 * AW, AY + y0 * AH), (AX + x1 * AW, AY + y1 * AH)
        c = (1.0, 1.0, 1.0, 0.3)
        for (ax, ay), (bx_, by_) in (((p0[0], p0[1]), (p1[0], p0[1])), ((p1[0], p0[1]), (p1[0], p1[1])),
                                     ((p1[0], p1[1]), (p0[0], p1[1])), ((p0[0], p1[1]), (p0[0], p0[1]))):
            for k in range(0, 10, 2):
                b.line(ax + (bx_ - ax) * k / 10, ay + (by_ - ay) * k / 10,
                       ax + (bx_ - ax) * (k + 1) / 10, ay + (by_ - ay) * (k + 1) / 10, 1.5, c)

    def _progress_ring(self, p, r, progress, color):
        b = self.batch
        b.ring(*p, r, 3, (*color, 0.25), 48)
        for k in range(max(1, int(48 * progress))):
            a0 = -math.pi / 2 + 2 * math.pi * k / 48
            a1 = -math.pi / 2 + 2 * math.pi * (k + 1) / 48
            b.line(p[0] + r * math.cos(a0), p[1] + r * math.sin(a0),
                   p[0] + r * math.cos(a1), p[1] + r * math.sin(a1), 5, (*color, 0.95))

    def _draw_gaze_calib(self, t):
        b = self.batch
        b.rect(AX, AY, AW, AH, (0.0, 0.02, 0.05, 0.55))
        cal = self.g.gaze.calib
        for i, p in enumerate(cal.points):
            b.circle(*p, 5, (0.3, 1.0, 0.6, 0.9) if i < cal.index else (0.5, 0.55, 0.65, 0.5), 12)
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

    def _draw_hand_calib(self, t):
        """同期フェーズ：マゼンタのターゲット（握る区間では、握る合図で塗りつぶす）と、自分の手の位置"""
        b = self.batch
        hc = self.g.hand_calib
        b.rect(AX, AY, AW, AH, (0.03, 0.0, 0.05, 0.35))
        p = hand_to_screen(hc.target)
        mag = (1.0, 0.3, 0.85)
        if hc.phase == "grip":
            closed = hc.grip_cue > 0.5
            b.circle(*p, 46 if not closed else 30, (*mag, 0.85 if closed else 0.15), 40)
            b.ring(*p, 46 if not closed else 30, 4, (*mag, 1.0), 40)
        else:
            b.circle(*p, 26, (*mag, 0.18))
            b.ring(*p, 26, 3, (*mag, 0.95))
            b.circle(*p, 4, (*mag, 1.0))
        h = self.g.hand
        if h is not None and h.pos is not None:
            q = hand_to_screen(h.pos)
            b.ring(*q, 18, 3, (1, 1, 1, 0.8), 28)
        b.rect(AX, AY + AH - 8, AW, 8, (0.15, 0.1, 0.18, 1.0))
        b.rect(AX, AY + AH - 8, AW * hc.progress, 8, (*mag, 1.0))

    def _grip_meter(self):
        """把持のメーター：本当の握り具合（白）とデコードした値（黄）、しきい値（ヒステリシス）"""
        b = self.batch
        x, y, w, h = BX + 230, BY + BH + 150, 160, 12
        b.rect(x, y, w, h, (0.1, 0.12, 0.18, 1.0))
        gv = float(np.clip(self.g.cortex.grip.value, 0, 1))
        b.rect(x, y, w * gv, h, (1.0, 0.8, 0.25, 0.9) if self.g.grip_switch.closed else (0.6, 0.6, 0.7, 0.8))
        for th in (config.GRIP_OPEN_TH, config.GRIP_CLOSE_TH):
            b.rect(x + w * th - 1, y - 3, 2, h + 6, (1, 1, 1, 0.7))
        tg = getattr(self.g, "last_g", 0.0) if not self.g.test_mode else (1.0 if self.g.mouse_down else 0.0)
        b.circle(x + w * tg, y + h + 6, 4, (1, 1, 1, 0.9), 10)

    # --------------------------------------------------------
    # グラフ（ステージクリア・結果）
    # --------------------------------------------------------
    def _axes(self, x, y, w, h, ymax, yticks):
        b = self.batch
        b.rect(x, y, w, h, (0.05, 0.06, 0.1, 0.9))
        for v in yticks:
            yy = y + h - h * v / ymax
            b.line(x, yy, x + w, yy, 1, (0.2, 0.24, 0.32, 1.0))
        b.line(x, y + h, x + w, y + h, 1.5, (0.45, 0.5, 0.6, 1.0))
        b.line(x, y, x, y + h, 1.5, (0.45, 0.5, 0.6, 1.0))

    def _polyline(self, pts, color, width=2.0):
        for p, q in zip(pts[:-1], pts[1:]):
            self.batch.line(*p, *q, width, (*color, 1.0))

    def graph_rect(self, kind):
        if kind == "stage":
            return (AX + 70, AY + 300, AW - 140, 120), (AX + 70, AY + 450, AW - 140, 120)
        return (AX + 70, AY + 330, AW - 140, 220)

    def _series_pts(self, acc, idx, rect, ymax, tmax):
        x, y, w, h = rect
        pts = []
        for tt, *vals in acc:
            v = vals[idx]
            if v is None or not np.isfinite(v):
                continue
            pts.append((x + w * tt / tmax, y + h - h * min(max(v, 0.0), ymax) / ymax))
        return pts

    def _graph_stage(self, res):
        """ステージクリア：相関係数と RMSE の推移（それぞれ別のグラフ）。電極劣化・再同期の時刻に縦線"""
        r_corr, r_rmse = self.graph_rect("stage")
        tmax = max(10.0, max((a[0] for a in res.accuracy), default=10.0))
        for rect, idx, ymax, ticks, color in ((r_corr, 0, 1.0, (0.25, 0.5, 0.75, 1.0), SERIES[0]),
                                              (r_rmse, 1, 1.0, (0.25, 0.5, 0.75, 1.0), SERIES[1])):
            self._axes(*rect, ymax, ticks)
            for tt, ev in res.events:
                if ev.startswith("degrade"):
                    xx = rect[0] + rect[2] * tt / tmax
                    self.batch.line(xx, rect[1], xx, rect[1] + rect[3], 2, (1.0, 0.35, 0.3, 0.9))
            self._polyline(self._series_pts(res.accuracy, idx, rect, ymax, tmax), color)

    def _graph_result(self):
        rect = self.graph_rect("result")
        tmax = max([10.0] + [a[0] for r in self.g.results for a in r.accuracy])
        self._axes(*rect, 1.0, (0.25, 0.5, 0.75, 1.0))
        for r in self.g.results:
            self._polyline(self._series_pts(r.accuracy, 0, rect, 1.0, tmax), SERIES[(r.stage - 1) % 4])

    # ========================================================
    # 文字
    # ========================================================
    @staticmethod
    def _t(img, text, x, y, scale=0.45, color=INK, thick=1):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), thick, cv2.LINE_AA)

    def _jt(self, img, ja, en, x, y, size=18, color=(255, 255, 255), anchor="left"):
        """日本語が書ければ日本語、書けなければ英語（cv2）。y は文字の上端"""
        if self.jtext.available:
            if anchor == "right":
                w, _ = self.jtext.size(ja, size)
                x -= w
                anchor = "left"
            self.jtext.draw(img, ja, x, y, size, color, anchor)
        else:
            (tw, _), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)
            if anchor == "center":
                x -= tw / 2
            elif anchor == "right":
                x -= tw
            self._t(img, en, x, y + size * 0.8, size / 30, color)

    def _text_game(self, t, gaze, items, fps):
        img = self.overlay_img
        g = self.g
        T = self._t
        cx, cy = AX + AW / 2, AY + AH / 2
        st = g.stage
        m = g.mission if g.state == "play" else None

        # ---- 状態ごとの大きな表示 ----
        if g.state == "connect":
            k = min(1.0, g.state_t / config.MORPH_SEC)
            cv2.rectangle(img, (int(cx - 230), int(cy - 50)), (int(cx + 230), int(cy + 50)), (6, 10, 24, 210), -1)
            self._jt(img, "脳と接続中…", "LINKING TO THE BRAIN...", cx, cy - 36, 32, ACCENT, "center")
            cv2.rectangle(img, (int(cx - 180), int(cy + 22)), (int(cx + 180), int(cy + 30)), (60, 70, 90, 255), -1)
            cv2.rectangle(img, (int(cx - 180), int(cy + 22)), (int(cx - 180 + 360 * k), int(cy + 30)),
                          (*ACCENT, 255), -1)
        elif g.state == "gaze_calib":
            cal = g.gaze.calib
            n = len(cal.points)
            cv2.rectangle(img, (int(cx - 260), int(AY + 18)), (int(cx + 260), int(AY + 92)), (6, 10, 24, 210), -1)
            self._jt(img, f"視線同期  {min(cal.index + 1, n)} / {n}", f"GAZE SYNC {min(cal.index + 1, n)}/{n}",
                     cx, AY + 24, 28, ACCENT, "center")
            face = g.vision.get_state().face_detected if g.vision else False
            self._jt(img, "光る点を見つめてください（頭は動かさずに）" if face else "顔が見つかりません：カメラの正面へ",
                     "Look at the glowing dot" if face else "Face not found", cx, AY + 62, 18,
                     INK if face else BAD, "center")
        elif g.state == "stage_intro":
            self._jt(img, st.title_ja, st.title_en, cx, cy - 120, 40, (255, 255, 255), "center")
            self._jt(img, st.note_ja, st.note_ja, cx, cy - 50, 20, ACCENT, "center")
            info = f"ニューロン {st.n_neurons} 個（+ 把持 {g.cortex.n_grip} 個）  ノイズ {st.noise_type} {st.noise:.0f} Hz  " \
                   f"ゴールの半径 {world_radius(st.goal_radius):.0f} 画素"
            self._jt(img, info, info, cx, cy - 10, 16, DIM, "center")
            self._jt(img, f"目標：{config.ITEMS_PER_STAGE} 個のアイテムをゴールへ（制限時間 {config.STAGE_TIME:.0f} 秒）",
                     "Deliver items to the GOAL", cx, cy + 30, 20, INK, "center")
            self._jt(img, "次は「同期フェーズ」：ターゲットを右手で追いかけ、合図で握る・開く",
                     "Next: hand sync", cx, cy + 80, 18, (255, 160, 230), "center")
            self._jt(img, "SPACE で開始", "SPACE to start", cx, cy + 120, 18, DIM, "center")
        elif g.state == "stage_clear":
            self._text_stage_clear(g.results[-1])
        elif g.state == "result":
            self._text_result()

        if g.hand_calib.active:
            hc = g.hand_calib
            title = "再同期フェーズ" if g.recalibrating else "同期フェーズ"
            cv2.rectangle(img, (int(cx - 300), int(AY + 14)), (int(cx + 300), int(AY + 96)), (14, 4, 20, 200), -1)
            self._jt(img, f"{title}  残り {hc.remaining:4.1f} 秒", f"HAND SYNC {hc.remaining:4.1f}s", cx, AY + 20,
                     26, (255, 150, 230), "center")
            if hc.phase == "grip":
                msg = "握る！" if hc.grip_cue > 0.5 else "開く！"
                self._jt(img, msg, "GRIP!" if hc.grip_cue > 0.5 else "OPEN!", cx, AY + 58, 26,
                         (255, 255, 255), "center")
            else:
                self._jt(img, "マゼンタのターゲットを右手で追いかける（白い輪 = あなたの手）",
                         "Follow the magenta target with your right hand", cx, AY + 60, 17, INK, "center")
            self._jt(img, f"記録 {hc.valid_sec:4.1f} 秒", f"rec {hc.valid_sec:4.1f}s", AX + AW - 12, AY + AH - 34,
                     14, DIM, "right")

        # ---- プレイ中の案内 ----
        if m is not None and not g.recalibrating:
            self._jt(img, self._hint(m), self._hint(m), cx, AY + 14, 20, (255, 255, 255), "center")
        if g.banner == "electrode" and time.monotonic() < g.banner_until:
            self._jt(img, "！ 電極劣化：ニューロンが脱落した", "! ELECTRODE FAILURE", cx, cy - 38, 30, (255, 230, 200),
                     "center")
            self._jt(img, "L で再同期（約 21 秒、時間は進む）／ そのまま続けることもできる",
                     "L: re-sync (~21 s) or keep going", cx, cy + 6, 18, (255, 230, 200), "center")
        elif m is not None and m.degraded and g.cortex.dropped_since_calib > 0 and not g.recalibrating:
            self._jt(img, "電極劣化中：L で再同期", "Degraded: press L to re-sync", AX + 12, AY + AH - 30, 16,
                     (255, 160, 110))

        # ---- アイテムとゴールの名前 ----
        if g.state == "play":
            for it in items:
                if it.delivered or it.held:
                    continue
                p = world_to_screen(it.pos)
                r = world_radius(it.radius) + 20
                T(img, it.color + (" (real)" if it.kind == "real" else ""), p[0] + r * 0.7, p[1] - r * 0.7, 0.42,
                  rgb255(it.rgb))
        if g.state not in ("stage_clear", "result"):
            gp = world_to_screen(g.goal.pos)
            T(img, "GOAL", gp[0] - 20, gp[1] + world_radius(g.goal.radius) + 18, 0.5, (90, 255, 140))
        if g.message and time.monotonic() < g.message_until and g.state not in ("stage_clear", "result"):
            self._jt(img, g.message, g.message, cx, AY + AH - 44, 20, (255, 255, 255), "center")

        self._text_brain_legend()
        self._text_hud(fps)
        rx, ry, rw, rh = config.RASTER_RECT
        T(img, f"M1 hand-knob spikes ({g.cortex.n} direction + {g.cortex.n_grip} grasp neurons, top rows white)",
          rx + 4, ry + 13, 0.36, (200, 205, 220))

    def _hint(self, m):
        g = self.g
        if m.held is None:
            if g.selected is None:
                return "アイテムを見つめてロックオン"
            if g.robot.moving:
                return "移動中…"
            return "手を動かしてアームを伸ばし、手を握ってつかむ"
        if g.selected != "goal":
            return "ゴールを見つめて移動"
        if g.robot.moving:
            return "移動中…"
        return "アームをゴールの上へ動かし、手を開いて置く"

    def _text_brain_legend(self):
        img = self.overlay_img
        T = self._t
        T(img, "BRAIN (left hemisphere = right hand)", BX + 8, BY + 18, 0.42, DIM)
        y = BY + BH - 30
        for name, label in (("motor", "motor (hand)"), ("somato", "somatosensory"), ("fef", "frontal eye field"),
                            ("visual", "visual")):
            gl_ = self.g.brain.glow[REGION_ID[name]]
            c = rgb255(REGION_COLORS[name])
            x = BX + 10 + (0 if name in ("motor", "fef") else 190)
            yy = y + (0 if name in ("motor", "somato") else 20)
            cv2.rectangle(img, (x, yy - 10), (x + 12, yy + 2), (*c, 255) if gl_ > 0.3 else (70, 75, 90, 255), -1)
            T(img, label, x + 18, yy, 0.42, c if gl_ > 0.3 else DIM)
        if self.g.model.simplified:
            cv2.rectangle(img, (BX, BY + 26), (BX + BW, BY + 50), (90, 30, 10, 220), -1)
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", BX + 6, BY + 28, 15, (255, 210, 120))

    def _text_hud(self, fps):
        img = self.overlay_img
        g = self.g
        T = self._t
        x, y = BX, BY + BH + 26
        st = g.stage
        m = g.mission
        T(img, f"STAGE {st.number}", x, y, 0.7, (255, 255, 255), 2)
        score = 0 if m is None else m.result.score
        T(img, f"SCORE {score:5d}", x + 140, y, 0.7, (255, 230, 120), 2)
        tl = config.STAGE_TIME if m is None else m.time_left
        T(img, f"TIME {tl:5.1f}", x + 300, y, 0.6, BAD if tl < 20 else INK, 2 if tl < 20 else 1)
        done = 0 if m is None else m.result.delivered
        fails = 0 if m is None else m.result.failures
        T(img, f"delivered {done}/{config.ITEMS_PER_STAGE}   failures {fails}", x, y + 26, 0.48)
        c = g.cortex
        c.metrics.compute()
        corr, rmse = c.metrics.corr, c.metrics.rmse
        dec = "LIN (calibrated)" if c.calibrated else "PV (not synced)"
        T(img, f"decoder: {dec}", x, y + 50, 0.45, ACCENT if c.calibrated else DIM)
        T(img, "corr " + ("--" if corr is None else f"{corr:.2f}") + "   RMSE " +
          ("--" if rmse is None else f"{rmse:.2f}"), x, y + 74, 0.55, (255, 230, 120))
        T(img, f"neurons {c.n_alive}/{c.n_all}", x + 260, y + 74, 0.45, BAD if c.n_alive < c.n_all else INK)
        T(img, "grip (white = hand, bar = decoded)", x, y + 132, 0.4, DIM)
        T(img, "CLOSED" if g.grip_switch.closed else "open", x + 400 - 60, y + 112, 0.45,
          (255, 210, 80) if g.grip_switch.closed else DIM)
        if g.test_mode:
            T(img, "TEST MODE: mouse = gaze & hand, click = grip", x, y + 166, 0.42, (255, 200, 110))
            T(img, f"render {fps:4.1f} fps", x, y + 186, 0.38, DIM)
        else:
            vs = g.vision.get_state()
            T(img, f"camera {vs.fps:4.1f} fps  face {'OK' if vs.face_detected else '--'}  "
                   f"hand {'OK' if vs.hand.detected else '--'}", x, y + 166, 0.42,
              GOOD if vs.face_detected and vs.hand.detected else BAD)
            T(img, f"render {fps:4.1f} fps   gaze drift corr ({g.gaze.offset[0]:+.0f}, {g.gaze.offset[1]:+.0f}) px",
              x, y + 186, 0.38, DIM)
        T(img, "L re-sync  C gaze sync  G gaze dot  V virtual/real  P color", x, y + 210, 0.38, DIM)
        T(img, "R reset robot  D debug  ESC quit", x, y + 228, 0.38, DIM)

    def _text_stage_clear(self, res):
        img = self.overlay_img
        cx = AX + AW / 2
        st = STAGES[res.stage - 1]
        head = f"ステージ {res.stage} クリア！" if res.cleared else f"ステージ {res.stage} タイムアップ"
        self._jt(img, head, f"STAGE {res.stage} {'CLEAR' if res.cleared else 'TIME UP'}", cx, AY + 40, 36,
                 GOOD if res.cleared else BAD, "center")
        bonus = f"（+{int(res.time_left * config.TIME_SCORE)}）" if res.cleared else ""
        line = f"運搬 {res.delivered} / {config.ITEMS_PER_STAGE}　　失敗 {res.failures} 回　　残り時間 {res.time_left:4.1f} 秒{bonus}"
        self._jt(img, line, line, cx, AY + 104, 20, INK, "center")
        self._jt(img, f"スコア  {res.score}", f"SCORE {res.score}", cx, AY + 146, 30, (255, 230, 120), "center")
        r_corr, r_rmse = self.graph_rect("stage")
        self._jt(img, "デコード精度の推移（直近 5 秒の相関係数）", "Decoding correlation (last 5 s)", r_corr[0], r_corr[1] - 26,
                 15, rgb255(SERIES[0]))
        self._jt(img, "速度の RMSE [画面/秒]（低いほど良い）", "Velocity RMSE [screen/s]", r_rmse[0], r_rmse[1] - 26, 15,
                 rgb255(SERIES[1]))
        for rect in (r_corr, r_rmse):
            for v in (0.0, 0.5, 1.0):
                self._t(img, f"{v:.1f}", rect[0] - 34, rect[1] + rect[3] - rect[3] * v + 4, 0.38, DIM)
        tmax = max(10.0, max((a[0] for a in res.accuracy), default=10.0))
        self._t(img, "0 s", r_rmse[0], r_rmse[1] + r_rmse[3] + 16, 0.38, DIM)
        self._t(img, f"{tmax:.0f} s (play time)", r_rmse[0] + r_rmse[2] - 120, r_rmse[1] + r_rmse[3] + 16, 0.38, DIM)
        if any(ev.startswith("degrade") for _, ev in res.events):
            self._jt(img, "赤い線 = 電極劣化", "red line = electrode failure", r_corr[0] + r_corr[2] - 140,
                     r_corr[1] - 26, 14, (255, 120, 110))
        self._jt(img, "SPACE で次へ", "SPACE: next", cx, AY + AH - 34, 18, DIM, "center")
        del st

    def _text_result(self):
        img = self.overlay_img
        cx = AX + AW / 2
        res = self.g.results
        total = sum(r.score for r in res)
        self._jt(img, "ミッション終了", "MISSION COMPLETE", cx, AY + 30, 36, (255, 255, 255), "center")
        self._jt(img, f"合計スコア  {total}", f"TOTAL {total}", cx, AY + 82, 30, (255, 230, 120), "center")
        y = AY + 140
        self._t(img, "stage     score   delivered   failures   mean corr", AX + 120, y, 0.5, DIM)
        for r in res:
            y += 26
            cs = [a[1] for a in r.accuracy if a[1] is not None]
            mc = f"{np.mean(cs):.2f}" if cs else "--"
            self._t(img, f"  {r.stage}       {r.score:6d}       {r.delivered}/{config.ITEMS_PER_STAGE}"
                         f"          {r.failures}          {mc}", AX + 120, y, 0.5, rgb255(SERIES[(r.stage - 1) % 4]))
        rect = self.graph_rect("result")
        self._jt(img, "デコード精度（相関係数）の推移：ステージ別", "Decoding correlation by stage", rect[0], rect[1] - 26, 15, INK)
        for v in (0.0, 0.5, 1.0):
            self._t(img, f"{v:.1f}", rect[0] - 34, rect[1] + rect[3] - rect[3] * v + 4, 0.38, DIM)
        for i, r in enumerate(res):
            self._t(img, f"stage {r.stage}", rect[0] + rect[2] - 330 + i * 82, rect[1] - 10, 0.42,
                    rgb255(SERIES[(r.stage - 1) % 4]))
        self._jt(img, "SPACE でタイトルへ", "SPACE: title", cx, AY + AH - 34, 18, DIM, "center")
