"""
game_view.py
------------
BCI オペレーターの描画（main.py の Game の状態を読んで描くだけ。ゲームの進行は変えない）。

  ・タイトル画面：本物の脳（pial）を大きく回転表示し、タイトルと遊び方
  ・ゲーム画面：左に AR（カメラ映像 + ロボット + アイテム）、右に脳（光る領域）と 3 枚のカード
      「ステージとスコア」「ミッションの手順（今どこか）」「BCI の状態（デコーダ・精度・ニューロン・把持）」、
      AR の下に小さなラスター。AR の上の帯に「今やること」、下にお知らせ
  ・重ねて表示する画面：視線同期、ステージの説明、同期フェーズ、電極劣化の警告、ステージクリア（スコアと
    デコード精度の推移グラフ）、結果（全ステージのスコアと精度の推移）

■ 解像度
  画面の配置は 1280×720 の「論理座標」で決めてあり、実際には RENDER_SCALE 倍の解像度のオフスクリーン
  （render_target.py）に描いてから、ウィンドウ（ディスプレイいっぱい）へ拡大する。
  文字の画像もオフスクリーンと同じ解像度で描く（論理座標 × k）ので、RENDER_SCALE を上げると文字がくっきりする。

描く順番：AR の背景（2D）→ ロボット・仮想アイテム（3D）→ 枠・リング・カードの背景（2D）→ 脳（3D）→ ラスター → 文字。
文字（英数字は cv2.putText、日本語は jtext）は 1 枚の透明な画像に 0.1 秒ごと（と画面が切りかわったとき）に描き直す。
"""

import math
import time

import cv2
import glfw
import numpy as np
from OpenGL import GL as gl

import config
from brain_model import REGION_ID, WARNING_TEXT
from brain_renderer import REGION_COLORS
from gfx2d import Batch2D, TexturedQuad, rgb255
from jtext import JText
from layout import AH, AW, AX, AY, BH, BW, BX, BY, H, W, hand_to_screen, world_radius, world_to_screen
from mission import STAGES, gripper_tip
from render_target import RenderTarget
from robot_view import RobotView

# グラフの系列の色（暗い背景用。ステージ 1〜4、相関係数・RMSE）
SERIES = [(0.22, 0.53, 0.90), (0.85, 0.35, 0.15), (0.10, 0.62, 0.44), (0.79, 0.52, 0.0)]
INK = (230, 235, 245)
DIM = (150, 160, 185)
GOOD = (140, 230, 160)
BAD = (255, 140, 120)
ACCENT = (130, 220, 255)
GOLD = (255, 225, 120)
MAGENTA = (255, 150, 230)
TITLE_BRAIN = (320, 96, 640, 340)          # タイトル画面の脳の表示 (x, y, 幅, 高さ)
START_BUTTON = (W / 2 - 180, 598, 360, 58)

# 右側のカード (x, y, 幅, 高さ)
CARD_STAGE = (BX, BY + BH + 8, BW, 96)
CARD_MISSION = (BX, CARD_STAGE[1] + CARD_STAGE[3] + 8, BW, 112)
CARD_BCI = (BX, CARD_MISSION[1] + CARD_MISSION[3] + 8, BW, H - (CARD_MISSION[1] + CARD_MISSION[3] + 8) - 58)

COLOR_JA = {"red": "赤", "blue": "青"}
STEPS = [("アイテムを見つめて選ぶ", "look at an item"), ("手でアームを伸ばし、握ってつかむ", "reach and grip"),
         ("ゴールを見つめて運ぶ", "look at the GOAL"), ("ゴールの上で手を開いて置く", "open your hand")]
PHASE_JA = {"ready": "準備：白い輪（あなたの手）をターゲットに重ねる", "reach": "8 方向へ往復：ターゲットを手で追う",
            "circle": "円を描く：ターゲットを手で追う", "grip": "合図に合わせて手を握る・開く"}


class GameView:
    def __init__(self, game):
        self.g = game
        self.rt = RenderTarget(round(W * config.RENDER_SCALE), round(H * config.RENDER_SCALE), config.MSAA_SAMPLES)
        self.k = self.rt.width / W                 # 論理座標 → オフスクリーンの画素
        self.robot_view = RobotView()
        self.batch = Batch2D()
        self.ar_quad = TexturedQuad(gl.GL_LINEAR)
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.overlay_img = np.zeros((self.rt.height, self.rt.width, 4), dtype=np.uint8)
        self._last_overlay = -1.0
        self._last_state = None
        self._ar_frame_id = -1
        self.jtext = JText()

    # --------------------------------------------------------
    def window_to_layout(self, mx, my):
        """ウィンドウの座標（マウス）→ 論理座標（拡大と黒い帯を戻す）"""
        ww, wh = glfw.get_window_size(self.g.window)
        x, y, w, h = self.rt.letterbox(ww, wh)
        return np.array([(mx - x) / max(w, 1) * W, (my - y) / max(h, 1) * H])

    def start_button_hit(self, p):
        x, y, w, h = START_BUTTON
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    def _vp(self, x, y, w, h):
        """論理座標の矩形 → オフスクリーンの glViewport（左下原点）"""
        k = self.k
        return (int(x * k), int((H - y - h) * k), int(w * k), int(h * k))

    def _full_viewport(self):
        gl.glViewport(0, 0, self.rt.width, self.rt.height)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # ========================================================
    def render(self, t, dt, gaze, items, fps):
        g = self.g
        self.rt.bind()
        gl.glClearColor(0.025, 0.03, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._full_viewport()

        if g.state == "title":
            self._render_title(t, dt)
        else:
            self._render_game(t, dt, gaze, items)

        if t - self._last_overlay >= config.OVERLAY_INTERVAL or g.state != self._last_state:
            self._last_state = g.state           # 画面が切りかわったら、すぐに文字を描き直す
            self.overlay_img[:] = 0
            if g.state == "title":
                self._text_title()
            else:
                self._text_game(items, fps)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        self._full_viewport()
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        self.rt.present(g.window)
        glfw.swap_buffers(g.window)

    def _brain(self, rect, t, dt):
        c = self.g.cortex
        self.g.brain.draw(self._vp(*rect), t, dt, c.flash, c.activity, c.alive)
        self._full_viewport()

    def _card(self, rect, accent=(0.35, 0.55, 0.8)):
        x, y, w, h = rect
        b = self.batch
        b.rect(x, y, w, h, (0.055, 0.065, 0.10, 0.95))
        b.rect(x, y, w, 2, (*accent, 1.0))
        b.rect(x, y + h - 1, w, 1, (0.18, 0.22, 0.3, 1.0))

    def _pill(self, cx, y, w, h, color=(0.03, 0.05, 0.10, 0.82)):
        self.batch.rect(cx - w / 2, y, w, h, color)

    def _text_w(self, text, size):
        if self.jtext.available:
            return self.jtext.size(text, size)[0]
        return len(text) * size * 0.55

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
        self._card((24, 460, 430, 120))
        self._card((W - 454, 460, 430, 120), (0.85, 0.65, 0.3))
        b.flush()
        self._brain(TITLE_BRAIN, t, dt)

    def _text_title(self):
        img = self.overlay_img
        g = self.g
        self._jt(img, "BCI オペレーター", "BCI OPERATOR", W / 2, 16, 48, (235, 245, 255), "center")
        self._jt(img, "視線で選び、脳の信号でロボットを動かせ", "Select with your eyes, move the robot with your brain",
                 W / 2, 82, 20, ACCENT, "center")
        self._jt(img, "スタート（SPACE / クリック）", "START (SPACE / click)", W / 2, START_BUTTON[1] + 15, 23,
                 (255, 255, 255), "center")
        lines = [("ハイブリッド BCI のしくみ", "Hybrid BCI"),
                 ("視線 → 行き先・つかむ物を選ぶ（前頭眼野・視覚野）", "gaze -> choose (FEF / visual)"),
                 ("手の動き → 運動野のニューロン → アーム", "hand -> motor neurons -> arm"),
                 ("手を握る・開く → 把持ニューロン → グリッパー", "grip -> grasp neurons -> gripper")]
        for i, (ja, en) in enumerate(lines):
            self._jt(img, ja, en, 40, 470 + i * 26, 18 if i == 0 else 16, ACCENT if i == 0 else INK)
        rx = W - 438
        self._jt(img, "テストモード（カメラなし）" if g.test_mode else "カメラモード", "TEST MODE" if g.test_mode else
                 "CAMERA MODE", rx, 470, 18, (255, 200, 110) if g.test_mode else GOOD)
        if g.test_mode:
            how = ["マウスの位置 = 視線、AR の上で動かす = 手", "左クリックを押している間 = 握る"]
        else:
            how = ["顔をカメラの正面に、右手をカメラに映す", "赤・青の物があればアイテムになる（なければ V キー）"]
        for i, s in enumerate(how):
            self._jt(img, s, s, rx, 498 + i * 24, 15, INK)
        keys = "T：カメラ / テスト切り替え　F11：全画面　ESC：終了" if g.vision is not None else "F11：全画面　ESC：終了"
        self._jt(img, keys, keys, rx, 550, 14, DIM)
        self._jt(img, "標準脳 fsaverage5（FreeSurfer）+ Destrieux アトラス。光る点 = 運動野の手の領域の擬似ニューロン",
                 "fsaverage5 + Destrieux atlas", W / 2, 432, 13, DIM, "center")
        if g.model.simplified:
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", W / 2, 118, 17, (255, 210, 120), "center")

    # ========================================================
    # ゲーム画面（図形）
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
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        b.circle(*gp, gr, (0.2, 1.0, 0.5, 0.12 + 0.08 * pulse))
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
        self._full_viewport()

        # ---- アイテムの枠、選択、ドウェルのリング、グリッパーの先 ----
        m = g.mission if g.state == "play" else None
        for it in items:
            if it.delivered:
                continue
            p = world_to_screen(it.pos)
            r = world_radius(it.radius) + 8
            c = it.rgb
            ok = m is None or m.selectable(it)
            if it.kind == "real":
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
            b.ring(*tip, 7, 1.5, (1.0, 0.9, 0.4, 0.85), 16)
            if g.dwell.focus is not None and g.dwell.focus_pos is not None:
                self._progress_ring(g.dwell.focus_pos, 40, g.dwell.progress, (1.0, 0.85, 0.3))
        if g.state == "gaze_calib":
            self._draw_gaze_calib(t)
        if g.hand_calib.active:
            self._draw_hand_calib()
        if g.show_gaze and gaze is not None and g.state in ("play", "stage_intro"):
            b.circle(*gaze, 6, (1.0, 1.0, 1.0, 0.85), 12)
            b.ring(*gaze, 11, 1.5, (0.2, 0.9, 1.0, 0.8), 20)

        # ---- AR の上の帯（今やること）と下のお知らせ ----
        if g.state == "play" and not g.recalibrating and m is not None:
            hint = self._hint(m)
            self._pill(AX + AW / 2, AY + 8, self._text_w(hint, 21) + 48, 40)
        if self._message() is not None:
            self._pill(AX + AW / 2, AY + AH - 54, self._text_w(self._message(), 20) + 40, 38, (0.02, 0.04, 0.08, 0.85))

        # ---- 重ねる画面 ----
        if g.state in ("stage_intro", "stage_clear", "result", "connect"):
            b.rect(AX, AY, AW, AH, (0.0, 0.01, 0.03, 0.78))
        if g.state == "stage_intro":
            self._card((AX + 90, AY + 120, AW - 180, 360), (1.0, 0.45, 0.85))
        if g.state == "stage_clear":
            self._graph_stage(g.results[-1])
        elif g.state == "result":
            self._graph_result()
        if g.banner == "electrode" and time.monotonic() < g.banner_until:
            a = 0.5 + 0.5 * math.sin(t * 10)
            b.rect(AX, AY + AH / 2 - 56, AW, 112, (0.5, 0.05, 0.02, 0.6 + 0.25 * a))

        for (x, y, w, h) in ((AX - 1, AY - 1, AW + 2, 1), (AX - 1, AY + AH, AW + 2, 1), (AX - 1, AY, 1, AH),
                             (AX + AW, AY, 1, AH)):
            b.rect(x, y, w, h, (0.3, 0.4, 0.55, 1))
        rx, ry, rw, rh = config.RASTER_RECT
        b.rect(rx - 1, ry - 1, rw + 2, rh + 2, (0.25, 0.3, 0.4, 1.0))

        # ---- 右側：脳の背景とカード ----
        b.rect(BX, BY, BW, BH, (0.035, 0.04, 0.065, 1.0))
        self._card(CARD_STAGE, (1.0, 0.85, 0.35))
        self._card(CARD_MISSION, (0.4, 0.9, 1.0))
        self._card(CARD_BCI, (1.0, 0.55, 0.2))
        self._time_bar()
        self._bci_bars()
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
        self._pill(AX + AW / 2, AY + 14, 540, 84)

    def _draw_hand_calib(self):
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
            b.circle(*p, 28, (*mag, 0.18))
            b.ring(*p, 28, 3, (*mag, 0.95))
            b.circle(*p, 4, (*mag, 1.0))
        h = self.g.hand
        if h is not None and h.pos is not None:
            q = hand_to_screen(h.pos)
            b.ring(*q, 18, 3, (1, 1, 1, 0.85), 28)
        b.rect(AX, AY + AH - 8, AW, 8, (0.15, 0.1, 0.18, 1.0))
        b.rect(AX, AY + AH - 8, AW * hc.progress, 8, (*mag, 1.0))
        self._pill(AX + AW / 2, AY + 12, 640, 92, (0.06, 0.01, 0.08, 0.85))

    def _time_bar(self):
        g = self.g
        m = g.mission if g.state == "play" else None
        x, y, w, h = CARD_STAGE
        tl = config.STAGE_TIME if m is None else m.time_left
        frac = tl / config.STAGE_TIME
        b = self.batch
        b.rect(x + 14, y + 74, w - 28, 8, (0.12, 0.14, 0.2, 1.0))
        col = (1.0, 0.35, 0.3) if tl < 20 else (0.4, 0.85, 1.0)
        b.rect(x + 14, y + 74, (w - 28) * frac, 8, (*col, 1.0))

    def _grip_range(self):
        gd = self.g.cortex.grip
        return min(0.0, gd.open_th - 0.3), max(1.0, gd.close_th + 0.3)

    def _bci_bars(self):
        """BCI の状態のカード：精度（相関）、ニューロンの生存、把持のメーター（しきい値つき）"""
        g = self.g
        b = self.batch
        x, y, w, h = CARD_BCI
        bx, bw = x + 120, w - 140
        c = g.cortex
        corr = c.metrics.corr
        b.rect(bx, y + 44, bw, 10, (0.12, 0.14, 0.2, 1.0))
        if corr is not None:
            b.rect(bx, y + 44, bw * max(0.0, corr), 10, (1.0, 0.85, 0.35, 1.0))
        b.rect(bx, y + 68, bw, 10, (0.12, 0.14, 0.2, 1.0))
        b.rect(bx, y + 68, bw * c.n_alive / c.n_all, 10, (0.45, 0.85, 0.55, 1.0) if c.n_alive == c.n_all
               else (1.0, 0.45, 0.3, 1.0))
        # 把持のメーター：バー = デコードした値、白い縦線 = しきい値（開いた / 握った）、白い点 = 本当の握り具合
        lo, hi = self._grip_range()
        gy = y + 100
        b.rect(bx, gy, bw, 14, (0.12, 0.14, 0.2, 1.0))
        val = float(np.clip((c.grip.value - lo) / (hi - lo), 0, 1))
        closed = g.grip_switch.closed
        b.rect(bx, gy, bw * val, 14, (1.0, 0.8, 0.25, 0.95) if closed else (0.65, 0.7, 0.85, 0.9))
        for th in (c.grip.open_th, c.grip.close_th):
            xx = bx + bw * (th - lo) / (hi - lo)
            b.rect(xx - 1, gy - 4, 2, 22, (1, 1, 1, 0.8))
        tg = (1.0 if g.mouse_down else 0.0) if g.test_mode else getattr(g, "last_g", 0.0)
        b.circle(bx + bw * float(np.clip((tg - lo) / (hi - lo), 0, 1)), gy + 22, 4, (1, 1, 1, 0.9), 10)

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

    @staticmethod
    def _series_pts(acc, idx, rect, ymax, tmax):
        x, y, w, h = rect
        pts = []
        for tt, *vals in acc:
            v = vals[idx]
            if v is None or not np.isfinite(v):
                continue
            pts.append((x + w * tt / tmax, y + h - h * min(max(v, 0.0), ymax) / ymax))
        return pts

    def _graph_stage(self, res):
        """ステージクリア：相関係数と RMSE の推移（それぞれ別のグラフ）。電極劣化の時刻に赤い縦線"""
        r_corr, r_rmse = self.graph_rect("stage")
        tmax = max(10.0, max((a[0] for a in res.accuracy), default=10.0))
        for rect, idx, color in ((r_corr, 0, SERIES[0]), (r_rmse, 1, SERIES[1])):
            self._axes(*rect, 1.0, (0.25, 0.5, 0.75, 1.0))
            for tt, ev in res.events:
                if ev.startswith("degrade"):
                    xx = rect[0] + rect[2] * tt / tmax
                    self.batch.line(xx, rect[1], xx, rect[1] + rect[3], 2, (1.0, 0.35, 0.3, 0.9))
            self._polyline(self._series_pts(res.accuracy, idx, rect, 1.0, tmax), color)

    def _graph_result(self):
        rect = self.graph_rect("result")
        tmax = max([10.0] + [a[0] for r in self.g.results for a in r.accuracy])
        self._axes(*rect, 1.0, (0.25, 0.5, 0.75, 1.0))
        for r in self.g.results:
            self._polyline(self._series_pts(r.accuracy, 0, rect, 1.0, tmax), SERIES[(r.stage - 1) % 4])

    # ========================================================
    # 文字（論理座標で指定し、k 倍してオフスクリーンの解像度で描く）
    # ========================================================
    def _t(self, img, text, x, y, scale=0.45, color=INK, thick=1):
        k = self.k
        cv2.putText(img, text, (int(x * k), int(y * k)), cv2.FONT_HERSHEY_SIMPLEX, scale * k, (*color, 255),
                    max(1, round(thick * k)), cv2.LINE_AA)

    def _jt(self, img, ja, en, x, y, size=18, color=(255, 255, 255), anchor="left"):
        """日本語が書ければ日本語、書けなければ英語（cv2）。y は文字の上端"""
        k = self.k
        if self.jtext.available:
            px = int(round(size * k))
            if anchor == "right":
                w, _ = self.jtext.size(ja, px)
                self.jtext.draw(img, ja, x * k - w, y * k, px, color)
            else:
                self.jtext.draw(img, ja, x * k, y * k, px, color, anchor)
        else:
            (tw, _), _ = cv2.getTextSize(en, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)
            if anchor == "center":
                x -= tw / 2
            elif anchor == "right":
                x -= tw
            self._t(img, en, x, y + size * 0.8, size / 30, color)

    def _box(self, img, x0, y0, x1, y1, color):
        k = self.k
        cv2.rectangle(img, (int(x0 * k), int(y0 * k)), (int(x1 * k), int(y1 * k)), color, -1)

    def _message(self):
        g = self.g
        if g.message and time.monotonic() < g.message_until and g.state not in ("stage_clear", "result", "title"):
            return g.message
        return None

    def _step(self, m):
        """ミッションの今の手順（0〜3）"""
        if m.held is None:
            return 0 if self.g.selected is None else 1
        return 2 if self.g.selected != "goal" else 3

    def _hint(self, m):
        g = self.g
        if g.robot.moving:
            return "ロボットが移動中…（その間に手をカメラの中央へ戻しておく）"
        return ["アイテムを約 1 秒見つめて選ぶ", "手を動かしてアームを伸ばし、手を握ってつかむ",
                "ゴールを約 1 秒見つめる", "アームをゴールの上へ動かし、手を開いて置く"][self._step(m)]

    def _text_game(self, items, fps):
        img = self.overlay_img
        g = self.g
        cx, cy = AX + AW / 2, AY + AH / 2
        st = g.stage
        m = g.mission if g.state == "play" else None

        # ---- 状態ごとの大きな表示 ----
        if g.state == "connect":
            k = min(1.0, g.state_t / config.MORPH_SEC)
            self._box(img, cx - 230, cy - 50, cx + 230, cy + 50, (6, 10, 24, 220))
            self._jt(img, "脳と接続中…", "LINKING TO THE BRAIN...", cx, cy - 38, 32, ACCENT, "center")
            self._box(img, cx - 180, cy + 22, cx + 180, cy + 30, (60, 70, 90, 255))
            self._box(img, cx - 180, cy + 22, cx - 180 + 360 * k, cy + 30, (*ACCENT, 255))
        elif g.state == "gaze_calib":
            cal = g.gaze.calib
            n = len(cal.points)
            self._jt(img, f"視線同期  {min(cal.index + 1, n)} / {n}", f"GAZE SYNC {min(cal.index + 1, n)}/{n}",
                     cx, AY + 20, 28, ACCENT, "center")
            face = g.vision.get_state().face_detected if g.vision else False
            self._jt(img, "光る点を見つめてください（頭は動かさずに）" if face else "顔が見つかりません：カメラの正面へ",
                     "Look at the glowing dot" if face else "Face not found", cx, AY + 62, 18,
                     INK if face else BAD, "center")
        elif g.state == "stage_intro":
            self._text_stage_intro(st)
        elif g.state == "stage_clear":
            self._text_stage_clear(g.results[-1])
        elif g.state == "result":
            self._text_result()

        if g.hand_calib.active:
            hc = g.hand_calib
            title = "再同期フェーズ" if g.recalibrating else "同期フェーズ"
            self._jt(img, f"{title}　残り {hc.remaining:4.1f} 秒", f"HAND SYNC {hc.remaining:4.1f}s", cx, AY + 18,
                     26, MAGENTA, "center")
            if hc.phase == "grip":
                msg = "握る！" if hc.grip_cue > 0.5 else "開く！"
                self._jt(img, msg, "GRIP!" if hc.grip_cue > 0.5 else "OPEN!", cx, AY + 56, 30, (255, 255, 255),
                         "center")
            else:
                self._jt(img, PHASE_JA[hc.phase], hc.phase, cx, AY + 60, 18, INK, "center")
            self._jt(img, f"記録 {hc.valid_sec:4.1f} 秒（手が映っている間だけ記録）", f"rec {hc.valid_sec:4.1f}s",
                     AX + AW - 12, AY + AH - 34, 14, DIM, "right")

        # ---- プレイ中の案内 ----
        if m is not None and not g.recalibrating:
            self._jt(img, self._hint(m), self._hint(m), cx, AY + 15, 21, (255, 255, 255), "center")
        if g.banner == "electrode" and time.monotonic() < g.banner_until:
            self._jt(img, "！ 電極劣化：ニューロンが脱落した", "! ELECTRODE FAILURE", cx, cy - 44, 30, (255, 235, 210),
                     "center")
            self._jt(img, "L で再同期（約 30 秒、制限時間は進む）／ そのまま手で補って続けることもできる",
                     "L: re-sync or keep going", cx, cy + 4, 17, (255, 235, 210), "center")

        # ---- アイテムとゴールの名前 ----
        if g.state == "play":
            for it in items:
                if it.delivered or it.held:
                    continue
                p = world_to_screen(it.pos)
                r = world_radius(it.radius) + 18
                name = COLOR_JA.get(it.color, "登録色") + ("（実物）" if it.kind == "real" else "")
                self._jt(img, name, it.color, p[0] + r * 0.7, p[1] - r * 0.7 - 16, 14, rgb255(it.rgb))
        if g.state not in ("stage_clear", "result", "stage_intro"):
            gp = world_to_screen(g.goal.pos)
            self._jt(img, "ゴール", "GOAL", gp[0], gp[1] + world_radius(g.goal.radius) + 4, 15, (90, 255, 140),
                     "center")
        msg = self._message()
        if msg is not None:
            self._jt(img, msg, msg, cx, AY + AH - 50, 20, (255, 255, 255), "center")

        self._text_brain()
        self._text_cards(fps)
        rx, ry, rw, rh = config.RASTER_RECT
        label = f"スパイク（運動野の手の領域：方向 {g.cortex.n} 個 + 把持 {g.cortex.n_grip} 個 = 上の白い行）"
        self._box(img, rx, ry, rx + self._text_w(label, 12) + 10, ry + 17, (6, 8, 16, 210))
        self._jt(img, label, "spikes", rx + 4, ry + 1, 12, (215, 220, 235))

    def _text_stage_intro(self, st):
        img = self.overlay_img
        g = self.g
        cx = AX + AW / 2
        y0 = AY + 140
        self._jt(img, f"ステージ {st.number} / {len(STAGES)}", f"STAGE {st.number}", cx, y0, 18, DIM, "center")
        self._jt(img, st.title_ja.split("：")[-1], st.title_en, cx, y0 + 26, 40, (255, 255, 255), "center")
        self._jt(img, st.note_ja, st.note_ja, cx, y0 + 90, 19, ACCENT, "center")
        info = (f"ニューロン {st.n_neurons} 個 + 把持 {g.cortex.n_grip} 個　／　ノイズ {st.noise:.0f} Hz　／　"
                f"ゴールの大きさ {'小' if st.goal_radius < 0.06 else '大'}")
        self._jt(img, info, info, cx, y0 + 128, 16, DIM, "center")
        self._jt(img, f"目標：アイテムを {config.ITEMS_PER_STAGE} 個ゴールへ運ぶ（制限時間 {config.STAGE_TIME:.0f} 秒）",
                 "Deliver items to the GOAL", cx, y0 + 172, 21, GOLD, "center")
        self._jt(img, "はじめに「同期フェーズ」：マゼンタのターゲットを右手でゆっくり追い、最後は合図で握る・開く",
                 "Next: hand sync", cx, y0 + 220, 16, MAGENTA, "center")
        self._jt(img, "SPACE で開始", "SPACE to start", cx, y0 + 262, 18, INK, "center")

    def _text_brain(self):
        img = self.overlay_img
        self._jt(img, "脳（左半球 ＝ 右手を動かす側）", "BRAIN (left hemisphere)", BX + 8, BY + 6, 14, DIM)
        y = BY + BH - 44
        self._box(img, BX, y - 4, BX + BW, BY + BH, (6, 8, 16, 205))     # 凡例の背景（脳の絵に重なっても読めるように）
        for name, label in (("motor", "運動野（手）"), ("somato", "体性感覚野"), ("fef", "前頭眼野"), ("visual", "視覚野")):
            gl_ = self.g.brain.glow[REGION_ID[name]]
            c = rgb255(REGION_COLORS[name])
            x = BX + 10 + (0 if name in ("motor", "fef") else 200)
            yy = y + (0 if name in ("motor", "somato") else 20)
            self._box(img, x, yy + 3, x + 12, yy + 15, (*c, 255) if gl_ > 0.3 else (70, 75, 90, 255))
            self._jt(img, label + (" ●" if gl_ > 0.3 else ""), name, x + 18, yy, 14, c if gl_ > 0.3 else DIM)
        if self.g.model.simplified:
            self._box(img, BX, BY + 26, BX + BW, BY + 50, (90, 30, 10, 220))
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", BX + 6, BY + 28, 15, (255, 210, 120))

    def _text_cards(self, fps):
        img = self.overlay_img
        g = self.g
        st = g.stage
        m = g.mission
        in_play = g.state == "play" and m is not None

        # ---- ステージとスコア ----
        x, y, w, h = CARD_STAGE
        self._jt(img, f"ステージ {st.number}/{len(STAGES)}", f"STAGE {st.number}", x + 14, y + 8, 14, DIM)
        self._jt(img, st.title_ja.split("：")[-1], st.title_en, x + 14, y + 28, 20, (255, 255, 255))
        score = m.result.score if in_play else 0
        self._jt(img, "スコア", "SCORE", x + w - 14, y + 8, 13, DIM, "right")
        self._jt(img, f"{score}", f"{score}", x + w - 14, y + 24, 30, GOLD, "right")
        tl = m.time_left if in_play else config.STAGE_TIME
        done = m.result.delivered if in_play else 0
        fails = m.result.failures if in_play else 0
        pips = "●" * done + "○" * (config.ITEMS_PER_STAGE - done)
        self._jt(img, f"残り {tl:3.0f} 秒", f"{tl:3.0f}s", x + 14, y + 54, 14, BAD if tl < 20 else INK)
        self._jt(img, f"運搬 {pips}　失敗 {fails}", f"{done}/{config.ITEMS_PER_STAGE}", x + w - 14, y + 54, 14, INK,
                 "right")

        # ---- ミッションの手順 ----
        x, y, w, h = CARD_MISSION
        self._jt(img, "ミッションの手順", "MISSION", x + 14, y + 8, 14, DIM)
        cur = self._step(m) if in_play else -1
        for i, (ja, en) in enumerate(STEPS):
            yy = y + 30 + i * 20
            if i < cur:
                mark, col = "◎", GOOD
            elif i == cur:
                mark, col = "▶", (255, 255, 255)
            else:
                mark, col = "・", DIM
            self._jt(img, f"{mark} {i + 1}. {ja}", en, x + 18, yy, 15 if i == cur else 14, col)

        # ---- BCI の状態 ----
        x, y, w, h = CARD_BCI
        c = g.cortex
        c.metrics.compute()
        dec = "線形デコーダ（同期ずみ）" if c.calibrated else "PV 法（同期前）"
        self._jt(img, "BCI の状態", "BCI STATUS", x + 14, y + 8, 14, DIM)
        self._jt(img, dec, dec, x + w - 14, y + 8, 14, ACCENT if c.calibrated else DIM, "right")
        corr = c.metrics.corr
        self._jt(img, "デコード精度", "accuracy", x + 14, y + 40, 14, INK)
        self._jt(img, "--" if corr is None else f"相関 {corr:.2f}", "", x + w - 14, y + 26, 12, DIM, "right")
        self._jt(img, "ニューロン", "neurons", x + 14, y + 64, 14, INK)
        self._jt(img, f"{c.n_alive}/{c.n_all} 個", "", x + w - 14, y + 80, 12, BAD if c.n_alive < c.n_all else DIM,
                 "right")
        closed = g.grip_switch.closed
        self._jt(img, "把持", "grip", x + 14, y + 98, 14, INK)
        self._jt(img, "握っている" if closed else "開いている", "closed" if closed else "open", x + 50, y + 99, 12,
                 GOLD if closed else DIM)
        self._jt(img, "開く", "open", x + 120, y + 122, 11, DIM)
        self._jt(img, "握る", "grip", x + w - 20, y + 122, 11, DIM, "right")
        if c.n_alive < c.n_all and in_play and m.degraded and not g.recalibrating:
            self._jt(img, "電極劣化中：L で再同期", "degraded: L", x + 14, y + 80, 12, (255, 170, 120))

        # ---- 下：操作とカメラの状態 ----
        x, y = BX, H - 50
        self._jt(img, "L 再同期　C 視線同期　G 注視点　V 仮想/実物", "keys", x, y - 2, 12, DIM)
        self._jt(img, "P 色の登録　R ロボットを戻す　F11 全画面　ESC 終了", "keys", x, y + 14, 12, DIM)
        y += 2
        if g.test_mode:
            st_txt = f"テストモード（マウス）　描画 {fps:4.1f} fps"
            col = (255, 200, 110)
        else:
            vs = g.vision.get_state()
            st_txt = (f"カメラ {vs.fps:4.1f} fps　顔 {'○' if vs.face_detected else '×'}　手 {'○' if vs.hand.detected else '×'}"
                      f"　描画 {fps:4.1f} fps　視線の補正 ({g.gaze.offset[0]:+.0f}, {g.gaze.offset[1]:+.0f})")
            col = GOOD if vs.face_detected and vs.hand.detected else BAD
        self._jt(img, st_txt, "status", x, y + 28, 11, col)

    def _text_stage_clear(self, res):
        img = self.overlay_img
        cx = AX + AW / 2
        head = f"ステージ {res.stage} クリア！" if res.cleared else f"ステージ {res.stage} タイムアップ"
        self._jt(img, head, f"STAGE {res.stage} {'CLEAR' if res.cleared else 'TIME UP'}", cx, AY + 36, 36,
                 GOOD if res.cleared else BAD, "center")
        bonus = f"（+{int(res.time_left * config.TIME_SCORE)}）" if res.cleared else ""
        line = f"運搬 {res.delivered} / {config.ITEMS_PER_STAGE}　　失敗 {res.failures} 回　　残り時間 {res.time_left:4.1f} 秒{bonus}"
        self._jt(img, line, line, cx, AY + 100, 20, INK, "center")
        self._jt(img, f"スコア  {res.score}", f"SCORE {res.score}", cx, AY + 140, 32, GOLD, "center")
        r_corr, r_rmse = self.graph_rect("stage")
        self._jt(img, "デコード精度の推移（直近 5 秒の相関係数。高いほど良い）", "correlation", r_corr[0], r_corr[1] - 24,
                 15, rgb255(SERIES[0]))
        self._jt(img, "速度の誤差 RMSE [画面/秒]（低いほど良い）", "RMSE", r_rmse[0], r_rmse[1] - 24, 15,
                 rgb255(SERIES[1]))
        for rect in (r_corr, r_rmse):
            for v in (0.0, 0.5, 1.0):
                self._t(img, f"{v:.1f}", rect[0] - 34, rect[1] + rect[3] - rect[3] * v + 4, 0.38, DIM)
        tmax = max(10.0, max((a[0] for a in res.accuracy), default=10.0))
        self._t(img, "0 s", r_rmse[0], r_rmse[1] + r_rmse[3] + 16, 0.38, DIM)
        self._jt(img, f"{tmax:.0f} 秒（プレイ時間）", f"{tmax:.0f} s", r_rmse[0] + r_rmse[2], r_rmse[1] + r_rmse[3] + 4,
                 13, DIM, "right")
        if any(ev.startswith("degrade") for _, ev in res.events):
            self._jt(img, "赤い線 = 電極劣化", "red = failure", r_corr[0] + r_corr[2], r_corr[1] - 24, 14,
                     (255, 120, 110), "right")
        self._jt(img, "SPACE で次へ", "SPACE: next", cx, AY + AH - 32, 18, INK, "center")

    def _text_result(self):
        img = self.overlay_img
        cx = AX + AW / 2
        res = self.g.results
        total = sum(r.score for r in res)
        self._jt(img, "ミッション終了", "MISSION COMPLETE", cx, AY + 26, 36, (255, 255, 255), "center")
        self._jt(img, f"合計スコア  {total}", f"TOTAL {total}", cx, AY + 76, 30, GOLD, "center")
        heads = ["ステージ", "スコア", "運搬", "失敗", "平均の精度"]
        xs = [AX + 130, AX + 270, AX + 400, AX + 510, AX + 620]
        for hx, hd in zip(xs, heads):
            self._jt(img, hd, hd, hx, AY + 134, 15, DIM, "center")
        for i, r in enumerate(res):
            yy = AY + 160 + i * 26
            cs = [a[1] for a in r.accuracy if a[1] is not None]
            vals = [f"{r.stage}", f"{r.score}", f"{r.delivered}/{config.ITEMS_PER_STAGE}", f"{r.failures}",
                    f"{np.mean(cs):.2f}" if cs else "--"]
            for hx, v in zip(xs, vals):
                self._jt(img, v, v, hx, yy, 17, rgb255(SERIES[(r.stage - 1) % 4]), "center")
        rect = self.graph_rect("result")
        self._jt(img, "デコード精度（相関係数）の推移：ステージ別", "correlation by stage", rect[0], rect[1] - 24, 15, INK)
        for v in (0.0, 0.5, 1.0):
            self._t(img, f"{v:.1f}", rect[0] - 34, rect[1] + rect[3] - rect[3] * v + 4, 0.38, DIM)
        for i, r in enumerate(res):
            self._jt(img, f"ステージ {r.stage}", f"stage {r.stage}", rect[0] + rect[2] - (len(res) - i) * 96,
                     rect[1] - 24, 14, rgb255(SERIES[(r.stage - 1) % 4]))
        self._jt(img, "SPACE でタイトルへ", "SPACE: title", cx, AY + AH - 32, 18, INK, "center")
