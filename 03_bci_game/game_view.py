"""
game_view.py
------------
BCI オペレーターの描画（main.py の Game の状態を読んで描くだけ。ゲームの進行は変えない）。

  ・タイトル / 接続中：本物の脳（pial）を大きく表示し、開始すると inflated へふくらむ（言葉は最小限）
  ・プレイ：画面いっぱいの 3D の場面（scene3d.py）。上に HUD の帯（スコア・残り時間・集めた数・コンボ・同期の状態）、
            右下に小さなカメラ映像（手の骨格つき）、得点のポップアップ、電極劣化の赤いフラッシュ
  ・重ねて表示：視線同期、同期フェーズ（同期パッド + 遊び方）、カウントダウン、B キーの「裏側」（脳とラスター）
  ・結果：スコアとハイスコア、プレイ中の脳の活動の振り返り（光った領域、ラスターの一部、デコード精度の推移と劣化の時刻）

■ 解像度と文字の読みやすさ
  画面の配置は 1280×720 の「論理座標」で決めてある。3D・脳・図形は軽さのために RENDER_SCALE 倍の解像度の
  オフスクリーン（render_target.py）に描いてから、ウィンドウ（ディスプレイいっぱい）へ拡大する。
  **文字だけは、拡大したあとの実際の画面の解像度で描いて最後に重ねる**（拡大でぼやけないように）。
  さらに、文字には薄い暗い縁取りをつけ、小さすぎる文字（13 未満）は 13 にそろえる。
■ フォント（jtext.py）：機械的・無機質な見た目にするため、役割ごとにフォントを分ける
  "display" = 大きな数字・見出し（英数字は Orbitron）、"tech" = ラベル・計器（英数字は Rajdhani。字間を空ける）、
  "jp" = 日本語の本文（Noto Sans CJK、細め）。英字のラベルに小さな日本語を添える
  文字（英数字は cv2.putText、日本語は jtext）は 1 枚の透明な画像に 0.1 秒ごと（と画面が切りかわったとき）に描き直す。
"""

import math
import time

import cv2
import glfw
import numpy as np
from OpenGL import GL as gl

import config
from brain_model import WARNING_TEXT
from brain_renderer import REGION_COLORS
from gfx2d import Batch2D, TexturedQuad, rgb255
from jtext import JText
from layout import PH, PW, PX, PY, H, W, pad_to_screen
from mission import gripper_tip
from render_target import RenderTarget
from robot_view import RobotView

INK = (235, 238, 245)
DIM = (150, 160, 185)
GOOD = (140, 230, 160)
BAD = (255, 120, 110)
ACCENT = (130, 220, 255)
GOLD = (255, 225, 120)
MAGENTA = (255, 150, 230)
TITLE_BRAIN = (290, 96, 700, 420)
START_BUTTON = (W / 2 - 170, 604, 340, 56)
# タイトル画面のモードの選択（ノーマル = 一人で集める、チャレンジ = CPU のライバルと対戦）
MODE_BUTTONS = {"normal": (W / 2 - 215, 550, 210, 46), "challenge": (W / 2 + 5, 550, 210, 46)}
MODE_TEXT = {"normal": ("NORMAL", "ノーマル　一人で集める"), "challenge": ("CHALLENGE", "チャレンジ　ライバルと対戦")}
# タイトル画面の左のルールの説明と、チャレンジのライバルの強さのボタン
# タイトル画面の中央のルールの説明（脳の表示の代わり）。高さはモードで変わり、上下の真ん中にそろえる
RULE_AREA = (106, 506)                     # カードを置く上下の範囲（題名の下 〜 カメラ / テストの表示の上）
RULE_CARD_W = {"normal": 560, "challenge": 760}   # カードの横幅（チャレンジは強さのボタンを横に並べるので広く）
RULE_LINE_H = 26                           # ルールの 1 行の高さ
RULES = {
    "normal": ["120 秒で、物をできるだけ多くカゴへ", "小さい物ほど高得点（箱 < 缶 < 玉 < 星）",
               "続けて成功するとコンボで点が増える"],
    "challenge": ["120 秒で、ライバル（赤）より多く点を取れば勝ち",
                  "ライバルも擬似 BCI で動く（たまにつかみ損ねる）",
                  "ライバルを約 1.5 秒見つめると、少しのあいだ止まる",
                  "あなたが狙った物を横取りしにくる",
                  "運んでいるときにぶつかられると、落とすことがある"],
}
RIVAL_RED = (255, 120, 95)
# カメラ調整の画面
SETUP_VIEW = (320, 96, 640, 480)           # カメラ映像の位置と大きさ（論理座標）
SETUP_BUTTON = (W / 2 - 170, 600, 340, 54)  # 「ゲームへ進む」
ICE = (180, 230, 255)
SCENE_RECT = (0, 0, W, H)
HUD = config.HUD_HEIGHT
STEPS = ["物を約 1 秒見つめる → ロボットが近づく", "手を動かしてアームを伸ばし、手を握ってつかむ",
         "カゴを約 1 秒見つめる → ロボットが運ぶ", "アームをカゴの上へ動かし、手を開いて置く"]
PHASE_JA = {"ready": "準備：白い輪（あなたの手）をターゲットに重ねる", "reach": "ターゲットを手でゆっくり追う（8 方向）",
            "circle": "ターゲットを手でゆっくり追う（円）", "grip": "合図に合わせて手を握る・開く"}
REGION_JA = {"motor": "運動野", "somato": "体性感覚野", "fef": "前頭眼野", "visual": "視覚野"}


class GameView:
    def __init__(self, game):
        self.g = game
        self.rt = RenderTarget(round(W * config.RENDER_SCALE), round(H * config.RENDER_SCALE), config.MSAA_SAMPLES)
        self.kv = self.rt.width / W                # 論理座標 → オフスクリーンの画素（3D・図形）
        self.k = 1.0                               # 論理座標 → 実際の画面の画素（文字。ウィンドウの大きさで変わる）
        self.batch = Batch2D()
        self.pip_quad = TexturedQuad(gl.GL_LINEAR)
        self.setup_quad = TexturedQuad(gl.GL_LINEAR)
        self._setup_id = -1
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.result_raster = TexturedQuad(gl.GL_NEAREST)
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.overlay_img = None                    # 文字の画像（実際の画面の解像度。render で用意する）
        self._last_overlay = -1.0
        self._last_state = None
        self._pip_id = -1
        self._result_uploaded = None
        self.jtext = JText()

    # --------------------------------------------------------
    def window_to_layout(self, mx, my):
        """ウィンドウの座標（マウス）→ 論理座標（拡大と黒い帯を戻す）"""
        ww, wh = glfw.get_window_size(self.g.window)
        x, y, w, h = self.rt.letterbox(ww, wh)
        return np.array([(mx - x) / max(w, 1) * W, (my - y) / max(h, 1) * H])

    def _rule_card(self):
        """ルールのカードの位置 (x, y, 幅, 高さ)：中央に置き、モードで高さを変えて上下の真ん中にそろえる"""
        mode = self.g.mode
        h = 54 + len(RULES[mode]) * RULE_LINE_H + (96 if mode == "challenge" else 6)
        top, bottom = RULE_AREA
        w = RULE_CARD_W[mode]
        return (W / 2 - w / 2, top + (bottom - top - h) / 2, w, h)

    def _level_buttons(self):
        """ライバルの強さのボタン（ルールのカードの下のほう）"""
        x, y, w, h = self._rule_card()
        y0 = y + 54 + len(RULES["challenge"]) * RULE_LINE_H + 34
        bw, gap = 210, 16                      # 3 つを横に並べる
        x0 = W / 2 - (3 * bw + 2 * gap) / 2
        return {lv: (x0 + i * (bw + gap), y0, bw, 40) for i, lv in enumerate(("weak", "normal", "strong"))}

    def level_button_hit(self, p):
        """タイトル画面のライバルの強さのボタンの上なら、その強さ（チャレンジを選んでいるときだけ）"""
        if self.g.mode != "challenge":
            return None
        for lv, (x, y, w, h) in self._level_buttons().items():
            if x <= p[0] <= x + w and y <= p[1] <= y + h:
                return lv
        return None

    def mode_button_hit(self, p):
        """タイトル画面のモードのボタンの上なら、そのモード（"normal" / "challenge"）"""
        for mode, (x, y, w, h) in MODE_BUTTONS.items():
            if x <= p[0] <= x + w and y <= p[1] <= y + h:
                return mode
        return None

    def _degrade_mode(self):
        """電極劣化ありのモードか（プレイ中はそのゲームのモード、それ以外はタイトルで選んでいるモード）"""
        m = self.g.mission
        return m.degrade_enabled if m is not None else config.DEGRADE_ENABLED

    def setup_button_hit(self, p):
        x, y, w, h = SETUP_BUTTON
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    def start_button_hit(self, p):
        x, y, w, h = START_BUTTON
        return x <= p[0] <= x + w and y <= p[1] <= y + h

    def _vp(self, x, y, w, h):
        k = self.kv
        return (int(x * k), int((H - y - h) * k), int(w * k), int(h * k))

    def _full_viewport(self):
        gl.glViewport(0, 0, self.rt.width, self.rt.height)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _brain(self, rect, t, dt):
        c = self.g.cortex
        self.g.brain.draw(self._vp(*rect), t, dt, c.flash, c.activity, c.alive)
        self._full_viewport()

    def _card(self, rect, accent=(0.35, 0.55, 0.8), alpha=0.92):
        x, y, w, h = rect
        b = self.batch
        b.rect(x, y, w, h, (0.045, 0.055, 0.09, alpha))
        b.rect(x, y, w, 2, (*accent, 1.0))

    def _text_w(self, text, size, font="jp", tracking=0.0):
        """文字の幅（論理座標）"""
        if not self.jtext.available:
            return len(text) * size * 0.55
        return self.jtext.size(text, size, role=font, tracking=int(round(tracking * size)))[0]

    # ========================================================
    def render(self, t, dt, gaze, fps):
        g = self.g
        self.rt.bind()
        bg = config.SCENE_BG if config.SCENE_SHADING == "full" else (0.035, 0.04, 0.06)   # 3D の背景（遠くの色）
        gl.glClearColor(*bg, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._full_viewport()

        if g.state in ("title", "connect"):
            self._render_title(t, dt)
        elif g.state == "camera_setup":
            self._render_setup(t)
        elif g.state == "result":
            self._render_result(t, dt)
        else:
            self._render_game(t, dt, gaze)

        gl.glDisable(gl.GL_BLEND)
        self.rt.present(g.window)              # オフスクリーンをウィンドウいっぱいに拡大

        # ---- 文字：実際の画面の解像度で描いて重ねる ----
        fb_w, fb_h = glfw.get_framebuffer_size(g.window)
        x, y, w, h = self.rt.letterbox(fb_w, fb_h)
        if w <= 0 or h <= 0:
            glfw.swap_buffers(g.window)
            return
        if self.overlay_img is None or self.overlay_img.shape[:2] != (h, w):
            self.overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
            self.k = w / W
            self._last_overlay = -1.0
        if t - self._last_overlay >= config.OVERLAY_INTERVAL or g.state != self._last_state:
            self._last_state = g.state
            self.overlay_img[:] = 0
            if g.state in ("title", "connect"):
                self._text_title()
            elif g.state == "camera_setup":
                self._text_setup()
            elif g.state == "result":
                self._text_result()
            else:
                self._text_game(fps)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        gl.glViewport(x, y, w, h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)
        glfw.swap_buffers(g.window)

    # ========================================================
    # カメラ調整の画面
    # ========================================================
    def _render_setup(self, t):
        """カメラ映像（左右反転）と「ゲームへ進む」ボタン"""
        b = self.batch
        g = self.g
        b.rect(0, 0, W, H, (0.02, 0.025, 0.045, 1.0))
        vx, vy, vw, vh = SETUP_VIEW
        b.rect(vx - 3, vy - 3, vw + 6, vh + 6, (0.3, 0.45, 0.65, 1))
        x, y, w, h = SETUP_BUTTON
        pulse = 0.5 + 0.5 * math.sin(t * 3)
        b.rect(x, y, w, h, (0.08, 0.25, 0.4, 0.8 + 0.2 * pulse))
        b.rect(x, y, w, 2, (0.5, 0.85, 1.0, 1))
        b.rect(x, y + h - 2, w, 2, (0.5, 0.85, 1.0, 1))
        b.flush()
        fid, img = g.vision.get_setup() if g.vision is not None else (-1, None)
        if img is not None:
            if fid != self._setup_id:
                self.setup_quad.upload(img)
                self._setup_id = fid
            self.setup_quad.draw(vx, vy, vw, vh)

    def _text_setup(self):
        img = self.overlay_img
        self._jt(img, "CAMERA SETUP", "CAMERA SETUP", W / 2, 18, 28, (225, 240, 255), "center", font="display",
                 bold=True, tracking=0.15)
        self._jt(img, "顔と上半身（手を動かす範囲）が入るように、カメラの角度を調整してください", "adjust camera",
                 W / 2, 62, 16, INK, "center")
        x, y, w, h = SETUP_BUTTON
        self._jt(img, "ゲームへ進む", "START", W / 2, y + 6, 18, (255, 255, 255), "center", bold=True)
        self._jt(img, "SPACE / ENTER / CLICK", "", W / 2, y + 32, 11, ACCENT, "center", font="tech", tracking=0.3)
        self._jt(img, "ESC  タイトルへ戻る", "ESC: title", W / 2, 672, 12, DIM, "center", font="tech", tracking=0.15)

    # ========================================================
    # タイトル・接続中
    # ========================================================
    def _render_title(self, t, dt):
        b = self.batch
        b.rect(0, 0, W, H, (0.02, 0.025, 0.045, 1.0))
        for k in range(24):
            r = 120 + k * 28 + (t * 20) % 28
            b.ring(W / 2, 310, r, 1, (0.2, 0.5, 0.8, max(0.0, 0.10 - k * 0.004)), 64)
        if self.g.state == "title":
            x, y, w, h = START_BUTTON
            pulse = 0.5 + 0.5 * math.sin(t * 3)
            b.rect(x, y, w, h, (0.08, 0.25, 0.4, 0.8 + 0.2 * pulse))
            b.rect(x, y, w, 2, (0.5, 0.85, 1.0, 1))
            b.rect(x, y + h - 2, w, 2, (0.5, 0.85, 1.0, 1))
            for mode, (x, y, w, h) in MODE_BUTTONS.items():       # モードの選択（選んでいる方を明るく）
                on = mode == self.g.mode
                acc = (1.0, 0.45, 0.3) if mode == "challenge" else (0.4, 0.9, 0.6)
                b.rect(x, y, w, h, (*[c * (0.35 if on else 0.08) for c in acc], 0.9))
                for yy in (y, y + h - 2):
                    b.rect(x, yy, w, 2, (*acc, 1.0 if on else 0.3))
            self._card(self._rule_card(), (1.0, 0.45, 0.3) if self.g.mode == "challenge" else (0.4, 0.9, 0.6), 0.85)
            if self.g.mode == "challenge":                        # ライバルの強さ
                for lv, (x, y, w, h) in self._level_buttons().items():
                    on = lv == self.g.rival_level
                    b.rect(x, y, w, h, (0.85, 0.25, 0.2, 0.55) if on else (0.2, 0.08, 0.08, 0.6))
                    b.rect(x, y, 3, h, (1.0, 0.5, 0.35, 1.0 if on else 0.3))
        else:
            k = min(1.0, self.g.state_t / config.MORPH_SEC)
            b.rect(W / 2 - 200, 610, 400, 8, (0.15, 0.2, 0.3, 1))
            b.rect(W / 2 - 200, 610, 400 * k, 8, (0.5, 0.85, 1.0, 1))
        b.flush()
        if self.g.state != "title":            # 脳は「脳と接続中」の画面だけ（タイトルでは中央にルールを出す）
            self._brain(TITLE_BRAIN, t, dt)

    def _text_title(self):
        img = self.overlay_img
        g = self.g
        self._jt(img, "BCI OPERATOR", "BCI OPERATOR", W / 2, 20, 52, (225, 240, 255), "center", font="display",
                 bold=True, tracking=0.12)
        if g.state == "title":
            self._jt(img, "START", "START", W / 2, START_BUTTON[1] + 4, 26, (255, 255, 255), "center", font="display",
                     bold=True, tracking=0.3)
            for mode, (x, y, w, h) in MODE_BUTTONS.items():
                on = mode == g.mode
                en, ja = MODE_TEXT[mode]
                col = (255, 255, 255) if on else DIM
                self._jt(img, ("▶ " if on else "") + en, en, x + w / 2, y + 5, 16, col, "center", font="tech", bold=True,
                         tracking=0.2)
                self._jt(img, ja, en, x + w / 2, y + 25, 12, col, "center")
            # ルールの説明（中央のカード）と、チャレンジならライバルの強さ
            mode = g.mode
            cx_, cy_, cw_, ch_ = self._rule_card()
            y0 = cy_ + 14
            col = (255, 170, 140) if mode == "challenge" else GOOD
            self._jt(img, "RULES", "RULES", W / 2 - 6, y0, 15, col, "right", font="tech", bold=True, tracking=0.22)
            self._jt(img, "ルール", "", W / 2 + 6, y0 + 1, 13, DIM)
            for i, line in enumerate(RULES[mode]):
                self._jt(img, line, "rule", W / 2, y0 + 38 + i * RULE_LINE_H, 15,
                         INK if not line.startswith("（") else DIM, "center")
            if mode == "challenge":
                lvb = self._level_buttons()
                self._jt(img, "RIVAL LEVEL  ―  1 / 2 / 3", "", W / 2, lvb["weak"][1] - 22, 12, DIM, "center",
                         font="tech", tracking=0.15)
                for lv, (x, y, w, h) in lvb.items():
                    on = lv == g.rival_level
                    en, ja = config.RIVAL_LEVEL_NAMES[lv]
                    self._jt(img, ("▶ " if on else "") + en, en, x + 14, y + 10, 15, (255, 255, 255) if on else DIM,
                             font="tech", bold=True, tracking=0.2)
                    self._jt(img, ja, en, x + w - 12, y + 12, 13, (255, 255, 255) if on else DIM, "right")
            self._jt(img, "SPACE / CLICK", "", W / 2, START_BUTTON[1] + 33, 13, ACCENT, "center", font="tech",
                     tracking=0.3)
            if g.test_mode:
                self._jt(img, "TEST MODE  ―  マウスで操作（カメラなし）", "", W / 2, 522, 15, (255, 200, 110), "center",
                         font="tech", tracking=0.08)
            else:
                self._jt(img, "CAMERA MODE  ―  顔と右手をカメラに映す", "", W / 2, 522, 15, GOOD, "center", font="tech",
                         tracking=0.08)
        else:
            self._jt(img, "LINKING TO CORTEX", "", W / 2, 556, 26, ACCENT, "center", font="display", tracking=0.18)
            self._jt(img, "脳と接続中", "", W / 2, 586, 15, DIM, "center")
        if g.model.simplified:
            self._jt(img, "！" + WARNING_TEXT, "! simplified brain model", W / 2, 134, 15, (255, 210, 120), "center")

    # ========================================================
    # ゲーム（3D の場面 + 重ねる表示）
    # ========================================================
    def _render_game(self, t, dt, gaze):
        g = self.g
        b = self.batch
        m = g.mission
        held = m.held if m is not None else None
        sel = g.selected if isinstance(g.selected, int) else None
        focus = (g.dwell.focus, g.dwell.progress) if g.state == "play" and not g.recalibrating else None
        g.scene.render(self._vp(*SCENE_RECT), g.robot, g.field.items, held, t, dt, sel, rival=g.rival, focus=focus)
        self._full_viewport()

        if g.state == "play" and not g.recalibrating:
            # グリッパーの先（どこでつかむか分かるように）と、見つめている対象のリング
            tip = g.scene.project(gripper_tip(g.robot), RobotView.ARM_HEIGHT, SCENE_RECT)
            if tip is not None:
                b.ring(*tip, 9, 2, (1.0, 0.9, 0.4, 0.9), 18)
            if g.dwell.focus is not None and g.dwell.focus_pos is not None:
                if g.dwell.focus == "rival":         # ライバルを見つめている：青白いリング（たまると止まる）
                    self._progress_ring(g.dwell.focus_pos, 44, g.dwell.progress, (0.65, 0.9, 1.0))
                else:
                    self._progress_ring(g.dwell.focus_pos, 34, g.dwell.progress, (1.0, 0.85, 0.3))
            if g.selected == "goal":
                p = g.scene.project(np.array(config.GOAL_POS), 0.05, SCENE_RECT)
                if p is not None:
                    b.ring(*p, 46, 2, (1, 1, 1, 0.7), 36)
        if g.show_gaze and gaze is not None and g.state == "play":
            b.circle(*gaze, 6, (1.0, 1.0, 1.0, 0.85), 12)
            b.ring(*gaze, 11, 1.5, (0.2, 0.9, 1.0, 0.8), 20)

        # ---- 電極劣化の赤いフラッシュ ----
        left = g.flash_until - time.monotonic()
        if left > 0:
            a = left / config.DEGRADE_FLASH_SEC
            b.rect(0, 0, W, H, (0.9, 0.05, 0.02, 0.45 * a))
            for k in range(6):          # 画面の縁を赤く
                b.rect(0, HUD + k * 4, W, 4, (1, 0.1, 0.05, 0.25 * a * (6 - k) / 6))
                b.rect(0, H - (k + 1) * 4, W, 4, (1, 0.1, 0.05, 0.25 * a * (6 - k) / 6))

        # ---- 重ねる画面 ----
        if g.state == "gaze_calib":
            self._draw_gaze_calib(t)
            # 金色の確認の点のときだけ、注視点を出す（どこに視線が当たっているか自分で確かめられるように）
            if g.gaze.calib.active and g.gaze.calib.phase == "check" and gaze is not None:
                b.circle(*gaze, 6, (1.0, 1.0, 1.0, 0.85), 12)
                b.ring(*gaze, 11, 1.5, (1.0, 0.8, 0.3, 0.9), 20)
        if g.hand_calib.active:
            self._draw_hand_calib()
        if g.state == "countdown":
            b.rect(0, 0, W, H, (0.0, 0.01, 0.03, 0.45))

        # ---- HUD の帯 ----
        b.rect(0, 0, W, HUD, (0.03, 0.04, 0.07, 0.86))
        b.rect(0, HUD, W, 2, (0.3, 0.45, 0.65, 0.9))
        if m is not None and g.state != "gaze_calib":
            frac = m.time_left / config.GAME_TIME
            col = (1.0, 0.35, 0.3) if m.time_left < 20 else (0.4, 0.85, 1.0)
            b.rect(470, 44, 300, 6, (0.15, 0.18, 0.25, 1))
            b.rect(470, 44, 300 * frac, 6, (*col, 1))
            for i in range(len(config.COMBO_LEVELS)):
                on = i <= m.combo
                b.rect(830 + i * 22, 44, 16, 6, (1.0, 0.75, 0.25, 1) if on else (0.2, 0.22, 0.3, 1))
        if self._sync_state()[0] == "degraded":
            a = 0.5 + 0.5 * math.sin(t * 6)
            b.rect(1040, 8, 232, 48, (0.6, 0.06, 0.04, 0.55 + 0.35 * a))

        # ---- お知らせ ----
        msg = self._message()
        if msg is not None:
            w = self._text_w(msg, 19) + 40
            b.rect(W / 2 - w / 2, H - 70, w, 38, (0.02, 0.04, 0.08, 0.85))

        # ---- 小さなカメラ映像（手の骨格つき）----
        px, py, pw, ph = self._pip_rect()
        if not g.test_mode:
            b.rect(px - 2, py - 2, pw + 4, ph + 4, (0.3, 0.45, 0.65, 1))
        # ---- 裏側（B キー）----
        if g.show_backstage:
            self._card(config.BRAIN_OVERLAY_RECT, (1.0, 0.55, 0.2), 0.62)
            self._card(config.RASTER_OVERLAY_RECT, (1.0, 0.55, 0.2), 0.62)
        b.flush()

        if not g.test_mode:
            fid, pip = g.vision.get_pip()
            if pip is not None:
                if fid != self._pip_id:
                    self.pip_quad.upload(pip)
                    self._pip_id = fid
                self.pip_quad.draw(px, py, pw, ph)
        if g.show_backstage:
            bx, by, bw, bh = config.BRAIN_OVERLAY_RECT
            self._brain((bx, by + 20, bw, bh - 20), t, dt)
            rx, ry, rw, rh = config.RASTER_OVERLAY_RECT
            self.raster_quad.upload(g.cortex.raster)
            self.raster_quad.draw(rx + 6, ry + 22, rw - 12, rh - 28)

        # ---- 得点のポップアップ（カゴの上へ浮かぶ）----
        now = time.monotonic()
        g.popups = [p for p in g.popups if now - p[2] < 1.4]

    def _progress_ring(self, p, r, progress, color):
        b = self.batch
        b.ring(*p, r, 3, (*color, 0.25), 48)
        for k in range(max(1, int(48 * progress))):
            a0 = -math.pi / 2 + 2 * math.pi * k / 48
            a1 = -math.pi / 2 + 2 * math.pi * (k + 1) / 48
            b.line(p[0] + r * math.cos(a0), p[1] + r * math.sin(a0),
                   p[0] + r * math.cos(a1), p[1] + r * math.sin(a1), 5, (*color, 0.95))

    def _draw_gaze_calib(self, t):
        """視線同期：学習の点（水色）→ 確認の点（金色）。終わった点は小さな点で残す"""
        b = self.batch
        b.rect(0, HUD, W, H - HUD, (0.0, 0.02, 0.05, 0.7))
        cal = self.g.gaze.calib
        check = cal.phase == "check"
        col = (1.0, 0.8, 0.3) if check else (0.4, 0.9, 1.0)
        for i, p in enumerate(cal.check_points if check else cal.points):
            b.circle(*p, 5, (*col, 0.9) if i < cal.index else (0.5, 0.55, 0.65, 0.5), 12)
        p = cal.current_point
        if p is None:
            return
        pulse = 0.5 + 0.5 * math.sin(t * 8)
        shrink = 1.0 - cal.settle_progress
        b.ring(*p, 16 + 40 * shrink, 2, (*col, 0.6 + 0.3 * pulse), 40)
        b.circle(*p, 7, (1.0, 1.0, 1.0, 1.0), 16)
        if cal.collecting:
            self._progress_ring(p, 26, cal.collect_progress, col)

    def _pip_rect(self):
        """小さなカメラ映像の位置。視線同期の間は、点を隠さないよう小さくして、点のない場所へ"""
        if self.g.state == "gaze_calib":     # 9 点・確認の点のどれとも重ならない位置（下の、中央と右の点の間）に小さく
            return (800, H - 14 - 120, 160, 120)
        return config.PIP_RECT

    def _draw_hand_calib(self):
        """同期フェーズ：画面中央の同期パッドに、マゼンタのターゲットと自分の手の位置（白い輪）"""
        b = self.batch
        hc = self.g.hand_calib
        b.rect(0, HUD, W, H - HUD, (0.03, 0.0, 0.05, 0.72))
        b.rect(PX, PY, PW, PH, (0.06, 0.04, 0.09, 0.95))
        for k in range(1, 4):
            b.line(PX + PW * k / 4, PY, PX + PW * k / 4, PY + PH, 1, (0.2, 0.15, 0.25, 1))
            b.line(PX, PY + PH * k / 4, PX + PW, PY + PH * k / 4, 1, (0.2, 0.15, 0.25, 1))
        p = pad_to_screen(hc.target)
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
            q = pad_to_screen(np.clip(h.pos, -0.05, 1.05))
            b.ring(*q, 16, 3, (1, 1, 1, 0.9), 28)
        b.rect(PX, PY + PH + 6, PW, 8, (0.15, 0.1, 0.18, 1.0))
        b.rect(PX, PY + PH + 6, PW * hc.progress, 8, (*mag, 1.0))
        if not self.g.recalibrating:
            self._card((40, 150, 330, 250), (0.4, 0.9, 1.0))       # 遊び方

    def _text_hud(self, img, g, m):
        """HUD の数字（スコア・ライバル / 集めた数・残り時間・コンボ・ライバル / 同期の状態）"""
        # ---- HUD：スコア・集めた数・残り時間・コンボ・同期の状態（計器の表示）----
        score = m.result.score if m else 0
        self._jt(img, "SCORE", "", 24, 7, 13, DIM, font="tech", bold=True, tracking=0.25)
        self._jt(img, f"{score:05d}", f"{score}", 24, 22, 30, GOLD, font="display", bold=True, tracking=0.06)
        rv = g.rival
        if rv is not None:                     # チャレンジ：集めた数の代わりにライバルのスコア
            self._jt(img, "RIVAL", "", 250, 7, 13, RIVAL_RED, font="tech", bold=True, tracking=0.25)
            self._jt(img, f"{rv.score:05d}", f"{rv.score}", 250, 22, 30, RIVAL_RED, font="display", bold=True,
                     tracking=0.06)
        else:
            self._jt(img, "ITEMS", "", 250, 7, 13, DIM, font="tech", bold=True, tracking=0.25)
            self._jt(img, f"{m.result.collected if m else 0:02d}", "", 250, 22, 30, INK, font="display", bold=True)
        tl = m.time_left if m else config.GAME_TIME
        self._jt(img, "TIME", "", 470, 7, 13, DIM, font="tech", bold=True, tracking=0.25)
        self._jt(img, f"{tl:05.1f}", f"{tl:3.0f}", 770, 10, 24, BAD if tl < 20 else INK, "right", font="display")
        self._jt(img, "COMBO", "", 830, 7, 13, DIM, font="tech", bold=True, tracking=0.25)
        self._jt(img, f"x{m.combo_mult:.1f}" if m else "x1.0", "", 925, 12, 24, GOLD if (m and m.combo > 0) else INK,
                 font="display")
        if rv is not None:                    # チャレンジ：ライバルの状態（動いている / 止まっている / 止められるまで）
            st, left = rv.status
            if st == "frozen":
                code, sub, col = "FROZEN", f"止まっている　残り {left:3.1f} 秒", ICE
            elif st == "cooldown":
                code, sub, col = "ACTIVE", f"止められるまで {left:3.0f} 秒", RIVAL_RED
            else:
                code, sub, col = "ACTIVE", "見つめて止められる", RIVAL_RED
            self._jt(img, f"RIVAL  {config.RIVAL_LEVEL_NAMES[rv.level][0]}", "", 1050, 7, 13, DIM, font="tech",
                     bold=True, tracking=0.25)
            self._jt(img, code, code, 1050, 22, 18, col, font="tech", bold=True, tracking=0.12)
            self._jt(img, sub, "", 1050, 42, 13, ICE if st == "ready" else DIM)
        elif not self._degrade_mode():        # ノーマル（劣化なし）：同期の状態は出さず、モードだけ
            self._jt(img, "MODE", "", 1050, 7, 13, DIM, font="tech", bold=True, tracking=0.25)
            self._jt(img, "NORMAL", "NORMAL", 1050, 22, 18, DIM, font="tech", bold=True, tracking=0.12)
        else:                                 # 電極劣化あり（config.DEGRADE_ENABLED）：同期の状態
            state, text, col = self._sync_state()
            code = {"good": "LINK  STABLE", "degraded": "LINK  DEGRADED", "resync": "RESYNC",
                    "none": "LINK  OFFLINE"}[state]
            self._jt(img, "NEURAL LINK", "", 1050, 7, 13, DIM if state != "degraded" else (255, 220, 210),
                     font="tech", bold=True, tracking=0.25)
            self._jt(img, code, code, 1050, 22, 18, col, font="tech", bold=True, tracking=0.12)
            sub = {"good": "同期 良好", "degraded": "L で再同期", "resync": f"残り {g.hand_calib.remaining:3.0f} 秒",
                   "none": "同期前"}[state]
            self._jt(img, sub, "", 1050, 42, 13, col if state == "degraded" else DIM)


    def _sync_state(self):
        """(状態, 表示する文字, 色)：同期の状態"""
        g = self.g
        if g.recalibrating:
            return "resync", f"再同期中 {g.hand_calib.remaining:3.0f} 秒", MAGENTA
        if not g.cortex.calibrated:
            return "none", "同期前", DIM
        if g.cortex.dropped_since_calib > 0:
            n = g.mission.degrade_count if g.mission else 0
            return "degraded", f"電極劣化{'×' + str(n) if n > 1 else ''}！ L で再同期", (255, 235, 220)
        return "good", "同期 良好", GOOD

    def _message(self):
        g = self.g
        if g.message and time.monotonic() < g.message_until:
            return g.message
        return None

    # ========================================================
    # 結果
    # ========================================================
    def _render_result(self, t, dt):
        b = self.batch
        f = self.g.final
        b.rect(0, 0, W, H, (0.03, 0.035, 0.06, 1.0))
        self._card((30, 30, 560, 640), (1.0, 0.85, 0.35))
        self._card((610, 30, 640, 640), (1.0, 0.55, 0.2))
        rank = f["rank"]
        if rank is not None:
            b.rect(46, 300 + rank * 30 - 4, 528, 28, (0.35, 0.28, 0.08, 0.9))
        # 領域の使われ方のバー
        for i, name in enumerate(("motor", "somato", "fef", "visual")):
            v = f["usage"].get(name, 0.0)
            x = 640 + (i % 2) * 300
            y = 352 + (i // 2) * 26
            b.rect(x + 100, y + 4, 150, 10, (0.15, 0.17, 0.24, 1))
            b.rect(x + 100, y + 4, 150 * v, 10, (*REGION_COLORS[name], 1))
        # デコード精度の推移（相関係数）と、劣化・再同期・ライバルを止めた時刻
        gx, gy, gw, gh = 660, 530, 560, 110
        b.rect(gx, gy, gw, gh, (0.06, 0.07, 0.11, 1))
        for v in (0.25, 0.5, 0.75):
            b.line(gx, gy + gh * (1 - v), gx + gw, gy + gh * (1 - v), 1, (0.18, 0.21, 0.3, 1))
        rec = f["record"]
        tmax = max(config.GAME_TIME, 1.0)
        for tt, what in rec.marks:                    # 劣化の印は、劣化が起きたとき（DEGRADE_ENABLED）だけある
            x = gx + gw * tt / tmax
            col = {"degrade": (1.0, 0.3, 0.25, 0.95), "freeze": (0.65, 0.9, 1.0, 0.9)}.get(what, (1.0, 0.45, 0.85, 0.8))
            b.line(x, gy, x, gy + gh, 2 if what in ("degrade", "freeze") else 1.5, col)
        pts = [(gx + gw * tt / tmax, gy + gh * (1 - min(max(c, 0), 1))) for tt, c, _ in rec.accuracy if c is not None]
        for p, q in zip(pts[:-1], pts[1:]):
            b.line(*p, *q, 2, (0.35, 0.65, 1.0, 1))
        b.flush()
        self._brain((630, 70, 600, 270), t, dt)
        if rec.raster is not None:
            if self._result_uploaded is not rec:
                self.result_raster.upload(rec.raster)
                self._result_uploaded = rec
            self.result_raster.draw(660, 428, 560, 64)

    def _text_result(self):
        img = self.overlay_img
        f = self.g.final
        r = f["result"]
        self._label(img, "RESULT", "結果", 50, 46)
        by = "　".join(f"{v['name']} {r.by_type.get(k, 0)}" for k, v in config.ITEM_TYPES.items())
        rv = f.get("rival")
        if rv is None:
            self._jt(img, f"{r.score}", f"{r.score}", 300, 74, 66, GOLD, "center", font="display", bold=True)
            self._jt(img, "PTS", "", 430, 120, 18, GOLD, font="tech", tracking=0.2)
            self._jt(img, f"集めた数 {r.collected}（{by}）", "collected", 310, 170, 16, INK, "center")
            self._jt(img, f"最高コンボ ×{r.max_combo:.1f}　　失敗 {r.failures}", "combo", 310, 198, 16, INK, "center")
        else:                                  # チャレンジ：勝ち負けと両者のスコア、止めた回数
            out_en, out_ja, out_col = {"win": ("YOU WIN", "勝ち", GOLD), "lose": ("YOU LOSE", "負け", RIVAL_RED),
                                       "draw": ("DRAW", "引き分け", INK)}[rv["outcome"]]
            self._jt(img, out_en, out_ja, 310, 64, 44, out_col, "center", font="display", bold=True, tracking=0.1)
            self._jt(img, f"{r.score}", "", 220, 122, 28, GOLD, "right", font="display", bold=True)
            self._jt(img, "VS", "", 310, 128, 16, DIM, "center", font="tech", tracking=0.2)
            self._jt(img, f"{rv['score']}", "", 400, 122, 28, RIVAL_RED, font="display", bold=True)
            self._jt(img, f"あなた {r.collected} 個　　ライバル（{config.RIVAL_LEVEL_NAMES[rv['level']][1]}）"
                     f"{rv['collected']} 個", "collected", 310, 170, 15, INK, "center")
            self._jt(img, f"止めた回数 {rv['freezes']}　　取られた {rv['steals']}　　落とされた {rv['knocked']}"
                     f"　　失敗 {r.failures}", "counts", 310, 198, 15, INK, "center")
        rank = f["rank"]
        key = f.get("key", "normal")
        mode_ja = "ノーマル" if key == "normal" else \
            f"チャレンジ・{config.RIVAL_LEVEL_NAMES[key.split('_', 1)[1]][1]}"
        self._label(img, "HIGH SCORES", f"ハイスコア（{mode_ja}）"
                    + (f"　今回 {rank + 1} 位" if rank is not None else ""), 50, 254, GOLD if rank is not None else ACCENT)
        for i, sc in enumerate(f["scores"]):
            y = 300 + i * 30
            col = GOLD if i == rank else INK
            sub = col if i == rank else DIM
            self._jt(img, f"{i + 1:02d}", "", 60, y, 17, sub, font="display")
            self._jt(img, f"{sc['score']}", "", 250, y, 18, col, "right", font="display")
            self._jt(img, f"{sc.get('collected', 0)} ITEMS", "", 360, y + 2, 14, sub, "right", font="tech", tracking=0.1)
            self._jt(img, sc["date"], "", 560, y + 2, 14, sub, "right", font="tech", tracking=0.08)
        self._label(img, "BRAIN ACTIVITY", "プレイ中の脳の活動", 630, 46, (255, 190, 140))
        self._jt(img, "光った領域（左半球。明るいほど長く使った）", "regions", 630, 80, 13, DIM)
        for i, name in enumerate(("motor", "somato", "fef", "visual")):
            x = 640 + (i % 2) * 300
            y = 352 + (i // 2) * 26
            self._jt(img, REGION_JA[name], name, x, y, 14, rgb255(REGION_COLORS[name]))
            self._jt(img, f"{f['usage'].get(name, 0.0) * 100:3.0f}%", "", x + 290, y, 14, DIM, "right", font="tech")
        self._label(img, "SPIKES", "最後に物をつかんだ直前の 4 秒（上の白い行 = 把持ニューロン）", 660, 404, DIM, 13)
        kinds = {w for _, w in f["record"].marks}
        legend = "相関係数の推移" + ("　赤 = 電極劣化" if "degrade" in kinds else "") \
            + ("　水色 = ライバルを止めた" if f.get("rival") else "") + "　ピンク = 再同期"
        self._label(img, "DECODING ACCURACY", legend, 660, 504, DIM, 13)
        self._jt(img, "0 S", "0", 660, 646, 13, DIM, font="tech", tracking=0.1)
        self._jt(img, f"{config.GAME_TIME:.0f} S", "", 1220, 646, 13, DIM, "right", font="tech", tracking=0.1)
        self._jt(img, "SPACE  ―  RETURN TO TITLE", "SPACE: title", W / 2, 684, 16, INK, "center", font="tech",
                 tracking=0.2)

    # ========================================================
    # 文字
    # ========================================================
    def _t(self, img, text, x, y, scale=0.45, color=INK, thick=1):
        k = self.k
        cv2.putText(img, text, (int(x * k), int(y * k)), cv2.FONT_HERSHEY_SIMPLEX, scale * k, (*color, 255),
                    max(1, round(thick * k)), cv2.LINE_AA)

    def _jt(self, img, ja, en, x, y, size=18, color=(255, 255, 255), anchor="left", bold=False, outline=True,
            font="jp", tracking=0.0):
        """
        文字を書く（日本語フォントがなければ英語を cv2 で）。y は文字の上端。
        font：役割（"jp" / "tech" / "display"）、tracking：字間（文字の大きさに対する割合）。
        読みやすさのため 13 未満の大きさは 13 にし、薄い暗い縁取りをつける
        """
        if not ja:
            return
        k = self.k
        size = max(size, 13)
        if self.jtext.available:
            px = int(round(size * k))
            stroke = max(1, round(px / 14)) if outline else 0
            tr = int(round(tracking * px))
            self.jtext.draw(img, ja, x * k, y * k, px, color, anchor, bold=bold, stroke=stroke, role=font,
                            tracking=tr)
        else:
            text = en or ja
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size / 30, 1)
            if anchor == "center":
                x -= tw / 2
            elif anchor == "right":
                x -= tw
            self._t(img, text, x, y + size * 0.8, size / 30, color)

    def _label(self, img, en, ja, x, y, color=ACCENT, size=15, anchor="left"):
        """計器のようなラベル：字間を空けた英字（Rajdhani）に、小さな日本語を添える"""
        self._jt(img, en, en, x, y, size, color, anchor, font="tech", bold=True, tracking=0.22)
        if ja:
            w = self._text_w(en, size, "tech", 0.22)
            jx = x + w + 10 if anchor == "left" else x + 10
            self._jt(img, ja, "", jx, y + 1, max(13, size - 2), DIM)

    def _text_game(self, fps):
        img = self.overlay_img
        g = self.g
        m = g.mission

        rv = g.rival
        if g.state != "gaze_calib":            # 視線同期の間は、HUD の帯に説明を出すので、数字は出さない
            self._text_hud(img, g, m)

        # ---- 視線の合わせ直しの案内（カメラモードのプレイ中、HUD の下に小さく）----
        if g.state == "play" and not g.test_mode:
            self._jt(img, "視線がずれてきたら C で合わせ直し", "C: gaze resync", 24, HUD + 6, 12, DIM)

        # ---- 止まっているライバルの上に「FROZEN」----
        if rv is not None and rv.frozen and g.state == "play":
            p = g.scene.project(rv.robot.pos, 0.13, SCENE_RECT)
            if p is not None:
                self._jt(img, f"FROZEN {rv.frozen_t:3.1f}", "FROZEN", p[0], p[1] - 24, 16, ICE, "center", font="tech",
                         bold=True, tracking=0.2)

        # ---- 得点のポップアップ ----
        now = time.monotonic()
        for text, pos, t0 in g.popups:
            age = now - t0
            p = g.scene.project(pos, 0.08 + 0.12 * age, SCENE_RECT)
            if p is not None:
                a = max(0.0, 1.0 - age / 1.4)
                c = tuple(int(v * a + 30 * (1 - a)) for v in GOLD)
                self._jt(img, text, text, p[0], p[1] - 30, 30, c, "center", font="display", bold=True)

        # ---- 視線同期・同期フェーズ・カウントダウン ----
        cx = W / 2
        if g.state == "gaze_calib":
            cal = g.gaze.calib
            # 何をしているかの説明（画面の上の中央。四隅の点・確認の点と重ならない高さ）
            check = cal.phase == "check"
            n = len(cal.check_points if check else cal.points)
            title = "ずれを確認します" if check else "視線を合わせます"
            # 説明は HUD の帯の中に（9 点のいちばん上の列・確認の点と重ならない）
            self._jt(img, f"{title}　{min(cal.index + 1, n)} / {n}", "GAZE SYNC", cx, 1, 19,
                     GOLD if check else ACCENT, "center", bold=True)
            face = g.vision.get_state().face_detected if g.vision else False
            if not face:
                guide = "顔が見つかりません：カメラの正面へ"
            elif check:
                guide = "金色の点を見つめてください。"
            else:
                guide = "光る点を、ゲーム中と同じように自然に見てください。あなたの目の動きと画面の位置を対応づけます。"
            self._jt(img, guide, "look at the dot", cx, 27, 12, INK if face else BAD, "center")
            if cal.tries > 0:
                last = f"前回のずれ 約 {cal.check_error:.0f} 画素（{config.GAZE_CHECK_MAX_ERR:.0f} 以下で合格）→ {cal.tries + 1} 回目"
                if cal.can_skip:
                    last += f"　　ENTER：今までで一番よい結果（約 {cal.best_error:.0f} 画素）で進む"
                self._jt(img, last, "retry", cx, 44, 11, GOLD, "center")
        if g.hand_calib.active:
            hc = g.hand_calib
            title = "RESYNC" if g.recalibrating else "HAND SYNC"
            self._jt(img, f"{title}  {hc.remaining:04.1f}", "HAND SYNC", cx, HUD + 10, 26, MAGENTA, "center",
                     font="display", tracking=0.12)
            if hc.phase == "grip":
                closed = hc.grip_cue > 0.5
                self._jt(img, "GRIP" if closed else "OPEN", "GRIP/OPEN", cx, PY + PH + 20, 32, (255, 255, 255),
                         "center", font="display", bold=True, tracking=0.3)
                self._jt(img, "握る" if closed else "開く", "", cx, PY + PH + 60, 15, MAGENTA, "center")
            else:
                self._jt(img, PHASE_JA[hc.phase], hc.phase, cx, PY + PH + 24, 18, INK, "center")
            self._jt(img, f"REC {hc.valid_sec:04.1f} S", "rec", PX + PW - 10, PY + 8, 13, DIM, "right", font="tech",
                     tracking=0.15)
            if not g.recalibrating:
                self._label(img, "HOW TO PLAY", "遊び方", 56, 162)
                for i, s in enumerate(STEPS):
                    self._jt(img, f"{i + 1}. {s}", s, 56, 196 + i * 40, 13, INK)
                self._jt(img, f"制限時間 {config.GAME_TIME:.0f} 秒。小さい物ほど高得点（箱 100・缶 150・玉 250・星 500）", "", 56, 360, 12, GOLD)
        if g.state == "countdown":
            n = int(math.ceil(config.COUNTDOWN_SEC - g.state_t))
            self._jt(img, str(n) if n > 0 else "GO", "GO", cx, H / 2 - 74, 110, (255, 255, 255), "center",
                     font="display", bold=True)
            self._jt(img, "物を見つめて選び、手でつかんで、カゴへ運ぼう", "", cx, H / 2 + 70, 18, ACCENT, "center")

        # ---- お知らせ ----
        msg = self._message()
        if msg is not None:
            self._jt(img, msg, msg, cx, H - 64, 19, (255, 255, 255), "center")

        # ---- 小さなカメラ映像・裏側の見出し ----
        px, py, pw, ph = self._pip_rect()
        if g.test_mode:
            self._jt(img, "TEST MODE  ―  マウス = 視線・手、クリック = 握る", "TEST MODE", W - 16, H - 28, 13,
                     (255, 200, 110), "right", font="tech", tracking=0.05)
        else:
            vs = g.vision.get_state()
            self._jt(img, f"HAND {'OK' if vs.hand.detected else '--'}   FACE {'OK' if vs.face_detected else '--'}",
                     "", px + 6, py + 4, 13, GOOD if vs.hand.detected and vs.face_detected else BAD, font="tech",
                     bold=True, tracking=0.15)
        if g.show_backstage:
            bx, by, bw, bh = config.BRAIN_OVERLAY_RECT
            self._label(img, "BACKSTAGE", "左半球の脳（光る点 = 運動野の擬似ニューロン）", bx + 8, by + 4,
                        (255, 200, 150), 13)
            rx, ry, rw, rh = config.RASTER_OVERLAY_RECT
            self._label(img, "SPIKES", f"{g.cortex.n_alive}/{g.cortex.n_all} 個を記録中", rx + 8, ry + 4,
                        (255, 200, 150), 13)
        self._jt(img, f"{fps:4.1f} FPS", "", 1272, HUD + 6, 11, DIM, "right", font="tech", tracking=0.1)
