"""
jtext.py
--------
文字を、文字の重ね表示用の画像（RGBA の numpy 配列）に書く。
OpenCV の cv2.putText は日本語を書けないので、Pillow（PIL）とフォントファイルを使う。

■ フォントの役割（role）：機械的・無機質な見た目にするため、英数字はテクノ系のフォントで書く
  "jp"      : 日本語の本文（Noto Sans CJK JP。システムのフォント）。英数字も同じフォント
  "tech"    : ラベル・計器の表示。英数字は Rajdhani（細身で角ばったフォント）、日本語は Noto Sans CJK
  "display" : 大きな数字・見出し。英数字は Orbitron（幅広で角ばった SF 風のフォント）、日本語は Noto Sans CJK
  1 つの文字列の中で、英数字（ASCII）と日本語を分けてそれぞれのフォントで描き、ベースラインをそろえる。
  tracking で字間を空けられる（英字のラベルは字間を空けると計器らしく見える）。
  Rajdhani・Orbitron は assets/fonts/ に同梱（SIL Open Font License 1.1）。なければ Noto Sans CJK で代用する。

・同じ文字列・大きさ・役割の組み合わせは、一度描いた形をキャッシュして使い回す（毎回描くと重いため）
・日本語フォントが見つからないときは available = False になる。そのときは呼び出し側で英語の文を使う
"""

import os
import subprocess

import numpy as np

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")
_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
)
# 英数字のフォント：(通常, 太字) のファイル名と、日本語のフォントと大文字の高さをそろえる倍率
LATIN = {
    "tech": (("Rajdhani-Medium.ttf", None), ("Rajdhani-SemiBold.ttf", None), 1.12),
    "display": (("Orbitron-Variable.ttf", 500), ("Orbitron-Variable.ttf", 700), 0.98),
}


def _find_font():
    for p in _CANDIDATES:
        if os.path.exists(p):
            return p
    try:   # fontconfig に聞く
        out = subprocess.run(["fc-match", "-f", "%{file}", ":lang=ja"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and os.path.exists(out.stdout.strip()):
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


class JText:
    """
    文字を描く。role でフォントの役割（"jp" / "tech" / "display"）、bold で太字、stroke（画素）で暗い縁取り、
    tracking（画素）で字間を指定する
    """

    def __init__(self):
        self.path = _find_font()
        self.bold_path = None
        self.available = False
        self._fonts = {}
        self._cache = {}
        if self.path is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont   # noqa: F401
                self.available = True
            except ImportError:
                pass
            bold = self.path.replace("Regular", "Bold")
            self.bold_path = bold if os.path.exists(bold) else self.path

    # --------------------------------------------------------
    def _font(self, size, bold=False, role="jp"):
        """(フォント, 倍率)。role の英数字フォントがなければ日本語フォントを返す"""
        from PIL import ImageFont
        key = (size, bold, role)
        if key not in self._fonts:
            font = None
            if role in LATIN:
                (name, wght), scale = LATIN[role][1 if bold else 0], LATIN[role][2]
                path = os.path.join(FONT_DIR, name)
                if os.path.exists(path):
                    font = ImageFont.truetype(path, max(1, int(round(size * scale))))
                    if wght is not None:
                        try:
                            font.set_variation_by_axes([wght])
                        except (OSError, ValueError):
                            pass
            if font is None:
                font = ImageFont.truetype(self.bold_path if bold else self.path, size)
            self._fonts[key] = font
        return self._fonts[key]

    def _runs(self, text, role):
        """文字列を (文字, フォント) に分ける。英数字は role のフォント、それ以外（日本語）は日本語フォント"""
        return [(ch, role if (role != "jp" and ord(ch) < 128) else "jp") for ch in text]

    def _render(self, text, size, bold=False, stroke=0, role="jp", tracking=0):
        """文字列の形（アルファ値）と縁取りの形を、ぴったりの大きさの画像に描く（キャッシュつき）"""
        key = (text, size, bold, stroke, role, tracking)
        if key in self._cache:
            return self._cache[key]
        from PIL import Image, ImageDraw
        chars = self._runs(text, role)
        fonts = {r: self._font(size, bold, r) for r in {r for _, r in chars}}
        asc = max(f.getmetrics()[0] for f in fonts.values())
        desc = max(f.getmetrics()[1] for f in fonts.values())
        # 1 文字ずつ並べる（字間 tracking を足しながら、ベースラインをそろえる）
        xs, x = [], stroke + 1
        for ch, r in chars:
            xs.append(x)
            x += fonts[r].getlength(ch) + tracking
        w = int(np.ceil(x - tracking + stroke + 2)) if chars else 1
        h = int(asc + desc + 2 * stroke + 2)
        base = stroke + 1 + asc
        fill = Image.new("L", (max(w, 1), h), 0)
        d = ImageDraw.Draw(fill)
        for (ch, r), cx in zip(chars, xs):
            d.text((cx, base), ch, font=fonts[r], fill=255, anchor="ls")
        outline = None
        if stroke > 0:
            o = Image.new("L", (max(w, 1), h), 0)
            od = ImageDraw.Draw(o)
            for (ch, r), cx in zip(chars, xs):
                od.text((cx, base), ch, font=fonts[r], fill=255, anchor="ls", stroke_width=stroke, stroke_fill=255)
            outline = np.asarray(o, dtype=np.float32) / 255.0
        if len(self._cache) > 800:
            self._cache.clear()
        self._cache[key] = (np.asarray(fill, dtype=np.float32) / 255.0, outline)
        return self._cache[key]

    def size(self, text, size, bold=False, stroke=0, role="jp", tracking=0):
        alpha, _ = self._render(text, size, bold, stroke, role, tracking)
        return alpha.shape[1], alpha.shape[0]

    def draw(self, img, text, x, y, size=18, color=(255, 255, 255), anchor="left", bold=False, stroke=0,
             stroke_color=(4, 8, 14), role="jp", tracking=0):
        """
        img（H×W×4 の RGBA、uint8）の (x, y) に文字を書く。y は文字の上端。
        anchor="center" なら x を文字の中心に、"right" なら右端にする。stroke > 0 なら先に暗い縁取りを描く
        """
        if not self.available or not text:
            return
        alpha, outline = self._render(text, size, bold, stroke, role, tracking)
        h, w = alpha.shape
        if anchor == "center":
            x = int(x - w / 2)
        elif anchor == "right":
            x = int(x - w)
        else:
            x = int(x)
        y = int(y)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1:
            return
        region = img[y0:y1, x0:x1].astype(np.float32)
        if outline is not None:
            o = outline[y0 - y:y1 - y, x0 - x:x1 - x, None] * 0.75
            region = region * (1 - o) + np.array([*stroke_color, 255], dtype=np.float32) * o
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None]
        col = np.array([*color, 255], dtype=np.float32)
        img[y0:y1, x0:x1] = (region * (1 - a) + col * a).astype(np.uint8)
