"""
jtext.py
--------
日本語の文字を、文字の重ね表示用の画像（RGBA の numpy 配列）に書く。
OpenCV の cv2.putText は日本語を書けないので、Pillow（PIL）と日本語フォント（Noto Sans CJK など）を使う。

・同じ文字列・大きさ・色の組み合わせは、一度描いた小さな画像をキャッシュして使い回す（毎回描くと重いため）
・日本語フォントが見つからないときは available = False になる。そのときは呼び出し側で英語の文を使う
"""

import os
import subprocess

import numpy as np

_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
)


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
    def __init__(self):
        self.path = _find_font()
        self.available = False
        self._fonts = {}
        self._cache = {}
        if self.path is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont   # noqa: F401
                self.available = True
            except ImportError:
                pass

    def _font(self, size):
        from PIL import ImageFont
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.path, size)
        return self._fonts[size]

    def _render(self, text, size, color):
        """文字列を、ぴったりの大きさの RGBA 画像に描く（キャッシュつき）"""
        key = (text, size, color)
        if key not in self._cache:
            from PIL import Image, ImageDraw
            font = self._font(size)
            left, top, right, bottom = font.getbbox(text)
            w, h = max(1, right - left + 2), max(1, bottom - top + 2)
            img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(img).text((1 - left, 1 - top), text, font=font, fill=255)
            alpha = np.asarray(img, dtype=np.float32) / 255.0
            if len(self._cache) > 400:
                self._cache.clear()
            self._cache[key] = (alpha, top)
        return self._cache[key]

    def size(self, text, size):
        alpha, _ = self._render(text, size, (255, 255, 255))
        return alpha.shape[1], alpha.shape[0]

    def draw(self, img, text, x, y, size=18, color=(255, 255, 255), anchor="left"):
        """
        img（H×W×4 の RGBA、uint8）の (x, y) に文字を書く。y は文字の上端。
        anchor="center" なら x を文字の中心にする
        """
        if not self.available or not text:
            return
        alpha, _ = self._render(text, size, tuple(color))
        h, w = alpha.shape
        x = int(x - w / 2) if anchor == "center" else int(x)
        y = int(y)
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1:
            return
        a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None]
        region = img[y0:y1, x0:x1].astype(np.float32)
        col = np.array([*color, 255], dtype=np.float32)
        img[y0:y1, x0:x1] = (region * (1 - a) + col * a).astype(np.uint8)
