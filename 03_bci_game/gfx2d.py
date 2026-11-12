"""
gfx2d.py
--------
（コピー元：02_pseudo_bci/renderer.py の Batch2D・TexturedQuad クラスと、行列の補助関数 ortho / translate /
  rotate_z / scale。03_bci_game で流用するために、そのままの形で別ファイルに取り出した）
  ・Batch2D      : 円・線・四角を三角形としてためて、1 回の描画命令で描く（画面の画素座標、左上が原点）
  ・TexturedQuad : 画像をテクスチャにして画面の四角形に貼る（文字の重ね表示、ラスタープロット、カメラ映像）
"""

import ctypes
import math

import numpy as np
from OpenGL import GL as gl

import config
from gl_utils import ShaderProgram

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT


def rgb255(c):
    """RGB 0〜1 → 文字の色 0〜255（overlay 画像は RGBA の順でテクスチャに送るので RGB のまま）"""
    return tuple(int(v * 255) for v in c)


def ortho(left, right, bottom, top, near, far):
    m = np.identity(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (far - near)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(far + near) / (far - near)
    return m


def translate(x, y, z=0.0):
    m = np.identity(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def rotate_z(a):
    m = np.identity(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def scale(x, y, z):
    return np.diag([x, y, z, 1.0]).astype(np.float32)



# ============================================================
# 2D 図形をまとめて描くためのクラス
# ============================================================
class Batch2D:
    """円・線・四角を三角形の頂点列としてためておき、1 回の描画命令で描く"""

    def __init__(self):
        self.shader = ShaderProgram("flat2d.vert", "flat2d.frag")
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)
        self.parts = []

    def _tri(self, pts, color):
        """pts: (k,2) の頂点（3 個ずつで三角形）"""
        c = np.broadcast_to(np.asarray(color, dtype=np.float32), (len(pts), 4))
        self.parts.append(np.hstack([np.asarray(pts, dtype=np.float32), c]))

    def rect(self, x, y, w, h, color):
        self._tri([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], color)

    def circle(self, cx, cy, r, color, seg=28):
        a = np.linspace(0, 2 * np.pi, seg + 1)
        rim = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
        pts = np.empty((seg * 3, 2))
        pts[0::3] = (cx, cy)
        pts[1::3] = rim[:-1]
        pts[2::3] = rim[1:]
        self._tri(pts, color)

    def ring(self, cx, cy, r, width, color, seg=36):
        a = np.linspace(0, 2 * np.pi, seg + 1)
        outer = np.stack([cx + (r + width / 2) * np.cos(a), cy + (r + width / 2) * np.sin(a)], axis=1)
        inner = np.stack([cx + (r - width / 2) * np.cos(a), cy + (r - width / 2) * np.sin(a)], axis=1)
        pts = np.empty((seg * 6, 2))
        pts[0::6], pts[1::6], pts[2::6] = outer[:-1], outer[1:], inner[1:]
        pts[3::6], pts[4::6], pts[5::6] = outer[:-1], inner[1:], inner[:-1]
        self._tri(pts, color)

    def line(self, x0, y0, x1, y1, width, color):
        d = np.array([x1 - x0, y1 - y0], dtype=np.float64)
        L = np.linalg.norm(d)
        if L < 1e-6:
            return
        n = np.array([-d[1], d[0]]) / L * width / 2
        p0, p1 = np.array([x0, y0]), np.array([x1, y1])
        self._tri([p0 + n, p1 + n, p1 - n, p0 + n, p1 - n, p0 - n], color)

    def arrow(self, x0, y0, x1, y1, width, color):
        """矢印（線 + 先端の三角形）"""
        d = np.array([x1 - x0, y1 - y0], dtype=np.float64)
        L = np.linalg.norm(d)
        if L < 2:
            self.circle(x0, y0, width, color, 12)
            return
        u = d / L
        head = min(12.0, L * 0.5)
        base = np.array([x1, y1]) - u * head
        self.line(x0, y0, base[0], base[1], width, color)
        n = np.array([-u[1], u[0]]) * head * 0.55
        self._tri([[x1, y1], base + n, base - n], color)

    def flush(self):
        """ためた図形をまとめて描く"""
        if not self.parts:
            return
        data = np.ascontiguousarray(np.vstack(self.parts), dtype=np.float32)
        self.parts = []
        self.shader.use()
        self.shader.set_vec2("uScreen", (W, H))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(data))
        gl.glBindVertexArray(0)


# ============================================================
# テクスチャを画面の四角形に貼るクラス（文字の重ね表示とラスタープロット用）
# ============================================================
class TexturedQuad:
    def __init__(self, filter_mode):
        self.shader = ShaderProgram("textured.vert", "textured.frag")
        quad = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=np.float32)
        self.vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        self.size = None

    def upload(self, img):
        """RGBA 画像 (高さ, 幅, 4) をテクスチャに転送する（サイズが変わったら作り直す）"""
        h, w = img.shape[:2]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        if self.size != (w, h):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img)
            self.size = (w, h)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, w, h, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img)

    def draw(self, x, y, w, h):
        self.shader.use()
        self.shader.set_vec2("uScreen", (W, H))
        self.shader.set_vec4("uRect", (x, y, w, h))
        self.shader.set_int("uTex", 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

