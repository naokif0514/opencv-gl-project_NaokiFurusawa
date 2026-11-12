"""
renderer.py
-----------
PyOpenGL + glfw による描画（シェーダー使用）。1 つのウィンドウを 3 つの領域に分けて表示する。

  ┌──────────────────────┬──────────────────────┐
  │ 左：運動野のニューロン（3D）│ 右：本当の手 と デコードしたカーソル │
  │   曲面（皮質）上に格子状に並んだ球 │   （2D 平面、ズレがわかるように線で結ぶ） │
  │   色 = 好きな方向、光 = スパイク  │   左側に速度の比較と色の凡例        │
  ├──────────────────────┴──────────────────────┤
  │ 下：スパイクのラスタープロット（横軸：時間、縦軸：ニューロン番号）            │
  └─────────────────────────────────────────────┘

・ニューロンの球はインスタンス描画（1 回の描画命令で N 個）で描く
・ラスタープロットは「ニューロン数 × 時間ビン数」の画像を毎ステップ 1 列ずつ左へずらし、
  テクスチャとして貼る（スパイクの数が多くても描画の重さが変わらない）
・文字（HUD・ラベル）は OpenCV の cv2.putText で透明な画像に描き、テクスチャとして重ねる
"""

import ctypes
import math
import os
import time
from collections import deque

# Wayland 環境では X11 互換モードで開く（理由は main.py の先頭を参照）。glfw / OpenGL の import 前に設定する
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import cv2           # noqa: E402
import glfw          # noqa: E402
import numpy as np   # noqa: E402
from OpenGL import GL as gl   # noqa: E402

import config        # noqa: E402
from gl_utils import ShaderProgram, hue_to_rgb, icosphere, look_at, perspective   # noqa: E402

# ------------------------------------------------------------
# 画面のレイアウト（ウィンドウ座標 [画素]、左上が原点。ウィンドウは 1280x720 固定）
# ------------------------------------------------------------
W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
TOP_H = int(H * 0.72)                                 # 上段（3D とカーソル）の高さ
LEFT_W = W // 2                                       # 左の 3D 表示の幅
PANEL = int(min(W - LEFT_W - 140, TOP_H - 90))        # 右の 2D 平面（正方形）の一辺
PANEL_X = LEFT_W + 120
PANEL_Y = 44
COMPASS = (LEFT_W + 60, PANEL_Y + 95, 44)             # 速度比較の円 (中心 x, 中心 y, 半径)
WHEEL = (LEFT_W + 60, PANEL_Y + 290, 38)              # 色の凡例の円
RASTER = (70, TOP_H + 26, W - 90, H - TOP_H - 52)     # ラスタープロット (x, y, 幅, 高さ)


def surface_height(x, z):
    """大脳皮質に見立てた曲面の高さ y = f(x, z)（なだらかに曲がった面に、軽いうねりを加える）"""
    return -0.30 * x ** 2 + 0.06 * np.sin(3.0 * z + 1.2 * x)


def surface_normal(x, z):
    """曲面の法線（数値微分）"""
    e = 1e-3
    dx = (surface_height(x + e, z) - surface_height(x - e, z)) / (2 * e)
    dz = (surface_height(x, z + e) - surface_height(x, z - e)) / (2 * e)
    n = np.stack([-dx, np.ones_like(dx), -dz], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


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


# ============================================================
# 左：皮質の曲面とニューロン（3D）
# ============================================================
class CortexView:
    def __init__(self):
        self.neuron_shader = ShaderProgram("neuron.vert", "neuron.frag")
        self.sheet_shader = ShaderProgram("sheet.vert", "sheet.frag")
        self._build_sheet()

        # 球のメッシュ（全ニューロンで共有）
        sphere_v, sphere_i = icosphere(config.NEURON_SPHERE_DETAIL)
        self.sphere_count = len(sphere_i)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sphere_v.nbytes, sphere_v, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, sphere_i.nbytes, sphere_i, gl.GL_STATIC_DRAW)

        # インスタンスごとの属性（位置・色・光り具合・生存）。divisor=1 で「1 インスタンスに 1 つ」
        self.inst_vbos = {}
        for loc, name, size in ((1, "offset", 3), (2, "color", 3), (3, "flash", 1), (4, "alive", 1),
                                (5, "activity", 1)):
            b = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, size * 4, ctypes.c_void_p(0))
            gl.glVertexAttribDivisor(loc, 1)
            self.inst_vbos[name] = b
        gl.glBindVertexArray(0)
        self.n = 0

    def _build_sheet(self, nx=48, nz=32):
        """皮質の曲面を格子状の三角形メッシュとして作る"""
        xs = np.linspace(-1.35, 1.35, nx)
        zs = np.linspace(-0.95, 0.95, nz)
        X, Z = np.meshgrid(xs, zs)
        Y = surface_height(X, Z)
        N = surface_normal(X, Z)
        verts = np.concatenate([np.stack([X, Y, Z], -1), N], -1).reshape(-1, 6).astype(np.float32)
        idx = []
        for j in range(nz - 1):
            for i in range(nx - 1):
                a = j * nx + i
                idx += [a, a + nx, a + 1, a + 1, a + nx, a + nx + 1]
        idx = np.array(idx, dtype=np.uint32)
        self.sheet_count = len(idx)
        self.sheet_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.sheet_vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)

    def set_population(self, pop, rng):
        """
        ニューロンを曲面上に格子状に並べる。
        実際の M1 では好きな方向がきれいな地図状には並んでいないので、格子の位置はランダムに割り当てる。
        """
        n = pop.n
        cols = int(math.ceil(math.sqrt(n * 1.6)))
        rows = int(math.ceil(n / cols))
        xs = np.linspace(-1.1, 1.1, cols)
        zs = np.linspace(-0.7, 0.7, rows) if rows > 1 else np.zeros(1)
        gx, gz = np.meshgrid(xs, zs)
        cells = np.stack([gx.ravel(), gz.ravel()], 1)[:n]
        cells = cells[rng.permutation(n)]
        spacing = min(2.2 / max(cols - 1, 1), 1.4 / max(rows - 1, 1))
        self.radius = 0.28 * spacing
        x, z = cells[:, 0], cells[:, 1]
        normal = surface_normal(x, z)
        pos = np.stack([x, surface_height(x, z), z], 1) + normal * self.radius
        colors = hue_to_rgb(pop.pd_angle / (2 * np.pi))

        self.n = n
        self._upload("offset", pos)
        self._upload("color", colors)
        self._upload("flash", np.zeros(n))
        self._upload("alive", np.ones(n))
        self._upload("activity", np.zeros(n))

    def _upload(self, name, arr):
        data = np.ascontiguousarray(arr, dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.inst_vbos[name])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def draw(self, flash, activity, alive, t, viewport):
        gl.glViewport(*viewport)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # カメラはゆっくり左右に揺れる
        a = math.radians(18.0 * math.sin(t * 0.25))
        cam = (3.3 * math.sin(a), 2.3, 3.3 * math.cos(a))
        view = look_at(cam, (0.0, 0.05, 0.1), (0.0, 1.0, 0.0))
        proj = perspective(38.0, viewport[2] / viewport[3], 0.1, 20.0)
        light = (0.3, 1.0, 0.5)

        self.sheet_shader.use()
        self.sheet_shader.set_mat4("uView", view)
        self.sheet_shader.set_mat4("uProj", proj)
        self.sheet_shader.set_vec3("uLightDir", light)
        self.sheet_shader.set_float("uActivity", float(np.mean(activity)) if len(activity) else 0.0)
        gl.glBindVertexArray(self.sheet_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.sheet_count, gl.GL_UNSIGNED_INT, None)

        self._upload("flash", flash)
        self._upload("alive", alive.astype(np.float32))
        self._upload("activity", activity)
        self.neuron_shader.use()
        self.neuron_shader.set_mat4("uView", view)
        self.neuron_shader.set_mat4("uProj", proj)
        self.neuron_shader.set_vec3("uCamPos", cam)
        self.neuron_shader.set_vec3("uLightDir", light)
        self.neuron_shader.set_float("uRadius", self.radius)
        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.sphere_count, gl.GL_UNSIGNED_INT, None, self.n)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_DEPTH_TEST)


# ============================================================
# 描画全体
# ============================================================
class Renderer:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("glfw の初期化に失敗しました")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, config.MSAA_SAMPLES)
        glfw.window_hint(glfw.RESIZABLE, gl.GL_FALSE)     # レイアウトを固定するため大きさは変えない
        self.window = glfw.create_window(W, H, config.WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("ウィンドウを作成できませんでした（OpenGL 3.3 に対応していない可能性）")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        if config.MSAA_SAMPLES > 0:
            gl.glEnable(gl.GL_MULTISAMPLE)
        print(f"[renderer] {gl.glGetString(gl.GL_VERSION).decode()} / {gl.glGetString(gl.GL_RENDERER).decode()}")

        self.cortex = CortexView()
        self.batch = Batch2D()
        self.overlay = TexturedQuad(gl.GL_LINEAR)
        self.raster_quad = TexturedQuad(gl.GL_NEAREST)
        self.overlay_img = np.zeros((H, W, 4), dtype=np.uint8)
        self._last_overlay = -1.0
        self.start = time.monotonic()

        self.true_trail = deque(maxlen=45)
        self.cursor_trail = deque(maxlen=45)

    # --------------------------------------------------------
    # ニューロン集団が変わったとき・スパイクが出たとき
    # --------------------------------------------------------
    def set_population(self, pop, rng):
        self.n = pop.n
        self.colors = hue_to_rgb(pop.pd_angle / (2 * np.pi))
        self.flash = np.zeros(pop.n)
        self.activity = np.zeros(pop.n)
        self.cortex.set_population(pop, rng)
        bins = int(round(config.RASTER_SEC / config.DT))
        self.raster_bg = np.array([14, 16, 26, 255], dtype=np.uint8)
        self.raster = np.empty((pop.n, bins, 4), dtype=np.uint8)
        self.raster[:] = self.raster_bg
        # 行の色（ニューロン 0 が一番下に来るよう、上下を逆にした順）
        self.raster_colors = (self.colors[::-1] * 255).astype(np.uint8)

    def reset_display(self):
        self.flash[:] = 0
        self.activity[:] = 0
        self.raster[:] = self.raster_bg
        self.true_trail.clear()
        self.cursor_trail.clear()

    def on_spikes(self, counts, alive, rate_hat):
        """
        1 ステップ分のスパイク数と、デコーダが推定した発火率を受け取り、表示を更新する。
          flash    : スパイクが出た瞬間に 1 になり、すぐ消える
          activity : 推定発火率がベースラインよりどれだけ高いか（0〜1）。手の動きの向きと
                     好きな方向が近いニューロンほど大きくなり、皮質上で「どこが活動しているか」が見える
        """
        fired = counts > 0
        self.flash[fired] = 1.0
        act = np.clip((rate_hat - config.BASE_RATE) / (config.MODULATION * 1.0), 0.0, 1.0)
        self.activity += 0.25 * (act - self.activity)
        # ラスター：1 列左へずらし、右端に今回のスパイクを書く
        self.raster[:, :-1] = self.raster[:, 1:]
        col = np.empty((self.n, 4), dtype=np.uint8)
        col[:] = self.raster_bg
        col[(~alive)[::-1]] = (34, 34, 38, 255)                       # 脱落したニューロンの行は灰色
        rows = fired[::-1]
        col[rows, :3] = self.raster_colors[rows]
        col[rows, 3] = 255
        self.raster[:, -1] = col

    # --------------------------------------------------------
    # 1 フレームの描画
    # --------------------------------------------------------
    def render(self, s, dt):
        """s: main.py が作る表示用の値の dict"""
        t = time.monotonic() - self.start
        self.flash *= math.exp(-dt / config.SPIKE_FLASH_TAU)

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w == 0 or fb_h == 0:
            return
        sx, sy = fb_w / W, fb_h / H

        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glClearColor(0.025, 0.03, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # ---- 左：3D のニューロン（ビューポートを左上の領域に限定）----
        self.cortex.draw(self.flash, self.activity, s["alive"], t,
                         (0, int((H - TOP_H) * sy), int(LEFT_W * sx), int(TOP_H * sy)))

        # ---- 2D の図形 ----
        gl.glViewport(0, 0, fb_w, fb_h)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self._draw_panel(s)
        self._draw_compass(s)
        self._draw_wheel()
        b = self.batch
        rx, ry, rw, rh = RASTER
        b.rect(rx - 2, ry - 2, rw + 4, rh + 4, (0.25, 0.3, 0.4, 1.0))
        b.line(LEFT_W, 8, LEFT_W, TOP_H - 8, 1, (0.25, 0.3, 0.4, 1.0))
        b.flush()

        # ---- ラスタープロット ----
        self.raster_quad.upload(self.raster)
        self.raster_quad.draw(rx, ry, rw, rh)

        # ---- 文字（0.1 秒ごとに描き直す）----
        if t - self._last_overlay >= config.OVERLAY_INTERVAL:
            self._draw_overlay(s)
            self.overlay.upload(self.overlay_img)
            self._last_overlay = t
        self.overlay.draw(0, 0, W, H)
        gl.glDisable(gl.GL_BLEND)

        glfw.swap_buffers(self.window)

    def _to_panel(self, p):
        """手の座標（0〜1, y 上向き）→ 右の 2D 平面の画素座標"""
        return PANEL_X + p[0] * PANEL, PANEL_Y + (1.0 - p[1]) * PANEL

    def _draw_panel(self, s):
        """右：本当の手（半透明のリング）とデコードしたカーソル（はっきりした点）"""
        b = self.batch
        b.rect(PANEL_X, PANEL_Y, PANEL, PANEL, (0.06, 0.07, 0.11, 1.0))
        for k in range(1, 4):
            g = PANEL * k / 4
            b.line(PANEL_X + g, PANEL_Y, PANEL_X + g, PANEL_Y + PANEL, 1, (0.14, 0.16, 0.22, 1.0))
            b.line(PANEL_X, PANEL_Y + g, PANEL_X + PANEL, PANEL_Y + g, 1, (0.14, 0.16, 0.22, 1.0))

        hand = s["hand"]
        cur = self._to_panel(s["cursor"])
        self.cursor_trail.append(cur)
        if hand.pos is not None:
            self.true_trail.append(self._to_panel(hand.pos))
        else:
            self.true_trail.clear()

        # 軌跡（古いほど薄く）
        for trail, color in ((self.true_trail, (1.0, 1.0, 1.0)), (self.cursor_trail, (1.0, 0.85, 0.2))):
            m = len(trail)
            for i, (x, y) in enumerate(trail):
                b.circle(x, y, 2.0, (*color, 0.35 * (i + 1) / m), 8)

        if hand.pos is not None:
            hp = self._to_panel(hand.pos)
            b.line(hp[0], hp[1], cur[0], cur[1], 1.5, (1.0, 0.4, 0.4, 0.6))   # ズレ
            b.circle(hp[0], hp[1], 20, (1.0, 1.0, 1.0, 0.12))
            b.ring(hp[0], hp[1], 20, 3, (1.0, 1.0, 1.0, 0.55))
        b.circle(cur[0], cur[1], 9, (1.0, 0.85, 0.2, 1.0))
        b.line(cur[0] - 16, cur[1], cur[0] + 16, cur[1], 1.5, (1.0, 0.85, 0.2, 0.9))
        b.line(cur[0], cur[1] - 16, cur[0], cur[1] + 16, 1.5, (1.0, 0.85, 0.2, 0.9))

    def _draw_compass(self, s):
        """速度の比較：本当の速度（白）とデコードした速度（黄）を同じ円の中に矢印で表示"""
        b = self.batch
        cx, cy, r = COMPASS
        b.circle(cx, cy, r, (0.06, 0.07, 0.11, 1.0))
        b.ring(cx, cy, r, 1.5, (0.3, 0.35, 0.45, 1.0))
        scale = r / 1.2      # 半径 = 速さ 1.2 [画面/秒]
        # 本当の速度は太い半透明の白、推定速度は細い黄色で上に重ねる（重なっていればデコードが正確）
        for v, color, w in ((s["true_v"], (1.0, 1.0, 1.0, 0.45), 7), (s["v_hat"], (1.0, 0.85, 0.2, 1.0), 2.5)):
            vx, vy = np.clip(v, -1.2, 1.2)
            b.arrow(cx, cy, cx + vx * scale, cy - vy * scale, w, color)

    def _draw_wheel(self):
        """色の凡例：円周上の位置（方向）に、その方向が好きなニューロンの色を並べる"""
        b = self.batch
        cx, cy, r = WHEEL
        seg = 36
        for k in range(seg):
            a0, a1 = 2 * np.pi * k / seg, 2 * np.pi * (k + 1) / seg
            c = hue_to_rgb((a0 + a1) / 2 / (2 * np.pi))
            # 画面の y は下向きなので、角度は y を反転して描く
            p0 = (cx + r * math.cos(a0), cy - r * math.sin(a0))
            p1 = (cx + r * math.cos(a1), cy - r * math.sin(a1))
            q0 = (cx + 0.6 * r * math.cos(a0), cy - 0.6 * r * math.sin(a0))
            q1 = (cx + 0.6 * r * math.cos(a1), cy - 0.6 * r * math.sin(a1))
            b._tri([p0, p1, q1, p0, q1, q0], (*c, 1.0))

    # --------------------------------------------------------
    # 文字（OpenCV で描く）
    # --------------------------------------------------------
    @staticmethod
    def _text(img, text, x, y, scale=0.45, color=(225, 230, 240), thick=1):
        cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (*color, 255), thick, cv2.LINE_AA)

    def _draw_overlay(self, s):
        img = self.overlay_img
        img[:] = 0
        T = self._text
        dim = (150, 160, 185)

        # ---- HUD（左上）----
        cv2.rectangle(img, (10, 10), (310, 186), (8, 10, 20, 200), -1)
        cv2.rectangle(img, (10, 10), (310, 186), (70, 90, 130, 255), 1)
        T(img, "PSEUDO BCI DECODER", 20, 32, 0.55, (255, 255, 255))
        T(img, s["mode"], 20, 54, 0.45, s["mode_color"])
        hand = s["hand"]
        T(img, "hand: detected" if hand.detected else "hand: NOT FOUND (v = 0)", 20, 76, 0.45,
          (140, 230, 160) if hand.detected else (255, 120, 120))
        T(img, f"neurons: {s['n_alive']} / {s['n']}", 20, 98)
        T(img, f"noise: {s['noise']:.1f} Hz", 175, 98)
        corr, rmse = s["corr"], s["rmse"]
        T(img, "corr: " + ("--" if corr is None else f"{corr:.2f}"), 20, 124, 0.6, (255, 230, 120))
        T(img, "RMSE: " + ("--" if rmse is None else f"{rmse:.2f}"), 175, 124, 0.6, (255, 230, 120))
        T(img, f"(last {config.METRIC_WINDOW:.0f} s, velocity [screen/s])", 20, 144, 0.38, dim)
        T(img, f"render {s['fps']:4.1f} fps   camera {hand.fps:4.1f} fps", 20, 172, 0.4, dim)

        T(img, "Motor cortex neurons   color = preferred direction / flash = spike", 14, TOP_H - 12, 0.42, dim)

        # ---- 右の平面 ----
        T(img, "Hand (white ring) vs Decoded cursor (yellow)", PANEL_X, PANEL_Y - 14, 0.48)
        T(img, "velocity", COMPASS[0] - 30, COMPASS[1] - COMPASS[2] - 10, 0.42, dim)
        T(img, "true", COMPASS[0] - 40, COMPASS[1] + COMPASS[2] + 18, 0.4, (255, 255, 255))
        T(img, "decoded", COMPASS[0] - 8, COMPASS[1] + COMPASS[2] + 18, 0.4, (255, 215, 60))
        T(img, "PD color", WHEEL[0] - 32, WHEEL[1] - WHEEL[2] - 10, 0.42, dim)
        T(img, "UP/DOWN neurons  N/M noise  K dropout  T test  D debug  R reset  ESC quit",
          PANEL_X - 110, TOP_H - 12, 0.40, (120, 135, 160))

        # ---- ラスタープロット ----
        rx, ry, rw, rh = RASTER
        T(img, f"Spike raster (last {config.RASTER_SEC:.0f} s, neurons sorted by preferred direction)",
          rx, ry - 8, 0.42, dim)
        T(img, f"#{s['n'] - 1}", 14, ry + 12, 0.4, dim)
        T(img, "#0", 14, ry + rh, 0.4, dim)
        T(img, f"-{config.RASTER_SEC:.0f} s", rx, ry + rh + 18, 0.4, dim)
        T(img, "now", rx + rw - 28, ry + rh + 18, 0.4, dim)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
