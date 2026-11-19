"""
shadows.py
----------
3D の場面の影（画質ごとに方式を変える。config.QUALITY_PRESETS の "shadow"）。

  "ellipse"（light）  : 物の下の床に置いた半透明の黒い楕円（scene3d.py の _shadow。以前のまま）
  "decal"（standard）: あらかじめ生成した柔らかい影の画像（assets/textures/shadow_soft.png）を机の上に貼る … ShadowDecals
                        ・ロボットは、車体・肩・上腕・肘・前腕・手首で別々に影を落とす。腕を伸ばすと影もついていく
                        ・影は、主な光（KEY_LIGHT）の向きに、その部分の高さに応じてずらす（高い所ほど遠くへ）
                        ・物を持ち上げると、影は小さく薄くなる
                        ・影の四角形は毎フレーム CPU でまとめて作り、1 回の描画で全部描く
  "map"（high）      : 主な光から見た深度の画像（シャドウマップ）を作り、scene_hq.frag で影かどうかを調べる … ShadowMap
                        ・縁は PCF（まわり 8 点で調べた平均）でぼかす
                        ・机に接している所が浮いて見えないよう、ShadowDecals の薄い「接地の影」も重ねる
"""

import ctypes
import os

import cv2
import numpy as np
from OpenGL import GL as gl

import config
from gl_utils import ShaderProgram, look_at

TEXTURE_PATH = "assets/textures/shadow_soft.png"
LIFT = 0.0009                  # 影を机の面からどれだけ浮かせるか（机と重なってちらつかないように）
SHADOW_COLOR = (0.03, 0.025, 0.035)


def light_dir():
    d = np.asarray(config.KEY_LIGHT[0], dtype=np.float64)
    return d / np.linalg.norm(d)


def light_shift(h):
    """高さ h の点の影が、真下からどれだけずれるか（3D の x, z）。主な光と反対向きに h × 傾き"""
    L = light_dir()
    return np.array([-L[0] / L[1] * h, -L[2] / L[1] * h])


def ortho(l, r, b, t, n, f):
    m = np.identity(4, dtype=np.float32)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


# ============================================================
# 柔らかい影の画像を貼る（画質 standard。high では薄い接地の影）
# ============================================================
class ShadowDecals:
    def __init__(self, base_dir):
        self.shader = ShaderProgram("decal.vert", "decal.frag")
        img = cv2.imread(os.path.join(base_dir, TEXTURE_PATH), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"影の画像がありません: {TEXTURE_PATH}（tools/make_shadow_textures.py で作る）")
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R8, img.shape[1], img.shape[0], 0, gl.GL_RED,
                        gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(img))
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        for k, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, k, v)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        stride = 6 * 4
        for loc, n, off in ((0, 3, 0), (1, 2, 12), (2, 1, 20)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, n, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(off))
        gl.glBindVertexArray(0)
        self.rows = []

    # --------------------------------------------------------
    def begin(self):
        self.rows = []

    def _quad(self, p, uv, a):
        """4 つの角（3D の x, z）と画像の座標で、四角形（三角形 2 つ）を足す"""
        for i in (0, 1, 2, 0, 2, 3):
            self.rows.append((p[i][0], LIFT, p[i][1], uv[i][0], uv[i][1], a))

    def rect(self, c, du, hu, hv, soft, alpha):
        """
        角の丸い長方形の影（画像を 9 つに分けて貼る）。
        c：中心（3D の x, z）、du：長い向きの単位ベクトル（x, z）、hu, hv：外側の縁までの半分の長さ・幅、
        soft：縁のぼやけた部分の幅
        """
        if alpha <= 0.005:
            return
        c, du = np.asarray(c, dtype=np.float64), np.asarray(du, dtype=np.float64)
        dv = np.array([-du[1], du[0]])
        soft = max(1e-4, min(soft, hu, hv))
        us = [-hu, -hu + soft, hu - soft, hu]
        vs = [-hv, -hv + soft, hv - soft, hv]
        tex = [0.0, 0.5, 0.5, 1.0]          # 四隅は画像の 4 分の 1、辺と真ん中は画像の中央の値
        for i in range(3):
            for j in range(3):
                if us[i + 1] - us[i] < 1e-6 or vs[j + 1] - vs[j] < 1e-6:
                    continue
                corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                self._quad([c + du * us[a] + dv * vs[b] for a, b in corners],
                           [(tex[a], tex[b]) for a, b in corners], alpha)

    def blob(self, c, radius, alpha):
        """丸い影"""
        self.rect(c, (1.0, 0.0), radius, radius, radius, alpha)

    def draw(self, view, proj):
        if not self.rows:
            return
        data = np.asarray(self.rows, dtype=np.float32)
        s = self.shader
        s.use()
        s.set_mat4("uView", view)
        s.set_mat4("uProj", proj)
        s.set_vec3("uColor", SHADOW_COLOR)
        s.set_int("uTex", 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(data))
        gl.glBindVertexArray(0)


# ============================================================
# シャドウマップ（画質 high）
# ============================================================
class ShadowMap:
    """主な光から見た深度の画像。机の全体（と、その上 0.35 の高さまで）が入るように正射影で作る"""

    def __init__(self, size):
        self.size = int(size)
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, self.size, self.size, 0, gl.GL_DEPTH_COMPONENT,
                        gl.GL_FLOAT, None)
        for k, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER),
                     (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE),
                     (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, k, v)
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, (1.0, 1.0, 1.0, 1.0))
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("シャドウマップのフレームバッファを作れませんでした")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self.depth_shader = ShaderProgram("scene.vert", "depth.frag")
        self._fit()

    def _fit(self):
        """光の向きの正射影：机の四隅（高さ 0〜0.35）が全部入る、いちばん小さな箱にする"""
        L = light_dir()
        eye = L * 3.0
        self.view = look_at(eye, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        m = 0.08
        pts = [(sx * (config.WORLD_W / 2 + m), y, sz * (config.WORLD_H / 2 + m), 1.0)
               for sx in (-1, 1) for sz in (-1, 1) for y in (0.0, 0.35)]
        v = np.array([self.view @ np.array(p) for p in pts])
        x0, x1, y0, y1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
        near, far = -v[:, 2].max() - 0.2, -v[:, 2].min() + 0.2
        self.proj = ortho(x0, x1, y0, y1, near, far)
        self.vp = (self.proj @ self.view).astype(np.float32)
        self.texel = 1.0 / self.size

    def begin(self):
        """深度だけを描く準備（今のフレームバッファと表示範囲を覚えておく）"""
        self._prev_fbo = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
        self._prev_vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.size, self.size)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(2.0, 4.0)          # 自分の影で表面がまだらになる（シャドウアクネ）のを防ぐ
        s = self.depth_shader
        s.use()
        s.set_mat4("uView", self.view)
        s.set_mat4("uProj", self.proj)
        return s

    def end(self):
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(self._prev_fbo))
        gl.glViewport(*[int(v) for v in self._prev_vp])

    def bind(self, shader, unit=1):
        """メインのシェーダーにシャドウマップを渡す"""
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        shader.set_int("uShadowMap", unit)
        shader.set_int("uShadowOn", 1)
        shader.set_mat4("uLightVP", self.vp)
        shader.set_float("uShadowTexel", self.texel)
