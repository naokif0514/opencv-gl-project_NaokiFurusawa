"""
gl_utils.py
-----------
OpenGL まわりの共通処理（前回の作品 01_focus_brain の renderer.py から流用）。
  ・透視投影・視点変換などの 4x4 行列（numpy）
  ・シェーダーの読み込み・コンパイル・uniform 変数の設定
  ・小さなアイコスフィア（ニューロンの球に使う）
"""

import math
import os

import numpy as np
from OpenGL import GL as gl

SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


# ============================================================
# 行列（行優先で作り、GL へは転置して渡す）
# ============================================================
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    eye, target, up = (np.asarray(v, dtype=np.float32) for v in (eye, target, up))
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.identity(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ============================================================
# シェーダー
# ============================================================
class ShaderProgram:
    """shaders/ のソースからプログラムを作り、uniform 変数の場所をキャッシュする"""

    def __init__(self, vert_name, frag_name):
        vs = self._compile(vert_name, gl.GL_VERTEX_SHADER)
        fs = self._compile(frag_name, gl.GL_FRAGMENT_SHADER)
        self.id = gl.glCreateProgram()
        gl.glAttachShader(self.id, vs)
        gl.glAttachShader(self.id, fs)
        gl.glLinkProgram(self.id)
        if not gl.glGetProgramiv(self.id, gl.GL_LINK_STATUS):
            raise RuntimeError("シェーダーのリンクに失敗\n" + gl.glGetProgramInfoLog(self.id).decode())
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        self._locs = {}

    @staticmethod
    def _compile(name, kind):
        path = os.path.join(SHADER_DIR, name)
        with open(path, encoding="utf-8") as fp:
            src = fp.read()
        sh = gl.glCreateShader(kind)
        gl.glShaderSource(sh, src)
        gl.glCompileShader(sh)
        if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
            raise RuntimeError(f"シェーダーのコンパイルに失敗: {path}\n" + gl.glGetShaderInfoLog(sh).decode())
        return sh

    def loc(self, name):
        if name not in self._locs:
            self._locs[name] = gl.glGetUniformLocation(self.id, name)
        return self._locs[name]

    def use(self):
        gl.glUseProgram(self.id)

    def set_mat4(self, name, m):
        gl.glUniformMatrix4fv(self.loc(name), 1, gl.GL_TRUE, np.asarray(m, dtype=np.float32))

    def set_float(self, name, v):
        gl.glUniform1f(self.loc(name), float(v))

    def set_int(self, name, v):
        gl.glUniform1i(self.loc(name), int(v))

    def set_vec2(self, name, v):
        gl.glUniform2f(self.loc(name), *map(float, v))

    def set_vec3(self, name, v):
        gl.glUniform3f(self.loc(name), *map(float, v))

    def set_vec4(self, name, v):
        gl.glUniform4f(self.loc(name), *map(float, v))


# ============================================================
# アイコスフィア（法線 = 位置なので、位置だけ返す）
# ============================================================
def icosphere(subdivisions=1):
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
             [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
             [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
             [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32).ravel()


def hue_to_rgb(h):
    """色相 h（0〜1, 配列可）を鮮やかな RGB に変換する（彩度・明度 1 の HSV）"""
    h = np.asarray(h, dtype=np.float64)
    k = (np.array([5.0, 3.0, 1.0]) + h[..., None] * 6.0) % 6.0
    return 1.0 - np.clip(np.minimum(k, 4.0 - k), 0.0, 1.0)
