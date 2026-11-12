"""
renderer.py
-----------
PyOpenGL + glfw による 3D 描画を担当する。固定機能パイプラインではなくシェーダーを使う。

・中央に「脳」を表示し、ゆっくり自動回転させる
・集中度（0〜100）をシェーダーに uniform 変数として渡し、発光の強さや色のくすみを変える
・画面左上に HUD（集中度の数値とバー、各指標の値）を重ねて表示する
    HUD の文字は OpenCV（cv2.putText）で RGBA 画像に描き、それをテクスチャとして貼っている
"""

import ctypes
import math
import os
import time

# Ubuntu の Wayland 環境では、pyglfw が同梱している Wayland 版 glfw が
# GNOME の設定と合わずに異常終了することがあるため、X11（XWayland）版を使う。
# X11 版 glfw は GLX でコンテキストを作るので、PyOpenGL にも GLX を使わせる
# （指定しないと Wayland 環境では EGL を探しに行き「no valid context」エラーになる）。
# これらの設定は glfw / OpenGL を import する前に行う必要がある。
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import cv2
import glfw
import numpy as np
from OpenGL import GL as gl

import config
from brain_mesh import load_brain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHADER_DIR = os.path.join(BASE_DIR, "shaders")


# ============================================================
# 行列計算（numpy で 4x4 行列を作る。行優先で作り、GL へは転置して渡す）
# ============================================================
def perspective(fovy_deg, aspect, near, far):
    """透視投影行列"""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    """視点変換行列（gluLookAt と同じもの）"""
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


def rotate_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float32)


def rotate_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float32)


# ============================================================
# シェーダー
# ============================================================
def load_shader_program(vert_name, frag_name):
    """shaders/ フォルダのソースを読み込み、コンパイル・リンクしてプログラム ID を返す"""

    def compile_shader(path, kind):
        with open(path, encoding="utf-8") as fp:
            src = fp.read()
        sh = gl.glCreateShader(kind)
        gl.glShaderSource(sh, src)
        gl.glCompileShader(sh)
        if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
            raise RuntimeError(f"シェーダーのコンパイルに失敗: {path}\n"
                               + gl.glGetShaderInfoLog(sh).decode())
        return sh

    vs = compile_shader(os.path.join(SHADER_DIR, vert_name), gl.GL_VERTEX_SHADER)
    fs = compile_shader(os.path.join(SHADER_DIR, frag_name), gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError("シェーダーのリンクに失敗\n" + gl.glGetProgramInfoLog(prog).decode())
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


class ShaderProgram:
    """uniform 変数の場所をキャッシュして、値を簡単にセットできるようにする小さなクラス"""

    def __init__(self, vert_name, frag_name):
        self.id = load_shader_program(vert_name, frag_name)
        self._locs = {}

    def loc(self, name):
        if name not in self._locs:
            self._locs[name] = gl.glGetUniformLocation(self.id, name)
        return self._locs[name]

    def use(self):
        gl.glUseProgram(self.id)

    def set_mat4(self, name, m):
        gl.glUniformMatrix4fv(self.loc(name), 1, gl.GL_TRUE, m.astype(np.float32))

    def set_float(self, name, v):
        gl.glUniform1f(self.loc(name), float(v))

    def set_vec2(self, name, v):
        gl.glUniform2f(self.loc(name), *map(float, v))

    def set_vec3(self, name, v):
        gl.glUniform3f(self.loc(name), *map(float, v))

    def set_vec4(self, name, v):
        gl.glUniform4f(self.loc(name), *map(float, v))

    def set_int(self, name, v):
        gl.glUniform1i(self.loc(name), int(v))


# ============================================================
# HUD（OpenCV で描いた画像をテクスチャとして重ねる）
# ============================================================
class Hud:
    """画面左上の情報パネル"""
    WIDTH = 380
    HEIGHT = 300

    def __init__(self):
        self.shader = ShaderProgram("hud.vert", "hud.frag")
        self.image = np.zeros((self.HEIGHT, self.WIDTH, 4), dtype=np.uint8)

        # 0〜1 の四角形（2 つの三角形）
        quad = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=np.float32)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.WIDTH, self.HEIGHT, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, self.image)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    @staticmethod
    def _text(img, text, pos, scale=0.5, color=(230, 235, 245, 255), thick=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    @staticmethod
    def _bar(img, x, y, w, h, value, color):
        """value(0〜1) の長さのバーを描く"""
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 80, 100, 255), 1)
        if value is not None:
            fill = int((w - 2) * max(0.0, min(1.0, value)))
            if fill > 0:
                cv2.rectangle(img, (x + 1, y + 1), (x + fill, y + h - 1), color, -1)

    def update(self, info):
        """表示内容（info: main.py で作る dict）から HUD 画像を描き直してテクスチャに転送する"""
        img = self.image
        img[:] = (10, 14, 26, 185)                       # 半透明の背景
        cv2.rectangle(img, (0, 0), (self.WIDTH - 1, self.HEIGHT - 1), (80, 110, 160, 255), 1)

        focus = info["focus"]
        t = focus / 100.0
        # 集中度に応じてバーの色を 青 → シアン → 黄緑 と変える（RGBA の順）
        bar_color = (int(60 + 140 * t), int(120 + 120 * t), int(255 - 120 * t), 255)

        self._text(img, "FOCUS", (14, 30), 0.6, (150, 170, 200, 255))
        self._text(img, f"{focus:5.1f}", (95, 42), 1.4, (255, 255, 255, 255), 2)
        self._bar(img, 14, 54, self.WIDTH - 28, 16, t, bar_color)
        self._text(img, info["mode"], (14, 92), 0.48, info.get("mode_color", (255, 210, 120, 255)))

        # 各指標：値とスコアのバー
        m = info.get("metrics")
        scores = info.get("scores") or {}

        def fmt(v, f):
            return "--" if v is None else f.format(v)

        rows = [
            ("Blink", fmt(m.blink_rate if m else None, "{:.1f} /min"), scores.get("blink")),
            ("Eye open", fmt(m.ear_avg if m else None, "EAR {:.3f}"), scores.get("eye_open")),
            ("Head", fmt(m.head_angle if m else None, "{:.1f} deg"), scores.get("head")),
            ("Gaze", fmt(m.gaze_std if m else None, "std {:.3f}"), scores.get("gaze")),
        ]
        y = 122
        for name, value, score in rows:
            self._text(img, name, (14, y), 0.48, (170, 185, 210, 255))
            self._text(img, value, (110, y), 0.48)
            self._bar(img, 250, y - 11, 110, 12, score, (110, 200, 255, 255))
            y += 26

        if m is not None:
            status = "ABSENT (no face)" if m.absent else "face detected"
            color = (255, 120, 120, 255) if m.absent else (140, 230, 160, 255)
            self._text(img, status, (14, y), 0.48, color)
            self._text(img, f"cam {m.camera_fps:4.1f} fps", (250, y), 0.42, (150, 160, 180, 255))
        y += 26
        self._text(img, f"render {info.get('render_fps', 0):4.0f} fps   {info.get('extra', '')}",
                   (14, y), 0.42, (150, 160, 180, 255))
        self._text(img, "ESC quit  D debug  R reset  1/2 test  0 cam  C calib",
                   (14, self.HEIGHT - 12), 0.40, (120, 135, 160, 255))

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.WIDTH, self.HEIGHT,
                           gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img)

    def draw(self, screen_w, screen_h):
        """左上にパネルを描く（アルファブレンドで背景と重ねる）"""
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.shader.use()
        self.shader.set_vec2("uScreen", (screen_w, screen_h))
        # 画面が小さいときはパネルも縮める
        scale = min(1.0, screen_w / 1280.0 * 1.4)
        self.shader.set_vec4("uRect", (16, 16, self.WIDTH * scale, self.HEIGHT * scale))
        self.shader.set_int("uTex", 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_BLEND)


# ============================================================
# 脳オブジェクト
# ============================================================
class Brain:
    """脳のメッシュを GPU に転送し、シェーダーで描画する"""

    def __init__(self, model_path):
        self.shader = ShaderProgram("brain.vert", "brain.frag")
        vertex_data, indices, self.source = load_brain(model_path, config.BRAIN_DETAIL)
        self.count = len(indices)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        stride = 7 * 4    # 1 頂点 = float 7 個
        gl.glEnableVertexAttribArray(0)   # 位置
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)   # 法線
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)   # 溝の深さ
        gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))
        gl.glBindVertexArray(0)

    def draw(self, model, view, proj, cam_pos, focus01, t):
        self.shader.use()
        self.shader.set_mat4("uModel", model)
        self.shader.set_mat4("uView", view)
        self.shader.set_mat4("uProj", proj)
        self.shader.set_vec3("uCamPos", cam_pos)
        self.shader.set_vec3("uLightDir", (0.4, 0.8, 0.6))
        self.shader.set_float("uFocus", focus01)
        self.shader.set_float("uTime", t)
        gl.glBindVertexArray(self.vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)


# ============================================================
# 描画全体
# ============================================================
class Renderer:
    """ウィンドウの作成と、1 フレーム分の描画をまとめたクラス"""

    def __init__(self):
        if not glfw.init():
            raise RuntimeError("glfw の初期化に失敗しました")
        # OpenGL 3.3 Core Profile（シェーダー必須のモダンな OpenGL）を要求する
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, config.MSAA_SAMPLES)   # アンチエイリアス

        self.window = glfw.create_window(config.WINDOW_WIDTH, config.WINDOW_HEIGHT,
                                         config.WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("ウィンドウを作成できませんでした（OpenGL 3.3 に対応していない可能性）")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)                        # 垂直同期

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self.brain = Brain(os.path.join(BASE_DIR, config.BRAIN_MODEL_PATH))
        self.hud = Hud()
        print(f"[renderer] OpenGL {gl.glGetString(gl.GL_VERSION).decode()}  brain: {self.brain.source}")

        self.start_time = time.monotonic()
        self.angle = 0.0
        self.display_focus = 0.0      # 描画用にさらになめらかにした集中度
        self._last_hud = 0.0

    def should_close(self):
        return glfw.window_should_close(self.window)

    def render(self, focus, info, dt):
        """
        1 フレーム描画する。
        focus: 集中度 0〜100、info: HUD に出す情報、dt: 前フレームからの経過時間 [秒]
        """
        t = time.monotonic() - self.start_time

        # 描画用の集中度：急な変化（モード切り替えなど）でもなめらかに変わるようにする
        k = 1.0 - math.exp(-dt / 0.25)
        self.display_focus += k * (focus - self.display_focus)
        f01 = self.display_focus / 100.0

        # 自動回転
        self.angle = (self.angle + config.AUTO_ROTATE_SPEED * dt) % 360.0

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        if fb_w == 0 or fb_h == 0:            # 最小化中
            return
        gl.glViewport(0, 0, fb_w, fb_h)

        # 背景色：集中度が高いほどわずかに青みが増す
        gl.glClearColor(0.02, 0.025 + 0.02 * f01, 0.05 + 0.05 * f01, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        cam_pos = (0.0, 0.9, 4.2)
        view = look_at(cam_pos, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        proj = perspective(40.0, fb_w / fb_h, 0.1, 100.0)
        model = rotate_y(self.angle) @ rotate_x(-8.0)

        self.brain.draw(model, view, proj, cam_pos, f01, t)

        # HUD は毎フレーム作り直すと重いので 0.1 秒ごとに更新する
        if t - self._last_hud > 0.1:
            self.hud.update(info)
            self._last_hud = t
        self.hud.draw(fb_w, fb_h)

        glfw.swap_buffers(self.window)

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()
