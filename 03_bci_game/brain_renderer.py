"""
brain_renderer.py
-----------------
本物の脳（fsaverage5 の皮質表面）と、その上に置いた擬似ニューロンを OpenGL で描く。

  ・左右の半球を、溝の深さによる濃淡と陰影つきで表示する
  ・領域（運動野・手の領域・体性感覚野・前頭眼野・視覚野）を、明るさ 0〜1 で個別に光らせる
  ・擬似ニューロン（02 と同じ見せ方：色 = 好きな方向、光 = 発火）を、左半球の運動野の手の領域の表面上に置く
      右手の運動は、主に反対側（左半球）の一次運動野が制御する（錐体路が延髄で交差するため）。
      このゲームでは右手で操作する想定なので、ニューロンを左半球の手の領域（hand knob）に置いている
  ・カメラはゆっくり自動で回り、マウスのドラッグで回転・ホイールで拡大縮小できる。
    Z キーで手の領域に寄る（ニューロンが小さいので、寄ると発火の様子がよく見える）

座標の変換：脳の座標（mm、x：右、y：前、z：上）→ OpenGL（x：右、y：上、z：手前）
    gl = ((x − cx), (z − cz), −(y − cy)) / 100      （回転なので、三角形の向きは変わらない）
"""

import ctypes
import math

import numpy as np
from OpenGL import GL as gl

import config
from brain_model import REGION_ID, BrainModel, vertex_normals
from gl_utils import ShaderProgram, hue_to_rgb, icosphere, look_at, perspective

MM = 0.01            # 1 mm = 0.01（OpenGL の単位）

# 領域の色（RGB 0〜1）。番号は brain_model.REGION_ID と同じ（0 は使わない）
REGION_COLORS = {
    "motor": (1.00, 0.45, 0.15),     # 運動野：オレンジ
    "hand": (1.00, 0.62, 0.20),      # 手の領域：明るいオレンジ
    "somato": (0.35, 0.80, 1.00),    # 体性感覚野：水色
    "fef": (0.70, 0.45, 1.00),       # 前頭眼野：紫
    "visual": (0.30, 1.00, 0.55),    # 視覚野：緑
}


class OrbitCamera:
    """注視点のまわりを回るカメラ（自動回転 + マウス操作）"""

    def __init__(self):
        self.yaw = -90.0           # -90°：左から見る（左半球の外側面が正面になる）
        self.pitch = 28.0          # 少し上から（手の領域は頭のてっぺんに近い）
        self.dist = 3.3
        self.target = np.zeros(3)
        self.goal_target = np.zeros(3)
        self.goal_dist = 3.3
        self.drag = None           # ドラッグ開始時の (マウス x, y, yaw, pitch)
        self.idle = 0.0            # 最後にマウスで操作してからの時間（しばらくは自動回転しない）

    def mouse_down(self, x, y):
        self.drag = (x, y, self.yaw, self.pitch)

    def mouse_up(self):
        self.drag = None

    def mouse_move(self, x, y):
        if self.drag is not None:
            x0, y0, yaw0, pitch0 = self.drag
            self.yaw = yaw0 + (x - x0) * 0.35
            self.pitch = float(np.clip(pitch0 + (y - y0) * 0.35, -85, 85))
            self.idle = 0.0

    def scroll(self, dy):
        self.goal_dist = float(np.clip(self.goal_dist * (0.9 ** dy), 0.35, 6.0))
        self.idle = 0.0

    def focus(self, target, dist):
        self.goal_target = np.asarray(target, dtype=np.float64)
        self.goal_dist = dist

    def update(self, dt):
        self.idle += dt
        if self.drag is None and self.idle > 3.0:
            self.yaw += config.BRAIN_AUTO_ROTATE * dt
        k = 1.0 - math.exp(-dt / 0.35)                 # 注視点と距離はなめらかに近づける
        self.target += (self.goal_target - self.target) * k
        self.dist += (self.goal_dist - self.dist) * k

    @property
    def eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])
        return self.target + self.dist * d

    def matrices(self, aspect):
        return look_at(self.eye, self.target, (0.0, 1.0, 0.0)), perspective(32.0, aspect, 0.05, 30.0)


class BrainView:
    def __init__(self, model: BrainModel):
        self.model = model
        self.shader = ShaderProgram("brain.vert", "brain.frag")
        self.neuron_shader = ShaderProgram("neuron.vert", "neuron.frag")
        self.camera = OrbitCamera()
        surf = config.BRAIN_SURFACE

        # 両半球の頂点をまとめて中心を求める（pial の座標で）
        allv = np.vstack([model.hemis[h]["pial"] for h in ("left", "right")])
        self.center = (allv.min(axis=0) + allv.max(axis=0)) / 2.0

        # ふくらませた表面（infl）は半球ごとに原点のまわりにあるので、左右に離して並べる（そのずらし量）
        self.shift = {}
        for hemi in ("left", "right"):
            v = model.hemis[hemi][surf].astype(np.float64)
            self.shift[hemi] = (-v.mean(axis=0) + np.array([-45.0 if hemi == "left" else 45.0, 0, 0])
                                + self.center) if surf == "infl" else np.zeros(3)

        self.meshes = []
        for hemi in ("left", "right"):
            h = model.hemis[hemi]
            v = h[surf].astype(np.float64) + self.shift[hemi]
            n = vertex_normals(v, h["faces"])
            data = np.hstack([self.to_gl(v), self.dir_to_gl(n), h["sulc"][:, None],
                              h["region"][:, None].astype(np.float32)]).astype(np.float32)
            self.meshes.append(self._upload_mesh(data, h["faces"].astype(np.uint32).ravel()))

        # 手の領域の中心（Z キーで寄るときの注視点）
        hv = model.region_vertices("left", "hand")
        self.hand_center = self.to_gl(model.hemis["left"][surf][hv].mean(axis=0) + self.shift["left"])
        self.focus_hand = False

        self._init_neuron_buffers()
        self.glow = np.zeros(6)
        self.n_neurons = 0

    # --------------------------------------------------------
    def to_gl(self, p):
        p = np.asarray(p, dtype=np.float64) - self.center
        return (np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1) * MM).astype(np.float32)

    @staticmethod
    def dir_to_gl(d):
        d = np.asarray(d, dtype=np.float64)
        return np.stack([d[..., 0], d[..., 2], -d[..., 1]], axis=-1).astype(np.float32)

    @staticmethod
    def _upload_mesh(data, idx):
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        stride = 8 * 4
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 1, 24), (3, 1, 28)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(off))
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)
        return vao, len(idx)

    def _init_neuron_buffers(self):
        """ニューロンの球（02 の CortexView と同じインスタンス描画）"""
        sphere_v, sphere_i = icosphere(config.NEURON_SPHERE_DETAIL)
        self.sphere_count = len(sphere_i)
        self.nvao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.nvao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, sphere_v.nbytes, sphere_v, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 12, ctypes.c_void_p(0))
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, sphere_i.nbytes, sphere_i, gl.GL_STATIC_DRAW)
        self.inst = {}
        for loc, name, size in ((1, "offset", 3), (2, "color", 3), (3, "flash", 1), (4, "alive", 1),
                                (5, "activity", 1)):
            b = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, size * 4, ctypes.c_void_p(0))
            gl.glVertexAttribDivisor(loc, 1)
            self.inst[name] = b
        gl.glBindVertexArray(0)

    def _upload(self, name, arr):
        data = np.ascontiguousarray(arr, dtype=np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.inst[name])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)

    def set_population(self, pop, rng):
        """
        擬似ニューロンを左半球の手の領域の表面上にランダムに置く（三角形の面積に比例した一様分布）。
        実際の M1 では好きな方向がきれいな地図状には並んでいないので、置き場所と PD は無関係にしている
        """
        pos, normal = self.model.sample_hand_points(pop.n, rng, config.BRAIN_SURFACE)
        pos = pos + normal * config.NEURON_LIFT_MM + self.shift["left"]
        self.n_neurons = pop.n
        self._upload("offset", self.to_gl(pos))
        self._upload("color", hue_to_rgb(pop.pd_angle / (2 * np.pi)))
        self._upload("flash", np.zeros(pop.n))
        self._upload("alive", np.ones(pop.n))
        self._upload("activity", np.zeros(pop.n))

    def toggle_focus(self):
        self.focus_hand = not self.focus_hand
        if self.focus_hand:
            self.camera.focus(self.hand_center, 0.55)
        else:
            self.camera.focus(np.zeros(3), 3.3)

    # --------------------------------------------------------
    def update_glow(self, targets, dt):
        """targets：{領域名: 目標の明るさ 0〜1}。なめらかに近づける"""
        k = 1.0 - math.exp(-dt / config.REGION_GLOW_TAU)
        goal = np.zeros(6)
        for name, v in targets.items():
            goal[REGION_ID[name]] = v
        self.glow += (goal - self.glow) * k

    def draw(self, viewport, t, dt, flash=None, activity=None, alive=None):
        gl.glViewport(*viewport)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.camera.update(dt)
        view, proj = self.camera.matrices(viewport[2] / viewport[3])
        eye = self.camera.eye
        light = eye / np.linalg.norm(eye) + np.array([0.3, 0.8, 0.0])     # カメラ側の斜め上から照らす

        s = self.shader
        s.use()
        s.set_mat4("uView", view)
        s.set_mat4("uProj", proj)
        s.set_vec3("uCamPos", eye)
        s.set_vec3("uLightDir", light)
        s.set_float("uHint", 0.16)
        s.set_float("uTime", t)
        gl.glUniform1fv(s.loc("uGlow"), 6, np.asarray(self.glow, dtype=np.float32))
        colors = np.zeros((6, 3), dtype=np.float32)
        for name, c in REGION_COLORS.items():
            colors[REGION_ID[name]] = c
        gl.glUniform3fv(s.loc("uRegionColor"), 6, colors)
        for vao, count in self.meshes:
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)

        if self.n_neurons > 0 and flash is not None:
            self._upload("flash", flash)
            self._upload("activity", activity)
            self._upload("alive", alive.astype(np.float32))
            ns = self.neuron_shader
            ns.use()
            ns.set_mat4("uView", view)
            ns.set_mat4("uProj", proj)
            ns.set_vec3("uCamPos", eye)
            ns.set_vec3("uLightDir", light)
            ns.set_float("uRadius", config.NEURON_RADIUS_MM * MM)
            gl.glBindVertexArray(self.nvao)
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.sphere_count, gl.GL_UNSIGNED_INT, None,
                                       self.n_neurons)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_DEPTH_TEST)

    def project(self, p_gl, viewport):
        """3D の点 → ビューポート内の画素座標（左上原点）。ラベルを書く位置に使う"""
        view, proj = self.camera.matrices(viewport[2] / viewport[3])
        c = proj @ view @ np.append(p_gl, 1.0)
        if c[3] <= 0:
            return None
        ndc = c[:3] / c[3]
        return ((ndc[0] + 1) / 2 * viewport[2], (1 - ndc[1]) / 2 * viewport[3])
