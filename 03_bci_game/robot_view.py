"""
robot_view.py
-------------
ロボット（台座 + 2 関節アーム + グリッパー）と仮想アイテムを 3D で描く。
（02_pseudo_bci/renderer.py の ArmView と同じ方法：円柱・球のメッシュを部品ごとの変換行列で使い回し、
  シェーダーは 02 の arm.vert / arm.frag をコピーしたもの。03 では台座・車輪・立方体を追加した）

AR 画面を真上から正射影で見る：ワールド座標 x 0〜WORLD_W、y 0〜WORLD_H が AR 画面にぴったり重なるので、
カメラ映像・視線の注視点・アイテムの位置と、ロボットの位置がそのまま対応する。z は高さ（手前が大きい）。
"""

import ctypes
import math

import numpy as np
from OpenGL import GL as gl

import config
from gfx2d import ortho, rotate_z, scale, translate
from gl_utils import ShaderProgram, icosphere


def rotate_y(a):
    m = np.identity(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def _cylinder_mesh(seg):
    """x 軸方向に 0〜1、半径 1 の円柱（両端のふたつき。車輪や台座を上から見ても穴があかないように）"""
    a = np.linspace(0, 2 * np.pi, seg + 1)
    verts, idx = [], []
    for x in (0.0, 1.0):
        for ang in a:
            c, s = math.cos(ang), math.sin(ang)
            verts.append((x, c, s, 0.0, c, s))
    for i in range(seg):
        j = i + seg + 1
        idx += [i, i + 1, j, i + 1, j + 1, j]
    for x, nx in ((0.0, -1.0), (1.0, 1.0)):            # ふた
        center = len(verts)
        verts.append((x, 0.0, 0.0, nx, 0.0, 0.0))
        for ang in a:
            verts.append((x, math.cos(ang), math.sin(ang), nx, 0.0, 0.0))
        for i in range(seg):
            idx += [center, center + 1 + i, center + 2 + i]
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


def _cube_mesh():
    """中心が原点、一辺 2（−1〜1）の立方体（面ごとに法線を分ける）"""
    verts, idx = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            u, v = np.roll(np.eye(3), axis + 1, axis=0)[:2]
            base = len(verts)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = n + du * u + dv * v
                verts.append((*p, *n))
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


class RobotView:
    BODY_COLOR = (0.22, 0.27, 0.36)
    ACCENT_COLOR = (0.15, 0.75, 0.95)
    LINK_COLOR = (0.66, 0.71, 0.80)
    JOINT_COLOR = (0.95, 0.55, 0.18)
    WHEEL_COLOR = (0.08, 0.08, 0.10)

    def __init__(self):
        self.shader = ShaderProgram("arm.vert", "arm.frag")
        self.cyl = self._make_mesh(*_cylinder_mesh(config.ARM_MESH_SEGMENTS))
        sv, si = icosphere(2)
        self.sph = self._make_mesh(np.hstack([sv, sv]).astype(np.float32), si)
        self.cube = self._make_mesh(*_cube_mesh())

    @staticmethod
    def _make_mesh(verts, idx):
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
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
        return vao, len(idx)

    # --------------------------------------------------------
    def _draw(self, mesh, model, rot, color, alpha=1.0):
        self.shader.set_mat4("uModel", model)
        self.shader.set_mat4("uRot", rot)
        self.shader.set_vec3("uColor", color)
        self.shader.set_float("uAlpha", alpha)
        gl.glBindVertexArray(mesh[0])
        gl.glDrawElements(gl.GL_TRIANGLES, mesh[1], gl.GL_UNSIGNED_INT, None)

    def _link(self, p0, p1, r, color, z):
        d = np.asarray(p1) - np.asarray(p0)
        length = float(np.hypot(*d))
        rot = rotate_z(math.atan2(d[1], d[0]))
        self._draw(self.cyl, translate(p0[0], p0[1], z) @ rot @ scale(length, r, r), rot, color)

    def _sphere(self, p, r, color, z):
        self._draw(self.sph, translate(p[0], p[1], z) @ scale(r, r, r), np.identity(4, dtype=np.float32), color)

    def _box(self, p, size, angle, color, z, alpha=1.0):
        rot = rotate_z(angle)
        self._draw(self.cube, translate(p[0], p[1], z) @ rot @ scale(*size), rot, color, alpha)

    # --------------------------------------------------------
    def draw(self, robot, items, viewport):
        gl.glViewport(*viewport)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        self.shader.use()
        self.shader.set_mat4("uProj", ortho(0.0, config.WORLD_W, 0.0, config.WORLD_H, -1.0, 1.0))

        # ---- 仮想アイテム（立方体）----
        for it in items:
            if it.kind != "virtual":
                continue
            h = it.radius * 0.8
            z = h + (0.09 if it.held else 0.0)
            self._box(it.pos, (h, h, h), it.angle, it.rgb, z)

        # ---- 台座 ----
        f = robot.heading
        fw, side = robot.forward, np.array([-math.sin(f), math.cos(f)])
        L, Wd = config.ROBOT_BODY
        self._box(robot.pos, (L / 2, Wd / 2, 0.018), f, self.BODY_COLOR, 0.022)
        self._box(robot.pos + fw * L * 0.42, (0.008, Wd * 0.42, 0.006), f, self.ACCENT_COLOR, 0.042)   # 前のライト
        axle = rotate_z(f + math.pi / 2)                          # 車輪（車軸は台座の左右方向）
        for sx in (-0.3, 0.3):
            for sy in (-1, 1):
                c = robot.pos + fw * L * sx + side * sy * (Wd / 2) - side * 0.007
                self._draw(self.cyl, translate(c[0], c[1], 0.016) @ axle @ scale(0.014, 0.016, 0.016), axle,
                           self.WHEEL_COLOR)

        # ---- アーム ----
        pose = robot.pose
        z = 0.06
        self._sphere(pose.base, 0.022, self.JOINT_COLOR, z)
        self._link(pose.base, pose.elbow, 0.012, self.LINK_COLOR, z)
        self._link(pose.elbow, pose.hand, 0.010, self.LINK_COLOR, z)
        self._sphere(pose.elbow, 0.016, self.JOINT_COLOR, z)
        self._sphere(pose.hand, 0.012, self.JOINT_COLOR, z)
        # グリッパー：前腕の向きから、開き具合に応じた角度で 2 本の指を出す
        fa = pose.theta1 + pose.theta2
        spread = math.radians(8 + 30 * robot.grip)
        for s in (1, -1):
            a = fa + s * spread
            tip = pose.hand + 0.03 * np.array([math.cos(a), math.sin(a)])
            self._link(pose.hand, tip, 0.005, self.LINK_COLOR, z)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
