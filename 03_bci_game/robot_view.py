"""
robot_view.py
-------------
3D の場面で使うメッシュ（立方体・ふたつきの円柱・球）と、ロボット（台座 + 2 関節アーム + グリッパー）の描画。
（02_pseudo_bci/renderer.py の ArmView と同じ方法：同じメッシュを部品ごとの変換行列で使い回す。
  03 では真上からの正射影をやめ、scene3d.py の透視投影の 3D 空間に置くように作り直した）

座標の対応（scene3d.to3d）：ワールド座標 (x, y)（机の上の平面、y は奥）→ 3D (X, 高さ, Z)
  X = x − WORLD_W/2、Z = WORLD_H/2 − y（奥ほど Z が小さい = カメラから遠い）
ワールド座標で角度 a の向き (cos a, sin a) は、3D では (cos a, 0, −sin a) になるので、Y 軸まわりに a 回す（rotate_y(a)）。
"""

import ctypes
import math

import numpy as np
from OpenGL import GL as gl

import config
from gfx2d import rotate_z, scale, translate
from gl_utils import icosphere


def rotate_y(a):
    """Y 軸まわりの回転（x 軸 → (cos a, 0, −sin a)）"""
    m = np.identity(4, dtype=np.float32)
    c, s = math.cos(a), math.sin(a)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def cylinder_mesh(seg):
    """x 軸方向に 0〜1、半径 1 の円柱（両端のふたつき）"""
    a = np.linspace(0, 2 * np.pi, seg + 1)
    verts, idx = [], []
    for x in (0.0, 1.0):
        for ang in a:
            c, s = math.cos(ang), math.sin(ang)
            verts.append((x, c, s, 0.0, c, s))
    for i in range(seg):
        j = i + seg + 1
        idx += [i, i + 1, j, i + 1, j + 1, j]
    for x, nx in ((0.0, -1.0), (1.0, 1.0)):
        center = len(verts)
        verts.append((x, 0.0, 0.0, nx, 0.0, 0.0))
        for ang in a:
            verts.append((x, math.cos(ang), math.sin(ang), nx, 0.0, 0.0))
        for i in range(seg):
            idx += [center, center + 1 + i, center + 2 + i]
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


def cube_mesh():
    """中心が原点、一辺 2（−1〜1）の立方体（面ごとに法線を分ける）"""
    verts, idx = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            u, v = np.eye(3)[(axis + 1) % 3], np.eye(3)[(axis + 2) % 3]     # 法線と直交する 2 つの軸
            base = len(verts)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = n + du * u + dv * v
                verts.append((*p, *n))
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(verts, dtype=np.float32), np.array(idx, dtype=np.uint32)


def make_mesh(verts, idx):
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


class Meshes:
    """場面で共有するメッシュ（ポリゴン数は少なめ：円柱 12 分割、球 80 三角形）"""

    def __init__(self):
        self.cyl = make_mesh(*cylinder_mesh(config.ARM_MESH_SEGMENTS))
        sv, si = icosphere(1)
        self.sph = make_mesh(np.hstack([sv, sv]).astype(np.float32), si)
        self.cube = make_mesh(*cube_mesh())


class RobotView:
    """ロボット（台座 + 車輪 + 2 関節アーム + グリッパー）を scene3d の空間に描く"""

    BODY_COLOR = (0.22, 0.27, 0.36)
    ACCENT_COLOR = (0.15, 0.75, 0.95)
    LINK_COLOR = (0.70, 0.74, 0.82)
    JOINT_COLOR = (0.95, 0.55, 0.18)
    WHEEL_COLOR = (0.08, 0.08, 0.10)
    ARM_HEIGHT = 0.075            # アームを動かす高さ（机の上の平面と平行）

    def __init__(self, scene):
        self.s = scene

    def _link(self, p0, p1, r, color, h):
        """ワールド座標の 2 点を、高さ h で結ぶ円柱"""
        s = self.s
        a, b = s.to3d(p0, h), s.to3d(p1, h)
        d = b - a
        length = float(np.hypot(d[0], d[2]))
        rot = rotate_y(math.atan2(-d[2], d[0]))
        s.draw(s.meshes.cyl, translate(*a) @ rot @ scale(length, r, r), rot, color, spec=0.6)

    def draw(self, robot):
        s = self.s
        f = robot.heading
        fw, side = robot.forward, np.array([-math.sin(f), math.cos(f)])
        L, Wd = config.ROBOT_BODY
        rot = rotate_y(f)
        # ---- 台座 ----
        c = s.to3d(robot.pos, 0.032)
        s.draw(s.meshes.cube, translate(*c) @ rot @ scale(L / 2, 0.022, Wd / 2), rot, self.BODY_COLOR, spec=0.3)
        fr = s.to3d(robot.pos + fw * L * 0.42, 0.05)
        s.draw(s.meshes.cube, translate(*fr) @ rot @ scale(0.008, 0.006, Wd * 0.42), rot, self.ACCENT_COLOR,
               emissive=0.6)
        axle = rotate_y(f + math.pi / 2)              # 車輪（車軸は台座の左右方向）
        for sx in (-0.3, 0.3):
            for sy in (-1, 1):
                w = robot.pos + fw * L * sx + side * sy * (Wd / 2) - side * 0.007
                s.draw(s.meshes.cyl, translate(*s.to3d(w, 0.016)) @ axle @ scale(0.014, 0.016, 0.016), axle,
                       self.WHEEL_COLOR)

        # ---- アーム（机と平行な高さ ARM_HEIGHT の平面で動く）----
        pose = robot.pose
        h = self.ARM_HEIGHT
        s.draw(s.meshes.cyl, translate(*s.to3d(pose.base, 0.05)) @ rotate_z(math.pi / 2) @ scale(h - 0.05, 0.014, 0.014),
               np.identity(4, dtype=np.float32), self.LINK_COLOR)      # 肩の柱
        for p, r in ((pose.base, 0.020), (pose.elbow, 0.016), (pose.hand, 0.012)):
            s.draw(s.meshes.sph, translate(*s.to3d(p, h)) @ scale(r, r, r), np.identity(4, dtype=np.float32),
                   self.JOINT_COLOR, spec=0.5)
        self._link(pose.base, pose.elbow, 0.011, self.LINK_COLOR, h)
        self._link(pose.elbow, pose.hand, 0.009, self.LINK_COLOR, h)
        # グリッパー：前腕の向きから、開き具合に応じた角度で 2 本の指を出し、先を少し下へ
        fa = pose.theta1 + pose.theta2
        spread = math.radians(8 + 30 * robot.grip)
        for sgn in (1, -1):
            a = fa + sgn * spread
            tip = pose.hand + 0.03 * np.array([math.cos(a), math.sin(a)])
            self._link(pose.hand, tip, 0.005, self.LINK_COLOR, h)
