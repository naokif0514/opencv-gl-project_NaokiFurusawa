"""
robot_view.py
-------------
3D の場面で使うメッシュ（立方体・ふたつきの円柱・球。物・カゴ・机に使う）と、ロボットの描画。
ロボットは、assets/models/robot.obj（tools/make_robot_model.py で Python のコードから生成した自作のモデル。
約 3,200 三角形）を部品ごとに読み込み、関節の位置・角度の変換行列で動かして描く
（部品ごとに変換行列を変えて描く方法は 02_pseudo_bci/renderer.py の ArmView と同じ）。

座標の対応（scene3d.to3d）：ワールド座標 (x, y)（机の上の平面、y は奥）→ 3D (X, 高さ, Z)
  X = x − WORLD_W/2、Z = WORLD_H/2 − y（奥ほど Z が小さい = カメラから遠い）
ワールド座標で角度 a の向き (cos a, sin a) は、3D では (cos a, 0, −sin a) になるので、Y 軸まわりに a 回す（rotate_y(a)）。
"""

import ctypes
import math
import os

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
    """場面で共有するメッシュ（細かさは画質の段階で変わる：円柱 ARM_MESH_SEGMENTS 分割、球 SPHERE_SUBDIV 回分割）"""

    def __init__(self):
        self.cyl = make_mesh(*cylinder_mesh(config.ARM_MESH_SEGMENTS))
        sv, si = icosphere(config.SPHERE_SUBDIV)
        self.sph = make_mesh(np.hstack([sv, sv]).astype(np.float32), si)
        self.cube = make_mesh(*cube_mesh())


# ============================================================
# ロボットのモデル（assets/models/robot.obj。tools/make_robot_model.py で生成）
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "models", "robot.obj")

# 質感：OBJ の usemtl の名前 → (色, 新しいシェーダーの質感, 以前のシェーダーの鏡面反射の強さ, 発光)
ROBOT_MATERIALS = {
    "graphite": ((0.17, 0.19, 0.23), "gunmetal", 0.35, 0.0),   # 車体・アームの横の板：暗い金属
    "panel":    ((0.80, 0.83, 0.87), "painted", 0.30, 0.0),    # デッキ・アームのはり：白い塗装
    "steel":    ((0.70, 0.73, 0.77), "steel", 0.60, 0.0),      # ホイール・指・ふた：鋼
    "rubber":   ((0.07, 0.07, 0.08), "matte", 0.05, 0.0),      # タイヤ・指先のパッド：ゴム
    "joint":    ((0.26, 0.29, 0.34), "anodized", 0.45, 0.0),   # 関節のハウジング：暗いアルマイト
    "accent":   ((0.30, 0.88, 1.00), "plastic", 0.30, 0.85),   # 光る線・表示灯：水色（HUD の色）
    "warn":     ((1.00, 0.60, 0.18), "painted", 0.30, 0.0),    # 注意の印：オレンジ
}


# ライバル（チャレンジ）の配色：白い塗装を赤に、光る線をオレンジに（プレイヤーの白・水色と補色に近い色）
RIVAL_PALETTE = {
    "panel":  ((0.86, 0.20, 0.17), "painted", 0.30, 0.0),
    "accent": ((1.00, 0.52, 0.18), "plastic", 0.30, 0.85),
    "warn":   ((0.95, 0.85, 0.25), "painted", 0.30, 0.0),
}
FROZEN_COLOR = (0.70, 0.88, 1.00)   # 止まっている（凍った）ときの色：青白く光る


def load_obj(path):
    """OBJ（v / vn / f v//vn、g 部品.質感）を読み、{部品: [(質感, (vao, 個数)), ...]} を返す"""
    vs, ns, groups, cur = [], [], {}, None
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "v":
                vs.append([float(x) for x in t[1:4]])
            elif t[0] == "vn":
                ns.append([float(x) for x in t[1:4]])
            elif t[0] == "g":
                cur = t[1]
                groups.setdefault(cur, [])
            elif t[0] == "f":
                groups[cur].append([tuple(int(i) - 1 for i in c.split("//")) for c in t[1:4]])
    vs, ns = np.array(vs, np.float32), np.array(ns, np.float32)
    parts = {}
    for name, faces in groups.items():
        part, mat = name.split(".")
        corners = np.array(faces).reshape(-1, 2)
        verts = np.hstack([vs[corners[:, 0]], ns[corners[:, 1]]]).astype(np.float32)
        idx = np.arange(len(verts), dtype=np.uint32)
        parts.setdefault(part, []).append((mat, make_mesh(verts, idx)))
    return parts


class RobotView:
    """
    ロボットを scene3d の空間に描く。部品ごとに、関節の位置と角度から変換行列を作る：
      車体・車輪：台座の位置と向き（車輪は進んだ距離と曲がった角度で回す）
      肩：肩の軸で θ1、上腕：肩から θ1、肘・前腕：肘から θ1 + θ2、手首・指：手先から θ1 + θ2
      指：開き具合（robot.grip）で左右に開く。物を持っているときは、物の大きさのところで止める
    """

    ARM_HEIGHT = 0.075            # アームを動かす高さ（机の上の平面と平行。モデルもこの高さで作ってある）
    WHEEL_R = 0.016               # 車輪の半径
    WHEEL_POS = ((0.33, 1), (0.33, -1), (-0.33, 1), (-0.33, -1))   # (台座の長さに対する前後の位置, 右 = 1 / 左 = −1)
    FINGER_GAP = (0.0016, 0.0115) # 指の内側の面の、中心からの距離（閉じた, 開いた）

    def __init__(self, scene, palette=None):
        """palette：質感の上書き（ライバルは RIVAL_PALETTE）"""
        self.s = scene
        self.materials = {**ROBOT_MATERIALS, **(palette or {})}
        self.frozen = 0.0             # 0〜1：凍った見た目の強さ（draw の引数で毎フレーム決める）
        self.parts = load_obj(MODEL_PATH)
        self.prev = None              # (位置, 向き)：車輪を回すため
        self.spin = {1: 0.0, -1: 0.0} # 右・左の車輪の回転角

    def _part(self, name, M, rot):
        s = self.s
        f = self.frozen
        for mat, mesh in self.parts[name]:
            color, material, spec, emissive = self.materials[mat]
            if f > 0:                 # 凍った見た目：色を青白く寄せて光らせる
                color = tuple(c * (1 - 0.75 * f) + fc * 0.75 * f for c, fc in zip(color, FROZEN_COLOR))
                emissive = emissive + 0.45 * f
            s.draw(mesh, M, rot, color, emissive=emissive, spec=spec, material=material)

    def _update_wheels(self, robot):
        """車輪の回転：前へ進んだ距離 ± 向きを変えた分（左右の車輪で逆向き）"""
        pos, hd = robot.pos.copy(), robot.heading
        if self.prev is not None:
            d = pos - self.prev[0]
            fwd = float(np.dot(d, robot.forward))
            dh = (hd - self.prev[1] + math.pi) % (2 * math.pi) - math.pi
            half = config.ROBOT_BODY[1] / 2
            for side in (1, -1):
                # 右の車輪（side = 1）は、左に曲がる（dh > 0）と多く進む
                self.spin[side] += (fwd + side * dh * half) / self.WHEEL_R
        self.prev = (pos, hd)

    def draw(self, robot, held_half=None, frozen=0.0):
        """held_half：持っている物の表面までの距離（指をそこで止める。持っていなければ None）、frozen：凍った見た目 0〜1"""
        s = self.s
        self.frozen = frozen
        if frozen <= 0:               # 止まっている間は車輪も回さない
            self._update_wheels(robot)
        L, Wd = config.ROBOT_BODY
        rot = rotate_y(robot.heading)
        base = translate(*s.to3d(robot.pos)) @ rot
        self._part("chassis", base, rot)
        for fx, side in self.WHEEL_POS:
            z = side * (Wd / 2 - 0.007)
            spin = rotate_z(-self.spin[side])
            self._part("wheel", base @ translate(fx * L, self.WHEEL_R, z) @ spin, rot @ spin)

        pose = robot.pose
        t1 = pose.theta1
        fa = pose.theta1 + pose.theta2
        r1, rf = rotate_y(t1), rotate_y(fa)
        self._part("shoulder", translate(*s.to3d(pose.base)) @ r1, r1)
        self._part("upper_arm", translate(*s.to3d(pose.base)) @ r1, r1)
        self._part("elbow", translate(*s.to3d(pose.elbow)) @ rf, rf)
        self._part("forearm", translate(*s.to3d(pose.elbow)) @ rf, rf)
        wrist = translate(*s.to3d(pose.hand)) @ rf
        self._part("wrist", wrist, rf)
        lo, hi = self.FINGER_GAP
        gap = lo + (hi - lo) * robot.grip
        if held_half is not None:             # 持っている物の表面で止める（めり込まない）
            gap = max(gap, held_half + 0.0006)
        mirror = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        self._part("finger", wrist @ translate(0, 0, gap), rf)
        self._part("finger", wrist @ mirror @ translate(0, 0, gap), rf @ mirror)
