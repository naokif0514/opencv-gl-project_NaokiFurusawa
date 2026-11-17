"""
tools/make_robot_model.py
-------------------------
ロボット（台座 + 車輪 + 2 関節アーム + 平行グリッパー）の 3D モデルを、Python のコードで組み立てて
assets/models/robot.obj に保存する（Blender がなくても作れるように。自作のモデル）。

  使い方（03_bci_game で）：../.venv/bin/python tools/make_robot_model.py

■ 部品（OBJ のグループ名 = "部品.質感"）。部品ごとに自分の座標系で作り、ゲーム側（robot_view.py）が
  関節の位置・角度で動かす。関節の中心を各部品の原点にしているので、関節が動いても部品の位置関係は崩れない
    chassis   : 車体（原点 = 台座の中心、床の高さ）。x = 前、y = 上、z = 右
    wheel     : 車輪 1 個（原点 = 車軸の中心。車軸は z 方向）。4 個を回転させて使う
    shoulder  : 肩の回転台（原点 = 肩の軸、床の高さ）。上腕の向き（θ1）で回る
    upper_arm : 上腕（原点 = 肩の軸、アームの高さ）。x 方向に長さ ARM_L1
    elbow     : 肘のハウジング（原点 = 肘の軸、アームの高さ）
    forearm   : 前腕（原点 = 肘の軸）。x 方向に長さ ARM_L2
    wrist     : 手首とグリッパーの台（原点 = 手首の軸）
    finger    : 指 1 本（原点 = 手首の軸。+z 側の指。−z 側は鏡に映して使う）。開き具合で z 方向に動かす
■ 形の作り方
    ・面取りした箱：24 個の点の凸包（scipy.spatial.ConvexHull）。面ごとに法線を分けて角ばった見た目にする
    ・回転体：断面の折れ線を軸のまわりに回す（車輪のタイヤ・ホイール、関節のハウジング）
■ 質感（usemtl）：graphite（暗い金属）/ panel（白い塗装）/ steel（鋼）/ rubber（ゴム）/ joint（関節の金属）/
  accent（水色の発光）/ warn（オレンジの塗装）。色と光り方は robot_view.py の ROBOT_MATERIALS で決める
"""

import math
import os
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "assets", "models", "robot.obj")
SEG = 16                       # 回転体の分割数（三角形の数を 3,000 個くらいに収めるため）
WHEEL_SEG = 12                 # 車輪の分割数（4 個あるので少なめ）
ARM_H = 0.075                  # アームの高さ（robot_view.RobotView.ARM_HEIGHT と同じ）
L, WD = config.ROBOT_BODY
L1, L2 = config.ARM_L1, config.ARM_L2
SHOULDER_X = L * 0.30          # 肩の位置（robot.Robot.shoulder と同じ：台座の中心から前へ）


class Mesh:
    """三角形の集まり（頂点ごとに法線を持つ）"""

    def __init__(self):
        self.v, self.n, self.f = [], [], []

    def tri(self, a, b, c, n=None):
        a, b, c = map(np.asarray, (a, b, c))
        if n is None:
            n = np.cross(b - a, c - a)
            n = n / max(np.linalg.norm(n), 1e-12)
        k = len(self.v)
        for p in (a, b, c):
            self.v.append(p)
            self.n.append(np.asarray(n, dtype=np.float64))
        self.f.append((k, k + 1, k + 2))

    def tri_smooth(self, ps, ns):
        k = len(self.v)
        for p, n in zip(ps, ns):
            self.v.append(np.asarray(p, dtype=np.float64))
            self.n.append(np.asarray(n, dtype=np.float64) / max(np.linalg.norm(n), 1e-12))
        self.f.append((k, k + 1, k + 2))

    def add(self, other):
        k = len(self.v)
        self.v += other.v
        self.n += other.n
        self.f += [(a + k, b + k, c + k) for a, b, c in other.f]
        return self

    def transformed(self, R=np.eye(3), t=(0, 0, 0)):
        m = Mesh()
        m.v = [R @ p + np.asarray(t) for p in self.v]
        m.n = [R @ n for n in self.n]
        m.f = list(self.f)
        return m


def hull(points):
    """凸包：面ごとに法線を分ける（角ばった見た目）"""
    P = np.asarray(points, dtype=np.float64)
    h = ConvexHull(P)
    c = P.mean(axis=0)
    m = Mesh()
    for simplex, eq in zip(h.simplices, h.equations):
        a, b, d = P[simplex]
        n = eq[:3]
        if np.dot(np.cross(b - a, d - a), n) < 0:
            b, d = d, b
        if np.dot(n, a - c) < 0:
            n = -n
        m.tri(a, b, d, n)
    return m


def cbox(x0, x1, y0, y1, z0, z1, c=0.002):
    """面取りした箱（範囲 x0〜x1 などの中に、角を c だけ落とす）"""
    pts = []
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
    for axis in range(3):
        for s0 in (0, 1):
            for s1 in (0, 1):
                for s2 in (0, 1):
                    p = np.where([s0, s1, s2], hi, lo).astype(np.float64)
                    for k in range(3):        # 軸 axis 以外は c だけ内側へ
                        if k != axis:
                            p[k] += c if p[k] == lo[k] else -c
                    pts.append(p)
    return hull(pts)


def box(x0, x1, y0, y1, z0, z1):
    """面取りしない箱（小さな部品用。12 三角形）"""
    return hull([(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def wedge(x0, x1, y0, y1, z0, z1, top_back, c=0.0015):
    """前が低く後ろが高い（または逆の）くさび形：上面の x0 側を top_back の高さまで下げる"""
    pts = []
    for x in (x0, x1):
        for z in (z0 + c, z1 - c):
            pts.append((x, y0, z))
            pts.append((x, (top_back if x == x0 else y1), z))
    for x in (x0 + c, x1 - c):
        for z in (z0, z1):
            pts.append((x, y0 + c, z))
            pts.append((x, (top_back if x == x0 + c else y1) - c, z))
    return hull(pts)


def lathe(profile, seg=SEG, axis="y", center=(0, 0, 0)):
    """
    回転体：profile = [(半径, 高さ), ...] の折れ線を、軸のまわりに回す。
    折れ線の 1 区間ごとに法線を分け（断面は角ばる）、まわりの向きにはなめらかにする
    """
    m = Mesh()
    ang = [2 * math.pi * i / seg for i in range(seg + 1)]
    for (r0, h0), (r1, h1) in zip(profile[:-1], profile[1:]):
        dr, dh = r1 - r0, h1 - h0
        nr, nh = dh, -dr                      # 区間の外向きの法線（断面の中で）
        ln = math.hypot(nr, nh) or 1.0
        nr, nh = nr / ln, nh / ln
        for a0, a1 in zip(ang[:-1], ang[1:]):
            def P(r, h, a):
                return np.array([r * math.cos(a), h, r * math.sin(a)])

            def N(a):
                return np.array([nr * math.cos(a), nh, nr * math.sin(a)])
            p00, p01, p10, p11 = P(r0, h0, a0), P(r0, h0, a1), P(r1, h1, a0), P(r1, h1, a1)
            if r0 > 1e-9:
                m.tri_smooth([p00, p10, p01], [N(a0), N(a0), N(a1)])
            if r1 > 1e-9:
                m.tri_smooth([p01, p10, p11], [N(a1), N(a0), N(a1)])
    if axis == "z":       # y 軸の回転体を z 軸へ（車輪）
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        m = m.transformed(R)
    return m.transformed(np.eye(3), center)


def cyl_profile(r, h0, h1, c=0.0):
    """ふたつきの円柱の断面（角を c だけ面取り）"""
    if c <= 0:
        return [(0, h1), (r, h1), (r, h0), (0, h0)]
    return [(0, h1), (r - c, h1), (r, h1 - c), (r, h0 + c), (r - c, h0), (0, h0)]


def beam(length, w, h, c, x0=0.0, end_round=True):
    """アームのはり：x = x0〜length、断面 幅 w（z）× 高さ h（y）、アームの高さが中心。両端は細めて丸みを出す"""
    pts = []
    y0, y1 = ARM_H - h / 2, ARM_H + h / 2
    taper = min(w, h) * 0.35 if end_round else 0.0
    for x, t in ((x0, taper), (x0 + taper * 1.5, 0.0), (length - taper * 1.5, 0.0), (length, taper)):
        for y in (y0 + t, y1 - t):
            for z in (-w / 2 + c + t, w / 2 - c - t):
                pts.append((x, y, z))
        for y in (y0 + c + t, y1 - c - t):
            for z in (-w / 2 + t, w / 2 - t):
                pts.append((x, y, z))
    return hull(pts)


# ============================================================
# 部品
# ============================================================
def chassis():
    parts = {}
    hl, hw = L / 2, WD / 2
    g = Mesh()
    # 下の車体（車輪の内側）と、車輪の上のフェンダー
    g.add(cbox(-hl, hl, 0.010, 0.040, -hw + 0.015, hw - 0.015, 0.004))
    for sx in (-1, 1):
        g.add(cbox(-hl + 0.004, hl - 0.004, 0.034, 0.041, -hw - 0.001, -hw + 0.016, 0.0025)
              .transformed(np.diag([1, 1, sx])))
    # 前のセンサーの台（くさび形）
    g.add(wedge(hl - 0.012, hl + 0.004, 0.016, 0.034, -0.022, 0.022, 0.040))
    parts["graphite"] = g

    p = Mesh()
    # 上のデッキ（白い板）と、後ろのバッテリー
    p.add(cbox(-hl + 0.002, hl - 0.010, 0.040, 0.046, -hw + 0.004, hw - 0.004, 0.002))
    p.add(cbox(-hl + 0.002, -hl + 0.030, 0.046, 0.058, -0.022, 0.022, 0.003))
    parts["panel"] = p

    a = Mesh()
    # 前のセンサーの光る帯、デッキの縁の光る線、バッテリーの表示灯
    a.add(box(hl + 0.0035, hl + 0.0055, 0.022, 0.029, -0.018, 0.018))
    for sz in (-1, 1):
        a.add(box(-hl + 0.010, hl - 0.016, 0.0455, 0.0475, sz * (hw - 0.006) - 0.0012, sz * (hw - 0.006) + 0.0012))
    for i in range(3):
        a.add(box(-hl + 0.008 + i * 0.006, -hl + 0.012 + i * 0.006, 0.058, 0.0595, -0.016, -0.010))
    parts["accent"] = a

    w = Mesh()
    # 後ろの角のオレンジの印（注意の印）
    for sz in (-1, 1):
        w.add(box(-hl - 0.0005, -hl + 0.006, 0.026, 0.036, sz * 0.012 - 0.008, sz * 0.012 + 0.008))
    parts["warn"] = w

    s = Mesh()
    # アンテナ（細い棒）と、前のセンサーの両脇のレンズ
    s.add(lathe(cyl_profile(0.0012, 0.058, 0.098), seg=10, center=(-hl + 0.008, 0, hw - 0.012)))
    s.add(lathe(cyl_profile(0.0028, 0.096, 0.101, 0.001), seg=12, center=(-hl + 0.008, 0, hw - 0.012)))
    for sz in (-1, 1):
        s.add(lathe(cyl_profile(0.0035, 0.0, 0.004, 0.0008), seg=14, axis="z")
              .transformed(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float), (hl + 0.002, 0.025, sz * 0.026)))
    parts["steel"] = s
    return parts


def wheel():
    r, w = 0.016, 0.014
    tire = lathe([(0.0105, w / 2), (r - 0.003, w / 2), (r, w / 2 - 0.003), (r, -w / 2 + 0.003),
                  (r - 0.003, -w / 2), (0.0105, -w / 2)], seg=WHEEL_SEG, axis="z")
    hub = lathe([(0, w / 2 + 0.0008), (0.0040, w / 2 + 0.0008), (0.0105, w / 2),
                 (0.0105, -w / 2), (0.0040, -w / 2 - 0.0008), (0, -w / 2 - 0.0008)], seg=WHEEL_SEG, axis="z")
    spokes = Mesh()
    for i in range(5):              # ホイールのスポーク（回っているのが分かるように）
        a = 2 * math.pi * i / 5
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        spokes.add(box(0.003, 0.0102, -0.0012, 0.0012, -w / 2 - 0.0006, w / 2 + 0.0006).transformed(R))
    return {"rubber": tire, "steel": hub, "graphite": spokes}


def shoulder():
    base_y = 0.046
    turn = lathe(cyl_profile(0.020, base_y, base_y + 0.005, 0.0015))
    housing = lathe([(0, ARM_H + 0.013), (0.011, ARM_H + 0.013), (0.0135, ARM_H + 0.010), (0.0135, base_y + 0.008),
                     (0.016, base_y + 0.006), (0.016, base_y + 0.005), (0, base_y + 0.005)])
    ring = lathe([(0.0137, ARM_H - 0.006), (0.0137, ARM_H - 0.0085)], seg=SEG)
    cap = lathe(cyl_profile(0.006, ARM_H + 0.013, ARM_H + 0.015, 0.001), seg=12)
    return {"graphite": turn, "joint": housing, "accent": ring, "steel": cap}


def upper_arm():
    body = beam(L1, 0.016, 0.017, 0.003)
    side = Mesh()
    for sz in (-1, 1):              # 横の黒い板（はりのくぼみ）
        side.add(box(0.020, L1 - 0.020, ARM_H - 0.0045, ARM_H + 0.0045, -0.0005, 0.0012)
                 .transformed(np.diag([1, 1, sz]), (0, 0, sz * 0.0081)))
    led = box(L1 * 0.5 - 0.012, L1 * 0.5 + 0.012, ARM_H + 0.0083, ARM_H + 0.0093, -0.0018, 0.0018)
    warn = box(0.014, 0.019, ARM_H + 0.0080, ARM_H + 0.0091, -0.0065, 0.0065)
    return {"panel": body, "graphite": side, "accent": led, "warn": warn}


def joint_housing(r, h, ring_y):
    housing = lathe([(0, ARM_H + h / 2), (r - 0.002, ARM_H + h / 2), (r, ARM_H + h / 2 - 0.002),
                     (r, ARM_H - h / 2 + 0.002), (r - 0.002, ARM_H - h / 2), (0, ARM_H - h / 2)])
    ring = lathe([(r + 0.0002, ARM_H + ring_y + 0.0012), (r + 0.0002, ARM_H + ring_y - 0.0012)])
    cap = lathe(cyl_profile(r * 0.45, ARM_H + h / 2, ARM_H + h / 2 + 0.0015, 0.0006), seg=12)
    return {"joint": housing, "accent": ring, "steel": cap}


def forearm():
    body = beam(L2, 0.012, 0.013, 0.0025)
    side = Mesh()
    for sz in (-1, 1):
        side.add(box(0.016, L2 - 0.016, ARM_H - 0.0032, ARM_H + 0.0032, -0.0005, 0.0010)
                 .transformed(np.diag([1, 1, sz]), (0, 0, sz * 0.0061)))
    led = box(L2 * 0.5 - 0.008, L2 * 0.5 + 0.008, ARM_H + 0.0063, ARM_H + 0.0072, -0.0014, 0.0014)
    return {"panel": body, "graphite": side, "accent": led}


def wrist():
    hub = joint_housing(0.0085, 0.016, 0.004)
    palm = cbox(0.004, 0.011, ARM_H - 0.0055, ARM_H + 0.0055, -0.0125, 0.0125, 0.0012)
    rail = box(0.009, 0.0125, ARM_H - 0.0025, ARM_H + 0.0025, -0.0140, 0.0140)
    out = {"joint": hub["joint"], "accent": hub["accent"], "graphite": palm, "steel": rail.add(hub["steel"])}
    return out


def finger():
    """+z 側の指 1 本。z = 0 が指の内側の面（ゲーム側で開き具合に応じて z 方向へずらす）"""
    jaw = cbox(0.009, 0.026, ARM_H - 0.0045, ARM_H + 0.0045, 0.0, 0.0035, 0.0007)
    tip = wedge(0.026, 0.032, ARM_H - 0.0045, ARM_H + 0.0045, 0.0, 0.0035, ARM_H + 0.0045)
    pad = box(0.014, 0.030, ARM_H - 0.0035, ARM_H + 0.0035, -0.0008, 0.0002)
    return {"steel": jaw.add(tip), "rubber": pad}


PARTS = {
    "chassis": chassis,
    "wheel": wheel,
    "shoulder": shoulder,
    "upper_arm": upper_arm,
    "elbow": lambda: joint_housing(0.0125, 0.024, 0.006),
    "forearm": forearm,
    "wrist": wrist,
    "finger": finger,
}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = ["# BCI Operator のロボット（tools/make_robot_model.py で生成。自作）",
             "# グループ名 = 部品.質感。単位はワールド座標（机の幅 = 1）。x = 前、y = 上、z = 右"]
    vi = 1
    total = {}
    for part, fn in PARTS.items():
        for mat, m in fn().items():
            lines.append(f"g {part}.{mat}")
            lines.append(f"usemtl {mat}")
            for p in m.v:
                lines.append("v %.6f %.6f %.6f" % tuple(p))
            for n in m.n:
                lines.append("vn %.5f %.5f %.5f" % tuple(n))
            for a, b, c in m.f:
                lines.append(f"f {a + vi}//{a + vi} {b + vi}//{b + vi} {c + vi}//{c + vi}")
            vi += len(m.v)
            total[part] = total.get(part, 0) + len(m.f)
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    # 実際に描く三角形の数（車輪は 4 個、指は 2 本）
    n = sum(v * (4 if k == "wheel" else 2 if k == "finger" else 1) for k, v in total.items())
    print(f"保存しました: {OUT}")
    print("部品ごとの三角形:", total)
    print(f"ロボット 1 台で描く三角形: {n}")


if __name__ == "__main__":
    main()
