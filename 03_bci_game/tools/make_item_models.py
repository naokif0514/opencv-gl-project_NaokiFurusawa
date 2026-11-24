"""
tools/make_item_models.py
-------------------------
机の上の物（箱・缶・玉・ボーナス）とゴールのカゴの 3D モデルを、Python のコードで組み立てて
assets/models/items.obj に保存する（自作。形の作り方は tools/make_robot_model.py と同じ部品を使う）。

  使い方（03_bci_game で）：../.venv/bin/python tools/make_item_models.py

■ 大きさ：物は「半径 1」の大きさで作り、ゲーム側（scene3d.py）が物の半径 r 倍に拡大する。原点は床の上の中心。
  今までの簡単な形と同じ大きさ・高さにしてあるので、視線で選ぶときの投影・吸い付く範囲、つかむ判定は変わらない
    box   ：立方体（半分の幅 0.85、高さ 0〜1.7）
    can   ：円柱（半径 1、高さ 0〜2.6）
    ball  ：球（半径 1、中心の高さ 1）
    bonus ：星（半径 1、中心の高さ 1.3。ゲーム側で回して上下にゆらす）
  カゴ（basket）だけは、ワールド座標の大きさ（GOAL_RADIUS）で作る
■ グループ名 = "物.役割"。役割 main は物の種類の色（config.ITEM_TYPES の rgb）、ほかは決まった色
  （色・質感は scene3d.py の ITEM_PARTS で決める）
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402
from make_robot_model import Mesh, box, cbox, cyl_profile, hull, lathe  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "assets", "models", "items.obj")


def uv_sphere(r, center, nu=28, nv=18):
    """なめらかな球"""
    m = Mesh()
    c = np.asarray(center, dtype=np.float64)

    def P(i, j):
        th, ph = math.pi * i / nv, 2 * math.pi * j / nu
        n = np.array([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
        return c + r * n, n
    for i in range(nv):
        for j in range(nu):
            (a, na), (b, nb), (d, nd), (e, ne) = P(i, j), P(i + 1, j), P(i, j + 1), P(i + 1, j + 1)
            if i > 0:
                m.tri_smooth([a, d, b], [na, nd, nb])
            if i < nv - 1:
                m.tri_smooth([d, e, b], [nd, ne, nb])
    return m


def tube(points, radius, sides=6, closed=True):
    """3D の折れ線に沿った細い管（玉の縫い目、ボーナスの輪）"""
    P = np.asarray(points, dtype=np.float64)
    n = len(P)
    m = Mesh()
    rings = []
    for i in range(n):
        t = P[(i + 1) % n] - P[i - 1] if closed else P[min(i + 1, n - 1)] - P[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        a = np.cross(t, [0.0, 1.0, 0.0])
        if np.linalg.norm(a) < 1e-3:
            a = np.cross(t, [1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        rings.append([(P[i] + radius * (math.cos(k * 2 * math.pi / sides) * a + math.sin(k * 2 * math.pi / sides) * b),
                       math.cos(k * 2 * math.pi / sides) * a + math.sin(k * 2 * math.pi / sides) * b)
                      for k in range(sides)])
    for i in range(n if closed else n - 1):
        r0, r1 = rings[i], rings[(i + 1) % n]
        for k in range(sides):
            (p00, n00), (p01, n01) = r0[k], r0[(k + 1) % sides]
            (p10, n10), (p11, n11) = r1[k], r1[(k + 1) % sides]
            m.tri_smooth([p00, p10, p01], [n00, n10, n01])
            m.tri_smooth([p01, p10, p11], [n01, n10, n11])
    return m


# ============================================================
# 物（半径 1 の大きさ）
# ============================================================
def cardboard_box():
    h = 0.85                      # 半分の幅
    H = 2 * h
    parts = {}
    parts["main"] = cbox(-h, h, 0.0, H, -h, h, 0.06)
    # 上のふたの合わせ目（細い溝）は、テープで隠れる中央以外の両端だけ見える
    seam = Mesh()
    for sx in (-1, 1):
        seam.add(box(sx * 0.18 if sx > 0 else -h + 0.06, h - 0.06 if sx > 0 else -0.18, H - 0.001, H + 0.004,
                     -0.012, 0.012))
    parts["print"] = seam
    # テープ：上面の中央を x 方向に横切り、前後の面を少し下りる
    tape = Mesh()
    w = 0.17
    tape.add(box(-h - 0.012, h + 0.012, H, H + 0.012, -w, w))
    for sx in (-1, 1):
        x0, x1 = (h, h + 0.012) if sx > 0 else (-h - 0.012, -h)
        tape.add(box(x0, x1, H - 0.55, H + 0.012, -w, w))
    parts["tape"] = tape
    # 横の面の白いラベルと、印刷の印（上向きの矢印 2 本・注意の帯）
    label = Mesh()
    label.add(box(-0.45, 0.25, 0.55, 1.10, h, h + 0.010))
    parts["label"] = label
    marks = parts["print"]
    marks.add(box(-0.38, 0.18, 0.70, 0.74, h + 0.010, h + 0.014))      # ラベルの文字の代わりの線
    marks.add(box(-0.38, 0.02, 0.84, 0.88, h + 0.010, h + 0.014))
    marks.add(box(-0.38, 0.10, 0.96, 1.00, h + 0.010, h + 0.014))
    for k, z in enumerate((-0.30, 0.15)):                               # 反対の面の矢印（↑↑）
        marks.add(box(z - 0.035, z + 0.035, 0.45, 1.05, -h - 0.010, -h).transformed(np.eye(3)))
        marks.add(hull([(z - 0.16, 1.05, -h - 0.010), (z + 0.16, 1.05, -h - 0.010), (z, 1.30, -h - 0.010),
                        (z - 0.16, 1.05, -h), (z + 0.16, 1.05, -h), (z, 1.30, -h)]))
    return parts


def soda_can():
    H = 2.6
    body = lathe([(0.80, 0.00), (0.95, 0.10), (1.00, 0.28), (1.00, H - 0.36), (0.86, H - 0.12)], seg=32)
    band = lathe([(1.006, H * 0.62), (1.006, H * 0.36)], seg=32)                 # 白いラベルの帯
    metal = Mesh()
    metal.add(lathe([(0.0, 0.03), (0.80, 0.03), (0.80, 0.0)], seg=32))         # 底
    metal.add(lathe([(0.86, H - 0.12), (0.88, H - 0.04), (0.86, H), (0.80, H), (0.78, H - 0.08),
                     (0.0, H - 0.08)], seg=32))                                    # 上のふち（巻き締め）とくぼんだふた
    tab = Mesh()
    # プルタブ：ふたの上の平たい輪っか（凸包を 2 つ）と、飲み口の線
    tab.add(cbox(-0.05, 0.42, H - 0.075, H - 0.040, -0.17, 0.17, 0.03))
    tab.add(cbox(-0.30, -0.05, H - 0.075, H - 0.050, -0.10, 0.10, 0.02))
    mouth = box(0.28, 0.62, H - 0.079, H - 0.074, -0.15, 0.15)
    return {"main": body, "band": band, "metal": metal, "tab": tab, "print": mouth}


def baseball():
    """野球のボール：白い球に、赤い縫い目（2 列の縫い糸と、その間を V 字にわたる糸）"""
    body = uv_sphere(1.0, (0, 1.0, 0), nu=24, nv=16)
    # 縫い目の中心の曲線（球の上で 2 つの U 字がかみ合う形。野球・テニスのボールと同じ）
    n = 96
    a, b = 0.75, 0.25
    C = []
    for i in range(n):
        t = 2 * math.pi * i / n
        p = np.array([a * math.cos(t) + b * math.cos(3 * t), a * math.sin(t) - b * math.sin(3 * t),
                      2 * math.sqrt(a * b) * math.sin(2 * t)])
        C.append(p / np.linalg.norm(p))
    C = np.array(C)
    W, LIFT = 0.075, 1.006                     # 縫い糸の列の、中心の曲線からのずれ・球の表面から少し浮かせる
    left, right, seam = [], [], Mesh()
    for i in range(n):
        tg = C[(i + 1) % n] - C[i - 1]
        side = np.cross(C[i], tg)
        side /= np.linalg.norm(side)
        left.append(C[i] + side * W)
        right.append(C[i] - side * W)
    up = np.array([0, 1.0, 0])

    def on(p):
        return p / np.linalg.norm(p) * LIFT + up

    seam.add(tube([on(p) for p in left], 0.035, sides=5))
    seam.add(tube([on(p) for p in right], 0.035, sides=5))
    for i in range(0, n, 2):                   # V 字の糸（両側の列から、少し先の中心へ）
        j = (i + 1) % n
        for s in (left, right):
            seam.add(tube([on(s[i]), on(C[j])], 0.026, sides=4, closed=False))
    return {"main": body, "seam": seam}


def star(outer=1.0, inner=0.45, depth=0.34, rim=0.10, cy=1.3):
    """ふくらんだ 5 角の星（xy 平面に立てる。z が厚み）。面ごとに法線を分けて宝石のように光らせる"""
    m = Mesh()
    pts = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        r = outer if k % 2 == 0 else inner
        pts.append(np.array([r * math.cos(a), cy + r * math.sin(a)]))
    c = np.array([0.0, cy])
    for s in (1, -1):
        apex = np.array([0.0, cy, s * depth])
        for k in range(10):
            p0, p1 = pts[k], pts[(k + 1) % 10]
            a = np.array([p0[0], p0[1], s * rim])
            b = np.array([p1[0], p1[1], s * rim])
            if s > 0:
                m.tri(apex, a, b)
            else:
                m.tri(apex, b, a)
    for k in range(10):                    # 側面（厚み 2 × rim）
        p0, p1 = pts[k], pts[(k + 1) % 10]
        a0, a1 = np.array([p0[0], p0[1], rim]), np.array([p1[0], p1[1], rim])
        b0, b1 = np.array([p0[0], p0[1], -rim]), np.array([p1[0], p1[1], -rim])
        m.tri(a0, b0, a1)
        m.tri(a1, b0, b1)
    _ = c
    return m


def bonus_star():
    body = star()
    core = star(outer=0.42, inner=0.19, depth=0.40, rim=0.12)     # 中心の光る芯（少し前に出す）
    ring = tube([(1.55 * math.cos(2 * math.pi * i / 48), 1.3, 1.55 * math.sin(2 * math.pi * i / 48))
                 for i in range(48)], 0.035, sides=5)               # まわりを回る輪（傾けて回す）
    return {"main": body, "core": core, "ring": ring}


# ============================================================
# カゴ（ワールド座標の大きさ）
# ============================================================
def basket():
    R = config.GOAL_RADIUS * 0.92
    hgt = 0.045
    t = 0.0045                     # ふちの太さ
    frame, mesh, bottom, rim = Mesh(), Mesh(), Mesh(), Mesh()
    bottom.add(cbox(-R, R, 0.0, 0.006, -R, R, 0.002))
    # 四隅の柱
    for sx in (-1, 1):
        for sz in (-1, 1):
            frame.add(cbox(sx * R - t, sx * R + t, 0.0, hgt, sz * R - t, sz * R + t, 0.0012))
    # 上のふち（太い枠）と、下のふち
    for y0, y1, tt in ((hgt - 0.006, hgt, t), (0.004, 0.009, t * 0.8)):
        for sx in (-1, 1):
            frame.add(cbox(sx * R - tt, sx * R + tt, y0, y1, -R - tt, R + tt, 0.0012))
        for sz in (-1, 1):
            frame.add(cbox(-R - tt, R + tt, y0, y1, sz * R - tt, sz * R + tt, 0.0012))
    # 網目の壁：縦の棒と横の棒（すき間から中が見える）
    n_v = 7
    for side in range(4):
        for k in range(1, n_v + 1):
            u = -R + 2 * R * k / (n_v + 1)
            b = box(u - 0.0012, u + 0.0012, 0.009, hgt - 0.006, R - 0.0012, R + 0.0012)
            mesh.add(b.transformed(_rot_y(side * math.pi / 2)))
        for y in (0.020, 0.031):
            b = box(-R, R, y - 0.0011, y + 0.0011, R - 0.0013, R + 0.0013)
            mesh.add(b.transformed(_rot_y(side * math.pi / 2)))
    # 上のふちの光る線（ゴールの目印）
    for sx in (-1, 1):
        rim.add(box(sx * R - 0.0022, sx * R + 0.0022, hgt, hgt + 0.0018, -R - 0.0022, R + 0.0022))
    for sz in (-1, 1):
        rim.add(box(-R - 0.0022, R + 0.0022, hgt, hgt + 0.0018, sz * R - 0.0022, sz * R + 0.0022))
    return {"frame": frame, "mesh": mesh, "bottom": bottom, "rim": rim}


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


PARTS = {"box": cardboard_box, "can": soda_can, "ball": baseball, "bonus": bonus_star, "basket": basket}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    lines = ["# BCI Operator の物とカゴ（tools/make_item_models.py で生成。自作）",
             "# グループ名 = 物.役割。物は半径 1 の大きさ（ゲーム側で半径倍に拡大）、カゴはワールド座標"]
    vi = 1
    total = {}
    for part, fn in PARTS.items():
        for role, m in fn().items():
            lines.append(f"g {part}.{role}")
            lines.append(f"usemtl {role}")
            lines += ["v %.6f %.6f %.6f" % tuple(p) for p in m.v]
            lines += ["vn %.5f %.5f %.5f" % tuple(n) for n in m.n]
            lines += [f"f {a + vi}//{a + vi} {b + vi}//{b + vi} {c + vi}//{c + vi}" for a, b, c in m.f]
            vi += len(m.v)
            total[part] = total.get(part, 0) + len(m.f)
    with open(OUT, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"保存しました: {OUT}")
    print("物ごとの三角形:", total)


if __name__ == "__main__":
    main()
