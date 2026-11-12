"""
brain_mesh.py
-------------
中央に表示する「脳」のメッシュ（頂点・法線・インデックス）を用意する。

・assets/brain.obj があればそれを読み込む（簡易 OBJ ローダ）
・なければ、ノイズで変形させた 2 つの半球をプロシージャルに生成する
    1. 正二十面体を細分化して、継ぎ目のない球（アイコスフィア）を作る
    2. 楕円体に引き伸ばし、内側（左右の半球が向かい合う面）を平らにつぶす
    3. 乱数で作った波の重ね合わせ（簡易ノイズ）の「0 付近」を溝（脳溝）としてへこませる

頂点ごとに [位置(3), 法線(3), 溝の深さ(1)] の 7 個の float を持つ。
「溝の深さ」はシェーダーで溝を暗くしたり、溝に沿って光らせたりするのに使う。
"""

import os

import numpy as np


# ------------------------------------------------------------
# アイコスフィア
# ------------------------------------------------------------
def icosphere(subdivisions=5):
    """単位球上に頂点がほぼ均等に並んだ三角形メッシュを作る"""
    t = (1.0 + 5 ** 0.5) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]

    for _ in range(subdivisions):
        cache = {}

        def midpoint(a, b):
            # 同じ辺の中点を二重に作らないよう辞書で管理する
            key = (a, b) if a < b else (b, a)
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces

    return np.array(verts), np.array(faces, dtype=np.uint32)


def compute_normals(positions, faces):
    """各三角形の法線を、その三角形に含まれる頂点に足し合わせて頂点法線を求める"""
    normals = np.zeros_like(positions)
    p0, p1, p2 = positions[faces[:, 0]], positions[faces[:, 1]], positions[faces[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)        # 面積で重み付けされた面法線
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return normals


# ------------------------------------------------------------
# 簡易ノイズ（ランダムな向き・位相の正弦波の重ね合わせ）
# ------------------------------------------------------------
def wave_noise(p, rng, n_waves=16, freq=11.0):
    """3 次元の点群 p (N,3) に対して、平均 0・分散およそ 1 のなめらかな値を返す"""
    dirs = rng.normal(size=(n_waves, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    freqs = freq * rng.uniform(0.7, 1.3, size=n_waves)
    phases = rng.uniform(0, 2 * np.pi, size=n_waves)
    f = np.sin(p @ (dirs.T * freqs) + phases).sum(axis=1)
    return f / np.sqrt(n_waves / 2.0)


# ------------------------------------------------------------
# プロシージャルな脳
# ------------------------------------------------------------
def make_hemisphere(side, rng, subdivisions=5):
    """
    片側の半球を作る。side = -1 で左、+1 で右。
    戻り値: (位置, 三角形, 溝の深さ 0〜1)
    """
    unit, faces = icosphere(subdivisions)
    p = unit.copy()

    # 楕円体に変形（x: 幅, y: 高さ, z: 前後の長さ）
    radii = np.array([0.46, 0.66, 0.92])
    p *= radii

    # 内側（もう片方の半球側）を平らにつぶして、2 つの半球が向かい合うようにする
    inner = (p[:, 0] * side) < 0
    p[inner, 0] *= 0.30
    # 下側を少しつぶす
    low = p[:, 1] < 0
    p[low, 1] *= 0.75

    # 溝（脳溝）：ノイズの値が 0 に近いところをへこませる
    f = wave_noise(unit * np.array([1.0, 1.2, 0.9]), rng)
    groove = np.exp(-(f * 3.2) ** 2)            # 0 付近で 1、離れると 0
    depth = 0.06 * groove
    # 内側の平らな面にはあまり溝を作らない
    depth[inner] *= 0.4
    p *= (1.0 - depth)[:, None]

    # 中心を左右にずらす
    # （内側の面が x = ±0.14 付近にあるので、少しすき間を空けて ±0.16 に置く）
    p[:, 0] += side * 0.16
    return p, faces, groove


def make_procedural_brain(seed=7, subdivisions=5):
    """左右の半球を 1 つのメッシュにまとめて返す"""
    rng = np.random.default_rng(seed)
    pos_list, face_list, groove_list = [], [], []
    offset = 0
    for side in (-1, 1):
        p, f, g = make_hemisphere(side, rng, subdivisions)
        pos_list.append(p)
        face_list.append(f + offset)
        groove_list.append(g)
        offset += len(p)
    positions = np.vstack(pos_list)
    faces = np.vstack(face_list)
    groove = np.concatenate(groove_list)
    normals = compute_normals(positions, faces)
    return positions, normals, groove, faces


# ------------------------------------------------------------
# OBJ 読み込み
# ------------------------------------------------------------
def load_obj(path):
    """
    簡易 OBJ ローダ（v と f のみ使用）。多角形の面は扇形に三角形分割する。
    法線は読み込んだ形状から計算し直す。大きさは半径 1 程度に正規化する。
    """
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = []
                for token in parts[1:]:
                    i = int(token.split("/")[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)   # 負の番号は末尾から
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    positions = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.uint32)

    # 中心を原点に移し、最も遠い点が 1.0 になるよう拡大縮小する
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2.0
    positions -= center
    positions /= np.linalg.norm(positions, axis=1).max()

    normals = compute_normals(positions, faces)
    groove = np.zeros(len(positions))      # OBJ には溝の情報がないので 0
    return positions, normals, groove, faces


def load_brain(path, detail=5):
    """
    脳のメッシュを用意する。
    戻り値: (頂点配列 float32 (N,7), インデックス uint32, 説明文字列)
    """
    if path and os.path.exists(path):
        positions, normals, groove, faces = load_obj(path)
        source = f"OBJ: {path}"
    else:
        positions, normals, groove, faces = make_procedural_brain(subdivisions=detail)
        source = "procedural"
    vertex_data = np.hstack([positions, normals, groove[:, None]]).astype(np.float32)
    return vertex_data, faces.astype(np.uint32).ravel(), source
