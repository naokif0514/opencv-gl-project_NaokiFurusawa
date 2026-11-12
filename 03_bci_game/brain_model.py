"""
brain_model.py
--------------
本物の脳の形（FreeSurfer の標準脳 fsaverage5 の大脳皮質の表面）と、部位のラベル（Destrieux アトラス）を読み込む。

■ データ
  ・fsaverage5：多くの人の MRI を平均して作った標準脳の皮質表面（片半球 10242 頂点・20480 三角形）。
    FreeSurfer の配布物で、nilearn に同梱されている
  ・Destrieux アトラス（aparc.a2009s）：皮質を 74 の回（G_）・溝（S_）に分けたラベル（Destrieux ら 2010）。
    nilearn が NITRC からダウンロードする
  ライセンスと出典は README.md を参照。

■ キャッシュ（オフラインで動かすため）
  初回だけ nilearn でデータを取得し（ダウンロードは assets/nilearn_data/ に保存）、
  必要な配列（頂点・三角形・溝の深さ・部位ラベル）を assets/brain_fsaverage5.npz にまとめて保存する。
  2 回目以降はこの npz だけを読むので、nilearn もネットワークも要らない。

■ 座標
  FreeSurfer の表面の座標（単位 mm、ほぼ MNI305 空間）。x：右、y：前、z：上。
  描画では OpenGL の座標（x：右、y：上、z：手前）に変換する（brain_renderer.py）。

■ 光らせる領域（REGIONS）
  motor     運動野（一次運動野 M1）     ：中心前回 G_precentral
  hand      運動野の手の領域（hand knob）：左半球の中心前回のうち、手の領域の中心から HAND_KNOB_RADIUS 以内
  somato    体性感覚野（S1）             ：中心後回 G_postcentral
  fef       前頭眼野（FEF）のあたり       ：中前頭回 G_front_middle の後ろ側（中心前溝に近い部分）
  visual    視覚野                       ：後頭極 Pole_occipital、鳥距溝 S_calcarine、楔部 G_cuneus、舌状回 G_oc-temp_med-Lingual
"""

import os

import numpy as np

import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")
CACHE_PATH = os.path.join(ASSET_DIR, "brain_fsaverage5.npz")
NILEARN_DIR = os.path.join(ASSET_DIR, "nilearn_data")

HEMIS = ("left", "right")
REGION_NAMES = ("motor", "hand", "somato", "fef", "visual")
REGION_ID = {name: i + 1 for i, name in enumerate(REGION_NAMES)}    # 0 はどこにも属さない

# 手の領域（hand knob）の中心。中心前回が後ろへ Ω 字型に張り出した部分で、MNI 座標ではおよそ (-38, -22, 58)
# （Yousry ら 1997。右手を動かすと左半球のここが活動する）
HAND_KNOB_CENTER = np.array([-38.0, -22.0, 58.0])
VISUAL_LABELS = ("Pole_occipital", "S_calcarine", "G_cuneus", "G_oc-temp_med-Lingual")


# ============================================================
# 読み込み（キャッシュがなければ作る）
# ============================================================
def _build_cache():
    """nilearn で fsaverage5 と Destrieux アトラスを取得し、npz にまとめる（初回だけ）"""
    print("[brain] 初回の準備：nilearn で fsaverage5 と Destrieux アトラスを取得します")
    from nilearn import datasets, surface    # 初回だけ使うので、ここで読み込む

    import warnings
    os.makedirs(NILEARN_DIR, exist_ok=True)
    fs = datasets.fetch_surf_fsaverage("fsaverage5", data_dir=NILEARN_DIR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # 「Unknown（番号 0）が地図にない」という無害な警告を出さない
        atlas = datasets.fetch_atlas_surf_destrieux(data_dir=NILEARN_DIR)
    labels = [str(lab) for lab in atlas["labels"]]

    out = {"label_names": np.array(labels)}
    for hemi in HEMIS:
        for surf, key in (("pial", "pial"), ("infl", "infl")):
            mesh = surface.load_surf_mesh(fs[f"{key}_{hemi}"])
            out[f"{surf}_{hemi}"] = np.asarray(mesh.coordinates, dtype=np.float32)
            out[f"faces_{hemi}"] = np.asarray(mesh.faces, dtype=np.int32)
        out[f"sulc_{hemi}"] = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), dtype=np.float32)
        out[f"labels_{hemi}"] = np.asarray(atlas[f"map_{hemi}"], dtype=np.int16)
    np.savez_compressed(CACHE_PATH, **out)
    print(f"[brain] キャッシュを保存しました: {CACHE_PATH}")


class BrainModel:
    """両半球の表面（pial / inflated）、溝の深さ、部位ラベル、光らせる領域"""

    def __init__(self):
        if not os.path.exists(CACHE_PATH):
            _build_cache()
        data = np.load(CACHE_PATH)
        self.label_names = [str(s) for s in data["label_names"]]
        self.hemis = {}
        for hemi in HEMIS:
            h = {
                "pial": data[f"pial_{hemi}"],
                "infl": data[f"infl_{hemi}"],
                "faces": data[f"faces_{hemi}"],
                "sulc": data[f"sulc_{hemi}"],
                "labels": data[f"labels_{hemi}"].astype(np.int32),
            }
            h["region"] = self._regions(hemi, h)
            self.hemis[hemi] = h

    # --------------------------------------------------------
    def label_mask(self, hemi, *names):
        idx = [self.label_names.index(n) for n in names]
        return np.isin(self.hemis[hemi]["labels"], idx)

    def _regions(self, hemi, h):
        """各頂点が、光らせる領域のどれに属するか（0 = なし）"""
        self.hemis[hemi] = h            # label_mask から参照するため先に登録
        region = np.zeros(len(h["labels"]), dtype=np.int32)
        pial = h["pial"]

        motor = self.label_mask(hemi, "G_precentral")
        region[motor] = REGION_ID["motor"]
        region[self.label_mask(hemi, "G_postcentral")] = REGION_ID["somato"]
        region[self.label_mask(hemi, *VISUAL_LABELS)] = REGION_ID["visual"]

        # 前頭眼野のあたり：中前頭回のうち、後ろ側（y が小さい側）の FEF_POSTERIOR_FRACTION の部分
        mfg = self.label_mask(hemi, "G_front_middle")
        y = pial[mfg, 1]
        cut = y.min() + config.FEF_POSTERIOR_FRACTION * (y.max() - y.min())
        fef = np.zeros_like(mfg)
        fef[np.flatnonzero(mfg)[y <= cut]] = True
        region[fef] = REGION_ID["fef"]

        # 手の領域（右手を使うので左半球だけ）
        if hemi == "left":
            d = np.linalg.norm(pial - HAND_KNOB_CENTER, axis=1)
            region[motor & (d <= config.HAND_KNOB_RADIUS)] = REGION_ID["hand"]
        return region

    def region_vertices(self, hemi, name):
        return np.flatnonzero(self.hemis[hemi]["region"] == REGION_ID[name])

    # --------------------------------------------------------
    def sample_hand_points(self, n, rng, surf="pial"):
        """
        左半球の手の領域の表面上に、n 個の点をランダムに置く（三角形の面積に比例した一様分布）。
        戻り値：位置 (n,3) と、その点での面の法線 (n,3)。擬似ニューロンの配置に使う
        """
        h = self.hemis["left"]
        v = h[surf]
        faces = h["faces"]
        hand = h["region"] == REGION_ID["hand"]
        tri = faces[hand[faces].all(axis=1)]              # 3 頂点とも手の領域にある三角形
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        k = rng.choice(len(tri), size=n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        pos = (1 - s)[:, None] * a[k] + (s * (1 - r2))[:, None] * b[k] + (s * r2)[:, None] * c[k]
        normal = cross[k] / np.linalg.norm(cross[k], axis=1, keepdims=True)
        return pos.astype(np.float32), normal.astype(np.float32)


def vertex_normals(coords, faces):
    """頂点の法線（周りの三角形の法線を面積で重みづけして平均）"""
    v = coords.astype(np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n.astype(np.float32)
