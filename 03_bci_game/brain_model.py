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

■ フォールバック（簡易モデル）
  キャッシュがなく、nilearn での取得もできない（ネットワークがない、nilearn が入っていない）ときは、
  左右の半球を楕円体で近似した簡易モデルで起動する（simplified = True）。
  領域は、MNI 座標でのおおよその位置（中心溝の位置、後頭葉の範囲など）を式で決める。
  画面には「標準脳データ未取得：簡易モデルで表示中」と警告を出す（WARNING_TEXT）。

■ pial と inflated
  pial（本物の脳のしわ）と inflated（しわを伸ばしてふくらませた表面）は、同じ頂点番号が皮質の同じ場所に
  対応している。そのため、頂点ごとに 2 つの位置をなめらかに補間すると、脳がふくらむ変形アニメーションになる。
  擬似ニューロンも「どの三角形の、どの重心座標の位置にあるか」で置くので、両方の表面での位置が求まる。

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
WARNING_TEXT = "標準脳データ未取得：簡易モデルで表示中"
WARNING_TEXT_EN = "Standard brain data not available: showing a simplified model"


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


def _simplified_hemi(hemi):
    """
    簡易モデルの半球：楕円体（外側は丸く、内側面は平ら、下面は少し平ら）。
    MNI 座標でのおおよその大きさ（左半球で x -68〜-2、y -105〜71、z -47〜77）に合わせる。
    しわの代わりに、表面を少し波打たせ、その凹凸を「溝の深さ」として濃淡に使う。
    """
    from gl_utils import icosphere
    u, faces = icosphere(config.FALLBACK_SUBDIV)
    u = u.astype(np.float64)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    # 左半球として作る：外側（x が負の側）は半径 37、内側（x が正の側）は平らに
    x = -32.0 + np.where(ux < 0, 37.0, 29.0) * ux
    y = -17.0 + 88.0 * uy
    z = 15.0 + np.where(uz < 0, 50.0, 62.0) * uz
    base = np.stack([x, y, z], axis=1)
    bump = np.sin(0.23 * y + 0.9 * np.sin(0.11 * z)) * np.sin(0.19 * z + 0.6 * np.sin(0.13 * y))
    center = np.array([-32.0, -17.0, 15.0])
    radial = (base - center) / np.linalg.norm(base - center, axis=1, keepdims=True)
    pial = base + radial * (2.2 * bump)[:, None]
    infl = center + (base - center) * np.array([1.05, 1.07, 1.06])    # しわのない、少しふくらんだ形
    if hemi == "right":
        pial[:, 0] *= -1
        infl[:, 0] *= -1
        faces = faces.reshape(-1, 3)[:, [0, 2, 1]]                        # 鏡に映すと裏表が逆になるので向きを戻す
    return {"pial": pial.astype(np.float32), "infl": infl.astype(np.float32),
            "faces": faces.reshape(-1, 3).astype(np.int32), "sulc": (-bump).astype(np.float32),
            "labels": np.zeros(len(u), dtype=np.int32)}


class BrainModel:
    """両半球の表面（pial / inflated）、溝の深さ、部位ラベル、光らせる領域"""

    def __init__(self):
        self.simplified = False
        if not os.path.exists(CACHE_PATH):
            try:
                _build_cache()
            except Exception as e:      # ネットワークがない・nilearn がない など
                print(f"[brain] 標準脳データを取得できませんでした（{type(e).__name__}: {e}）。簡易モデルで起動します")
                self._init_simplified()
                return
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

    def _init_simplified(self):
        self.simplified = True
        self.label_names = ["Unknown"]
        self.hemis = {}
        for hemi in HEMIS:
            h = _simplified_hemi(hemi)
            h["region"] = self._regions_simplified(hemi, h)
            self.hemis[hemi] = h

    @staticmethod
    def _regions_simplified(hemi, h):
        """
        簡易モデルの領域：MNI 座標でのおおよその位置を式で決める。
          中心溝の y 座標を、高さ z とともに後ろへ傾く直線 y_cs(z) = −8 − 0.45 (z − 20) で近似し、
          その前 14 mm を中心前回（運動野）、後ろ 14 mm を中心後回（体性感覚野）、
          さらに前の 14〜30 mm・高さ 35〜65 mm を前頭眼野のあたり、y < −82 mm を後頭葉（視覚野）とする
        """
        p = h["pial"].astype(np.float64)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        lateral = np.abs(x) > 8.0
        d = y - (-8.0 - 0.45 * (z - 20.0))            # 中心溝からの前後の距離 [mm]
        region = np.zeros(len(p), dtype=np.int32)
        motor = lateral & (z > 5) & (d > 0) & (d <= 14)
        region[motor] = REGION_ID["motor"]
        region[lateral & (z > 5) & (d <= 0) & (d > -14)] = REGION_ID["somato"]
        region[lateral & (d > 14) & (d <= 30) & (z > 35) & (z < 65)] = REGION_ID["fef"]
        region[y < -82] = REGION_ID["visual"]
        if hemi == "left":
            # 手の領域：手の領域の中心に最も近い表面の点から HAND_KNOB_RADIUS 以内の運動野
            c = p[np.argmin(np.linalg.norm(p - HAND_KNOB_CENTER, axis=1))]
            region[motor & (np.linalg.norm(p - c, axis=1) <= config.HAND_KNOB_RADIUS)] = REGION_ID["hand"]
        return region

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
    def sample_hand_points(self, n, rng):
        """
        左半球の手の領域の表面上に、n 個の点をランダムに置く（pial での三角形の面積に比例した一様分布）。
        点は「三角形の番号 + 重心座標」で決めるので、pial と inflated の両方での位置が求まる。
        戻り値：{"pial": (位置 (n,3), 法線 (n,3)), "infl": (位置, 法線)}。擬似ニューロンの配置に使う
        """
        h = self.hemis["left"]
        faces = h["faces"]
        hand = h["region"] == REGION_ID["hand"]
        tri = faces[hand[faces].all(axis=1)]              # 3 頂点とも手の領域にある三角形
        if len(tri) == 0:                                 # 念のため（領域が小さすぎるとき）
            tri = faces[hand[faces].any(axis=1)]
        pa, pb, pc = (h["pial"][tri[:, i]].astype(np.float64) for i in range(3))
        area = 0.5 * np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
        k = rng.choice(len(tri), size=n, p=area / area.sum())
        r1, r2 = rng.random(n), rng.random(n)
        s = np.sqrt(r1)
        w = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)          # 重心座標
        out = {}
        for surf in ("pial", "infl"):
            a, b, c = (h[surf][tri[k, i]].astype(np.float64) for i in range(3))
            pos = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
            cross = np.cross(b - a, c - a)
            normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
            out[surf] = (pos.astype(np.float32), normal.astype(np.float32))
        return out


def vertex_normals(coords, faces):
    """頂点の法線（周りの三角形の法線を面積で重みづけして平均）"""
    v = coords.astype(np.float64)
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n.astype(np.float32)
