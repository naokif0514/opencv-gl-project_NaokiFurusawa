"""
tools/face_mesh_snapshot.py
---------------------------
発表スライド用：自分の顔に MediaPipe Face Mesh の点（468 点）と黒目（虹彩）の輪を重ねた写真を撮り、
「① 顔と黒目を検出」の図（800×600。docs/slides/gaze_1_detect.png と同じ形）を作る。

  使い方（03_bci_game で）：
    ../.venv/bin/python tools/face_mesh_snapshot.py                 # カメラ。プレビューの窓で SPACE = 撮る、ESC = やめる
    ../.venv/bin/python tools/face_mesh_snapshot.py --out 保存先.png # 保存先を変える（既定はホームの gaze_1_face.png）
    ../.venv/bin/python tools/face_mesh_snapshot.py --image 写真.jpg # カメラの代わりに写真から作る
    ../.venv/bin/python tools/face_mesh_snapshot.py --bg dark        # 背景：white（白。既定）/ dark（紺）/ keep（そのまま）
  背景は MediaPipe Selfie Segmentation で人とそれ以外に分けて塗る（髪・肩は人として残る）

  保存するもの：
    保存先.png         … スライド用の図（ラベルつき）
    保存先_photo.png   … 点を重ねた写真だけ（ラベルなし）
    保存先_raw.png     … 撮ったままの写真（--image に渡せば、撮り直さずに背景などを変えて作り直せる）
  既定の保存先はリポジトリの外（ホーム）。顔の写真をリポジトリに入れるかどうかは自分で決める。
"""

import argparse
import math
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config  # noqa: E402

FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
IRIS_CENTERS = (468, 473)          # 虹彩の中心（refine_landmarks=True のとき）
MESH = (140, 205, 240)             # 点・線の色（RGB）
YELLOW = (255, 205, 80)
INK = (29, 43, 69)
BG = (245, 247, 251)


def detect(face_mesh, frame_bgr):
    res = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None


BG_COLORS = {"white": (255, 255, 255), "dark": (22, 30, 50)}


def person_mask(seg, frame_bgr):
    """人の部分 = 1、背景 = 0 のなめらかなマスク（frame と同じ大きさ）"""
    m = seg.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5)
    return np.clip((m - 0.35) / 0.3, 0.0, 1.0)


def overlay(frame_bgr, lm, scale=2, mask=None, bg="keep"):
    """点・線・黒目の輪を描いた写真（RGB、scale 倍の大きさ）と、点の座標を返す。
    mask（人の部分のマスク）と bg（white / dark）を渡すと、背景をその色で塗る"""
    img = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    P = np.array([(p.x * w, p.y * h) for p in lm])
    dim = img * 0.55                                 # 写真を少し暗くして、点を見やすく
    if mask is not None and bg in BG_COLORS:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        dim = dim * m + np.array(BG_COLORS[bg], dtype=np.float32) * (1 - m)
    dim = dim.astype(np.uint8)
    layer = dim.copy()
    for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        cv2.line(layer, tuple(np.int32(P[a])), tuple(np.int32(P[b])), (90, 150, 190), 1, cv2.LINE_AA)
    img = cv2.addWeighted(layer, 0.55, dim, 0.45, 0)
    for p in P[:468]:
        cv2.circle(img, tuple(np.int32(p)), max(2, scale), MESH, -1, cv2.LINE_AA)
    for a, b in mp.solutions.face_mesh.FACEMESH_IRISES:
        cv2.line(img, tuple(np.int32(P[a])), tuple(np.int32(P[b])), YELLOW, 2 * scale, cv2.LINE_AA)
    for i in IRIS_CENTERS:
        cv2.circle(img, tuple(np.int32(P[i])), 3 * scale, YELLOW, -1, cv2.LINE_AA)
    return img, P


def figure(photo_rgb, P, out_path, border=False):
    """スライド用の図：点を重ねた写真（顔のまわりを切り出す）＋ ラベル"""
    W, H, SS = 800, 600, 2
    tile = (30, 24, 770, 500)                      # 写真を置く枠
    tw, th = tile[2] - tile[0], tile[3] - tile[1]
    # 顔のまわり（点の範囲 + 余白）を、枠と同じ縦横比で切り出す。顔は枠の左寄りに置き、右にラベルの場所を空ける
    x0, y0 = P[:468].min(0)
    x1, y1 = P[:468].max(0)
    fh = (y1 - y0) * 1.25
    ch = fh
    cw = ch * tw / th
    cx = (x0 + x1) / 2 + cw * 0.18
    cy = (y0 + y1) / 2
    ih, iw = photo_rgb.shape[:2]
    left, top = cx - cw / 2, cy - ch / 2
    corner = tuple(int(v) for v in photo_rgb[2, 2])               # はみ出す分は、写真の角の色（= 背景の色）で埋める
    pad = Image.new("RGB", (iw * 3, ih * 3), corner)
    pad.paste(Image.fromarray(photo_rgb), (iw, ih))
    crop = pad.crop((int(left + iw), int(top + ih), int(left + iw + cw), int(top + ih + ch)))
    crop = crop.resize((tw * SS, th * SS), Image.LANCZOS)
    k = tw / cw

    def to_fig(p):
        return (tile[0] + (p[0] - left) * k, tile[1] + (p[1] - top) * k)

    im = Image.new("RGB", (W * SS, H * SS), BG)
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, crop.size[0] - 1, crop.size[1] - 1), 24 * SS, fill=255)
    im.paste(crop, (tile[0] * SS, tile[1] * SS), mask)
    d = ImageDraw.Draw(im)
    if border:                                     # 白い背景のときは、枠の線で写真の範囲を示す
        d.rounded_rectangle(tuple(v * SS for v in tile), 24 * SS, outline=(214, 222, 234), width=2 * SS)

    def f(size):
        return ImageFont.truetype(FONT_B, size * SS, index=0)

    def box(b, color, text):
        d.rounded_rectangle(tuple(v * SS for v in b), 14 * SS, fill=color)
        d.multiline_text(((b[0] + b[2]) / 2 * SS, (b[1] + b[3]) / 2 * SS), text, font=f(24), fill=INK, anchor="mm",
                         align="center", spacing=5 * SS)

    # 画面の右側に写る目（ラベルに近い方）の黒目から線を引く
    eye = max((to_fig(P[i]) for i in IRIS_CENTERS), key=lambda q: q[0])
    edge = to_fig(P[454])                                 # 顔の右の輪郭あたりの点
    d.line([(eye[0] + 14) * SS, eye[1] * SS, 590 * SS, 154 * SS], fill=YELLOW, width=3 * SS)
    box((560, 112, 760, 196), YELLOW, "黒目（虹彩）\nの輪")
    d.line([(edge[0] + 6) * SS, edge[1] * SS, 590 * SS, 332 * SS], fill=MESH, width=3 * SS)
    box((560, 290, 760, 374), MESH, "顔の点\n468 個")
    d.text((400 * SS, 528 * SS), "カメラ映像から、顔の点と黒目の輪を見つける", font=f(26), fill=INK, anchor="ma")
    im.resize((W, H), Image.LANCZOS).save(out_path)


def main():
    ap = argparse.ArgumentParser(description="Face Mesh を重ねた顔の写真と、スライド用の図を作る")
    ap.add_argument("--out", default=os.path.expanduser("~/gaze_1_face.png"))
    ap.add_argument("--image", default=None, help="カメラの代わりに使う写真")
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--bg", choices=("white", "dark", "keep"), default="white", help="背景の塗り方")
    a = ap.parse_args()
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=a.image is not None, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.5)
    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            sys.exit(f"写真を読めません: {a.image}")
        lm = detect(fm, frame)
        if lm is None:
            sys.exit("顔が見つかりませんでした")
    else:
        cap = cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)           # 写真用なので、できれば大きく撮る
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        lm = frame = None
        print("プレビューの窓で SPACE = 撮る、ESC = やめる（顔をカメラの正面に）")
        while True:
            ok, raw = cap.read()
            if not ok:
                continue
            raw = cv2.flip(raw, 1)                         # ゲームと同じく鏡のように
            cur = detect(fm, raw)
            view = raw
            if cur is not None:
                v, _ = overlay(raw, cur, scale=1)
                view = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
            cv2.imshow("face mesh  (SPACE = save, ESC = quit)", view)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit("やめました")
            if key == 32 and cur is not None:
                frame, lm = raw, cur
                break
        cap.release()
        cv2.destroyAllWindows()
    photo, P = overlay(frame, lm, scale=2, mask=person_mask(seg, frame), bg=a.bg)
    root, ext = os.path.splitext(a.out)
    if not a.image:
        cv2.imwrite(f"{root}_raw{ext or '.png'}", frame)
    Image.fromarray(photo).save(f"{root}_photo{ext or '.png'}")
    figure(photo, P, a.out, border=a.bg == "white")
    print(f"保存しました: {a.out}（図）、{root}_photo{ext or '.png'}（写真だけ）"
          + ("" if a.image else f"、{root}_raw{ext or '.png'}（撮ったまま）"))


if __name__ == "__main__":
    main()
