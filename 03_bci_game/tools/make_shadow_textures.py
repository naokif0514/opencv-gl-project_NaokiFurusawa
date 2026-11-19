"""
tools/make_shadow_textures.py
-----------------------------
柔らかい影の画像（画質 standard の影に使う）を生成して assets/textures/shadow_soft.png に保存する（自作）。

  使い方（03_bci_game で）：../.venv/bin/python tools/make_shadow_textures.py

画像は 128×128 の白黒（白 = 影が濃い）。中心から半径 CORE までは真っ白（影の芯）、そこから外側の縁まで
なめらかに薄くなる。shadows.py が、この 1 枚を
  ・そのまま貼る（物の下の丸い影）
  ・9 つに分けて貼る（角が丸い長方形の影：ロボットの車体・カゴ、腕の細長い影）
のどちらかで使う。9 つに分けるとき、四隅には画像の 4 分の 1、辺には画像の中央の行・列、
真ん中には中心の値を使うので、どんな縦横比でも縁の柔らかさが変わらない。
"""

import os

import cv2
import numpy as np

SIZE = 128
CORE = 0.30          # 影の芯（濃さが一定の部分）の半径（画像の半分の大きさに対する割合）
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "textures",
                   "shadow_soft.png")


def main():
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    c = (SIZE - 1) / 2
    r = np.hypot(x - c, y - c) / c                     # 中心 0 〜 辺の中点 1
    t = np.clip((r - CORE) / (1.0 - CORE), 0.0, 1.0)
    a = 1.0 - t * t * (3 - 2 * t)                      # なめらかに 1 → 0（smoothstep）
    a = a ** 1.6                                       # 外側をより薄く（光の回り込みの感じ）
    a[r >= 1.0] = 0.0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, np.round(a * 255).astype(np.uint8))
    print(f"保存しました: {OUT}")


if __name__ == "__main__":
    main()
