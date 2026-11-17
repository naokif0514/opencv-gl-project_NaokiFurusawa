"""
layout.py
---------
ゲーム画面の配置（論理座標 1280×720、左上原点）と座標の変換。main.py と game_view.py で共通に使う。
メインの画面は 3D の場面（画面いっぱい）。上に HUD の帯、右下に小さなカメラ映像を重ねる。
"""

import numpy as np

import config

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
PX, PY, PW, PH = config.SYNC_PAD


def pad_to_screen(p):
    """手の位置の座標（カメラ画像の 0〜1、y 上向き）→ 同期パッドの上の画素（同期フェーズの表示）"""
    return np.array([PX + p[0] * PW, PY + (1.0 - p[1]) * PH])


def check_points():
    """視線同期のあとの確認の点（学習に使わない 3 点）：中央の左右と、中央の少し上"""
    top = config.HUD_HEIGHT
    cy = (top + H) / 2
    return [(W * 0.30, cy + (H - top) * 0.18), (W * 0.70, cy + (H - top) * 0.18), (W / 2, top + (H - top) * 0.25)]


def calib_points():
    """視線同期の点：画面の四隅（少し内側）と中央の 5 点。GAZE_CALIB_POINTS = 9 なら辺の中点も加える"""
    m = config.GAZE_CALIB_INSET
    top = config.HUD_HEIGHT
    xs = (W * m, W * (1 - m))
    ys = (top + (H - top) * m, H - (H - top) * m)
    cx, cy = W / 2, (top + H) / 2
    pts = [(xs[0], ys[0]), (xs[1], ys[0]), (cx, cy), (xs[0], ys[1]), (xs[1], ys[1])]
    if config.GAZE_CALIB_POINTS >= 9:
        pts = [(xs[0], ys[0]), (cx, ys[0]), (xs[1], ys[0]), (xs[0], cy), (cx, cy), (xs[1], cy),
               (xs[0], ys[1]), (cx, ys[1]), (xs[1], ys[1])]
    return pts
