"""
layout.py
---------
ゲーム画面の配置と、座標の変換（ワールド座標・手の座標 ⇔ 画面の画素）。main.py と game_view.py で共通に使う。
"""

import numpy as np

import config

W, H = config.WINDOW_WIDTH, config.WINDOW_HEIGHT
AX, AY, AW, AH = config.AR_RECT
BX, BY, BW, BH = config.BRAIN_RECT


def world_to_screen(p):
    """ワールド座標（AR 画面の幅を 1、y 上向き）→ 画面の画素"""
    return np.array([AX + p[0] / config.WORLD_W * AW, AY + (1.0 - p[1] / config.WORLD_H) * AH])


def screen_to_world(q):
    return np.array([(q[0] - AX) / AW * config.WORLD_W, (1.0 - (q[1] - AY) / AH) * config.WORLD_H])


def hand_to_screen(p):
    """手の位置の座標（カメラ画像の 0〜1、y 上向き）→ 画面の画素。AR はカメラ映像なので、映った手と重なる"""
    return np.array([AX + p[0] * AW, AY + (1.0 - p[1]) * AH])


def world_radius(r):
    return r / config.WORLD_W * AW


def calib_points():
    """視線同期の点：AR 画面の四隅（少し内側）と中央の 5 点。GAZE_CALIB_POINTS = 9 なら辺の中点も加える"""
    m = config.GAZE_CALIB_INSET
    xs, ys = (AX + AW * m, AX + AW * (1 - m)), (AY + AH * m, AY + AH * (1 - m))
    cx, cy = AX + AW / 2, AY + AH / 2
    pts = [(xs[0], ys[0]), (xs[1], ys[0]), (cx, cy), (xs[0], ys[1]), (xs[1], ys[1])]
    if config.GAZE_CALIB_POINTS >= 9:
        pts = [(xs[0], ys[0]), (cx, ys[0]), (xs[1], ys[0]), (xs[0], cy), (cx, cy), (xs[1], cy),
               (xs[0], ys[1]), (cx, ys[1]), (xs[1], ys[1])]
    return pts
