"""
items.py
--------
現実の色の物（赤・青、P キーで登録した色）を、カメラの映像の中で追いかける。
ゲームでは、現実の物を一定時間カメラに見せると、3D の世界にボーナスアイテムが降ってくる（main.py / field.py）。

■ RealItemTracker
  vision.py が見つけた色の塊を、フレームをまたいで同じ物として追いかける。
    ・前のフレームの物のうち、同じ色で最も近いもの（MATCH_DIST 以内）を同じ物とみなす
    ・位置はなめらかにする（ITEM_SMOOTH_SEC）
    ・一瞬見えなくなっても ITEM_LOST_SEC 秒は残す（手で隠れたときなど）
    ・first_seen：最初に見えた時刻（BONUS_SHOW_SEC 秒続けて見えたらボーナスを降らせるのに使う）

座標：カメラ画像の正規化座標 (u, v)（0〜1、v は下向き）をそのまま使う。
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

import config

MATCH_DIST = 0.15
_ids = itertools.count(1)


@dataclass
class RealItem:
    id: int
    color: str                 # config.ITEM_COLORS の名前
    pos: np.ndarray            # カメラ画像の正規化座標
    radius: float
    first_seen: float = 0.0
    last_seen: float = 0.0
    used: bool = False         # この物でボーナスを降らせたか（同じ物を見せ続けても 1 回だけ）


class RealItemTracker:
    def __init__(self):
        self.items = []

    def update(self, blobs, now, dt):
        k = 1.0 - math.exp(-dt / config.ITEM_SMOOTH_SEC) if dt > 0 else 1.0
        unused = list(self.items)
        for b in blobs:
            cand = [it for it in unused if it.color == b.color]
            best = min(cand, key=lambda it: np.linalg.norm(it.pos - b.pos), default=None)
            if best is not None and np.linalg.norm(best.pos - b.pos) < MATCH_DIST:
                best.pos += (b.pos - best.pos) * k
                best.radius += (b.radius - best.radius) * k
                best.last_seen = now
                unused.remove(best)
            else:
                self.items.append(RealItem(next(_ids), b.color, np.array(b.pos, dtype=np.float64), b.radius,
                                           first_seen=now, last_seen=now))
        self.items = [it for it in self.items if now - it.last_seen <= config.ITEM_LOST_SEC]
        return self.items
