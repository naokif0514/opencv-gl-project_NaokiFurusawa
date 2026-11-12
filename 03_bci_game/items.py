"""
items.py
--------
ゲーム内のアイテム（つかんで運ぶ物）とゴール。

■ 現実の物をアイテムにする（RealItemTracker）
  vision.py が見つけた色の塊（赤・青など）を、フレームをまたいで同じ物として追いかける。
    ・前のフレームのアイテムのうち、同じ色で最も近いもの（MATCH_DIST 以内）を同じ物とみなす
    ・位置はなめらかにする（ITEM_SMOOTH_SEC）。現実の物を動かすと、ゲーム内のアイテムも追従する
    ・一瞬見えなくなっても ITEM_LOST_SEC 秒は残す（手で隠れたときなど）
  画面には、カメラに写っている現実の物の上に、色の枠と名前を重ねて表示する（AR）。

■ 仮想アイテム（VirtualItems）
  色の物が手元にないとき用。AR 画面の上に 3D の立方体として表示する（V キーで切り替え）。

座標：AR 画面の幅を 1 とした「ワールド座標」（x 右、y 上、高さ WORLD_H）。
カメラ画像の正規化座標 (u, v)（v は下向き）からは、x = u × WORLD_W、y = (1 − v) × WORLD_H。
"""

import itertools
import math
from dataclasses import dataclass

import numpy as np

import config

MATCH_DIST = 0.15
_ids = itertools.count(1)


@dataclass
class Item:
    id: int
    kind: str                  # "real"（カメラに写った色の物）/ "virtual"（仮想アイテム）
    color: str                 # config.ITEM_COLORS の名前
    pos: np.ndarray            # ワールド座標
    radius: float              # ワールド座標での半径
    last_seen: float = 0.0
    held: bool = False         # ロボットがつかんでいるか（段階 3）
    angle: float = 0.0         # 仮想アイテムの向き（見た目用）

    @property
    def rgb(self):
        return config.ITEM_COLORS[self.color]["rgb"]


@dataclass
class Goal:
    pos: np.ndarray
    radius: float


def cam_to_world(uv):
    return np.array([uv[0] * config.WORLD_W, (1.0 - uv[1]) * config.WORLD_H])


class RealItemTracker:
    def __init__(self):
        self.items = []

    def update(self, blobs, now, dt):
        k = 1.0 - math.exp(-dt / config.ITEM_SMOOTH_SEC) if dt > 0 else 1.0
        unused = list(self.items)
        for b in blobs:
            p = cam_to_world(b.pos)
            r = b.radius * config.WORLD_W
            cand = [it for it in unused if it.color == b.color]
            best = min(cand, key=lambda it: np.linalg.norm(it.pos - p), default=None)
            if best is not None and np.linalg.norm(best.pos - p) < MATCH_DIST:
                best.pos += (p - best.pos) * k
                best.radius += (r - best.radius) * k
                best.last_seen = now
                unused.remove(best)
            else:
                self.items.append(Item(next(_ids), "real", b.color, p, r, last_seen=now))
        self.items = [it for it in self.items if now - it.last_seen <= config.ITEM_LOST_SEC]
        return self.items


class VirtualItems:
    def __init__(self, rng):
        self.rng = rng
        self.items = []

    def spawn(self, n=config.VIRTUAL_ITEMS, avoid=()):
        """AR 画面の上半分あたりに、互いに・avoid の点から離して n 個置く"""
        self.items = []
        colors = list(config.ITEM_COLORS)
        for i in range(n):
            for _ in range(200):
                p = np.array([self.rng.uniform(0.12, 0.88), self.rng.uniform(0.32, 0.65)])
                others = [it.pos for it in self.items] + [np.asarray(a) for a in avoid]
                if all(np.linalg.norm(p - q) > 0.18 for q in others):
                    break
            self.items.append(Item(next(_ids), "virtual", colors[i % len(colors)], p, 0.035,
                                   angle=self.rng.uniform(0, math.pi / 2)))
        return self.items
