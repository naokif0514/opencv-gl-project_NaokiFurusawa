"""
field.py
--------
フィールド（机の上）の物の管理：最初の配置、補充、ボーナスアイテム、落ちてくる動き。

  ・物の種類（config.ITEM_TYPES）：箱・缶・玉は形・色・大きさ・基本点が違う（小さい物ほど高得点）。
    ボーナス（星形）は、現実の色の物をカメラに見せたときだけ空から降ってくる
  ・最初に FIELD_START_ITEMS 個を、互いに・ロボット・カゴから離して置く
  ・1 個運ぶと、REFILL_DELAY 秒後に 1 個補充する（上から落ちてくる）
  ・カゴの外に落とした物は消さずに床に残り、拾い直せる

座標はワールド座標（x 右、y 奥、机の大きさ WORLD_W × WORLD_H）。高さ h は 3D の表示と落下の演出だけに使う。
"""

import itertools
from dataclasses import dataclass, field

import numpy as np

import config

_ids = itertools.count(1)
GRAVITY = 2.5          # 落ちてくる物の加速度（見た目用）[ワールド/秒²]


@dataclass
class FieldItem:
    id: int
    type: str                  # config.ITEM_TYPES の名前
    pos: np.ndarray            # ワールド座標（床の上の位置）
    height: float = 0.0        # 床からの高さ（落ちてくる途中なら > 0）
    vz: float = 0.0
    angle: float = 0.0         # 向き（見た目用）
    held: bool = False         # ロボットがつかんでいるか
    pickup_pos: np.ndarray = field(default=None)   # つかんだ位置（カゴからの距離で点数の倍率を決める）

    @property
    def spec(self):
        return config.ITEM_TYPES[self.type]

    @property
    def radius(self):
        return self.spec["radius"]

    @property
    def rgb(self):
        return self.spec["rgb"]

    @property
    def landed(self):
        return self.height <= 0.0 and not self.held


class Field:
    def __init__(self, rng):
        self.rng = rng
        self.items = []
        self.refill_timers = []         # 補充までの残り時間
        self.extra_avoid = []           # ほかに避ける場所（チャレンジ：ライバルのカゴ）

    def reset(self, robot_pos, more_avoid=()):
        self.items = []
        self.refill_timers = []
        for _ in range(config.FIELD_START_ITEMS):
            self.spawn(self._random_type(), avoid=[robot_pos, *more_avoid], drop=False)

    def _random_type(self):
        names = [k for k, v in config.ITEM_TYPES.items() if v["weight"] > 0]
        w = np.array([config.ITEM_TYPES[k]["weight"] for k in names])
        return names[self.rng.choice(len(names), p=w / w.sum())]

    def _free_position(self, avoid=()):
        """他の物・ロボット・カゴから離れた、机の上の位置"""
        m = 0.06
        others = ([it.pos for it in self.items] + [np.asarray(a) for a in avoid] + [np.asarray(config.GOAL_POS)]
                  + [np.asarray(a) for a in self.extra_avoid])
        for _ in range(300):
            p = np.array([self.rng.uniform(m, config.WORLD_W - m), self.rng.uniform(m + 0.04, config.WORLD_H - m)])
            if all(np.linalg.norm(p - q) > config.ITEM_MIN_GAP for q in others):
                return p
        return p

    def spawn(self, kind, avoid=(), drop=True, pos=None):
        """物を 1 個置く。drop なら上から落ちてくる"""
        p = self._free_position(avoid) if pos is None else np.asarray(pos, dtype=np.float64)
        it = FieldItem(next(_ids), kind, p, height=0.6 if drop else 0.0, angle=self.rng.uniform(0, np.pi / 2))
        self.items.append(it)
        return it

    def remove(self, item):
        if item in self.items:
            self.items.remove(item)

    def schedule_refill(self):
        self.refill_timers.append(config.REFILL_DELAY)

    def update(self, dt, avoid=()):
        """落ちてくる物を動かし、補充の時間が来たら物を足す"""
        for it in self.items:
            if it.height > 0 and not it.held:
                it.vz -= GRAVITY * dt
                it.height = max(0.0, it.height + it.vz * dt)
                if it.height == 0.0:
                    it.vz = 0.0
        self.refill_timers = [t - dt for t in self.refill_timers]
        while self.refill_timers and self.refill_timers[0] <= 0:
            self.refill_timers.pop(0)
            normal = [it for it in self.items if it.type != "bonus"]
            if len(normal) < config.FIELD_MAX_ITEMS:
                self.spawn(self._random_type(), avoid=avoid)

    def selectable(self):
        """視線で選べる物（床に着いていて、つかまれていないもの）"""
        return [it for it in self.items if it.landed]
