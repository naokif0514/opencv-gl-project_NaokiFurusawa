"""
robot_arm.py
------------
（コピー元：02_pseudo_bci/robot_arm.py。03 では台座が動くので、肩の位置 base を毎フレーム書きかえて使う）
2 関節（肩・肘）のロボットアーム（平面上の 2 リンク）の逆運動学。
デコードしたカーソルの位置を「手先の目標」として、肩と肘の角度を求める。
実際の BCI でも、デコードした速度・位置をロボットアームの手先の目標にして動かす
（Velliste ら 2008 のサルの給餌、Hochberg ら 2012 のヒトのロボットアーム操作など）。

■ 順運動学（関節の角度 → 手先の位置）
      肘   e = base + L1 (cos θ1, sin θ1)
      手先 p = e    + L2 (cos(θ1+θ2), sin(θ1+θ2))
  θ1：肩の角度（x 軸から）、θ2：肘の角度（上腕に対する前腕の角度。0 で伸びきった状態）

■ 逆運動学（手先の目標 → 関節の角度）
  肩から目標までの距離を r とすると、余弦定理より
      cos θ2 = (r² − L1² − L2²) / (2 L1 L2)
      θ2 = ± arccos(cos θ2)             （符号で肘の曲がる向きが決まる。ARM_ELBOW で選ぶ）
      θ1 = atan2(dy, dx) − atan2(L2 sin θ2, L1 + L2 cos θ2)
  ・r > L1 + L2（遠すぎて届かない）：r = L1 + L2 として解く → θ2 = 0 で伸びきり、目標の方向を向く
  ・r < |L1 − L2|（近すぎて届かない）：r = |L1 − L2| として解く → 折りたたんだ状態で目標の方向を向く
"""

import math
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class ArmPose:
    theta1: float          # 肩の角度 [rad]
    theta2: float          # 肘の角度 [rad]
    base: np.ndarray       # 肩の位置
    elbow: np.ndarray      # 肘の位置
    hand: np.ndarray       # 手先の位置
    reachable: bool        # 目標に届いたか


class TwoLinkArm:
    def __init__(self, base=config.ARM_BASE, l1=config.ARM_L1, l2=config.ARM_L2, elbow=config.ARM_ELBOW):
        self.base = np.asarray(base, dtype=np.float64)
        self.l1, self.l2 = float(l1), float(l2)
        self.elbow_sign = 1.0 if elbow >= 0 else -1.0
        self.r_min = abs(self.l1 - self.l2)
        self.r_max = self.l1 + self.l2

    def forward(self, theta1, theta2):
        """順運動学：関節の角度から、肘と手先の位置を求める"""
        elbow = self.base + self.l1 * np.array([math.cos(theta1), math.sin(theta1)])
        hand = elbow + self.l2 * np.array([math.cos(theta1 + theta2), math.sin(theta1 + theta2)])
        return elbow, hand

    def solve(self, target):
        """逆運動学：手先の目標から関節の角度を求める（届かないときは目標の方向を向く）"""
        d = np.asarray(target, dtype=np.float64) - self.base
        r = float(np.hypot(*d))
        reachable = self.r_min <= r <= self.r_max
        direction = math.atan2(d[1], d[0]) if r > 1e-9 else math.pi / 2
        r_c = min(max(r, self.r_min), self.r_max)            # 届く範囲に収める

        c2 = (r_c ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2)
        theta2 = self.elbow_sign * math.acos(min(max(c2, -1.0), 1.0))
        theta1 = direction - math.atan2(self.l2 * math.sin(theta2), self.l1 + self.l2 * math.cos(theta2))
        elbow, hand = self.forward(theta1, theta2)
        return ArmPose(theta1, theta2, self.base.copy(), elbow, hand, reachable)
