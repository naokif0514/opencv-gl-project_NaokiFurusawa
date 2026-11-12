"""
robot.py
--------
動く台座 + 2 関節アーム（02 の robot_arm.py）のロボット。上から見下ろした平面（ワールド座標）で動く。

■ 台座の移動（視線で選んだ場所・物の近くへ）
  approach(target) で、対象の手前 ROBOT_STANDOFF の位置（アームが届く距離）を目的地にする。
  台座は、目的地までの距離に応じて減速しながら（到着でちょうど止まるように v = sqrt(2 a d) を上限に）、
  加速度・最高速度の上限つきでなめらかに移動し、進む向きへ少しずつ旋回する。到着したら対象の方を向く。

■ アーム
  肩は台座の前寄りにあり、台座といっしょに動く。手先の目標を arm_target に入れると、
  逆運動学（robot_arm.TwoLinkArm）で肘・肩の角度を求める。段階 2 では待機の姿勢（前に軽く伸ばす）。
"""

import math

import numpy as np

import config
from robot_arm import TwoLinkArm


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self):
        self.arm = TwoLinkArm(base=config.ROBOT_START, l1=config.ARM_L1, l2=config.ARM_L2, elbow=config.ARM_ELBOW)
        self.reset()

    def reset(self):
        self.pos = np.array(config.ROBOT_START, dtype=np.float64)
        self.heading = math.pi / 2                 # 上（画面の奥）を向く
        self.speed = 0.0
        self.goal = None                           # 目的地（ワールド座標）
        self.face_point = None                     # 到着後に向く点
        self.arm_target = None                     # アームの手先の目標（None なら待機の姿勢）
        self.grip = 1.0                            # グリッパーの開き 0（閉）〜 1（開）（段階 3）
        self.pose = None
        self._update_arm()

    @property
    def forward(self):
        return np.array([math.cos(self.heading), math.sin(self.heading)])

    @property
    def shoulder(self):
        return self.pos + self.forward * config.ROBOT_BODY[0] * 0.30

    @property
    def moving(self):
        return self.goal is not None

    def approach(self, target):
        """対象の手前（アームが届く距離）まで移動する"""
        target = np.asarray(target, dtype=np.float64)
        d = target - self.pos
        dist = np.linalg.norm(d)
        if dist > 1e-6:
            goal = target - d / dist * config.ROBOT_STANDOFF if dist > config.ROBOT_STANDOFF else self.pos.copy()
        else:
            goal = self.pos.copy()
        m = 0.06                                     # 画面の端からはみ出さないように
        self.goal = np.clip(goal, [m, m], [config.WORLD_W - m, config.WORLD_H - m])
        self.face_point = target

    def stop(self):
        self.goal = None
        self.speed = 0.0

    def update(self, dt):
        if self.goal is not None:
            d = self.goal - self.pos
            dist = np.linalg.norm(d)
            if dist < 0.004 and self.speed < 0.05:
                self.goal = None
                self.speed = 0.0
            else:
                # 到着でちょうど止まれる速さ sqrt(2 a d) と最高速度のうち小さい方へ、加速度の上限つきで近づける
                want = min(config.ROBOT_MAX_SPEED, math.sqrt(2 * config.ROBOT_ACCEL * dist))
                dv = np.clip(want - self.speed, -config.ROBOT_ACCEL * dt * 2, config.ROBOT_ACCEL * dt)
                self.speed = max(0.0, self.speed + dv)
                step = min(dist, self.speed * dt)
                self.pos += d / max(dist, 1e-9) * step
                if dist > 0.02:
                    self._turn_toward(math.atan2(d[1], d[0]), dt)
        if self.goal is None and self.face_point is not None:     # 到着したら対象の方を向く
            d = self.face_point - self.pos
            if np.linalg.norm(d) > 1e-6:
                self._turn_toward(math.atan2(d[1], d[0]), dt)
        self._update_arm()

    def _turn_toward(self, angle, dt):
        diff = _wrap(angle - self.heading)
        self.heading += float(np.clip(diff, -config.ROBOT_TURN_RATE * dt, config.ROBOT_TURN_RATE * dt))

    def _update_arm(self):
        self.arm.base = self.shoulder.copy()
        target = self.arm_target
        if target is None:     # 待機の姿勢：前方、届く距離の 55% のところ
            target = self.shoulder + self.forward * (config.ARM_L1 + config.ARM_L2) * 0.55
        self.pose = self.arm.solve(target)
