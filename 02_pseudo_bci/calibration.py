"""
calibration.py
--------------
線形デコーダを学習するための「キャリブレーション」の課題。

画面にターゲット（円）を表示し、ユーザーが手でそれを追いかける。その間の
  ・各ニューロンの発火率（スパイク数から推定したもの）  → X
  ・本当の手の速度                                     → Y
を 20 ms ごとに記録し、終わったら decoder.LinearDecoder.fit(X, Y) で重みを学習する。

実際の BCI でも、使い始める前（や毎日の使用前）に、画面のターゲットを追う課題で
デコーダを学習する。これをキャリブレーション（または open-loop の学習）という。

■ ターゲットの動き（合計 約 26 秒）。方向がかたよらないようにしている
  1. 準備（2 秒）        : 中心で止まる。手をターゲットに合わせてもらう（この間は記録しない）
  2. 8 方向への往復（12 秒）: 中心から 8 方向へ出て戻る（1 回 1.5 秒）。順番は隣り合わない方向を交互に
                           （center-out 課題。サルの実験で PD を測るのにも使われる動き）
  3. 円運動（12 秒）       : 左回りと右回りを半分ずつ。速さがほぼ一定で、すべての方向を連続して通る
"""

import math

import numpy as np

import config

READY_SEC = 2.0
REACH_ORDER = (0, 4, 2, 6, 1, 5, 3, 7)     # 8 方向（45° 刻み）の順番。反対向き・離れた向きを交互に
CENTER = np.array([0.5, 0.5])


def total_duration():
    return READY_SEC + len(REACH_ORDER) * config.CALIB_REACH_SEC + config.CALIB_CIRCLE_SEC


def target_position(t):
    """キャリブレーション開始から t 秒後のターゲットの位置（0〜1、y は上向き）"""
    R = config.CALIB_RADIUS
    if t < READY_SEC:
        return CENTER.copy()
    t -= READY_SEC

    # ---- 8 方向への往復：中心 → 外 → 中心 を sin で滑らかに ----
    reach_total = len(REACH_ORDER) * config.CALIB_REACH_SEC
    if t < reach_total:
        k = min(int(t / config.CALIB_REACH_SEC), len(REACH_ORDER) - 1)
        tau = (t - k * config.CALIB_REACH_SEC) / config.CALIB_REACH_SEC        # 0〜1
        a = REACH_ORDER[k] * math.pi / 4
        d = R * math.sin(math.pi * tau)
        return CENTER + d * np.array([math.cos(a), math.sin(a)])
    t -= reach_total

    # ---- 円運動：前半は左回り、後半は右回り。最初の 0.75 秒で中心から半径 R へ広がる ----
    half = config.CALIB_CIRCLE_SEC / 2
    omega = 2 * math.pi / config.CALIB_CIRCLE_PERIOD
    if t < half:
        a = omega * t
    else:
        a = omega * half - omega * (t - half)
    r = R * min(1.0, t / 0.75)
    return CENTER + r * np.array([math.cos(a), math.sin(a)])


def target_velocity(t, h=1e-3):
    """ターゲットの速度（数値微分）。実験モードで「手の動き」として使う"""
    return (target_position(t + h) - target_position(t - h)) / (2 * h)


class CalibrationTask:
    """キャリブレーションの進行と、データの記録"""

    def __init__(self):
        self.active = False
        self.duration = total_duration()
        self.t = 0.0
        self.X, self.Y = [], []

    def start(self):
        self.active = True
        self.t = 0.0
        self.X, self.Y = [], []

    def cancel(self):
        self.active = False
        self.X, self.Y = [], []

    @property
    def target(self):
        return target_position(self.t)

    @property
    def remaining(self):
        return max(0.0, self.duration - self.t)

    @property
    def progress(self):
        return min(1.0, self.t / self.duration)

    @property
    def recording(self):
        return self.t >= READY_SEC

    @property
    def valid_sec(self):
        """記録できた（手が検出されていた）時間 [秒]"""
        return len(self.X) * config.DT

    def step(self, rate_hat, v_true, detected, dt=config.DT):
        """
        20 ms ごとに呼ぶ。手が検出されているときだけ (発火率, 本当の速度) を記録する。
        終わったら True を返す（そのあと data() で X, Y を受け取る）。
        """
        if not self.active:
            return False
        if self.recording and detected:
            self.X.append(np.array(rate_hat, dtype=np.float64))
            self.Y.append(np.array(v_true, dtype=np.float64))
        self.t += dt
        if self.t >= self.duration:
            self.active = False
            return True
        return False

    def data(self):
        return np.array(self.X), np.array(self.Y)
