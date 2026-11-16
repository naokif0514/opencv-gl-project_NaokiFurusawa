"""
hand_calib.py
-------------
手の「同期フェーズ」（キャリブレーション）。ステージの最初と、電極劣化のあとの再同期で行う。
（02_pseudo_bci/calibration.py をもとに、ゲーム向けに短くし、把持（握る・開く）の区間を追加したもの）

画面のターゲット（マゼンタの円）を手で追いかけ、その間の
  ・方向のニューロンの発火率 → 手の速度        （速度の線形デコーダの学習データ）
  ・把持ニューロンの発火率   → 握り具合 g        （把持のデコーダの学習データ）
を 20 ms ごとに記録する。実際の BCI でも、使う前に「画面の動きを追う・つかむ」課題でデコーダを学習する。

■ ターゲットの動き（合計 約 21 秒。時間は config で変えられる）
  1. 準備（HC_READY_SEC）     : 中心で止まる（記録しない）
  2. 8 方向への往復            : 中心から 8 方向へ出て戻る（1 回 HC_REACH_SEC 秒）
  3. 円運動（HC_CIRCLE_SEC）   : 左回り・右回りを半分ずつ
  4. 握る・開く（HC_GRIP_SEC）  : 中心で止まったまま、表示に合わせて手を握る・開くをくり返す
座標は手の位置の座標（0〜1、y は上向き）。AR 画面ではカメラ映像と同じ位置に表示する。
"""

import math

import numpy as np

import config

REACH_ORDER = (0, 4, 2, 6, 1, 5, 3, 7)
CENTER = np.array([0.5, 0.5])


def _reach_total():
    return len(REACH_ORDER) * config.HC_REACH_SEC


def total_duration():
    return config.HC_READY_SEC + _reach_total() + config.HC_CIRCLE_SEC + config.HC_GRIP_SEC


def phase(t):
    """その時刻の区間の名前：ready / reach / circle / grip"""
    if t < config.HC_READY_SEC:
        return "ready"
    t -= config.HC_READY_SEC
    if t < _reach_total():
        return "reach"
    t -= _reach_total()
    return "circle" if t < config.HC_CIRCLE_SEC else "grip"


def grip_target(t):
    """握る区間で、握ってほしいとき 1、開いてほしいとき 0（それ以外の区間は 0）"""
    if phase(t) != "grip":
        return 0.0
    tg = t - config.HC_READY_SEC - _reach_total() - config.HC_CIRCLE_SEC
    return 1.0 if int(tg / config.HC_GRIP_PERIOD) % 2 == 1 else 0.0


def target_position(t):
    R = config.CALIB_RADIUS
    p = phase(t)
    if p in ("ready", "grip"):
        return CENTER.copy()
    t -= config.HC_READY_SEC
    if p == "reach":
        k = min(int(t / config.HC_REACH_SEC), len(REACH_ORDER) - 1)
        tau = (t - k * config.HC_REACH_SEC) / config.HC_REACH_SEC
        a = REACH_ORDER[k] * math.pi / 4
        return CENTER + R * math.sin(math.pi * tau) * np.array([math.cos(a), math.sin(a)])
    t -= _reach_total()
    half = config.HC_CIRCLE_SEC / 2
    omega = 2 * math.pi / config.HC_CIRCLE_PERIOD
    a = omega * t if t < half else omega * half - omega * (t - half)
    r = R * min(1.0, t / 0.6)
    return CENTER + r * np.array([math.cos(a), math.sin(a)])


def target_velocity(t, h=1e-3):
    return (target_position(t + h) - target_position(t - h)) / (2 * h)


class HandCalibration:
    def __init__(self):
        self.active = False
        self.duration = total_duration()
        self.t = 0.0
        self.X, self.Y, self.Xg, self.Yg = [], [], [], []

    def start(self):
        self.active = True
        self.t = 0.0
        self.X, self.Y, self.Xg, self.Yg = [], [], [], []

    def cancel(self):
        self.active = False

    @property
    def target(self):
        return target_position(self.t)

    @property
    def phase(self):
        return phase(self.t)

    @property
    def grip_cue(self):
        return grip_target(self.t)

    @property
    def remaining(self):
        return max(0.0, self.duration - self.t)

    @property
    def progress(self):
        return min(1.0, self.t / self.duration)

    @property
    def valid_sec(self):
        return len(self.X) * config.DT

    def step(self, rate_hat, grip_rate_hat, v_true, g_true, detected, dt=config.DT):
        """20 ms ごとに呼ぶ。手が見えているときだけ記録する。終わったら True"""
        if not self.active:
            return False
        if self.phase != "ready" and detected:
            self.X.append(np.array(rate_hat))
            self.Y.append(np.array(v_true))
            self.Xg.append(np.array(grip_rate_hat))
            self.Yg.append(float(g_true))
        self.t += dt
        if self.t >= self.duration:
            self.active = False
            return True
        return False

    def data(self):
        return np.array(self.X), np.array(self.Y), np.array(self.Xg), np.array(self.Yg)

    @property
    def enough(self):
        """学習に十分なデータか（手が見えていた時間と、握った・開いたの両方があるか）"""
        if self.valid_sec < config.HC_MIN_SEC or not self.Yg:
            return False
        yg = np.array(self.Yg)
        return (yg > 0.6).sum() >= 10 and (yg < 0.4).sum() >= 10
