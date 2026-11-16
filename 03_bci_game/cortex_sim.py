"""
cortex_sim.py
-------------
左半球の運動野（手の領域）の擬似ニューロン集団を 20 ms ごとに動かし、デコードと表示用の値をまとめて持つ。
（02_pseudo_bci の main.py の sim_step と renderer.py の on_spikes を、ゲームで使いやすい形にまとめ、
  把持ニューロンと把持のデコーダを追加したもの）

■ 2 種類のニューロン
  ・方向のニューロン（neuron_model.NeuronPopulation、02 と同じ）：手を動かす向き・速さにコサインチューニング
  ・把持ニューロン（GripPopulation、03 で追加）：手を握るほど発火が増える
        rate_i = b0_i + m_i × g + (ノイズ)，  g = 1 − 手の開き具合（0 = 開く、1 = 握る）
    実際の運動野・運動前野（腹側 PMv など）にも、つかむ動作や手の形に応じて発火するニューロンがあり、
    BCI でもロボットハンドの開閉（把持）がデコードされている（Carmena ら 2003、Velliste ら 2008、Collinger ら 2013）。

■ デコード
  ・速度：同期フェーズ（キャリブレーション）の前は PV 法、後は線形デコーダ（リッジ回帰。decoder.py、02 と同じ）
  ・把持：同期フェーズの前は「把持ニューロンの (r − b0)/m の平均」（モデルの知識だけを使う）、
          後は把持ニューロンの発火率 → 握る・開くの合図 のリッジ回帰。どちらも少しなめらかにしてから、
          しきい値にヒステリシスをつけて「握った / 開いた」を決める（mission.py の GripSwitch）
          学習には、同期フェーズの「握る・開く」の区間（合図が交互に出て、握る・開くのデータが半分ずつある）
          だけを使う。ターゲットを追う区間も使うと、そのあいだ手が少し丸まっていると「握っている」データばかりに
          なり、リッジ回帰の推定が平均（握る側）に引き寄せられて、開いても「開いた」のしきい値まで下がらなくなる
          （実際のプレイログで起きた問題）。さらに、学習後に「開く」「握る」の合図のときの推定値の中央値を求め、
          その間の GRIP_TH_LO・GRIP_TH_HI の位置をしきい値にする（人ごと・ステージごとの差に合わせる）
  ・脱落（電極劣化イベント）は両方のニューロンに起こる。デコーダはどれが脱落したかを知らないので、
    同期フェーズをやり直す（再キャリブレーション）まで推定がかたよる
"""

import math

import numpy as np

import config
from decoder import DecodingMetrics, LinearDecoder, PopulationVectorDecoder, RateEstimator
from gl_utils import hue_to_rgb
from neuron_model import NeuronPopulation

GRIP_COLOR = (0.95, 0.95, 1.0)      # 把持ニューロンの表示色（方向のニューロンは好きな方向の色）


class GripPopulation:
    """把持ニューロン：握るほど発火が増える（ノイズ・脱落は NeuronPopulation と同じ扱い）"""

    def __init__(self, n, rng):
        self.rng = rng
        self.n = n
        h = config.TUNING_HETEROGENEITY
        self.b0 = config.BASE_RATE * (1.0 + h * rng.uniform(-1, 1, n))
        self.m = config.GRIP_MODULATION * (1.0 + h * rng.uniform(-1, 1, n))
        self.noise_std = config.NOISE_STD
        self.noise_type = config.NOISE_TYPE
        self.xi = rng.normal(size=n)
        self.alive = np.ones(n, dtype=bool)

    def step(self, g, dt=config.DT):
        tau = config.NOISE_TAU
        self.xi += -self.xi * dt / tau + np.sqrt(2.0 * dt / tau) * self.rng.normal(size=self.n)
        rate = self.b0 + self.m * g
        if self.noise_type in ("drift", "both"):
            rate = rate + self.noise_std * self.xi
        if self.noise_type in ("independent", "both"):
            rate = rate + self.noise_std * self.rng.normal(size=self.n)
        counts = self.rng.poisson(np.maximum(rate, 0.0) * dt).astype(np.int32)
        counts[~self.alive] = 0
        return counts

    def dropout(self, fraction):
        idx = np.flatnonzero(self.alive)
        k = int(round(len(idx) * fraction))
        if k > 0:
            self.alive[self.rng.choice(idx, size=k, replace=False)] = False
        return k


class GripDecoder:
    """把持の推定（0 = 開く、1 = 握る）。学習前はモデルの知識だけ、学習後はリッジ回帰"""

    def __init__(self, n):
        self.n = n
        self.w = None
        self.c = 0.0
        self.value = 0.0          # なめらかにした推定値
        self.close_th = config.GRIP_CLOSE_TH
        self.open_th = config.GRIP_OPEN_TH
        self.levels = None        # 学習データでの (開くときの推定値, 握るときの推定値)

    @property
    def ready(self):
        return self.w is not None

    def fit(self, X, cue, alpha=config.GRIP_RIDGE_ALPHA):
        """X：把持ニューロンの発火率、cue：合図（1 = 握る、0 = 開く、−1 = 使わない）"""
        X, cue = np.asarray(X, dtype=np.float64), np.asarray(cue, dtype=np.float64)
        use = cue >= 0
        X, y = X[use], cue[use]
        if len(y) < 20 or y.min() == y.max():
            return False
        xm, ym = X.mean(axis=0), y.mean()
        Xc, yc = X - xm, y - ym
        G = Xc.T @ Xc
        lam = alpha * max(np.trace(G) / self.n, 1e-9)
        self.w = np.linalg.solve(G + lam * np.eye(self.n), Xc.T @ yc)
        self.c = ym - xm @ self.w
        # しきい値を、この人のデータに合わせる（開く・握るのときの推定値の中央値の間）
        pred = X @ self.w + self.c
        lo, hi = float(np.median(pred[y == 0])), float(np.median(pred[y == 1]))
        self.levels = (lo, hi)
        self.open_th = lo + config.GRIP_TH_LO * (hi - lo)
        self.close_th = lo + config.GRIP_TH_HI * (hi - lo)
        return True

    def step(self, rate_hat, dt=config.DT):
        if self.w is None:
            raw = float(np.mean((rate_hat - config.BASE_RATE) / config.GRIP_MODULATION))
        else:
            raw = float(rate_hat @ self.w + self.c)
        self.value += (raw - self.value) * (1.0 - math.exp(-dt / config.GRIP_SMOOTH_SEC))
        return self.value


class CortexSim:
    def __init__(self, n, rng, noise=config.NOISE_STD, noise_type=config.NOISE_TYPE, n_grip=None):
        self.rng = rng
        self.pop = NeuronPopulation(n, rng)
        n_grip = n_grip if n_grip is not None else max(config.GRIP_MIN, int(round(n * config.GRIP_FRACTION)))
        self.grip_pop = GripPopulation(n_grip, rng)
        self.set_noise(noise, noise_type)
        self.rates = RateEstimator(n)
        self.grip_rates = RateEstimator(n_grip)
        self.pv = PopulationVectorDecoder(self.pop)
        self.lin = LinearDecoder(n)
        self.grip = GripDecoder(n_grip)
        self.metrics = DecodingMetrics()
        self.v_hat = np.zeros(2)
        self.rate_hat = self.rates.rate_hat
        self.grip_rate_hat = self.grip_rates.rate_hat
        self.dropped_since_calib = 0

        n_all = n + n_grip
        self.flash = np.zeros(n_all)
        self.activity = np.zeros(n_all)
        self.motor_drive = 0.0
        self.grip_drive = 0.0
        bins = int(round(config.RASTER_SEC / config.DT))
        self.raster_bg = np.array([14, 16, 26, 255], dtype=np.uint8)
        self.raster = np.empty((n_all, bins, 4), dtype=np.uint8)
        self.raster[:] = self.raster_bg
        # ラスターの行：下から方向のニューロン（好きな方向の順）、上に把持ニューロン（白）
        colors = np.vstack([hue_to_rgb(self.pop.pd_angle / (2 * np.pi)), np.tile(GRIP_COLOR, (n_grip, 1))])
        self.colors = colors
        self.raster_colors = (colors[::-1] * 255).astype(np.uint8)

    # --------------------------------------------------------
    @property
    def n(self):
        return self.pop.n

    @property
    def n_grip(self):
        return self.grip_pop.n

    @property
    def n_all(self):
        return self.pop.n + self.grip_pop.n

    @property
    def alive(self):
        return np.concatenate([self.pop.alive, self.grip_pop.alive])

    @property
    def n_alive(self):
        return int(self.alive.sum())

    @property
    def calibrated(self):
        return self.lin.ready

    def set_noise(self, std, kind):
        for p in (self.pop, self.grip_pop):
            p.noise_std, p.noise_type = std, kind

    # --------------------------------------------------------
    def step(self, v, g, hand_detected=True):
        """手の速度 v・握り具合 g（0〜1）→ スパイク → 発火率 → 速度と把持の推定"""
        counts = self.pop.step(v)
        gcounts = self.grip_pop.step(g)
        self.rate_hat = self.rates.step(counts)
        self.grip_rate_hat = self.grip_rates.step(gcounts)
        if self.lin.ready:
            self.v_hat = self.lin.decode(self.rate_hat)
        else:
            self.v_hat = self.pv.decode(self.rate_hat)
        self.grip.step(self.grip_rate_hat)
        if hand_detected:
            self.metrics.add(v, self.v_hat)

        # 表示用
        allc = np.concatenate([counts, gcounts])
        fired = allc > 0
        self.flash[fired] = 1.0
        act = np.clip((self.rate_hat - config.BASE_RATE) / config.MODULATION, 0.0, 1.0)
        gact = np.clip((self.grip_rate_hat - config.BASE_RATE) / config.GRIP_MODULATION, 0.0, 1.0)
        self.activity += 0.25 * (np.concatenate([act, gact]) - self.activity)
        a, ga = self.pop.alive, self.grip_pop.alive
        self.motor_drive = float(np.clip(config.MOTOR_GLOW_GAIN * act[a].mean(), 0, 1)) if a.any() else 0.0
        self.grip_drive = float(np.clip(gact[ga].mean() * 1.4, 0, 1)) if ga.any() else 0.0
        alive = self.alive
        self.raster[:, :-1] = self.raster[:, 1:]
        col = np.empty((self.n_all, 4), dtype=np.uint8)
        col[:] = self.raster_bg
        col[(~alive)[::-1]] = (34, 34, 38, 255)
        rows = fired[::-1]
        col[rows, :3] = self.raster_colors[rows]
        self.raster[:, -1] = col
        return self.v_hat

    def decay(self, dt):
        self.flash *= math.exp(-dt / config.SPIKE_FLASH_TAU)

    # --------------------------------------------------------
    def dropout(self, fraction, grip_fraction=None):
        """
        電極劣化：ニューロンの一部が記録できなくなる。grip_fraction で把持ニューロンの脱落の割合を別に決められる
        （ゲームでは、腕の動きのニューロンが主に脱落する設定にしている。mission.py を参照）
        """
        grip_fraction = fraction if grip_fraction is None else grip_fraction
        k = self.pop.dropout(fraction) + (self.grip_pop.dropout(grip_fraction) if grip_fraction > 0 else 0)
        self.dropped_since_calib += k
        return k

    def fit(self, X, Y, Xg, cue):
        """同期フェーズで集めたデータで、速度と把持の線形デコーダを学習する"""
        info = self.lin.fit(X, Y, n_alive=self.pop.n_alive)
        self.grip.fit(Xg, cue)
        self.metrics.reset()
        self.dropped_since_calib = 0
        return info
