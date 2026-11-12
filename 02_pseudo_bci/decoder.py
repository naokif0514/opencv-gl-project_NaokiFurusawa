"""
decoder.py
----------
ニューロン集団のスパイク数「だけ」から手の速度を推定（デコード）し、カーソルを動かす。
本当の手の速度は、精度の評価（相関係数・RMSE）にだけ使い、推定には使わない。

■ 手順（実際の BCI と同じ流れ）
  1. 発火率の推定：直近 DECODE_WINDOW 秒（100 ms）のスパイク数を数え、時間で割る
         r̂_i = (窓内のスパイク数) / (窓の長さ)
     実際の BCI でも、数十〜百 ms の窓でスパイクを数えて発火率とする。

  2. ポピュレーションベクトル法（Georgopoulos ら, 1986）
     各ニューロンが「自分の好きな方向 pd_i」に、発火の強さに応じた票を入れ、
     全員の票を足し合わせた向きを運動方向とする。
         v̂ = (2 / N) × Σ_i ((r̂_i − b0) / m) × pd_i
     ・(r̂_i − b0) / m は、コサインチューニングの式を逆に解いた |v| cos(θ_v − θ_pd_i) の推定値
     ・PD が円周上に一様に分布していれば Σ_i cos(θ_v − θ_pd_i) pd_i ≈ (N/2) × (θ_v の向きの単位ベクトル)
       なので、2/N を掛けて正規化すると v̂ がそのまま速度（大きさも含む）の推定値になる
     ・b0, m はニューロン集団の平均値を使う（個々のばらつきや、脱落したニューロンはデコーダには分からない）
     Velliste ら（2008）のサルによるロボットアームの操作や、初期のヒト BCI でも使われた方法。

  3. 速度の積分によるカーソル位置
         p(t + Δt) = p(t) + v̂ Δt − λ (p(t) − 中心) Δt
     速度の推定誤差は積分すると少しずつたまり、カーソルが端に張り付いてしまう（ドリフト）。
     そこで、λ（CURSOR_LEAK）の強さでゆるやかに中心へ引き戻す（リーク積分）。
     位置は画面内（0〜1）に制限する。

  4. 精度の評価（直近 METRIC_WINDOW 秒）
     ・相関係数：本当の速度と推定速度の x 成分どうし・y 成分どうしのピアソン相関の平均
       （BCI 研究でデコード精度の指標としてよく使われる。1 に近いほど良い）
     ・RMSE：速度ベクトルの誤差の二乗平均平方根 sqrt(mean(|v − v̂|²))（0 に近いほど良い）
"""

from collections import deque

import numpy as np

import config


class PopulationVectorDecoder:
    """ポピュレーションベクトル法によるデコーダとカーソル"""

    def __init__(self, population):
        self.pd = population.pd.copy()     # デコーダが知っているのは各ニューロンの PD だけ
        self.n = population.n
        self.b0 = config.BASE_RATE         # 集団の平均的なベースライン
        self.m = config.MODULATION         # 集団の平均的な変調の深さ
        self.window_bins = max(1, int(round(config.DECODE_WINDOW / config.DT)))
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window_bins)   # 直近のビンのスパイク数
        self.v_hat = np.zeros(2)
        self.rate_hat = np.full(self.n, self.b0)
        self.cursor = np.array([0.5, 0.5])

    def step(self, counts, dt=config.DT):
        """1 ビン分のスパイク数を受け取り、推定速度を更新してカーソルを動かす"""
        self.history.append(counts.astype(np.float64))

        # 1. 発火率の推定
        window = len(self.history) * dt
        self.rate_hat = np.sum(self.history, axis=0) / window

        # 2. ポピュレーションベクトル
        weights = (self.rate_hat - self.b0) / self.m
        self.v_hat = (2.0 / self.n) * (weights @ self.pd)

        # 3. 積分してカーソル位置へ（中心へのリーク付き）
        self.cursor += self.v_hat * dt - config.CURSOR_LEAK * (self.cursor - 0.5) * dt
        self.cursor = np.clip(self.cursor, 0.0, 1.0)
        return self.v_hat


class DecodingMetrics:
    """本当の速度と推定速度を記録し、相関係数と RMSE を計算する"""

    def __init__(self):
        n = int(config.METRIC_WINDOW / config.DT)
        self.true_v = deque(maxlen=n)
        self.est_v = deque(maxlen=n)
        self.corr = None     # 計算できないとき（手がほとんど動いていないとき）は None
        self.rmse = None

    def reset(self):
        self.true_v.clear()
        self.est_v.clear()
        self.corr = None
        self.rmse = None

    def add(self, v_true, v_est):
        self.true_v.append(np.asarray(v_true, dtype=np.float64))
        self.est_v.append(np.asarray(v_est, dtype=np.float64))

    def compute(self):
        """直近の記録から相関係数と RMSE を計算する"""
        if len(self.true_v) < 10:
            self.corr, self.rmse = None, None
            return
        t = np.array(self.true_v)
        e = np.array(self.est_v)
        self.rmse = float(np.sqrt(np.mean(np.sum((t - e) ** 2, axis=1))))

        # x, y それぞれのピアソン相関。本当の速度がほとんど変化していない成分は除く
        rs = []
        for k in range(2):
            if t[:, k].std() >= config.METRIC_MIN_STD and e[:, k].std() > 1e-9:
                rs.append(np.corrcoef(t[:, k], e[:, k])[0, 1])
        self.corr = float(np.mean(rs)) if rs else None
