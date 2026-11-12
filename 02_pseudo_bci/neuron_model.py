"""
neuron_model.py
---------------
一次運動野（M1）のニューロン集団が、手の動きに応じてどう発火するかをシミュレーションする。

■ 使っているモデル：コサインチューニング（Georgopoulos ら, 1982）
  サルに腕を 8 方向へ動かさせながら M1 のニューロンを記録すると、各ニューロンには
  「最もよく発火する運動方向（preferred direction, PD）」があり、発火率は
  運動方向と PD のなす角のコサインに比例して変わることが見つかった。

      rate_i(t) = b0_i + m_i × |v(t)| × cos(θ_v(t) − θ_pd_i)      （負になったら 0）

      b0_i    : ベースライン発火率（動いていないときの平均発火率）[Hz]
      m_i     : 変調の深さ（速さ 1 あたりどれだけ発火率が変わるか）[Hz/(画面/秒)]
      v(t)    : 手の速度ベクトル。|v| は速さ、θ_v はその向き
      θ_pd_i  : ニューロン i の好きな方向。円周上にランダムに割り当てる

  |v| を掛けているのは、速く動くほど発火の変化が大きくなるという
  「速度チューニング」（Moran & Schwartz, 1999）を取り入れたもの。

■ スパイクの生成：非一様ポアソン過程
  実際のニューロンは発火率どおりに規則正しく発火するのではなく、確率的に発火する。
  時間を Δt = 20 ms の区間（ビン）に区切り、各区間のスパイク数を
      n_i ~ Poisson(rate_i × Δt)
  で生成する。これは電極で記録したスパイク列を「ビンごとのスパイク数」として扱う
  実際の BCI の前処理（spike binning）に対応する。

■ ノイズ（1/2/3 キーで種類を切り替え、N/M キーで強さ σ [Hz] を変える）
  実際のニューロンの発火には、運動とは関係のないゆらぎがある。ここでは 2 種類を用意する。

  (1) drift：ベースライン発火のゆっくりしたゆらぎ（覚醒度・注意・他の脳活動の影響に相当）
      各ニューロンのベースラインを、ゆっくり変動するランダムな値（オルンシュタイン＝
      ウーレンベック過程、時定数 τ = 0.5 秒）で揺らす。
          b_i(t) = b0_i + σ × ξ_i(t)
          ξ_i(t+Δt) = ξ_i(t) − ξ_i(t) Δt/τ + sqrt(2Δt/τ) × N(0, 1)       （ξ の標準偏差は 1）
      数百 ms〜数秒のあいだ同じ向きにずれ続けるので、速度の推定に「ゆっくりしたかたより」が乗り、
      積分したカーソルが流されやすい。

  (2) independent：ニューロンごとに独立で、ビン（20 ms）ごとに毎回新しく入るゆらぎ
      （シナプス入力の細かい変動など、ニューロンどうし・時刻どうしで相関のない成分に相当）
          rate_i(t) = b_i(t) + m_i (v · pd_i) + σ × η_i(t)，  η_i(t) ~ N(0, 1)（毎ビン独立）
      ニューロンどうしで独立なので、デコーダが N 個の票を足し合わせると打ち消し合い、
      推定誤差の標準偏差はおよそ 1/√N で小さくなる（集団符号化の「平均化の効果」）。
      ニューロン数を増やすと精度がはっきり上がるのはこのノイズのとき。

  (3) both：(1) と (2) の両方（同じ σ）

  どちらも発火率に加えたあとで 0 未満を 0 に切り、そこからポアソン分布でスパイクを作る。
  そのため、スパイクはいつも整数で、ラスタープロットの見た目も自然なままになる。
  ただし σ が大きいと 0 で切られる時間が増え、変調（速度による発火率の変化）が小さく見えるという
  かたよりも生じる。これは平均化では消えない誤差で、ゲインを学習する線形デコーダでは補正される。

■ ニューロンの脱落（dropout）
  慢性埋め込み電極では、時間がたつと組織の反応や電極の劣化で、拾えるニューロンが減っていく。
  脱落したニューロンはスパイクを出さない（記録されない）ものとして扱う。
  デコーダはどれが脱落したかを知らないので、推定にかたよりが生じる。

■ ニューロンごとのばらつき
  実際のニューロンは b0・m が 1 つ 1 つ違う。ここでも ±TUNING_HETEROGENEITY の割合でばらつかせる。
  ポピュレーションベクトル法のデコーダは、全員が平均的な b0・m を持つと仮定して推定する。
"""

import numpy as np

import config

NOISE_TYPES = ("drift", "independent", "both")


class NeuronPopulation:
    """運動野ニューロン集団"""

    def __init__(self, n, rng=None):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n = n

        # 好きな方向（PD）を円周上にランダムに割り当てる。
        # 番号順に並べ替えておくと、ラスタープロットで「方向ごとの帯」が見やすくなる
        self.pd_angle = np.sort(self.rng.uniform(0.0, 2.0 * np.pi, n))
        self.pd = np.stack([np.cos(self.pd_angle), np.sin(self.pd_angle)], axis=1)   # 単位ベクトル (n,2)

        # ニューロンごとのばらつき
        h = config.TUNING_HETEROGENEITY
        self.b0 = config.BASE_RATE * (1.0 + h * self.rng.uniform(-1, 1, n))
        self.m = config.MODULATION * (1.0 + h * self.rng.uniform(-1, 1, n))

        self.noise_std = config.NOISE_STD        # ノイズの強さ σ [Hz]
        self.noise_type = config.NOISE_TYPE       # "drift" / "independent" / "both"
        self.xi = self.rng.normal(size=n)          # ベースラインのゆらぎ ξ_i
        self.alive = np.ones(n, dtype=bool)        # 脱落していないか

        self.rate = self.b0.copy()                 # 直近の発火率 [Hz]
        self.counts = np.zeros(n, dtype=np.int32)  # 直近のビンのスパイク数

    # --------------------------------------------------------
    def firing_rate(self, v):
        """
        速度ベクトル v=(vx, vy) に対する各ニューロンの発火率（コサインチューニング）。
            rate_i = b0_i + m_i × (v · pd_i) + （ノイズ）
        v · pd_i = |v| cos(θ_v − θ_pd_i) なので、上の式と同じもの。
        """
        rate = self.b0 + self.m * (self.pd @ np.asarray(v, dtype=np.float64))
        if self.noise_type in ("drift", "both"):
            rate = rate + self.noise_std * self.xi                            # ゆっくりしたゆらぎ
        if self.noise_type in ("independent", "both"):
            rate = rate + self.noise_std * self.rng.normal(size=self.n)       # 毎ビン独立なゆらぎ
        return np.maximum(rate, 0.0)               # 発火率は負にならない

    def step(self, v, dt=config.DT):
        """
        Δt だけ時間を進め、各ニューロンのスパイク数を返す。
        1. ゆっくりしたノイズ ξ を更新（オルンシュタイン＝ウーレンベック過程）
        2. コサインチューニングで発火率を計算（ノイズの種類に応じてゆらぎを加える）
        3. ポアソン分布からスパイク数を生成（脱落したニューロンは 0）
        """
        tau = config.NOISE_TAU
        self.xi += -self.xi * dt / tau + np.sqrt(2.0 * dt / tau) * self.rng.normal(size=self.n)

        self.rate = self.firing_rate(v)
        self.counts = self.rng.poisson(self.rate * dt).astype(np.int32)
        self.counts[~self.alive] = 0
        return self.counts

    # --------------------------------------------------------
    def dropout(self, fraction=config.DROPOUT_FRACTION):
        """生きているニューロンのうち、指定した割合をランダムに脱落させる"""
        alive_idx = np.flatnonzero(self.alive)
        k = max(1, int(round(len(alive_idx) * fraction))) if len(alive_idx) > 0 else 0
        if k > 0:
            dead = self.rng.choice(alive_idx, size=k, replace=False)
            self.alive[dead] = False
        return k

    @property
    def n_alive(self):
        return int(self.alive.sum())
