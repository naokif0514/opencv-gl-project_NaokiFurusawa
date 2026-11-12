"""
decoder.py
----------
（コピー元：02_pseudo_bci/decoder.py。03_bci_game で流用するためにコピーした）
ニューロン集団のスパイク数「だけ」から手の速度を推定（デコード）し、カーソルを動かす。
本当の手の速度は、精度の評価（相関係数・RMSE）と、線形デコーダの学習（キャリブレーション）にだけ使う。

2 種類のデコーダを用意し、同じスパイクから同時に推定して比べられるようにしている。

  ┌──────────────┬────────────────────────────────┬───────────────────────────────┐
  │              │ ポピュレーションベクトル法（PV）    │ 線形デコーダ（最小二乗法・リッジ回帰） │
  ├──────────────┼────────────────────────────────┼───────────────────────────────┤
  │ 事前に必要なもの │ 各ニューロンの「好きな方向」pd_i と、 │ キャリブレーション（手の速度と発火を   │
  │              │ 集団の平均的な b0・m（モデルの知識）  │ 同時に記録した 20〜30 秒のデータ）    │
  │ 重み           │ 決め打ち（pd_i そのもの）          │ データから学習（ニューロンごとに違う） │
  │ 個々のばらつき   │ 知らない → かたよりの原因          │ 学習で吸収できる                     │
  │ 脱落           │ 知らない → かたより（直せない）      │ 再キャリブレーションで重みが 0 になる  │
  │ 実際の研究      │ Georgopoulos ら 1986、Velliste ら │ Wessberg ら 2000、Hochberg ら 2006、 │
  │              │ 2008（サルのロボットアーム）など     │ カルマンフィルタ系（Gilja ら 2012）など │
  └──────────────┴────────────────────────────────┴───────────────────────────────┘

■ 共通の手順
  1. 発火率の推定（RateEstimator）：直近 DECODE_WINDOW 秒（100 ms）のスパイク数を数え、時間で割る
         r̂_i = (窓内のスパイク数) / (窓の長さ)
     実際の BCI でも、数十〜百 ms の窓でスパイクを数えて（spike binning）特徴量とする。

  2. 速度の推定：PV 法 または 線形デコーダ（下で説明）

  3. 速度の積分によるカーソル位置（_CursorDecoder）
         p(t + Δt) = p(t) + v̂ Δt − λ (p(t) − 中心) Δt
     速度の推定誤差は積分すると少しずつたまり、カーソルが端に張り付いてしまう（ドリフト）。
     そこで、λ（CURSOR_LEAK）の強さでゆるやかに中心へ引き戻す（リーク積分）。
     位置は画面内（0〜1）に制限する。

  4. 精度の評価（DecodingMetrics、直近 METRIC_WINDOW 秒）
     ・相関係数：本当の速度と推定速度の x 成分どうし・y 成分どうしのピアソン相関の平均
       （BCI 研究でデコード精度の指標としてよく使われる。1 に近いほど良い）
     ・RMSE：速度ベクトルの誤差の二乗平均平方根 sqrt(mean(|v − v̂|²))（0 に近いほど良い）
     相関は「向きの変化についていけているか」、RMSE は「大きさ・かたよりも含めて合っているか」を見る。
     たとえば推定が常に一定方向にずれていても、相関は高いまま RMSE だけ大きくなる。

■ ポピュレーションベクトル法（Georgopoulos ら, 1986）
  各ニューロンが「自分の好きな方向 pd_i」に、発火の強さに応じた票を入れ、
  全員の票を足し合わせた向きを運動方向とする。
      v̂ = (2 / N) × Σ_i ((r̂_i − b0) / m) × pd_i
  ・(r̂_i − b0) / m は、コサインチューニングの式を逆に解いた |v| cos(θ_v − θ_pd_i) の推定値
  ・PD が円周上に一様に分布していれば Σ_i cos(θ_v − θ_pd_i) pd_i ≈ (N/2) × (θ_v の向きの単位ベクトル)
    なので、2/N を掛けて正規化すると v̂ がそのまま速度（大きさも含む）の推定値になる
  ・b0, m はニューロン集団の平均値を使う（個々のばらつきや、脱落したニューロンはデコーダには分からない）
  この方法は「各ニューロンの好きな方向が事前に分かっている」ことが前提になっている。
  実験では、サルに腕を 8 方向へ動かさせて各ニューロンの PD を測ってから使う（Velliste ら 2008 など）。
  しかし、PD の分布が一様でない・b0 や m がニューロンごとに違う・ニューロンが脱落する、
  といった状況では、上の正規化の前提がくずれて推定がかたよる。

■ 線形デコーダ（最小二乗法・リッジ回帰）
  発火率のベクトル r̂(t) = (r̂_1, …, r̂_N) から速度を 1 次式で予測する。
      v̂(t) = W^T r̂(t) + c            W : (N × 2) の重み行列、c : 切片（2 次元）
  W と c は、キャリブレーションで記録した T 個のサンプル（発火率 X: T×N、本当の速度 Y: T×2）から、
  二乗誤差に正則化項を加えたものが最小になるように求める（リッジ回帰）。
      min_W  Σ_t |y_t − W^T x_t − c|²  +  α' ‖W‖²
  X, Y からそれぞれの平均を引いたもの（Xc, Yc）を使うと、解は次の式で一度に求まる。
      W = (Xc^T Xc + α' I)^(-1) Xc^T Yc，   c = ȳ − W^T x̄
      α' = RIDGE_ALPHA × (Xc^T Xc の対角成分の平均)    （特徴量の大きさに合わせて正則化の強さを決める）
  正則化（α' I）は、ニューロン数がサンプル数に比べて多いときや、発火がほとんど変化しない
  ニューロン（脱落して 0 のままのものなど）があるときに、重みが極端な値になるのを防ぐ。
  脱落したニューロンは X の列がすべて 0 になるので、再キャリブレーションするとその重みは 0 になり、
  残ったニューロンだけで推定するように学習し直される。

  実際の BCI（BrainGate など）では、PD を個別に測るのではなく、このように
  「キャリブレーション（ユーザーが画面のターゲットを追う／追うつもりになる）→ 回帰で重みを学習」
  という方法が基本になっている。カルマンフィルタ（Wu ら 2006、Gilja ら 2012 の ReFIT）も、
  観測モデルを同じく最小二乗法で学習する線形デコーダの一種である。
  なお、手が動かせない患者では本当の手の速度が測れないので、「ターゲットの方向へ動かそうとしている」
  と仮定した速度（意図した速度）を Y に使う。この作品では本当の手の速度を使っている。
"""

from collections import deque

import numpy as np

import config


# ============================================================
# 1. 発火率の推定（両方のデコーダで共通）
# ============================================================
class RateEstimator:
    """直近 DECODE_WINDOW 秒のスパイク数から発火率 r̂ [Hz] を求める"""

    def __init__(self, n):
        self.n = n
        self.window_bins = max(1, int(round(config.DECODE_WINDOW / config.DT)))
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window_bins)
        self.rate_hat = np.full(self.n, config.BASE_RATE)

    def step(self, counts, dt=config.DT):
        self.history.append(np.asarray(counts, dtype=np.float64))
        self.rate_hat = np.sum(self.history, axis=0) / (len(self.history) * dt)
        return self.rate_hat


def rates_from_counts(counts, dt=config.DT):
    """
    スパイク数の時系列 (T, N) から、各時刻の推定発火率 (T, N) をまとめて求める（実験モード用）。
    RateEstimator.step() を順に呼んだのと同じ結果（最初の数ビンは、そこまでのビンだけで割る）。
    """
    counts = np.asarray(counts, dtype=np.float64)
    w = max(1, int(round(config.DECODE_WINDOW / config.DT)))
    csum = np.cumsum(counts, axis=0)
    out = csum.copy()
    out[w:] -= csum[:-w]
    n_bins = np.minimum(np.arange(1, len(counts) + 1), w)[:, None]
    return out / (n_bins * dt)


# ============================================================
# 3. カーソル（推定速度の積分）。各デコーダの共通部分
# ============================================================
class _CursorDecoder:
    """推定速度 v̂ を求める decode() を子クラスで定義し、ここで積分してカーソルを動かす"""

    name = "?"

    def __init__(self):
        self.reset()

    def reset(self):
        self.v_hat = np.zeros(2)
        self.cursor = np.array([0.5, 0.5])

    def decode(self, rate_hat):
        raise NotImplementedError

    def step(self, rate_hat, dt=config.DT):
        """推定した発火率を受け取り、推定速度を更新してカーソルを動かす"""
        self.v_hat = self.decode(rate_hat)
        self.cursor += self.v_hat * dt - config.CURSOR_LEAK * (self.cursor - 0.5) * dt
        self.cursor = np.clip(self.cursor, 0.0, 1.0)
        return self.v_hat


# ============================================================
# 2a. ポピュレーションベクトル法
# ============================================================
class PopulationVectorDecoder(_CursorDecoder):
    name = "PV"

    def __init__(self, population):
        self.pd = population.pd.copy()     # デコーダが知っているのは各ニューロンの PD だけ
        self.n = population.n
        self.b0 = config.BASE_RATE         # 集団の平均的なベースライン
        self.m = config.MODULATION         # 集団の平均的な変調の深さ
        super().__init__()

    def decode(self, rate_hat):
        weights = (rate_hat - self.b0) / self.m              # 各ニューロンの「票の強さ」
        return (2.0 / self.n) * (weights @ self.pd)          # 好きな方向へ票を足し合わせる


# ============================================================
# 2b. 線形デコーダ（リッジ回帰）
# ============================================================
class LinearDecoder(_CursorDecoder):
    name = "LIN"

    def __init__(self, n):
        self.n = n
        self.W = None          # (N, 2) の重み行列。学習前は None
        self.c = np.zeros(2)   # 切片
        self.fit_info = None   # 学習のときの情報（サンプル数、生きていたニューロン数、学習データでの R²）
        super().__init__()

    @property
    def ready(self):
        return self.W is not None

    def fit(self, X, Y, alpha=config.RIDGE_ALPHA, n_alive=None):
        """
        X: (T, N) 発火率、Y: (T, 2) 本当の速度。リッジ回帰で W, c を求める。
        式の意味はファイル先頭の説明を参照。
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - x_mean, Y - y_mean
        G = Xc.T @ Xc                                       # (N, N)
        lam = alpha * max(np.trace(G) / self.n, 1e-9)       # 正則化の強さを特徴量の大きさに合わせる
        self.W = np.linalg.solve(G + lam * np.eye(self.n), Xc.T @ Yc)
        self.c = y_mean - x_mean @ self.W

        # 学習データに対する決定係数 R²（どれだけ説明できたか。過学習の目安にもなる）
        resid = Yc - Xc @ self.W
        r2 = 1.0 - np.sum(resid ** 2) / max(np.sum(Yc ** 2), 1e-12)
        self.fit_info = {"samples": len(X), "n_alive": n_alive, "r2": float(r2)}
        self.reset()
        return self.fit_info

    def forget(self):
        """学習結果を捨てる（ニューロン集団を作り直したとき）"""
        self.W = None
        self.c = np.zeros(2)
        self.fit_info = None
        self.reset()

    def decode(self, rate_hat):
        if self.W is None:
            return np.zeros(2)           # 学習前は動かない
        return rate_hat @ self.W + self.c


# ============================================================
# 4. 精度の評価
# ============================================================
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
        self.corr, self.rmse = evaluate(self.true_v, self.est_v)


def evaluate(true_v, est_v):
    """速度の時系列 (T, 2) 2 つから (相関係数, RMSE) を返す。計算できなければ None"""
    if len(true_v) < 10:
        return None, None
    t = np.asarray(true_v, dtype=np.float64)
    e = np.asarray(est_v, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(np.sum((t - e) ** 2, axis=1))))
    # x, y それぞれのピアソン相関。本当の速度がほとんど変化していない成分は除く
    rs = []
    for k in range(2):
        if t[:, k].std() >= config.METRIC_MIN_STD and e[:, k].std() > 1e-9:
            rs.append(np.corrcoef(t[:, k], e[:, k])[0, 1])
    return (float(np.mean(rs)) if rs else None), rmse
