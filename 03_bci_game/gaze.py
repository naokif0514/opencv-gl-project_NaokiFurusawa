"""
gaze.py
-------
視線（Face Mesh の虹彩と顔の向き）から、画面上の注視点を推定して、対象を選ぶ。
ハイブリッド BCI の「どこへ・何を」を選ぶ部分（前頭眼野・視覚野に対応）。

■ 1. キャリブレーション（GazeCalibration、ゲームの演出では「視線同期」）
  画面の 5 点（AR 画面の四隅 + 中央）を順に見てもらい、そのときの特徴量 f（vision.gaze_features）を記録する。
  各点で、目が動き終わるまで待ってから（SETTLE）、COLLECT 秒のあいだの特徴量の中央値を使う。
  そこから、特徴量 → 画面座標 の 1 次式を最小二乗法（弱いリッジ回帰）で求める。
      x = a0 + a1 × (虹彩の左右) + a2 × (顔の向きの左右)
      y = b0 + b1 × (虹彩の上下) + b2 × (顔の向きの上下)
  5 点しかないので、軸ごとに 2 つの特徴量だけを使う（多くすると 5 点では決まらず、外れやすい）。

■ 2. 平滑化（OneEuroFilter）
  Web カメラの視線推定は 1 フレームごとに大きく揺れる。One Euro フィルタ（Casiez ら 2012）は、
  ゆっくり動くときは強くなめらかにし（揺れを消す）、速く動くときは弱くする（遅れを減らす）。

■ 3. 吸い付き + ドウェル選択（DwellSelector）
  Web カメラでは精度が低い（数十〜百画素ずれる）前提で、注視点から SNAP_PX 以内で最も近い対象に「吸い付く」。
  同じ対象を DWELL_SEC 秒（約 1 秒）見つめ続けたら決定する（ドウェル選択。視線入力の基本的な方法で、
  まばたきやクリックなしで選べる）。見つめている間はリングがたまっていく。
"""

import math

import numpy as np

import config


class OneEuroFilter:
    """One Euro フィルタ（2 次元）。min_cutoff [Hz] が小さいほど静止時になめらか、beta が大きいほど速い動きに追従"""

    def __init__(self, min_cutoff=config.GAZE_MIN_CUTOFF, beta=config.GAZE_BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = None
        self.dx = np.zeros(2)

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or dt <= 0:
            self.x = x.copy()
            return self.x
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x


class GazeCalibration:
    """5 点の視線キャリブレーション（視線同期）"""

    def __init__(self, points):
        self.points = [np.asarray(p, dtype=np.float64) for p in points]   # 画面座標（画素）
        self.active = False
        self.model = None           # (Ax, Ay)：x = Ax · [1, f0, f2]、y = Ay · [1, f1, f3]
        self.residual = None        # 学習した点での平均のずれ [画素]

    def start(self):
        self.active = True
        self.index = 0
        self.t = 0.0
        self.samples = [[] for _ in self.points]

    def cancel(self):
        self.active = False

    @property
    def current_point(self):
        return self.points[self.index] if self.active else None

    @property
    def collecting(self):
        return self.active and self.t >= config.GAZE_CALIB_SETTLE

    @property
    def point_progress(self):
        """今の点での進み具合 0〜1（演出用）"""
        return min(1.0, self.t / (config.GAZE_CALIB_SETTLE + config.GAZE_CALIB_COLLECT))

    def step(self, feat, dt):
        """毎フレーム呼ぶ。終わったら True を返す（失敗なら self.model は None のまま）"""
        if not self.active:
            return False
        self.t += dt
        if self.collecting and feat is not None:
            self.samples[self.index].append(np.asarray(feat))
        if self.t >= config.GAZE_CALIB_SETTLE + config.GAZE_CALIB_COLLECT:
            if len(self.samples[self.index]) < 3:       # 顔が見えていなかったら、同じ点をやり直す
                self.t = 0.0
                self.samples[self.index] = []
                return False
            self.index += 1
            self.t = 0.0
            if self.index >= len(self.points):
                self.active = False
                self._fit()
                return True
        return False

    def _fit(self):
        F = np.array([np.median(s, axis=0) for s in self.samples])        # 点ごとの中央値 (5, 4)
        P = np.array(self.points)                                          # (5, 2)
        models = []
        for axis, cols in ((0, (0, 2)), (1, (1, 3))):
            X = np.column_stack([np.ones(len(F)), F[:, cols[0]], F[:, cols[1]]])
            # 特徴量の大きさをそろえてから弱いリッジ回帰（切片には正則化をかけない）
            scale = np.maximum(X.std(axis=0), 1e-6)
            scale[0] = 1.0
            Xs = X / scale
            reg = config.GAZE_RIDGE * np.diag([0.0, 1.0, 1.0])
            a = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ P[:, axis]) / scale
            models.append(a)
        self.model = models
        pred = np.array([self.predict(f) for f in F])
        self.residual = float(np.mean(np.linalg.norm(pred - P, axis=1)))

    def predict(self, feat):
        if self.model is None or feat is None:
            return None
        ax, ay = self.model
        return np.array([ax @ [1.0, feat[0], feat[2]], ay @ [1.0, feat[1], feat[3]]])


class GazeTracker:
    """特徴量 → 画面上の注視点（キャリブレーション + 平滑化）。テストモードではマウスの位置をそのまま使う"""

    def __init__(self, calib_points):
        self.calib = GazeCalibration(calib_points)
        self.filter = OneEuroFilter()
        self.point = None           # 平滑化した注視点（画素）。分からなければ None
        self.raw = None

    @property
    def calibrated(self):
        return self.calib.model is not None

    def update(self, feat, dt, mouse=None):
        if mouse is not None:                       # テストモード：マウスの位置 = 視線
            self.raw = np.asarray(mouse, dtype=np.float64)
            self.point = self.raw.copy()
            return self.point
        self.raw = self.calib.predict(feat)
        if self.raw is None:
            self.point = None
            self.filter.reset()
            return None
        lo, hi = np.array(config.GAZE_CLAMP[:2]), np.array(config.GAZE_CLAMP[2:])
        self.point = np.clip(self.filter(self.raw, dt), lo, hi)
        return self.point


class DwellSelector:
    """
    注視点の近くの対象に吸い付き、DWELL_SEC 秒見つめ続けたら選ぶ。
    対象は (id, 画面座標, 半径) のリストで毎フレーム渡す（アイテムは動くので）
    """

    def __init__(self):
        self.focus = None           # 吸い付いている対象の id
        self.progress = 0.0         # 0〜1（1 で決定）
        self.focus_pos = None

    def reset(self):
        self.focus, self.progress, self.focus_pos = None, 0.0, None

    def update(self, gaze, targets, dt, exclude=None):
        """
        決定したらその id を返す（それ以外は None）。
        exclude：今選ばれている対象の id（見続けても、同じ対象を何度も選び直さないように除く）
        """
        best, best_d = None, None
        if gaze is not None:
            for tid, pos, radius in targets:
                if tid == exclude:
                    continue
                d = np.linalg.norm(np.asarray(pos) - gaze) - radius
                if d <= config.SNAP_PX and (best_d is None or d < best_d):
                    best, best_d, best_pos = tid, d, pos
        if best is None:
            # 視線が外れたら、たまったリングはすぐには消さずに減らす（一瞬の揺れで最初からにならないよう）
            self.progress = max(0.0, self.progress - dt / config.DWELL_DECAY_SEC)
            if self.progress == 0.0:
                self.focus, self.focus_pos = None, None
            return None
        if best != self.focus:
            self.focus, self.progress = best, 0.0
        self.focus_pos = best_pos
        self.progress = min(1.0, self.progress + dt / config.DWELL_SEC)
        if self.progress >= 1.0:
            chosen = self.focus
            self.reset()
            return chosen
        return None
