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

■ 2. 安定化（同じ点を見つめているときに、注視点がその点で止まるように）
  Web カメラの視線推定は 1 フレームごとに大きく揺れる（実測：虹彩の位置の揺れが、画面上で数十画素）。
  次の 3 段で揺れを抑える。
  (a) 特徴量の平均：直近 GAZE_FEAT_WINDOW 秒（約 0.3 秒、9 フレームほど）の特徴量を平均してから画面座標に直す
      （実測で、1 フレームごとの揺れが約 1/5 になる）
  (b) One Euro フィルタ（Casiez ら 2012）：ゆっくり動くときは強くなめらかにし、速く動くときは弱くする
  (c) 固視フィルタ（FixationFilter）：人の目は「止まる（固視）」と「跳ぶ（サッカード）」をくり返す。
      注視点が GAZE_FIX_RADIUS 以内に留まっている間は固視とみなし、表示する点をほぼ止める（ゆっくりだけ追う）。
      その範囲の外に GAZE_FIX_BREAK 秒以上出たらサッカードとみなし、新しい位置へ跳ぶ
      （視線研究で固視とサッカードを分ける「分散のしきい値による方法（I-DT）」の考え方）
  (d) 表示・選択に使う点は、(c) の点へ時定数 GAZE_GLIDE_SEC で滑らかに近づける（跳ぶときも滑って見える）

■ 3. ずれの補正（ドリフト補正）
  Web カメラの視線推定は、時間がたつと頭の位置の変化などで全体がずれてくる。
  ドウェル選択で対象が決まったとき、「ユーザーはその対象を見ていた」とみなし、
  見つめていた間の注視点と対象の位置の差の一部（GAZE_DRIFT_GAIN）を、以後の注視点の補正（offset）に加える。
  使えば使うほど、ずれが小さくなる（視線入力の暗黙のキャリブレーション）。

■ 4. 吸い付き + ドウェル選択（DwellSelector）
  Web カメラでは精度が低い（100 画素ほどずれる）前提で、注視点から SNAP_PX 以内で最も近い対象に「吸い付く」。
  一度吸い付いたら、別の対象がはっきり近く（FOCUS_SWITCH_PX 以上）ならないかぎり離れない（ヒステリシス）。
  同じ対象を DWELL_SEC 秒（約 1 秒）見つめ続けたら決定する（ドウェル選択。視線入力の基本的な方法で、
  まばたきやクリックなしで選べる）。見つめている間はリングがたまっていく。
"""

import math
from collections import deque

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
    """
    視線キャリブレーション（視線同期。GAZE_CALIB_POINTS = 9 なら 3×3 の 9 点で 2 次式、5 点なら 1 次式）と、そのあとの確認。
      1. 学習（"calib"）：points を順に見つめてもらい、特徴量 → 画面の位置の線形モデルを作る
      2. 確認（"check"）：学習に使っていない check_points を見つめてもらい、モデルで予測した位置との
         平均のずれ（check_error）を測る。学習した点での誤差（residual）は実際より小さく出るので、こちらで判定する
    best_model / best_error には、やり直したなかで一番よかった結果を残す
    """

    def __init__(self, points, check_points=()):
        self.points = [np.asarray(p, dtype=np.float64) for p in points]   # 画面座標（画素）
        self.check_points = [np.asarray(p, dtype=np.float64) for p in check_points]
        self.active = False
        self.phase = "calib"
        self.model = None           # (Ax, Ay)：x = Ax · [1, f0, f2]、y = Ay · [1, f1, f3]
        self.residual = None        # 学習した点での平均のずれ [画素]
        self.check_error = None     # 確認の点での平均のずれ [画素]
        self.check_dx, self.check_dy = [], []   # 確認の点ごとの左右・上下のずれ [画素]（ログ用）
        self.dropped_point = None   # 外れ値として除いた学習の点の番号（ログ用）
        self.best_model, self.best_error = None, None
        self.tries = 0              # 確認で失敗した回数

    def start(self):
        self.active = True
        self.phase = "calib"
        self.index = 0
        self.t = 0.0
        self.samples = [[] for _ in self.points]
        self.check_samples = [[] for _ in self.check_points]

    def reset_tries(self):
        self.best_model, self.best_error, self.tries = None, None, 0

    def cancel(self):
        self.active = False

    @property
    def _targets(self):
        return self.points if self.phase == "calib" else self.check_points

    @property
    def _settle(self):
        return config.GAZE_CALIB_SETTLE if self.phase == "calib" else config.GAZE_CHECK_SETTLE

    @property
    def _collect(self):
        return config.GAZE_CALIB_COLLECT if self.phase == "calib" else config.GAZE_CHECK_COLLECT

    @property
    def current_point(self):
        return self._targets[self.index] if self.active else None

    @property
    def collecting(self):
        return self.active and self.t >= self._settle

    @property
    def settle_progress(self):
        return min(1.0, self.t / self._settle)

    @property
    def collect_progress(self):
        return min(1.0, max(0.0, (self.t - self._settle) / self._collect))

    @property
    def point_progress(self):
        """今の点での進み具合 0〜1（演出用）"""
        return min(1.0, self.t / (self._settle + self._collect))

    def step(self, feat, dt):
        """毎フレーム呼ぶ。学習と確認が終わったら True を返す（判定は呼び出し側で check_error を見る）"""
        if not self.active:
            return False
        self.t += dt
        buf = self.samples if self.phase == "calib" else self.check_samples
        if self.collecting and feat is not None:
            buf[self.index].append(np.asarray(feat))
        if self.t >= self._settle + self._collect:
            if len(buf[self.index]) < 3:       # 顔が見えていなかったら、同じ点をやり直す
                self.t = 0.0
                buf[self.index] = []
                return False
            self.index += 1
            self.t = 0.0
            if self.phase == "calib" and self.index >= len(self.points):
                self._fit()
                if not self.check_points:
                    self.check_error = self.residual
                    return self._finish()
                self.phase, self.index = "check", 0
            elif self.phase == "check" and self.index >= len(self.check_points):
                F = np.array([np.median(self._clean(s), axis=0) for s in self.check_samples])
                pred = np.array([self.predict(f) for f in F])
                diff = pred - np.array(self.check_points)
                self.check_dx, self.check_dy = diff[:, 0].tolist(), diff[:, 1].tolist()   # ログ用（左右・上下のずれ）
                self.check_error = float(np.mean(np.linalg.norm(diff, axis=1)))
                return self._finish()
        return False

    def _finish(self):
        self.active = False
        if self.best_error is None or self.check_error < self.best_error:
            self.best_model, self.best_error = self.model, self.check_error
        if self.check_error > config.GAZE_CHECK_MAX_ERR:
            self.tries += 1
        return True

    @property
    def passed(self):
        return self.check_error is not None and self.check_error <= config.GAZE_CHECK_MAX_ERR

    @property
    def can_skip(self):
        """ENTER で「今までで一番よかった結果」で進めるか"""
        return self.tries >= config.GAZE_CHECK_SKIP_AFTER and self.best_model is not None

    def use_best(self):
        self.model, self.check_error = self.best_model, self.best_error

    def _clean(self, samples):
        """1 点ぶんの特徴量から外れ値を除く（中央値から、中央値からのずれの中央値の GAZE_OUTLIER_MAD 倍以上離れたもの）"""
        S = np.asarray(samples)
        if len(S) < 5:
            return S
        med = np.median(S, axis=0)
        mad = np.maximum(np.median(np.abs(S - med), axis=0), 1e-6)
        keep = np.all(np.abs(S - med) <= config.GAZE_OUTLIER_MAD * 1.4826 * mad, axis=1)
        return S[keep] if keep.sum() >= 3 else S

    def _fit(self):
        """
        特徴量 → 画面の位置の変換式を学習する。
          ・各点の特徴量から外れ値を除いて中央値をとる（点が移った直後の GAZE_CALIB_SETTLE 秒はもともと集めていない）
          ・9 点なら 2 次式、5 点なら 1 次式（リッジ回帰。2 次の項は強めに正則化）
          ・9 点のとき、ずれがほかより大きく外れた点は 1 個まで除いて学び直す
        """
        F = np.array([np.median(self._clean(s), axis=0) for s in self.samples])
        P = np.array(self.points)
        use = np.ones(len(F), dtype=bool)
        self.model = _fit_model(F, P)
        if len(F) >= 9:
            err = np.linalg.norm(np.array([self.predict(f) for f in F]) - P, axis=1)
            worst = int(np.argmax(err))
            if err[worst] > config.GAZE_POINT_OUTLIER * max(np.median(err), 1.0):
                use[worst] = False
                self.model = _fit_model(F[use], P[use])
        self.dropped_point = None if use.all() else int(np.argmin(use))
        pred = np.array([self.predict(f) for f in F[use]])
        self.residual = float(np.mean(np.linalg.norm(pred - P[use], axis=1)))

    def predict(self, feat):
        if self.model is None or feat is None:
            return None
        return _predict_model(self.model, feat)


def _design(F, quad):
    """
    変換式の説明変数。特徴量 f = [虹彩の左右, 上下, 顔の向きの左右, 上下] を標準化した z から：
      x（左右）：1, z0, z2（+ 2 次式なら z0², z0·z1, z1²）
      y（上下）：1, z1, z3（+ 2 次式なら z1², z0·z1, z0²）
    """
    one = np.ones(len(F))
    cols_x = [one, F[:, 0], F[:, 2]]
    cols_y = [one, F[:, 1], F[:, 3]]
    if quad:
        cols_x += [F[:, 0] ** 2, F[:, 0] * F[:, 1], F[:, 1] ** 2]
        cols_y += [F[:, 1] ** 2, F[:, 0] * F[:, 1], F[:, 0] ** 2]
    return np.column_stack(cols_x), np.column_stack(cols_y)


def _fit_model(F, P):
    quad = len(F) >= 9
    mu, sd = F.mean(axis=0), np.maximum(F.std(axis=0), 1e-6)
    Z = (F - mu) / sd
    model = {"mu": mu, "sd": sd, "quad": quad}
    for axis, X in zip(("ax", "ay"), _design(Z, quad)):
        k = X.shape[1]
        reg = np.diag([0.0] + [config.GAZE_RIDGE] * 2 + [config.GAZE_RIDGE_QUAD] * (k - 3))
        model[axis] = np.linalg.solve(X.T @ X + reg, X.T @ P[:, 0 if axis == "ax" else 1])
    return model


def _predict_model(model, feat):
    z = ((np.asarray(feat, dtype=np.float64) - model["mu"]) / model["sd"])[None, :]
    X, Y = _design(z, model["quad"])
    return np.array([float(X[0] @ model["ax"]), float(Y[0] @ model["ay"])])


class FixationFilter:
    """固視フィルタ：小さな範囲に留まっている間は点をほぼ止め、範囲の外に一定時間出たら新しい位置へ跳ぶ"""

    def __init__(self, radius=config.GAZE_FIX_RADIUS, break_sec=config.GAZE_FIX_BREAK,
                 follow_tau=config.GAZE_FIX_FOLLOW):
        self.radius, self.break_sec, self.follow_tau = radius, break_sec, follow_tau
        self.reset()

    def reset(self):
        self.fix = None
        self.out_t = 0.0
        self.out_pts = []

    def __call__(self, p, dt):
        p = np.asarray(p, dtype=np.float64)
        if self.fix is None:
            self.fix = p.copy()
            return self.fix
        if np.linalg.norm(p - self.fix) <= self.radius:
            self.out_t, self.out_pts = 0.0, []
            self.fix += (p - self.fix) * (1.0 - math.exp(-dt / self.follow_tau))   # 範囲内ではゆっくり追う
        else:
            self.out_t += dt
            self.out_pts.append(p)
            if self.out_t >= self.break_sec:                  # サッカード：外に出ていた間の平均の位置へ跳ぶ
                self.fix = np.mean(self.out_pts, axis=0)
                self.out_t, self.out_pts = 0.0, []
        return self.fix


class GazeTracker:
    """
    特徴量 → 画面上の注視点（平均 → キャリブレーションの 1 次式 → ずれの補正 → One Euro → 固視フィルタ）。
    テストモードではマウスの位置をそのまま使う
    """

    def __init__(self, calib_points, check_points=()):
        self.calib = GazeCalibration(calib_points, check_points)
        self.filter = OneEuroFilter()
        self.fixation = FixationFilter()
        self.feats = deque()        # (時刻, 特徴量)：平均をとるための直近の特徴量
        self.offset = np.zeros(2)   # ずれの補正（ドリフト補正）[画素]
        self.recent = deque()       # (時刻, 補正後・固視フィルタ前の注視点)：ずれの補正に使う
        self.point = None           # 表示・選択に使う注視点（画素）。分からなければ None
        self.raw = None             # 1 次式で直しただけの点（平均・平滑化の前）
        self.time = 0.0

    @property
    def calibrated(self):
        return self.calib.model is not None

    def reset_filters(self):
        self.filter.reset()
        self.fixation.reset()
        self.feats.clear()
        self.recent.clear()

    def reset_offset(self):
        self.offset = np.zeros(2)

    def update(self, feat, dt, mouse=None, feat_time=None):
        """
        feat_time：その特徴量を計算したカメラのフレームの時刻。同じフレームを何度も数えないために使う
        （描画はカメラより速く回るので、同じ特徴量が何回も渡される）
        """
        self.time += dt
        if mouse is not None:                       # テストモード：マウスの位置 = 視線
            self.raw = np.asarray(mouse, dtype=np.float64)
            self.point = self.raw.copy()
            return self.point
        if feat is None or not self.calibrated:
            self.point = None
            self.reset_filters()
            return None
        t = feat_time if feat_time is not None else self.time
        if not self.feats or self.feats[-1][0] != t:
            self.feats.append((t, np.asarray(feat, dtype=np.float64)))
        while self.feats and t - self.feats[0][0] > config.GAZE_FEAT_WINDOW:
            self.feats.popleft()
        mean_feat = np.mean([f for _, f in self.feats], axis=0)          # (a) 特徴量の平均
        self.raw = self.calib.predict(feat)
        p = self.calib.predict(mean_feat) + self.offset
        lo, hi = np.array(config.GAZE_CLAMP[:2]), np.array(config.GAZE_CLAMP[2:])
        p = np.clip(self.filter(p, dt), lo, hi)                          # (b) One Euro フィルタ
        self.recent.append((self.time, p.copy()))
        while self.recent and self.time - self.recent[0][0] > config.DWELL_SEC:
            self.recent.popleft()
        fix = self.fixation(p, dt)                                       # (c) 固視フィルタ
        # (d) 表示・選択に使う点は、固視フィルタの点へ滑らかに近づける（跳んだときも、ぱっと飛ばずに滑る）
        if self.point is None:
            self.point = fix.copy()
        else:
            self.point = self.point + (fix - self.point) * (1.0 - math.exp(-dt / config.GAZE_GLIDE_SEC))
        return self.point

    def correct(self, target):
        """
        ドリフト補正：対象が選ばれたとき、見つめていた間（直近 DWELL_SEC 秒）の注視点の平均と
        対象の位置の差の一部を、以後の補正に加える
        """
        if not self.recent:
            return
        mean = np.mean([p for _, p in self.recent], axis=0)
        err = np.asarray(target, dtype=np.float64) - mean
        self.offset = np.clip(self.offset + config.GAZE_DRIFT_GAIN * err, -config.GAZE_DRIFT_MAX,
                              config.GAZE_DRIFT_MAX)
        self.fixation.reset()


class DwellSelector:
    """
    注視点の近くの対象に吸い付き、DWELL_SEC 秒見つめ続けたら選ぶ。
    対象は (id, 画面座標, 半径) または (id, 画面座標, 半径, 設定) のリストで毎フレーム渡す（アイテムは動くので）。
    設定（dict）で、その対象だけ吸い付く範囲 "snap"・吸い付いたままにする範囲 "keep"・見つめる時間 "dwell" を変えられる
    （チャレンジのライバル：物を見ているつもりで誤って止めないよう、範囲を狭く、時間を長くする）
    """

    def __init__(self):
        self.focus = None           # 吸い付いている対象の id
        self.progress = 0.0         # 0〜1（1 で決定）
        self.focus_pos = None

    def reset(self):
        self.focus, self.progress, self.focus_pos = None, 0.0, None

    def update(self, gaze, targets, dt, current=None, dwell_sec=None):
        """
        決定したらその id を返す（それ以外は None）。
        current：今の行き先の id。候補から外さずに残し、CURRENT_PULL_PX だけ近いものとして扱う。
                 視線が今の行き先に吸い付いているあいだは何も起きない（隣の物に乗り換えない・選び直さない）
        dwell_sec：決定までに見つめる時間（行き先を乗り換えるときは長くする）
        """
        dwell_sec = config.DWELL_SEC if dwell_sec is None else dwell_sec
        best, best_d, best_pos = None, None, None
        cur_d, cur_pos, cur_opt = None, None, {}
        opts = {}
        if gaze is not None:
            for t in targets:
                tid, pos, radius = t[:3]
                opt = t[3] if len(t) > 3 else {}
                opts[tid] = opt
                d = np.linalg.norm(np.asarray(pos) - gaze) - radius
                if tid == current:
                    d -= config.CURRENT_PULL_PX
                if tid == self.focus:
                    cur_d, cur_pos, cur_opt = d, pos, opt
                if d <= opt.get("snap", config.SNAP_PX) and (best_d is None or d < best_d):
                    best, best_d, best_pos = tid, d, pos
            # ヒステリシス：吸い付いている対象は、少し離れても（SNAP_PX + FOCUS_KEEP_PX まで）、
            # 別の対象がはっきり近く（FOCUS_SWITCH_PX 以上）ならないかぎり離さない
            keep = cur_opt.get("snap", config.SNAP_PX) + cur_opt.get("keep", config.FOCUS_KEEP_PX)
            if cur_d is not None and cur_d <= keep:
                if best is None or best == self.focus or best_d > cur_d - config.FOCUS_SWITCH_PX:
                    best, best_d, best_pos = self.focus, cur_d, cur_pos
        if best is None or best == current:
            # 視線が外れた・今の行き先を見ている（何も決定しない）。
            # たまったリングはすぐには消さずに減らす（一瞬の揺れで最初からにならないよう）
            self.progress = max(0.0, self.progress - dt / config.DWELL_DECAY_SEC)
            if self.progress == 0.0:
                self.focus, self.focus_pos = None, None
            return None
        if best != self.focus:
            self.focus, self.progress = best, 0.0
        self.focus_pos = best_pos
        self.progress = min(1.0, self.progress + dt / opts.get(best, {}).get("dwell", dwell_sec))
        if self.progress >= 1.0:
            chosen = self.focus
            self.reset()
            return chosen
        return None
