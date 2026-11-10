"""
focus_estimator.py
------------------
カメラ処理で得られた各指標（まばたき・目の開き具合・顔の向き・視線の安定性）を
0〜1 のスコアに正規化し、重み付きで合計して「集中度」0〜100 を算出する。

    集中度(生) = 100 × Σ(w_i × s_i) / Σ(w_i)       （s_i: 各スコア, w_i: 重み）
    集中度     = 指数移動平均（EMA）でなめらかにした値

・顔が検出されない（離席）ときは、集中度(生) = 0 とする。
・起動直後などで計算できない指標は、その項を合計から除外する（重みも除く）。
"""

import math
from dataclasses import dataclass, field

import config


@dataclass
class Metrics:
    """カメラ処理で 1 フレームごとに計算される指標（未計算の値は None）"""
    face_detected: bool = False       # このフレームで顔が見つかったか
    absent: bool = True               # 一定時間顔が見つからない＝離席
    ear: float | None = None          # 現在の EAR（両目の平均）
    ear_avg: float | None = None      # EAR の移動平均（目の開き具合）
    blink_rate: float | None = None   # まばたき回数/分（直近 60 秒）
    blink_count: int = 0              # 起動からのまばたき総数
    yaw: float | None = None          # 顔の左右の向き [度]（基準からの差）
    pitch: float | None = None        # 顔の上下の向き [度]（基準からの差）
    head_angle: float | None = None   # 正面からのズレ角 [度]
    gaze_std: float | None = None     # 虹彩位置の標準偏差（目の幅で正規化）
    camera_fps: float = 0.0           # カメラ処理のフレームレート


@dataclass
class FocusState:
    """集中度の推定結果"""
    focus: float = 0.0                # なめらかにした集中度 0〜100
    raw: float = 0.0                  # その瞬間の集中度 0〜100
    scores: dict = field(default_factory=dict)   # 各指標のスコア 0〜1（None は未計算）


def linear_score(value, good, bad):
    """
    値を 0〜1 のスコアに線形変換する。
    good 側にあるほど 1、bad 側にあるほど 0。good > bad でも good < bad でも使える。
    """
    if value is None:
        return None
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, t))


class FocusEstimator:
    """各指標から集中度を計算し、EMA でなめらかにするクラス"""

    def __init__(self):
        self.state = FocusState()
        self._last_time = None

    def compute_scores(self, m: Metrics) -> dict:
        """各指標を 0〜1 のスコアに正規化する"""
        return {
            # まばたきが少ないほど高スコア
            "blink": linear_score(m.blink_rate, config.BLINK_RATE_GOOD, config.BLINK_RATE_BAD),
            # 目がしっかり開いているほど高スコア
            "eye_open": linear_score(m.ear_avg, config.EAR_OPEN_GOOD, config.EAR_OPEN_BAD),
            # 顔が正面を向いているほど高スコア
            "head": linear_score(m.head_angle, config.HEAD_ANGLE_GOOD, config.HEAD_ANGLE_BAD),
            # 視線が安定しているほど高スコア
            "gaze": linear_score(m.gaze_std, config.GAZE_STD_GOOD, config.GAZE_STD_BAD),
        }

    def update(self, m: Metrics, now: float) -> FocusState:
        """
        新しい指標を受け取り、集中度を更新して返す。
        now は time.monotonic() の値（EMA の時間間隔の計算に使う）。
        """
        if m.absent:
            # 離席中は集中度 0 に向かって下がっていく
            scores = {k: None for k in config.WEIGHTS}
            raw = 0.0
        else:
            scores = self.compute_scores(m)
            # 計算できた指標だけで重み付き平均をとる
            num = 0.0
            den = 0.0
            for key, w in config.WEIGHTS.items():
                s = scores.get(key)
                if s is not None:
                    num += w * s
                    den += w
            raw = 100.0 * num / den if den > 0 else self.state.raw

        # 指数移動平均：alpha = 1 - exp(-dt / tau)
        #   フレームレートが変わっても同じ速さでなめらかになるよう、時間間隔から係数を決める
        if self._last_time is None:
            focus = raw
        else:
            dt = max(0.0, now - self._last_time)
            alpha = 1.0 - math.exp(-dt / config.FOCUS_SMOOTHING_TAU)
            focus = self.state.focus + alpha * (raw - self.state.focus)
        self._last_time = now

        self.state = FocusState(focus=focus, raw=raw, scores=scores)
        return self.state
