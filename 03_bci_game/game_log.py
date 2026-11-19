"""
game_log.py
-----------
プレイログ。描画の 1 フレームごとに 1 行（時刻、状態、注視点、本当の手の速度、デコード結果、把持、
集めた数・スコアとその変化・コンボ、チャレンジのライバルの位置・状態・スコア、イベント）を記録し、
終了時に logs/game_YYYYmmdd_HHMMSS.csv に保存する（02_pseudo_bci/session_logger.py と同じ仕組み）。
"""

import csv
import os
from datetime import datetime

import config

COLUMNS = ["time_sec", "state", "mode", "gaze_x", "gaze_y", "dwell_target", "dwell_progress",
           "hand_detected", "true_vx", "true_vy", "decoded_vx", "decoded_vy", "decoder",
           "true_grip", "decoded_grip", "gripper_closed", "holding", "holding_type", "robot_x", "robot_y",
           "n_alive", "n_neurons", "degrade_count", "collected", "score", "score_delta", "combo", "time_left",
           # チャレンジ（ライバルとの対戦）：ライバルの位置・状態（行動 / frozen）・スコア・持っている物・止めた回数
           "rival_level", "rival_x", "rival_y", "rival_state", "rival_score", "rival_holding", "freezes",
           # 手の映る大きさ（手首 → 中指の付け根、画像の幅に対する割合。速度をこれで割る。config.HAND_REF_SIZE を参照）
           "hand_size",
           # 視線が吸い付いている対象の画面の位置（gaze_x, gaze_y と比べて、ずれの向きを調べる）
           "focus_x", "focus_y",
           # 視線の特徴量（vision.gaze_features）：黒目の左右・上下（目の中の位置 / 目の幅）、顔の向き（鼻先のずれ / 両目の間の距離）
           "iris_x", "iris_y", "head_yaw", "head_pitch",
           "event"]


class GameLogger:
    def __init__(self, base_dir):
        self.start = datetime.now()
        self.path = os.path.join(base_dir, config.LOG_DIR, f"game_{self.start.strftime('%Y%m%d_%H%M%S')}.csv")
        self.rows = []

    def record(self, **kw):
        def r(v, k=4):
            if v is None:
                return ""
            if isinstance(v, float):
                return round(v, k)
            return v
        self.rows.append([r(kw.get(c)) for c in COLUMNS])

    def save(self):
        if not self.rows:
            return None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(COLUMNS)
            w.writerows(self.rows)
        return self.path
