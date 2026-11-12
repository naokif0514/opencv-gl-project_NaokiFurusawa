"""
session_logger.py
-----------------
集中度の推移を一定間隔で記録し、終了時に CSV ファイルとして保存する。
保存先: logs/focus_YYYYmmdd_HHMMSS.csv（起動した時刻をファイル名にする）
"""

import csv
import os
from datetime import datetime

import config

COLUMNS = [
    "time", "elapsed_sec", "mode", "focus", "focus_raw", "absent",
    "blink_rate", "ear_avg", "yaw", "pitch", "head_angle", "gaze_std",
    "score_blink", "score_eye_open", "score_head", "score_gaze", "growth_sec",
]


class SessionLogger:
    def __init__(self, base_dir):
        self.start = datetime.now()
        self.log_dir = os.path.join(base_dir, config.LOG_DIR)
        self.rows = []
        self._last_elapsed = -1e9

    def record(self, elapsed, mode, focus, raw, metrics, scores, growth_sec=0.0):
        """LOG_INTERVAL_SEC ごとに 1 行記録する（それより短い間隔の呼び出しは無視）"""
        if elapsed - self._last_elapsed < config.LOG_INTERVAL_SEC:
            return
        self._last_elapsed = elapsed

        def r(v, n=4):
            return "" if v is None else round(float(v), n)

        m = metrics
        s = scores or {}
        self.rows.append([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            r(elapsed, 2), mode, r(focus, 2), r(raw, 2),
            "" if m is None else int(m.absent),
            r(m.blink_rate if m else None, 2), r(m.ear_avg if m else None),
            r(m.yaw if m else None, 2), r(m.pitch if m else None, 2),
            r(m.head_angle if m else None, 2), r(m.gaze_std if m else None, 5),
            r(s.get("blink"), 3), r(s.get("eye_open"), 3), r(s.get("head"), 3), r(s.get("gaze"), 3),
            r(growth_sec, 2),
        ])

    def save(self):
        """記録した内容を CSV に書き出し、そのパスを返す"""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"focus_{self.start.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(COLUMNS)
            writer.writerows(self.rows)
        return path
