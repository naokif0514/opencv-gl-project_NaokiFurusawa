"""
session_logger.py
-----------------
シミュレーションの 1 ステップ（20 ms）ごとに、時刻・本当の速度・推定速度・誤差などを記録し、
終了時に CSV に保存する（01_focus_brain の仕組みを流用）。
保存先: logs/bci_YYYYmmdd_HHMMSS.csv（起動した時刻をファイル名にする）
"""

import csv
import os
from datetime import datetime

import numpy as np

import config

COLUMNS = [
    "time_sec", "mode", "hand_detected",
    "true_vx", "true_vy", "decoded_vx", "decoded_vy", "error",
    "true_x", "true_y", "cursor_x", "cursor_y",
    "n_neurons", "n_alive", "noise_hz", "total_spikes",
]


class SessionLogger:
    def __init__(self, base_dir):
        self.start = datetime.now()
        self.log_dir = os.path.join(base_dir, config.LOG_DIR)
        self.rows = []

    def record(self, t, mode, hand, true_v, v_hat, cursor, n, n_alive, noise, total_spikes):
        err = float(np.linalg.norm(np.asarray(true_v) - np.asarray(v_hat)))
        pos = hand.pos if hand.pos is not None else (None, None)

        def r(v, k=4):
            return "" if v is None else round(float(v), k)

        self.rows.append([
            r(t, 3), mode, int(hand.detected),
            r(true_v[0]), r(true_v[1]), r(v_hat[0]), r(v_hat[1]), r(err),
            r(pos[0]), r(pos[1]), r(cursor[0]), r(cursor[1]),
            n, n_alive, r(noise, 2), int(total_spikes),
        ])

    def save(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"bci_{self.start.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(COLUMNS)
            w.writerows(self.rows)
        return path
