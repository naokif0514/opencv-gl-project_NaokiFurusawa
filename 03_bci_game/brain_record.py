"""
brain_record.py
---------------
結果画面の「プレイ中の脳の活動」の振り返り用に、プレイ中の脳の側の記録をためる。

  ・領域ごとの使われ方：各領域の光の強さ（brain.glow）をプレイ時間で平均したもの
    （手を動かす・握る → 運動野、握る → 体性感覚野、視線で選ぶ → 前頭眼野・視覚野）
  ・ラスターの一部：最後に物をつかんだ瞬間（なければ終了時）のラスター画像（直近 RASTER_SEC 秒）
  ・デコード精度の推移：ACCURACY_LOG_SEC 秒ごとの (プレイ時間, 相関係数, RMSE)
  ・イベントの時刻：電極劣化、再同期の開始・終了
"""

import numpy as np

import config
from brain_model import REGION_NAMES, REGION_ID


class BrainRecord:
    def __init__(self):
        self.glow_sum = np.zeros(6)
        self.time = 0.0
        self.accuracy = []          # (t, corr, rmse)
        self.marks = []             # (t, "degrade" / "recal_start" / "recal_end")
        self.raster = None
        self.next_acc = 0.0

    def update(self, dt, glow, metrics, play_time):
        self.glow_sum += np.asarray(glow) * dt
        self.time += dt
        if play_time >= self.next_acc:
            metrics.compute()
            self.accuracy.append((play_time, metrics.corr, metrics.rmse))
            self.next_acc = play_time + config.ACCURACY_LOG_SEC

    def mark(self, play_time, what):
        self.marks.append((play_time, what))

    def snapshot_raster(self, raster):
        self.raster = raster.copy()

    def usage(self):
        """領域ごとの平均の光の強さ（0〜1）。いちばん使った領域が 1 になるようにそろえる"""
        avg = self.glow_sum / max(self.time, 1e-6)
        top = max(avg[1:].max(), 1e-6)
        return {name: float(avg[REGION_ID[name]] / top) for name in REGION_NAMES}
