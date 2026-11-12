"""
cortex_sim.py
-------------
左半球の運動野（手の領域）の擬似ニューロン集団を 20 ms ごとに動かし、表示用の値をまとめて持つ。
（02_pseudo_bci の main.py の sim_step と renderer.py の on_spikes を、ゲームで使いやすい形にまとめたもの）

  step(v)  : 手の速度 v → スパイク（neuron_model）→ 発火率の推定（decoder.RateEstimator）
             → PV 法の推定速度、表示用の flash（スパイクで光る）・activity（発火率の高さ）、ラスター画像
  motor_drive : 集団がどれだけベースラインより発火しているか（0〜1）。脳の運動野の光らせ方に使う
"""

import math

import numpy as np

import config
from decoder import PopulationVectorDecoder, RateEstimator
from gl_utils import hue_to_rgb
from neuron_model import NeuronPopulation


class CortexSim:
    def __init__(self, n, rng, noise=config.NOISE_STD, noise_type=config.NOISE_TYPE):
        self.rng = rng
        self.pop = NeuronPopulation(n, rng)
        self.pop.noise_std = noise
        self.pop.noise_type = noise_type
        self.rates = RateEstimator(n)
        self.pv = PopulationVectorDecoder(self.pop)
        self.rate_hat = self.rates.rate_hat
        self.counts = np.zeros(n, dtype=np.int32)
        self.flash = np.zeros(n)
        self.activity = np.zeros(n)
        self.motor_drive = 0.0
        bins = int(round(config.RASTER_SEC / config.DT))
        self.raster_bg = np.array([14, 16, 26, 255], dtype=np.uint8)
        self.raster = np.empty((n, bins, 4), dtype=np.uint8)
        self.raster[:] = self.raster_bg
        self.raster_colors = (hue_to_rgb(self.pop.pd_angle / (2 * np.pi))[::-1] * 255).astype(np.uint8)

    @property
    def n(self):
        return self.pop.n

    def step(self, v):
        self.counts = self.pop.step(v)
        self.rate_hat = self.rates.step(self.counts)
        self.pv.step(self.rate_hat)
        fired = self.counts > 0
        self.flash[fired] = 1.0
        act = np.clip((self.rate_hat - config.BASE_RATE) / config.MODULATION, 0.0, 1.0)
        self.activity += 0.25 * (act - self.activity)
        alive = self.pop.alive
        self.motor_drive = float(np.clip(config.MOTOR_GLOW_GAIN * act[alive].mean(), 0, 1)) if alive.any() else 0.0
        # ラスター：1 列左へずらし、右端に今回のスパイクを書く（ニューロン 0 が一番下）
        self.raster[:, :-1] = self.raster[:, 1:]
        col = np.empty((self.n, 4), dtype=np.uint8)
        col[:] = self.raster_bg
        col[(~alive)[::-1]] = (34, 34, 38, 255)
        rows = fired[::-1]
        col[rows, :3] = self.raster_colors[rows]
        self.raster[:, -1] = col
        return self.rate_hat

    def decay(self, dt):
        """描画のフレームごとに、スパイクの光を弱める"""
        self.flash *= math.exp(-dt / config.SPIKE_FLASH_TAU)
