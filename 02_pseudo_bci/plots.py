"""
plots.py
--------
実験モード（experiment.py）の結果をグラフ（PNG）にする。レポートにそのまま貼れるように、
軸ラベル・凡例・エラーバー（標準偏差）をつける。
日本語フォント（Noto Sans CJK JP など）が見つかれば日本語で、なければ英語で文字を書く。

  fig1_neurons.png      : ニューロン数 vs 相関・RMSE（デコーダ別、ノイズの種類別）
  fig2_noise.png        : 独立ノイズの強さ vs RMSE（ニューロン数別、デコーダ別）
  fig3_dropout.png      : 脱落率 vs 相関・RMSE（PV / 線形・再学習なし / 線形・再キャリブレーション）
  fig4_assumptions.png  : PD のかたより・ゲインのばらつき vs RMSE（PV vs 線形）
  fig5_traces.png       : 代表的な条件での、本当の速度と推定速度の時系列
"""

import os

import matplotlib

matplotlib.use("Agg")                      # 画面なしで PNG を書く
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402
from matplotlib import font_manager        # noqa: E402

# ---- 日本語フォントがあれば使う ----
_JA_FONTS = ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic", "TakaoGothic", "Hiragino Sans", "Yu Gothic")
_available = {f.name for f in font_manager.fontManager.ttflist}
_ja = next((f for f in _JA_FONTS if f in _available), None)
JA = _ja is not None
if JA:
    plt.rcParams["font.family"] = _ja


def L(ja, en):
    """日本語フォントがあれば日本語、なければ英語のラベル"""
    return ja if JA else en


plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    "errorbar.capsize": 3,
})

# デコーダの色（カテゴリ。どのグラフでも同じデコーダは同じ色）
DEC_COLOR = {"PV": "#2a78d6", "LIN": "#eb6834", "LIN_recal": "#1baf7a"}
DEC_LABEL = {
    "PV": L("PV 法", "Population vector"),
    "LIN": L("線形デコーダ", "Linear decoder"),
    "LIN_recal": L("線形デコーダ（再キャリブレーション）", "Linear (recalibrated)"),
}
DEC_MARKER = {"PV": "o", "LIN": "s", "LIN_recal": "^"}
# ニューロン数（大きさの順序がある量）は 1 色の濃淡で
N_COLOR = {20: "#9ec3ee", 100: "#4a8ddf", 400: "#174a8c"}
TRUE_COLOR = "#2b2b29"
INK = "#2b2b29"
MUTED = "#6b6a66"


def _select(summary, experiment, **kw):
    out = [r for r in summary if r["experiment"] == experiment]
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def _series(rows, xkey, metric):
    rows = sorted(rows, key=lambda r: r[xkey])
    x = np.array([r[xkey] for r in rows], dtype=float)
    return x, np.array([r[f"{metric}_mean"] for r in rows]), np.array([r[f"{metric}_std"] for r in rows])


def _errline(ax, x, y, e, color, label, marker="o", ls="-"):
    ax.errorbar(x, y, yerr=e, color=color, label=label, marker=marker, ls=ls,
                markeredgecolor="white", markeredgewidth=1.0, elinewidth=1.2)


METRIC_LABEL = {
    "corr": L("相関係数（高いほど良い）", "Correlation (higher is better)"),
    "rmse": L("RMSE [画面/秒]（低いほど良い）", "RMSE [screen/s] (lower is better)"),
}


def _trials_note(summary):
    n = summary[0]["trials"] if summary else 0
    return L(f"各点：{n} 回の試行の平均 ± 標準偏差", f"mean ± SD over {n} trials")


# ============================================================
def fig_neurons(summary, path):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True)
    for row, nt in enumerate(("drift", "independent")):
        rows = _select(summary, "neurons", noise_type=nt)
        noise = rows[0]["noise"]
        for col, metric in enumerate(("corr", "rmse")):
            ax = axes[row, col]
            for dec in ("PV", "LIN"):
                x, y, e = _series([r for r in rows if r["decoder"] == dec], "n", metric)
                _errline(ax, x, y, e, DEC_COLOR[dec], DEC_LABEL[dec], DEC_MARKER[dec])
            ax.set_xscale("log")
            ax.set_xticks([10, 20, 50, 100, 200, 400])
            ax.set_xticklabels(["10", "20", "50", "100", "200", "400"])
            ax.set_ylabel(METRIC_LABEL[metric])
            title = L(f"ノイズ：{'drift（ゆっくり）' if nt == 'drift' else 'independent（独立）'} {noise:.0f} Hz",
                      f"Noise: {nt} {noise:.0f} Hz")
            ax.set_title(title, loc="left", fontsize=10, color=INK)
            if row == 1:
                ax.set_xlabel(L("ニューロン数（対数目盛）", "Number of neurons (log scale)"))
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(L("図 1　ニューロン数とデコード精度", "Fig. 1  Number of neurons vs decoding accuracy"),
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.98, 0.005, _trials_note(summary), ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def fig_noise(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, dec in zip(axes, ("PV", "LIN")):
        for n, color in N_COLOR.items():
            rows = _select(summary, "noise", n=n, decoder=dec)
            x, y, e = _series(rows, "noise", "rmse")
            _errline(ax, x, y, e, color, L(f"N = {n}", f"N = {n}"))
        ax.set_title(DEC_LABEL[dec], loc="left", fontsize=10, color=INK)
        ax.set_xlabel(L("独立ノイズの強さ σ [Hz]", "Independent noise σ [Hz]"))
    axes[0].set_ylabel(METRIC_LABEL["rmse"])
    axes[0].legend(title=L("ニューロン数", "Neurons"), loc="upper left")
    fig.suptitle(L("図 2　独立ノイズの強さと RMSE（ニューロンが多いほど平均化で強い）",
                   "Fig. 2  Independent noise vs RMSE"), x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.98, 0.005, _trials_note(summary), ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def fig_dropout(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric in zip(axes, ("corr", "rmse")):
        for dec in ("PV", "LIN", "LIN_recal"):
            x, y, e = _series(_select(summary, "dropout", decoder=dec), "dropout", metric)
            _errline(ax, x * 100, y, e, DEC_COLOR[dec],
                     DEC_LABEL[dec] if dec != "LIN" else L("線形デコーダ（再学習なし）", "Linear (not retrained)"),
                     DEC_MARKER[dec])
        ax.set_xticks([0, 10, 30, 50])
        ax.set_xlabel(L("脱落したニューロンの割合 [%]", "Dropped-out neurons [%]"))
        ax.set_ylabel(METRIC_LABEL[metric])
    axes[1].legend(loc="upper left")
    fig.suptitle(L("図 3　ニューロンの脱落と精度（N = 100、drift ノイズ 5 Hz）",
                   "Fig. 3  Neuron dropout vs accuracy (N = 100, drift 5 Hz)"),
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.98, 0.005, _trials_note(summary), ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def fig_assumptions(summary, path):
    """
    差が小さい相関係数も見やすいように、棒ではなく「点 + エラーバー」で描く
    （棒グラフは 0 から描かないと長さの比が誤解を招くため）
    """
    rows = _select(summary, "assumptions")
    groups = sorted({(r["pd_bias"], r["het"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, metric in zip(axes, ("corr", "rmse")):
        for k, dec in enumerate(("PV", "LIN")):
            y = [next(r for r in rows if (r["pd_bias"], r["het"]) == g and r["decoder"] == dec) for g in groups]
            xs = np.arange(len(groups)) + (k - 0.5) * 0.28
            ax.errorbar(xs, [r[f"{metric}_mean"] for r in y], yerr=[r[f"{metric}_std"] for r in y],
                        color=DEC_COLOR[dec], label=DEC_LABEL[dec], marker=DEC_MARKER[dec], ls="none",
                        markersize=8, markeredgecolor="white", markeredgewidth=1.0, elinewidth=1.5)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([L(f"±{h * 100:.0f}%", f"±{h * 100:.0f}%") for _, h in groups])
        ax.set_xlim(-0.5, len(groups) - 0.5)
        ax.axvline(2.5, color="#b5b4ae", lw=1, ls="--")
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_xlabel(L("ニューロンごとのゲイン（b0・m）のばらつき", "Per-neuron gain (b0, m) heterogeneity"))
        if metric == "rmse":
            ax.set_ylim(0, None)
        # グループ名（PD の分布）を上に書く
        for gx, text in ((1.0, L("PD 一様", "Uniform PD")), (4.0, L("PD 偏り（半分が右 ±20°）", "Biased PD (half near 0°)"))):
            ax.text(gx, 1.02, text, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=9.5, color=INK)
    axes[1].legend(loc="lower right")
    fig.suptitle(L("図 4　PV 法の前提が崩れた条件（N = 100、drift ノイズ 5 Hz）",
                   "Fig. 4  When the assumptions of the PV method break (N = 100, drift 5 Hz)"),
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.text(0.98, 0.005, _trials_note(summary), ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def fig_traces(traces, path, sec=6.0):
    from experiment import DT, WARMUP_BINS
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.0), sharex=True, sharey=True)
    cases = [("base", ("PV", "LIN"), L("既定の条件（N = 100、drift 5 Hz）", "Default (N = 100, drift 5 Hz)")),
             ("dropout", ("PV", "LIN", "LIN_recal"), L("30% 脱落後", "After 30% dropout"))]
    for row, (key, decs, title) in enumerate(cases):
        res, v, est = traces[key]
        T = int(sec / DT)
        t = (np.arange(T) + WARMUP_BINS) * DT
        for col, ax_name in enumerate(("x", "y")):
            ax = axes[row, col]
            ax.plot(t, v[:T, col], color=TRUE_COLOR, lw=2.6, alpha=0.35, label=L("本当の速度", "True velocity"))
            for dec in decs:
                lab = DEC_LABEL[dec] if dec != "LIN" or key == "base" else L("線形（再学習なし）", "Linear (not retrained)")
                ax.plot(t, est[dec][:T, col], color=DEC_COLOR[dec], lw=1.3, label=lab)
            ax.axhline(0, color="#b5b4ae", lw=0.8)
            metr = "　".join(f"{d}: r={res[d][0]:.2f}, RMSE={res[d][1]:.2f}" for d in decs)
            ax.set_title(f"{title}　v{ax_name}\n{metr}", loc="left", fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel(L("時間 [秒]", "Time [s]"))
            if col == 0:
                ax.set_ylabel(L(f"速度 [画面/秒]", "Velocity [screen/s]"))
    # 凡例は図の下にまとめる（線に重ならないように）。下段には 4 本すべての線がある
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(L("図 5　本当の速度とデコードした速度（評価用の動きの最初の 6 秒、シード 0）",
                   "Fig. 5  True vs decoded velocity (first 6 s of the evaluation movement)"),
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def make_all(summary, traces, out_dir, names):
    files = []
    jobs = [("neurons", "fig1_neurons.png", fig_neurons), ("noise", "fig2_noise.png", fig_noise),
            ("dropout", "fig3_dropout.png", fig_dropout), ("assumptions", "fig4_assumptions.png", fig_assumptions)]
    for name, fname, fn in jobs:
        if name in names:
            fn(summary, os.path.join(out_dir, fname))
            files.append(fname)
    fig_traces(traces, os.path.join(out_dir, "fig5_traces.png"))
    files.append("fig5_traces.png")
    return files
