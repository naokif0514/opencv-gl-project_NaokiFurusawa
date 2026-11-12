"""
experiment.py
-------------
自動の実験モード（画面なし）。決まった手の動きを入力し、条件を自動で変えながらデコード精度を測る。

使い方:
    cd 02_pseudo_bci
    ../.venv/bin/python main.py --experiment                 # 各条件 20 回（約 1 分）
    ../.venv/bin/python main.py --experiment --trials 50     # 試行回数を変える
    ../.venv/bin/python experiment.py --only neurons,dropout # 一部の実験だけ（直接起動も可）

■ 1 回の試行（run_trial）の流れ
  1. ニューロン集団を作る（ニューロン数・ノイズ・ばらつき・PD のかたよりは条件で決める）
  2. キャリブレーション：画面のキャリブレーションと同じターゲットの動き（8 方向の往復 + 円運動）を
     「手の動き」として入力し、線形デコーダを学習する。手はターゲットを完全には追えないので、
     ゆっくりした小さなずれ（追従誤差）を加える
  3. 脱落：条件の割合だけニューロンを脱落させる。脱落があれば、もう一度キャリブレーションして
     「再キャリブレーションした線形デコーダ」も作る（再学習なしの線形デコーダも残しておく）
  4. 評価：キャリブレーションとは別の手の動き（リサージュ曲線 10 秒 + ランダムな滑らかな動き 10 秒）
     を入力し、PV 法・線形デコーダ（・再キャリブレーション版）で速度を推定して、相関係数と RMSE を測る
     → 学習データと評価データが分かれているので、線形デコーダの「覚えただけ」の精度にはならない

■ 実験（EXPERIMENTS）
  neurons     : ニューロン数（10〜400）× ノイズの種類（drift / independent、どちらも 30 Hz）
  noise       : 独立ノイズの強さ（0〜100 Hz）× ニューロン数（20 / 100 / 400）
  dropout     : 脱落率（0 / 10 / 30 / 50 %）。PV 法 / 線形（再学習なし）/ 線形（再キャリブレーション）
  assumptions : PV 法の前提が崩れた条件。PD の分布（一様 / 半分が右方向に集中）× ゲインのばらつき（±15/50/80 %）
  （ほかの条件は既定値：N = 100、drift ノイズ 5 Hz、ばらつき ±15 %、PD 一様、脱落なし）

各条件をシードを変えて --trials 回試行し（シード i はどの条件でも同じ乱数の使い方をするので、
条件どうしの差が見やすい）、平均と標準偏差を求める。
結果は results/実行日時/ に CSV（trials.csv, summary.csv, traces.csv）と PNG のグラフで保存する。
"""

import argparse
import csv
import os
import time
from datetime import datetime

# 小さな行列計算を大量にくり返すので、数値計算ライブラリは 1 スレッドの方が速い（numpy の読み込み前に設定）
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np   # noqa: E402

import calibration as C                                                                  # noqa: E402
import config                                                                            # noqa: E402
from decoder import LinearDecoder, PopulationVectorDecoder, evaluate, rates_from_counts  # noqa: E402
from neuron_model import NeuronPopulation                                                # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DT = config.DT
EVAL_SEC = 20.0
WARMUP_BINS = 10            # 評価の最初の 0.2 秒は、発火率の窓がうまるまでなので使わない

BASE = {"n": 100, "noise_type": "drift", "noise": 5.0, "dropout": 0.0, "het": 0.15, "pd_bias": 0.0}


def _cond(**kw):
    c = dict(BASE)
    c.update(kw)
    return c


EXPERIMENTS = {
    "neurons": [_cond(n=n, noise_type=nt, noise=30.0)
                for nt in ("drift", "independent") for n in (10, 20, 50, 100, 200, 400)],
    "noise": [_cond(n=n, noise_type="independent", noise=s)
              for n in (20, 100, 400) for s in (0, 10, 20, 40, 60, 80, 100)],
    "dropout": [_cond(dropout=d) for d in (0.0, 0.1, 0.3, 0.5)],
    "assumptions": [_cond(pd_bias=b, het=h) for b in (0.0, 0.5) for h in (0.15, 0.5, 0.8)],
}


# ============================================================
# 手の動き（自動生成）
# ============================================================
def smooth_noise(rng, T, tau, std):
    """時定数 tau のなめらかなランダム信号 (T, 2)（オルンシュタイン＝ウーレンベック過程）"""
    a, b = 1.0 - DT / tau, std * np.sqrt(2.0 * DT / tau)
    x = np.zeros(2)
    out = np.empty((T, 2))
    eps = rng.normal(size=(T, 2))
    for t in range(T):
        x = a * x + b * eps[t]
        out[t] = x
    return out


def calibration_velocity(rng):
    """
    キャリブレーション中の手の速度 (T, 2) と、記録に使うビンの印 (T,)。
    ターゲットの速度に、追従誤差（ゆっくりした小さなずれ）を加える。
    """
    t = np.arange(0.0, C.total_duration(), DT)
    v = np.array([C.target_velocity(ti) for ti in t]) + smooth_noise(rng, len(t), 0.2, 0.05)
    return v, t >= C.READY_SEC


def evaluation_velocity(rng):
    """
    評価用の手の速度 (T, 2)。キャリブレーションとは違う動きにする。
      前半：リサージュ曲線（x と y で周波数が違う 8 の字のような動き。位相はランダム）
      後半：ランダムな滑らかな動き（向きも速さも不規則）
    """
    half = int(EVAL_SEC / 2 / DT)
    t = np.arange(half) * DT
    fx, fy = 0.37, 0.53
    phx, phy = rng.uniform(0, 2 * np.pi, 2)
    A = 0.25
    v1 = np.stack([A * 2 * np.pi * fx * np.cos(2 * np.pi * fx * t + phx),
                   A * 2 * np.pi * fy * np.cos(2 * np.pi * fy * t + phy)], axis=1)
    v2 = smooth_noise(rng, half, 0.4, 0.45)
    v2 *= np.minimum(1.0, np.arange(half) * DT / 0.3)[:, None]      # つなぎ目で速度が跳ばないように
    v1 *= np.minimum(1.0, (half - np.arange(half)) * DT / 0.3)[:, None]
    v = np.vstack([v1, v2])
    return np.clip(v, -config.SPEED_CLIP, config.SPEED_CLIP)


# ============================================================
# 1 回の試行
# ============================================================
def run_trial(cond, seed, keep_traces=False):
    """
    1 つの条件・1 つのシードで試行し、{デコーダ名: (相関, RMSE)} を返す。
    デコーダ名：PV / LIN（脱落前に学習）/ LIN_recal（脱落後に再キャリブレーション。脱落なしなら LIN と同じ）
    """
    rng = np.random.default_rng([seed, 0])          # ニューロンとスパイクの乱数
    traj_rng = np.random.default_rng([seed, 1])     # 手の動きの乱数（どの条件でも同じ動きになる）

    pop = NeuronPopulation(cond["n"], rng, heterogeneity=cond["het"], pd_bias=cond["pd_bias"])
    pop.noise_type = cond["noise_type"]
    pop.noise_std = cond["noise"]
    pv = PopulationVectorDecoder(pop)

    def calibrate():
        v, mask = calibration_velocity(traj_rng)
        R = rates_from_counts(pop.run(v))
        lin = LinearDecoder(pop.n)
        lin.fit(R[mask], v[mask], n_alive=pop.n_alive)
        return lin

    lin = calibrate()
    lin_recal = lin
    if cond["dropout"] > 0:
        pop.dropout(cond["dropout"])
        lin_recal = calibrate()

    v = evaluation_velocity(traj_rng)
    R = rates_from_counts(pop.run(v))[WARMUP_BINS:]
    v = v[WARMUP_BINS:]
    est = {"PV": pv.decode(R), "LIN": lin.decode(R), "LIN_recal": lin_recal.decode(R)}
    result = {name: evaluate(v, e) for name, e in est.items()}
    if keep_traces:
        return result, v, est
    return result


# ============================================================
# 実験の実行と保存
# ============================================================
COND_KEYS = ["n", "noise_type", "noise", "dropout", "het", "pd_bias"]


def run_experiments(names, trials, base_seed=0):
    rows = []
    total = sum(len(EXPERIMENTS[n]) for n in names) * trials
    done, t0 = 0, time.monotonic()
    for name in names:
        print(f"[experiment] {name}: {len(EXPERIMENTS[name])} 条件 × {trials} 回")
        for cond in EXPERIMENTS[name]:
            for i in range(trials):
                res = run_trial(cond, base_seed + i)
                for dec, (corr, rmse) in res.items():
                    rows.append({"experiment": name, **cond, "seed": base_seed + i, "decoder": dec,
                                 "corr": np.nan if corr is None else corr, "rmse": rmse})
                done += 1
            el = time.monotonic() - t0
            desc = ", ".join(f"{k}={cond[k]}" for k in COND_KEYS if cond[k] != BASE[k]) or "既定の条件"
            pv = np.mean([r["rmse"] for r in rows[-3 * trials:] if r["decoder"] == "PV"])
            ln = np.mean([r["rmse"] for r in rows[-3 * trials:] if r["decoder"] == "LIN"])
            print(f"  [{done:4d}/{total}] {desc:<45s} RMSE PV {pv:.3f} / LIN {ln:.3f}   "
                  f"（経過 {el:5.1f} 秒、残り約 {el / done * (total - done):5.1f} 秒）")
    return rows


def summarize(rows):
    """条件 × デコーダごとに、相関・RMSE の平均と標準偏差をまとめる"""
    groups = {}
    for r in rows:
        key = (r["experiment"], *(r[k] for k in COND_KEYS), r["decoder"])
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        corr = np.array([r["corr"] for r in rs], dtype=float)
        rmse = np.array([r["rmse"] for r in rs], dtype=float)
        out.append({"experiment": key[0], **dict(zip(COND_KEYS, key[1:-1])), "decoder": key[-1],
                    "trials": len(rs),
                    "corr_mean": np.nanmean(corr), "corr_std": np.nanstd(corr),
                    "rmse_mean": rmse.mean(), "rmse_std": rmse.std()})
    return out


def write_csv(path, rows, fmt=6):
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, fmt) if isinstance(v, float) else v) for k, v in r.items()})


def trace_examples(seed=0):
    """グラフ 5 用：代表的な条件での、本当の速度と推定速度の時系列"""
    base = run_trial(_cond(), seed, keep_traces=True)
    drop = run_trial(_cond(dropout=0.3), seed, keep_traces=True)
    return {"base": base, "dropout": drop}


def main(argv=None):
    parser = argparse.ArgumentParser(description="擬似 BCI の自動実験モード")
    parser.add_argument("--trials", type=int, default=20, help="各条件の試行回数（シードの数）。既定 20 回で約 1 分")
    parser.add_argument("--seed", type=int, default=0, help="最初のシード")
    parser.add_argument("--only", default=",".join(EXPERIMENTS), help="実行する実験（カンマ区切り）: "
                        + ", ".join(EXPERIMENTS))
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.only.split(",") if n.strip()]
    for n in names:
        if n not in EXPERIMENTS:
            parser.error(f"実験 {n} はありません（{', '.join(EXPERIMENTS)}）")

    out_dir = os.path.join(BASE_DIR, "results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    rows = run_experiments(names, args.trials, args.seed)
    summary = summarize(rows)
    write_csv(os.path.join(out_dir, "trials.csv"), rows)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)

    traces = trace_examples(args.seed)
    trace_rows = []
    for key, (res, v, est) in traces.items():
        for t in range(len(v)):
            trace_rows.append({"condition": key, "time_sec": round((t + WARMUP_BINS) * DT, 3),
                               "true_vx": v[t, 0], "true_vy": v[t, 1],
                               **{f"{d}_v{ax}": est[d][t, k] for d in est for k, ax in enumerate("xy")}})
    write_csv(os.path.join(out_dir, "traces.csv"), trace_rows, fmt=4)

    import plots      # matplotlib は実験モードのときだけ読み込む
    files = plots.make_all(summary, traces, out_dir, names)
    print(f"[experiment] 完了（{time.monotonic() - t0:.1f} 秒）。保存先: {out_dir}")
    for f in ["trials.csv", "summary.csv", "traces.csv"] + files:
        print(f"  - {f}")
    return out_dir


if __name__ == "__main__":
    main()
