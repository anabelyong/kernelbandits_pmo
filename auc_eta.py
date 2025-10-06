#!/usr/bin/env python
"""
Compute Top-K and AUC metrics for BUCB+MI results across eta values and trials.
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

OUT_DIR = "auc_results"
os.makedirs(OUT_DIR, exist_ok=True)

K = 5  # Top-5

def load_truef(csv_path):
    df = pd.read_csv(csv_path)
    if "True f" not in df.columns:
        raise ValueError(f"No 'True f' column in {csv_path}")
    return np.asarray(df["True f"], dtype=float)

def topk_running_mean(y, K):
    T = len(y)
    out = np.empty(T, dtype=float)
    running = []
    for t in range(T):
        running.append(y[t])
        arr = np.sort(running)[::-1]
        k = min(K, t+1)
        out[t] = arr[:k].mean()
    return out

def auc_topk(y, K):
    curve = topk_running_mean(y, K)
    return float(curve.mean())

def final_topk(y, K):
    arr = np.sort(y)[::-1]
    K = min(K, len(arr))
    return float(arr[:K].mean())

def process_csv(csv_path):
    y = load_truef(csv_path)
    running_curve = topk_running_mean(y, K)
    auc_val = auc_topk(y, K)
    final_val = final_topk(y, K)

    base = os.path.basename(csv_path).replace(".csv", "")

    # Save per-trial running curve
    curve_df = pd.DataFrame({
        "Iteration": np.arange(1, len(running_curve)+1),
        f"Top{K}_running_mean": running_curve
    })
    curve_out = os.path.join(OUT_DIR, f"{base}_running_top{K}.csv")
    curve_df.to_csv(curve_out, index=False)

    return running_curve, auc_val, final_val, len(y)

def aggregate_results():
    results = {"Eta": [], "Trial": [], "AUC": [], "FinalTopK": [], "NumEvals": []}
    all_curves = {}

    for csv_path in sorted(glob("csv_results/*.csv")):
        base = os.path.basename(csv_path)

        if not base.startswith("BUCB_MI_eta"):
            continue

        # detect eta
        parts = base.split("_")
        try:
            eta = float(parts[3])   # BUCB_MI_eta_0.05_logs_trial1.csv
        except Exception:
            print(f"[WARN] Could not parse eta from {base}")
            continue

        # detect trial
        if "logs_trial1" in base:
            trial = 1
        elif "logs_trial2" in base:
            trial = 2
        elif "logs_trial3" in base:
            trial = 3
        else:
            trial = 0

        try:
            running_curve, auc_val, final_val, nevals = process_csv(csv_path)
            results["Eta"].append(eta)
            results["Trial"].append(trial)
            results["AUC"].append(auc_val)
            results["FinalTopK"].append(final_val)
            results["NumEvals"].append(nevals)

            all_curves.setdefault(eta, []).append(running_curve)

            print(f"Processed {base}: η={eta}, AUC={auc_val:.4f}, FinalTopK={final_val:.4f}")
        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    # Save per-trial summary
    df = pd.DataFrame(results)
    trial_out = os.path.join(OUT_DIR, "per_trial_eta_results.csv")
    df.to_csv(trial_out, index=False)
    print(f"Saved per-trial results → {trial_out}")

    # Aggregate mean ± std by eta
    summary = df.groupby("Eta").agg({
        "AUC": ["mean", "std"],
        "FinalTopK": ["mean", "std"],
        "NumEvals": "first"
    })
    summary_out = os.path.join(OUT_DIR, "summary_BUCBMI_eta.csv")
    summary.to_csv(summary_out)
    print(f"Saved aggregated summary → {summary_out}")

    # Plot curves (mean ± std)
    plt.figure(figsize=(8,6))
    for eta, curves in sorted(all_curves.items()):
        min_len = min(len(c) for c in curves)
        curves = np.array([c[:min_len] for c in curves])  # align length
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        iters = np.arange(1, min_len+1)
        plt.plot(iters, mean_curve, label=f"η={eta}", linewidth=2)
        plt.fill_between(iters, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)

    plt.xlabel("Number of Evaluations")
    plt.ylabel(f"Running Top-{K} Mean (True f)")
    plt.title("BUCB+MI: Running Top-K curves across η (mean ± std, 3 trials)")
    plt.legend()
    plt.grid(alpha=0.3)
    out_plot = os.path.join(OUT_DIR, "bucb_mi_eta_curves.pdf")
    plt.savefig(out_plot, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved plot → {out_plot}")

def main():
    aggregate_results()

if __name__ == "__main__":
    main()
