#!/usr/bin/env python
"""
Compute Top-K and AUC metrics for BUCB vs BUCB+MI results across trials.
"""

import os
import numpy as np
import pandas as pd
from glob import glob

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

def process_csv(csv_path, method, trial):
    y = load_truef(csv_path)
    running_curve = topk_running_mean(y, K)
    auc_val = auc_topk(y, K)
    final_val = final_topk(y, K)

    base = os.path.basename(csv_path).replace(".csv", "")

    # Save running curve
    curve_df = pd.DataFrame({
        "Iteration": np.arange(1, len(running_curve)+1),
        f"Top{K}_running_mean": running_curve
    })
    curve_out = os.path.join(OUT_DIR, f"{base}_running_top{K}.csv")
    curve_df.to_csv(curve_out, index=False)

    return auc_val, final_val, len(y)

def aggregate_results():
    results = {"Method": [], "Trial": [], "AUC": [], "FinalTopK": [], "NumEvals": []}

    for csv_path in sorted(glob("csv_results/*.csv")):
        base = os.path.basename(csv_path)

        # detect method
        if base.startswith("BUCB_MI"):
            method = "BUCB+MI"
        elif base.startswith("BUCB"):
            method = "BUCB"
        else:
            continue

        # detect trial number
        if "logs_trial1" in base:
            trial = 1
        elif "logs_trial2" in base:
            trial = 2
        elif "logs_trial3" in base:
            trial = 3
        else:
            trial = 0  # fallback

        try:
            auc_val, final_val, nevals = process_csv(csv_path, method, trial)
            results["Method"].append(method)
            results["Trial"].append(trial)
            results["AUC"].append(auc_val)
            results["FinalTopK"].append(final_val)
            results["NumEvals"].append(nevals)
            print(f"Processed {base}: AUC={auc_val:.4f}, FinalTopK={final_val:.4f}")
        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

    # Save per-trial summary
    df = pd.DataFrame(results)
    trial_out = os.path.join(OUT_DIR, "per_trial_results.csv")
    df.to_csv(trial_out, index=False)
    print(f"Saved per-trial results → {trial_out}")

    # Aggregate mean ± std by method
    summary = df.groupby("Method").agg({
        "AUC": ["mean", "std"],
        "FinalTopK": ["mean", "std"],
        "NumEvals": "first"
    })
    summary_out = os.path.join(OUT_DIR, "summary_BUCB_vs_BUCBMI.csv")
    summary.to_csv(summary_out)
    print(f"Saved aggregated summary → {summary_out}")

def main():
    aggregate_results()

if __name__ == "__main__":
    main()

