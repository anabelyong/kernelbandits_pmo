#!/usr/bin/env python
"""
Compute Top-K and AUC metrics for UCB+eta constant results across trials.
"""

import os
import numpy as np
import pandas as pd
from glob import glob

OUT_DIR = "auc_results"
os.makedirs(OUT_DIR, exist_ok=True)

K = 5  # we want Top-5

def load_truef(csv_path):
    df = pd.read_csv(csv_path)
    if "True f" not in df.columns:
        raise ValueError(f"No 'True f' column in {csv_path}")
    return np.asarray(df["True f"], dtype=float)

def topk_running_mean(y, K):
    """Running mean of top-K values seen so far."""
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
    """Area under the running top-K curve (average value)."""
    curve = topk_running_mean(y, K)
    return float(curve.mean())

def final_topk(y, K):
    """Final Top-K at the end of run."""
    arr = np.sort(y)[::-1]
    K = min(K, len(arr))
    return float(arr[:K].mean())

def process_csvs_for_config(file_list, config_name):
    curves = []
    aucs, finals, lengths = [], [], []

    for path in file_list:
        y = load_truef(path)
        running_curve = topk_running_mean(y, K)
        auc_val = auc_topk(y, K)
        final_val = final_topk(y, K)

        curves.append(running_curve)
        aucs.append(auc_val)
        finals.append(final_val)
        lengths.append(len(y))

        # Save per-trial running curve
        base = os.path.basename(path).replace(".csv", "")
        curve_df = pd.DataFrame({
            "Iteration": np.arange(1, len(running_curve)+1),
            f"Top{K}_running_mean": running_curve
        })
        curve_out = os.path.join(OUT_DIR, f"{base}_running_top{K}.csv")
        curve_df.to_csv(curve_out, index=False)

    # aggregate across trials
    max_len = min(map(len, curves))  # align by shortest trial
    aligned = np.stack([c[:max_len] for c in curves], axis=0)
    mean_curve = aligned.mean(axis=0)
    std_curve = aligned.std(axis=0)

    # Save mean curve
    agg_df = pd.DataFrame({
        "Iteration": np.arange(1, max_len+1),
        f"Top{K}_running_mean": mean_curve,
        f"Top{K}_running_std": std_curve,
    })
    agg_out = os.path.join(OUT_DIR, f"{config_name}_mean_running_top{K}.csv")
    agg_df.to_csv(agg_out, index=False)

    # Save summary stats
    summary_df = pd.DataFrame([{
        "Config": config_name,
        f"AUC_Top{K}_mean": np.mean(aucs),
        f"AUC_Top{K}_std": np.std(aucs),
        f"Final_Top{K}_mean": np.mean(finals),
        f"Final_Top{K}_std": np.std(finals),
        "Num_evals": min(lengths)
    }])
    summary_out = os.path.join(OUT_DIR, f"{config_name}_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    print(f"Processed {config_name}")
    print(f"  AUC-Top{K}: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Final-Top{K}: {np.mean(finals):.4f} ± {np.std(finals):.4f}")

def main():
    # find all eta-constant CSVs across trials
    files = sorted(glob("csv_results/*eta_constant*.csv"))
    if not files:
        print("[ERROR] No eta constant CSVs found in csv_results/")
        return

    process_csvs_for_config(files, "ucb_eta_constant")

if __name__ == "__main__":
    main()
