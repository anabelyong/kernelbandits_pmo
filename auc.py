#!/usr/bin/env python
"""
Compute Top-K and AUC metrics for BO CSV results.
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

def process_csv(csv_path):
    y = load_truef(csv_path)
    running_curve = topk_running_mean(y, K)
    auc_val = auc_topk(y, K)
    final_val = final_topk(y, K)

    # Save running curve
    curve_df = pd.DataFrame({
        "Iteration": np.arange(1, len(running_curve)+1),
        f"Top{K}_running_mean": running_curve
    })
    base = os.path.basename(csv_path).replace(".csv", "")
    curve_out = os.path.join(OUT_DIR, f"{base}_running_top{K}.csv")
    curve_df.to_csv(curve_out, index=False)

    # Save summary
    summary_df = pd.DataFrame([{
        "File": base,
        f"AUC_Top{K}": auc_val,
        f"Final_Top{K}": final_val,
        "Num_evals": len(y)
    }])
    summary_out = os.path.join(OUT_DIR, f"{base}_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    print(f"Processed {csv_path}")
    print(f"  AUC-Top{K}: {auc_val:.4f}, Final-Top{K}: {final_val:.4f}")

def main():
    for csv_path in sorted(glob("csv_results/*.csv")):
        try:
            process_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] {csv_path}: {e}")

if __name__ == "__main__":
    main()
