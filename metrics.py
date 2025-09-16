#!/usr/bin/env python
"""
Compute PMO-metrics from BO logs. Metrics are AUC Top-K and Final Top-K, Top-1, Best-so-far at a specified bufget. 
"""
import argparse 
import numpy as np 
import pandas as pd 
from pathlib import Path 
from typing import List

def load_scores(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "True f" not in df.columns: 
        raise ValueError(f"{csv_path} must contain a 'True f' column")
    y = np.asarray(df["True f"], dtype= float)
    return y 

def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)

def topk_running_mean(y: np.ndarray, K: int) -> np.ndarray:
    """
    For t=1..T: take y[:t], sort descending, average top min(K,t).
    Returns an array of length T.
    """
    T = len(y)
    out = np.empty(T, dtype=float)
    running = []
    for t in range(T):
        running.append(y[t])
        arr = np.sort(running)[::-1]
        k = min(K, t + 1)
        out[t] = arr[:k].mean()
    return out

def auc_topk(y: np.ndarray, K: int) -> float:
    """
    PMO-style sample efficiency: area under Top-K running mean curve.
    We use the mean over timesteps (sum / T).
    """
    curve = topk_running_mean(y, K)
    return float(curve.mean())


def final_topk(y: np.ndarray, K: int) -> float:
    arr = np.sort(y)[::-1]
    K = min(K, len(arr))
    return float(arr[:K].mean())

def summarize_at_budget(y: np.ndarray, budget: int, K_list: List[int]) -> dict:
    y_bud = y[: min(len(y), budget)]
    bsf = best_so_far(y_bud)[-1]
    out = {
        "evals": len(y_bud),
        "best_so_far": float(bsf),
        "Top1": float(np.sort(y_bud)[::-1][0]),
    }
    for K in K_list:
        out[f"Top{K}"] = final_topk(y_bud, K)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", nargs="+", required=True,
                   help="CSV files to compare (ucb_fex, ucb_aug_fex).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels for the runs (same order as CSVs). Defaults to file stems.")
    p.add_argument("--K", type=int, default=10,
                   help="K for AUC-Top-K and final Top-K.")
    p.add_argument("--budgets", type=str, default="50,100,200",
                   help="Comma-separated budgets to summarize at.")
    p.add_argument("--out", type=str, default="bo_metrics_summary.csv",
                   help="Output summary CSV.")
    args = p.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    paths = [Path(c) for c in args.csv]
    labels = args.labels or [p.stem for p in paths]

    if len(labels) != len(paths):
        raise ValueError("Number of --labels must match number of --csv files.")

    rows = []
    for label, path in zip(labels, paths):
        y = load_scores(str(path))
        T = len(y)

        aucK = auc_topk(y, args.K)

        for B in budgets:
            s = summarize_at_budget(y, B, K_list=[1, args.K])
            rows.append({
                "method": label,
                "file": path.name,
                "T_total": T,
                "budget": B,
                "AUC_TopK_fullrun": aucK,
                "best_so_far@B": s["best_so_far"],
                "Top1@B": s["Top1"],
                f"Top{args.K}@B": s[f"Top{args.K}"],
            })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(by=["budget", "method"]).reset_index(drop=True)

    print("\n=== Summary ===")
    print(df_out.to_string(index=False))

    df_out.to_csv(args.out, index=False)
    print(f"\nWrote summary CSV to: {args.out}")

if __name__ == "__main__":
    main()



