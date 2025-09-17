#!/usr/bin/env python
"""
Compute PMO-metrics from BO logs.

Metrics:
  - PRIMARY: AUC Top-K (full run) and (optionally) AUC Top-K@B across a budget grid
  - SECONDARY: Final Top-K, Top-1, Best-so-far at specified budgets
Also plots:
  - Running Top-K curves for each CSV
  - AUC-Top-K@B vs. Budget (if --auc-grid is provided)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

# ----------------- IO & helpers -----------------
def load_scores(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # Expect "True f" per your logs; fall back to a few common names
    if "True f" in df.columns:
        col = "True f"
    else:
        for c in ["true_f", "y", "score", "f", "value", "objective"]:
            if c in df.columns:
                col = c
                break
        else:
            raise ValueError(f"{csv_path} must contain a score column (e.g. 'True f'). "
                             f"Available: {list(df.columns)}")
    y = np.asarray(df[col], dtype=float)
    # guard against NaNs
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=-np.inf)
    return y

def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)

def topk_running_mean(y: np.ndarray, K: int) -> np.ndarray:
    """
    For t=0..T-1: take y[:t+1], sort desc, average top min(K, t+1).
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
    """Mean of the Top-K running-mean curve over the full run."""
    curve = topk_running_mean(y, K)
    return float(curve.mean())

def auc_topk_at_budget(y: np.ndarray, K: int, B: int) -> float:
    """AUC-TopK computed only on the first B evaluations."""
    B = min(B, len(y))
    if B <= 0:
        return np.nan
    curve = topk_running_mean(y[:B], K)
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

def parse_budget_grid(s: str) -> list[int]:
    """
    Accepts '10,20,50' or '10:200:10' (start:end:step).
    """
    if ":" in s:
        start, end, step = [int(x) for x in s.split(":")]
        return list(range(start, end + 1, step))
    else:
        return [int(x) for x in s.split(",") if x.strip()]

# ----------------- Plotting -----------------
def plot_running_topk(curves: dict[str, np.ndarray], K: int, out_pdf: str):
    plt.figure(figsize=(7,4.5))
    for label, curve in curves.items():
        x = np.arange(1, len(curve)+1)
        plt.plot(x, curve, marker=None, linewidth=1.8, label=label)
    plt.xlabel("Evaluations")
    plt.ylabel(f"Running Top-{K} mean")
    plt.title(f"Running Top-{K} vs. evaluations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

def plot_auc_vs_budget(df_auc: pd.DataFrame, K: int, out_pdf: str):
    plt.figure(figsize=(7,4.5))
    for label in sorted(df_auc["method"].unique()):
        sub = df_auc[df_auc["method"] == label]
        plt.plot(sub["budget"], sub[f"AUC_Top{K}@B"], marker="o", linewidth=1.8, label=label)
    plt.xlabel("Budget (evaluations)")
    plt.ylabel(f"AUC-Top-{K} (prefix mean)")
    plt.title(f"AUC-Top-{K} vs. budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser(description="PMO-style metrics and plots for BO logs.")
    p.add_argument("--csv", nargs="+", required=True,
                   help="CSV files to compare (ucb_fex, ucb_aug_fex, ...).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels for the runs (same order as CSVs). Defaults to file stems.")
    p.add_argument("--K", type=int, default=10,
                   help="K for AUC-Top-K and final Top-K.")
    p.add_argument("--budgets", type=str, default="50,100,200",
                   help="Comma-separated budgets to summarize at.")
    p.add_argument("--auc-grid", type=str, default=None,
                   help="Budgets for AUC-TopK@B. E.g., '10:200:10' or '10,20,30'.")
    p.add_argument("--plot", action="store_true",
                   help="If set, save plots (running Top-K curves, and AUC@B vs budget if --auc-grid given).")
    p.add_argument("--out", type=str, default="bo_metrics_summary.csv",
                   help="Output summary CSV.")
    args = p.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    paths = [Path(c) for c in args.csv]
    labels = args.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError("Number of --labels must match number of --csv files.")

    # Load scores and precompute running curves
    ys = {}
    run_topk_curves = {}
    for label, path in zip(labels, paths):
        y = load_scores(str(path))
        ys[label] = y
        run_topk_curves[label] = topk_running_mean(y, args.K)

    # ----- Summary table (like before) -----
    rows = []
    for label, path in zip(labels, paths):
        y = ys[label]
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
    df_out = pd.DataFrame(rows).sort_values(by=["budget", "method"]).reset_index(drop=True)
    print("\n=== Summary ===")
    print(df_out.to_string(index=False))
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote summary CSV to: {args.out}")

    # ----- AUC grid (optional) -----
    if args.auc_grid:
        grid = parse_budget_grid(args.auc_grid)
        auc_rows = []
        for label, path in zip(labels, paths):
            y = ys[label]
            for B in grid:
                aucB = auc_topk_at_budget(y, args.K, B)
                auc_rows.append({
                    "method": label,
                    "file": path.name,
                    "budget": B,
                    f"AUC_Top{args.K}@B": aucB,
                })
        df_auc = pd.DataFrame(auc_rows).sort_values(["budget", "method"]).reset_index(drop=True)
        print("\n=== AUC-TopK@B Grid ===")
        print(df_auc.to_string(index=False))
        auc_out = Path(args.out).with_name(Path(args.out).stem + f"_aucgrid.csv")
        df_auc.to_csv(auc_out, index=False)
        print(f"\nWrote AUC grid CSV to: {auc_out}")
    else:
        df_auc = None

    # ----- Plots (optional) -----
    if args.plot:
        prefix = Path(args.out).with_suffix("").as_posix()
        plot_running_topk(run_topk_curves, args.K, out_pdf=f"{prefix}_running_topK_curves_K{args.K}.pdf")
        if df_auc is not None:
            plot_auc_vs_budget(df_auc, args.K, out_pdf=f"{prefix}_auc_topK_vs_budget_K{args.K}.pdf")

if __name__ == "__main__":
    main()
