#!/usr/bin/env python
"""
Compute PMO-metrics from BO logs with trial aggregation.

Metrics:
  - PRIMARY: AUC Top-K (full run) and (optionally) AUC Top-K@B across a budget grid
  - SECONDARY: Final Top-K, Top-1, Best-so-far at specified budgets
Also plots:
  - Running Top-K curves (averaged across trials per method)
  - AUC-Top-K@B vs. Budget (if --auc-grid is provided)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------- IO & helpers -----------------
def load_scores(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "True f" in df.columns:
        col = "True f"
    else:
        for c in ["true_f", "y", "score", "f", "value", "objective"]:
            if c in df.columns:
                col = c
                break
        else:
            raise ValueError(f"{csv_path} must contain a score column. "
                             f"Available: {list(df.columns)}")
    y = np.asarray(df[col], dtype=float)
    if np.isnan(y).any():
        y = np.nan_to_num(y, nan=-np.inf)
    return y

def best_so_far(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)

def topk_running_mean(y: np.ndarray, K: int) -> np.ndarray:
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
    return float(topk_running_mean(y, K).mean())

def auc_topk_at_budget(y: np.ndarray, K: int, B: int) -> float:
    B = min(B, len(y))
    if B <= 0:
        return np.nan
    return float(topk_running_mean(y[:B], K).mean())

def final_topk(y: np.ndarray, K: int) -> float:
    arr = np.sort(y)[::-1]
    return float(arr[:min(K, len(arr))].mean())

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
        plt.plot(x, curve, linewidth=1.8, label=label)
    plt.xlabel("Evaluations")
    plt.ylabel(f"Running Top-{K} mean")
    plt.title(f"Running Top-{K} vs. evaluations (averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

def plot_auc_vs_budget(df_auc: pd.DataFrame, K: int, out_pdf: str):
    plt.figure(figsize=(7,4.5))
    for label in sorted(df_auc["method"].unique()):
        sub = df_auc[df_auc["method"] == label]
        x = sub["budget"].to_numpy()
        y = sub[f"AUC_Top{K}@B"].to_numpy()
        y_std = sub[f"AUC_Top{K}@B_std"].to_numpy()
        plt.plot(x, y, marker="o", linewidth=1.8, label=label)
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.2)
    plt.xlabel("Budget (evaluations)")
    plt.ylabel(f"AUC-Top-{K} (prefix mean)")
    plt.title(f"AUC-Top-{K} vs. budget (mean ± std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

def average_curves(curves: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        pad = np.full(max_len, np.nan)
        pad[:len(c)] = c
        padded.append(pad)
    stacked = np.vstack(padded)
    return np.nanmean(stacked, axis=0)

# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser(description="PMO-style metrics and plots for BO logs (with trial aggregation).")
    p.add_argument("--csv", nargs="+", required=True, help="CSV files to compare.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Labels (methods). Same length as --csv, can repeat (e.g. BUCB BUCB BUCB BUCB+MI BUCB+MI BUCB+MI).")
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--budgets", type=str, default="50,100,200")
    p.add_argument("--auc-grid", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default="bo_metrics_summary.csv")
    args = p.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    paths = [Path(c) for c in args.csv]
    labels = args.labels or [p.stem for p in paths]
    if len(labels) != len(paths):
        raise ValueError("Number of --labels must match number of --csv files.")

    # --- group by method (aggregate trials) ---
    ys_grouped = defaultdict(list)
    curves_grouped = defaultdict(list)
    for label, path in zip(labels, paths):
        y = load_scores(str(path))
        ys_grouped[label].append(y)
        curves_grouped[label].append(topk_running_mean(y, args.K))

    # --- aggregate curves ---
    agg_curves = {m: average_curves(c_list) for m, c_list in curves_grouped.items()}

    # --- summary metrics per method ---
    rows = []
    for method, y_list in ys_grouped.items():
        for B in budgets:
            auc_vals = [auc_topk(y, args.K) for y in y_list]
            s_vals = [summarize_at_budget(y, B, K_list=[1, args.K]) for y in y_list]
            rows.append({
                "method": method,
                "budget": B,
                "T_total_mean": np.mean([len(y) for y in y_list]),
                "AUC_TopK_fullrun_mean": np.mean(auc_vals),
                "best_so_far@B_mean": np.mean([s["best_so_far"] for s in s_vals]),
                "Top1@B_mean": np.mean([s["Top1"] for s in s_vals]),
                f"Top{args.K}@B_mean": np.mean([s[f"Top{args.K}"] for s in s_vals]),
            })
    df_out = pd.DataFrame(rows).sort_values(by=["budget","method"]).reset_index(drop=True)
    print("\n=== Aggregated Summary ===")
    print(df_out.to_string(index=False))
    df_out.to_csv(args.out, index=False)
    print(f"\nWrote aggregated summary CSV to: {args.out}")

    # --- AUC grid (optional) ---
    if args.auc_grid:
        grid = parse_budget_grid(args.auc_grid)
        auc_rows = []
        for method, y_list in ys_grouped.items():
            for B in grid:
                auc_vals = [auc_topk_at_budget(y, args.K, B) for y in y_list]
                auc_rows.append({
                    "method": method,
                    "budget": B,
                    f"AUC_Top{args.K}@B": np.mean(auc_vals),
                    f"AUC_Top{args.K}@B_std": np.std(auc_vals, ddof=1) if len(auc_vals) > 1 else 0.0,
                })
        df_auc = pd.DataFrame(auc_rows).sort_values(["budget","method"]).reset_index(drop=True)
        print("\n=== Aggregated AUC-TopK@B Grid ===")
        print(df_auc.to_string(index=False))
        auc_out = Path(args.out).with_name(Path(args.out).stem + f"_aucgrid.csv")
        df_auc.to_csv(auc_out, index=False)
        print(f"\nWrote aggregated AUC grid CSV to: {auc_out}")
    else:
        df_auc = None

    # --- Plots ---
    if args.plot:
        prefix = Path(args.out).with_suffix("").as_posix()
        plot_running_topk(agg_curves, args.K, out_pdf=f"{prefix}_running_topK_curves_K{args.K}.pdf")
        if df_auc is not None:
            plot_auc_vs_budget(df_auc, args.K, out_pdf=f"{prefix}_auc_topK_vs_budget_K{args.K}.pdf")

if __name__ == "__main__":
    main()
