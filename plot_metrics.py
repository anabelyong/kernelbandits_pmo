#!/usr/bin/env python
"""
PMO-style metrics & plots for BO logs with multi-trial aggregation.

Inputs:
  --csv   : one or more CSV files (multiple trials allowed)
  --labels: same length as --csv; trials that share the same label are grouped
  --K     : Top-K used in running curve, AUC-Top-K, and final Top-K
  --budgets   : budgets for snapshot summaries (comma list)
  --auc-grid  : budgets for AUC-Top-K@B (range 'start:end:step' or comma list)
  --plot  : save plots (running Top-K mean±std; AUC-Top-K@B mean±std)

Outputs:
  - summary.csv: mean±std at snapshot budgets (per label)
  - summary_aucgrid.csv: mean±std AUC-Top-K@B (if --auc-grid given)
  - summary_running_topK_curves_K{K}.pdf
  - summary_auc_topK_vs_budget_K{K}.pdf (if --auc-grid given)
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

# ----------------- IO & helpers -----------------
COMMON_SCORE_COLS = ["True f", "true_f", "y", "score", "f", "value", "objective"]

def load_scores(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    col = None
    for c in COMMON_SCORE_COLS:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(
            f"{csv_path} must contain a score column. "
            f"Looked for: {COMMON_SCORE_COLS}. Found: {list(df.columns)}"
        )
    y = np.asarray(df[col], dtype=float)
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
    """Accepts '10,20,50' or '10:200:10' (start:end:step)."""
    if ":" in s:
        start, end, step = [int(x) for x in s.split(":")]
        return list(range(start, end + 1, step))
    else:
        return [int(x) for x in s.split(",") if x.strip()]

def pad_to_length(arr: np.ndarray, L: int) -> np.ndarray:
    """Pad 1D array with NaN to length L."""
    if len(arr) >= L:
        return arr[:L]
    out = np.full(L, np.nan, dtype=float)
    out[:len(arr)] = arr
    return out

# ----------------- Plotting -----------------
def plot_running_topk_mean_std(mean_curves: Dict[str, np.ndarray],
                               std_curves: Dict[str, np.ndarray],
                               K: int,
                               out_pdf: str):
    plt.figure(figsize=(7,4.5))
    for label in mean_curves:
        mean = mean_curves[label]
        std = std_curves[label]
        x = np.arange(1, len(mean) + 1)
        plt.plot(x, mean, linewidth=2.0, label=f"{label} (mean)")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Evaluations")
    plt.ylabel(f"Running Top-{K} mean")
    plt.title(f"Running Top-{K}: mean ± std over trials")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

def plot_auc_vs_budget_mean_std(df_auc: pd.DataFrame, K: int, out_pdf: str):
    """
    df_auc columns: method, budget, AUC_TopK_mean, AUC_TopK_std
    """
    plt.figure(figsize=(7,4.5))
    for label in sorted(df_auc["method"].unique()):
        sub = df_auc[df_auc["method"] == label].sort_values("budget")
        x = sub["budget"].values
        m = sub[f"AUC_Top{K}@B_mean"].values
        s = sub[f"AUC_Top{K}@B_std"].values
        plt.plot(x, m, marker="o", linewidth=2.0, label=f"{label} (mean)")
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.xlabel("Budget (evaluations)")
    plt.ylabel(f"AUC-Top-{K} (prefix mean)")
    plt.title(f"AUC-Top-{K} vs budget: mean ± std over trials")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    print(f"Saved plot: {out_pdf}")

# ----------------- Main -----------------
def main():
    p = argparse.ArgumentParser(description="PMO-style metrics & plots with multi-trial aggregation.")
    p.add_argument("--csv", nargs="+", required=True,
                   help="CSV files (multiple trials allowed).")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Labels for the runs (same order as CSVs). Trials sharing a label are grouped.")
    p.add_argument("--K", type=int, default=10,
                   help="K for AUC-Top-K and final Top-K.")
    p.add_argument("--budgets", type=str, default="50,100,200",
                   help="Comma-separated budgets to summarize at.")
    p.add_argument("--auc-grid", type=str, default=None,
                   help="Budgets for AUC-TopK@B. E.g., '10:200:10' or '10,20,30'.")
    p.add_argument("--plot", action="store_true",
                   help="If set, save plots (running Top-K mean±std; AUC@B mean±std).")
    p.add_argument("--out", type=str, default="summary.csv",
                   help="Output summary CSV basename.")
    args = p.parse_args()

    paths = [Path(c) for c in args.csv]
    labels = args.labels
    if len(labels) != len(paths):
        raise ValueError("Number of --labels must match number of --csv files.")
    budgets = [int(x) for x in args.budgets.split(",")]
    auc_grid = parse_budget_grid(args.auc_grid) if args.auc_grid else None

    # Group trials by label
    trials: Dict[str, List[np.ndarray]] = {}
    for label, path in zip(labels, paths):
        y = load_scores(str(path))
        trials.setdefault(label, []).append(y)

    # ---------- Running Top-K curves (mean ± std) ----------
    mean_curves, std_curves = {}, {}
    max_T = max(len(y) for ys in trials.values() for y in ys)
    for label, ys in trials.items():
        curves = [topk_running_mean(y, args.K) for y in ys]
        # Pad with NaN to align lengths (nanmean across trials)
        curves_pad = np.vstack([pad_to_length(c, max_T) for c in curves])
        mean_curves[label] = np.nanmean(curves_pad, axis=0)
        std_curves[label]  = np.nanstd(curves_pad, axis=0)

    # ---------- Summary table at budgets (mean ± std) ----------
    summary_rows = []
    for label, ys in trials.items():
        # Full-run AUC-TopK per trial
        auc_full = np.array([auc_topk(y, args.K) for y in ys], dtype=float)
        for B in budgets:
            bests, top1s, topKs = [], [], []
            eval_counts = []
            for y in ys:
                s = summarize_at_budget(y, B, K_list=[1, args.K])
                eval_counts.append(s["evals"])
                bests.append(s["best_so_far"])
                top1s.append(s["Top1"])
                topKs.append(s[f"Top{args.K}"])
            summary_rows.append({
                "method": label,
                "budget": B,
                "trials": len(ys),
                "evals_min": int(np.min(eval_counts)),
                "evals_max": int(np.max(eval_counts)),
                "AUC_TopK_fullrun_mean": float(np.mean(auc_full)),
                "AUC_TopK_fullrun_std":  float(np.std(auc_full, ddof=1)) if len(ys)>1 else 0.0,
                "best_so_far@B_mean": float(np.mean(bests)),
                "best_so_far@B_std":  float(np.std(bests, ddof=1)) if len(ys)>1 else 0.0,
                "Top1@B_mean": float(np.mean(top1s)),
                "Top1@B_std":  float(np.std(top1s, ddof=1)) if len(ys)>1 else 0.0,
                f"Top{args.K}@B_mean": float(np.mean(topKs)),
                f"Top{args.K}@B_std":  float(np.std(topKs, ddof=1)) if len(ys)>1 else 0.0,
            })
    df_summary = pd.DataFrame(summary_rows).sort_values(["budget","method"]).reset_index(drop=True)
    print("\n=== Summary (mean ± std over trials) ===")
    print(df_summary.to_string(index=False))
    out_summary = Path(args.out).with_suffix(".csv")
    df_summary.to_csv(out_summary, index=False)
    print(f"\nWrote summary CSV to: {out_summary}")

    # ---------- AUC-TopK@B grid (mean ± std) ----------
    df_auc = None
    if auc_grid:
        auc_rows = []
        for label, ys in trials.items():
            for B in auc_grid:
                aucs = [auc_topk_at_budget(y, args.K, B) for y in ys]
                auc_rows.append({
                    "method": label,
                    "budget": B,
                    f"AUC_Top{args.K}@B_mean": float(np.mean(aucs)),
                    f"AUC_Top{args.K}@B_std":  float(np.std(aucs, ddof=1)) if len(ys)>1 else 0.0,
                    "trials": len(ys),
                })
        df_auc = pd.DataFrame(auc_rows).sort_values(["budget","method"]).reset_index(drop=True)
        print("\n=== AUC-TopK@B Grid (mean ± std) ===")
        print(df_auc.to_string(index=False))
        out_auc = Path(args.out).with_name(Path(args.out).stem + f"_aucgrid.csv")
        df_auc.to_csv(out_auc, index=False)
        print(f"\nWrote AUC grid CSV to: {out_auc}")

    # ---------- Plots ----------
    if args.plot:
        prefix = Path(args.out).with_suffix("").as_posix()
        plot_running_topk_mean_std(mean_curves, std_curves, args.K,
                                   out_pdf=f"{prefix}_running_topK_curves_K{args.K}.pdf")
        if df_auc is not None:
            plot_auc_vs_budget_mean_std(df_auc, args.K,
                                        out_pdf=f"{prefix}_auc_topK_vs_budget_K{args.K}.pdf")

if __name__ == "__main__":
    main()
