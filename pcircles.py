#!/usr/bin/env python
"""
Plot mean ± std of NumCircles across thresholds
for BUCB (β = 1.0, 5.0, 10.0) and UCB (β = 1.0),
from `circles_results_BUCB_UCB_vary_beta/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ===== CONFIG =====
BASE_DIR = "circles_results_BUCB_UCB_vary_beta"
OUT_PLOT = os.path.join(BASE_DIR, "circles_vs_thresholds_vary_beta.pdf")

# Define all conditions
conditions = {
    "BUCB (β=1.0)": os.path.join(BASE_DIR, "logs_bucb_beta_1.0"),
    "BUCB (β=5.0)": os.path.join(BASE_DIR, "logs_bucb_beta_5.0"),
    "BUCB (β=10.0)": os.path.join(BASE_DIR, "logs_bucb_beta_10.0"),
    "UCB (β=1.0)": os.path.join(BASE_DIR, "logs_ucb"),
}

# Define colors
colors = {
    "BUCB (β=1.0)": "#1f77b4",   # blue
    "BUCB (β=5.0)": "#2ca02c",   # green
    "BUCB (β=10.0)": "#ff7f0e",  # orange
    "UCB (β=1.0)": "#d62728",    # red
}

plt.figure(figsize=(8, 6))

# ===== LOAD & PLOT =====
for label, base_path in conditions.items():
    dfs = []

    # Each trial has: trial_X/beta_Y/*.csv
    csv_paths = glob(os.path.join(base_path, "trial_*", "beta_*", "*_pareto_fex_circles.csv"))

    if not csv_paths:
        print(f"[WARN] No CSVs found for {label}")
        continue

    for csv in csv_paths:
        try:
            df = pd.read_csv(csv)
            if "Threshold" in df.columns and "NumCircles" in df.columns:
                dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Failed reading {csv}: {e}")

    if not dfs:
        print(f"[WARN] No valid dataframes for {label}")
        continue

    # Combine all trials and compute mean/std by threshold
    merged = pd.concat(dfs, axis=0)
    grouped = merged.groupby("Threshold")["NumCircles"].agg(["mean", "std"]).reset_index()

    # Plot mean ± std shaded
    plt.plot(
        grouped["Threshold"],
        grouped["mean"],
        label=label,
        color=colors[label],
        linewidth=2
    )
    plt.fill_between(
        grouped["Threshold"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        color=colors[label],
        alpha=0.2
    )

# ===== STYLE =====
plt.xlabel("Tanimoto Threshold", fontsize=12)
plt.ylabel("NumCircles (Pareto Set)", fontsize=12)
plt.title("BUCB vs UCB — Circles Metric Across β", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(title="Method", fontsize=10)
plt.tight_layout()

plt.savefig(OUT_PLOT)
plt.show()
print(f"✅ Saved plot to {OUT_PLOT}")
