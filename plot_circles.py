#!/usr/bin/env python
"""
Plot mean ± std of NumCircles across thresholds
for BUCB (no constraint) and BUCB + Tanimoto packing (ρ = 0.65, 0.70, 0.75)
from `circles_results_BUCB_and_pack/`.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ===== CONFIG =====
BASE_DIR = "circles_results_BUCB_and_pack"
OUT_PLOT = os.path.join(BASE_DIR, "circles_vs_thresholds_mean_std.pdf")

# Define four conditions
conditions = {
    "BUCB": os.path.join(BASE_DIR, "logs_bucb_beta_1.0"),
    "ρ=0.65": os.path.join(BASE_DIR, "logs_bucb_pack"),
    "ρ=0.70": os.path.join(BASE_DIR, "logs_bucb_pack"),
    "ρ=0.75": os.path.join(BASE_DIR, "logs_bucb_pack"),
}

# Define colors
colors = {
    "BUCB": "#1f77b4",      # blue
    "ρ=0.65": "#2ca02c",    # green
    "ρ=0.70": "#ff7f0e",    # orange
    "ρ=0.75": "#9467bd",    # purple
}

plt.figure(figsize=(8, 6))

# ===== LOAD & PLOT =====
for label, base_path in conditions.items():
    dfs = []

    if label == "BUCB":
        # BUCB baseline — e.g. logs_bucb_beta_1.0/trial_X/beta_1.00/
        csv_paths = glob(os.path.join(base_path, "trial_*", "beta_1.00", "*_pareto_fex_circles.csv"))
    else:
        # BUCB with packing constraint — e.g. logs_bucb_pack/trial_X/rho_0.65/beta_1.00/pack_batch/
        rho_val = label.replace("ρ=", "rho_")
        csv_paths = glob(os.path.join(base_path, "trial_*", rho_val, "beta_1.00", "pack_batch", "*_pareto_fex_circles.csv"))

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

    # Merge and aggregate across trials
    merged = pd.concat(dfs, axis=0)
    grouped = merged.groupby("Threshold")["NumCircles"].agg(["mean", "std"]).reset_index()

    # Plot mean ± std
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
plt.ylabel("NumCircles", fontsize=12)
plt.title("BUCB vs BUCB + Packing Constraint — #Circles Metric", fontsize=13)
plt.grid(alpha=0.3)
plt.legend(title="Method", fontsize=10)
plt.tight_layout()

plt.savefig(OUT_PLOT)
plt.show()
print(f"✅ Saved plot to {OUT_PLOT}")
