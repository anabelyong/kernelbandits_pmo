#!/usr/bin/env python
"""
Plot mean ± std running Top-5 curves across trials
for BUCB (β = 1.0, 5.0, 10.0) and UCB (β = 1.0 baseline).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ===== CONFIG =====
IN_DIR = "auc_results_bucb_vary_beta"
OUT_PLOT_PDF = os.path.join(IN_DIR, "bucb_vary_beta_top5_curves.pdf")

# Assign colors for each method
COLORS = {
    "gpucb_beta_1.0": "#d62728",  # red
    "beta_1.0": "#1f77b4",        # blue
    "beta_5.0": "#2ca02c",        # green
    "beta_10.0": "#ff7f0e"        # orange
}

LABELS = {
    "gpucb_beta_1.0": "UCB (β = 1.0)",
    "beta_1.0": "BUCB (β = 1.0)",
    "beta_5.0": "BUCB (β = 5.0)",
    "beta_10.0": "BUCB (β = 10.0)"
}

# ===== HELPER =====
def categorize_file(fname):
    """Return condition key from filename (check gpucb first)."""
    if "gpucb_beta_1.0" in fname:
        return "gpucb_beta_1.0"
    elif "beta_10.0" in fname:
        return "beta_10.0"
    elif "beta_5.0" in fname:
        return "beta_5.0"
    elif "beta_1.0" in fname:
        return "beta_1.0"
    else:
        return None

# ===== LOAD ALL FILES =====
groups = {key: [] for key in COLORS.keys()}

for path in sorted(glob(os.path.join(IN_DIR, "*_running_top5.csv"))):
    condition = categorize_file(path)
    if condition:
        groups[condition].append(path)

# ===== PLOT =====
plt.figure(figsize=(8, 6))

for condition, files in groups.items():
    if not files:
        continue

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "Top5_running_mean" not in df.columns:
            continue
        y = df["Top5_running_mean"].values[:100]  # limit to first 100 molecules
        dfs.append(y)

    if not dfs:
        continue

    # Pad / align
    min_len = min(len(y) for y in dfs)
    ys = np.array([y[:min_len] for y in dfs])
    mean = ys.mean(axis=0)
    std = ys.std(axis=0)
    x = np.arange(min_len)

    plt.plot(
        x,
        mean,
        label=LABELS[condition],
        color=COLORS[condition],
        linewidth=2.0
    )
    plt.fill_between(
        x,
        mean - std,
        mean + std,
        color=COLORS[condition],
        alpha=0.2
    )

# ===== STYLE =====
plt.xlabel("Number of Evaluations (Molecules)")
plt.ylabel("Running Top-5 Mean (True f)")
plt.title("Top-5 Running Mean Curves — BUCB vs UCB (vary β)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_PLOT_PDF)
plt.show()

print(f"✅ Saved plot to: {OUT_PLOT_PDF}")
