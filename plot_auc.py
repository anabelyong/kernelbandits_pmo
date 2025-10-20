#!/usr/bin/env python
"""
Plot mean ± std running Top-5 curves across trials
for BUCB (no packing) and BUCB+ρ variants (ρ=0.65, 0.70, 0.75).
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ===== CONFIG =====
IN_DIR = "auc_results_bucb_tanimoto_constraint"
OUT_PLOT_PDF = os.path.join(IN_DIR, "bucb_top5_curves.pdf")
OUT_PLOT_PNG = os.path.join(IN_DIR, "bucb_top5_curves.png")

# Assign colors for each method
COLORS = {
    "no_rho": "#1f77b4",   # blue
    "rho_0.65": "#2ca02c", # green
    "rho_0.70": "#ff7f0e", # orange
    "rho_0.75": "#d62728"  # red
}

LABELS = {
    "no_rho": "BUCB (no packing)",
    "rho_0.65": "BUCB + ρ=0.65",
    "rho_0.70": "BUCB + ρ=0.70",
    "rho_0.75": "BUCB + ρ=0.75"
}

# ===== HELPER =====
def categorize_file(fname):
    """Return condition key from filename."""
    if "rho_0.65" in fname:
        return "rho_0.65"
    elif "rho_0.70" in fname:
        return "rho_0.70"
    elif "rho_0.75" in fname:
        return "rho_0.75"
    elif "beta_1.0_trial" in fname:
        return "no_rho"
    else:
        return None

# ===== LOAD ALL FILES =====
groups = {"no_rho": [], "rho_0.65": [], "rho_0.70": [], "rho_0.75": []}

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
        # handle possible column name differences
        if "Iteration" in df.columns:
            x = np.arange(len(df))
            y = df["Top5_running_mean"].values
        else:
            x = df["Molecule_index"].values
            y = df["Top5_running_mean"].values
        dfs.append(y[:100])  # keep 0–100 evaluations

    # pad arrays to same length
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
plt.title("Top-5 Running Mean Curves — BUCB vs BUCB + ρ")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(OUT_PLOT_PDF)
plt.savefig(OUT_PLOT_PNG, dpi=300)
plt.show()

print(f"✅ Saved plots to:\n{OUT_PLOT_PDF}\n{OUT_PLOT_PNG}")
