#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

IN_DIR = "auc_results"
OUT_PLOT = "auc_results/top5_mean_curves.pdf"

plt.figure(figsize=(8,6))

# Methods and display properties
methods = {
    "BUCB": {"color": "tab:blue", "label": "BUCB"},
    "BUCB_MI": {"color": "tab:orange", "label": "BUCB+MI"}
}

# Collect files by method
files_by_method = {m: [] for m in methods}
for path in sorted(glob(os.path.join(IN_DIR, "*_running_top5.csv"))):
    name = os.path.basename(path)
    if name.startswith("BUCB_MI"):
        files_by_method["BUCB_MI"].append(path)
    elif name.startswith("BUCB_beta"):
        files_by_method["BUCB"].append(path)

# Plot mean ± std for each method
for method, files in files_by_method.items():
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df.set_index("Iteration")["Top5_running_mean"])
    
    if not dfs:
        continue

    # Align by index (Iteration)
    merged = pd.concat(dfs, axis=1)
    mean = merged.mean(axis=1)
    std = merged.std(axis=1)

    x = mean.index.values
    y = mean.values
    yerr = std.values

    plt.plot(
        x, y,
        label=methods[method]["label"],
        color=methods[method]["color"],
        linewidth=2
    )
    plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=methods[method]["color"])

plt.xlabel("Number of Evaluations")
plt.ylabel("Running Top-5 Mean (True f)")
plt.title("Top-5 Running Mean Curves (BUCB vs BUCB+MI)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT)
plt.show()

print(f"Saved plot: {OUT_PLOT}")
