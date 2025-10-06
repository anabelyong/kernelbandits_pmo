#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Path to circle results
circle_dir = Path("circles_results")

# Eta values you want to compare
etas = [0.05, 0.1, 0.25, 0.5]

# Colors and markers for each eta
colors = {
    0.05: "tab:blue",
    0.1:  "tab:orange",
    0.25: "tab:green",
    0.5:  "tab:red",
}
markers = {
    0.05: "o",
    0.1: "s",
    0.25: "D",
    0.5: "^",
}

plt.figure(figsize=(8, 6))

for eta in etas:
    dfs = []
    for trial in [1, 2, 3]:
        fname = circle_dir / f"BUCB_MI_eta_{eta}_logs_trial{trial}_circles.csv"
        if fname.exists():
            dfs.append(pd.read_csv(fname))
        else:
            print(f"[WARN] Missing file: {fname}")
    
    if not dfs:
        continue

    # Concatenate all trials
    merged = pd.concat(dfs, keys=range(len(dfs)), names=["trial"])
    grouped = merged.groupby("Threshold")["NumCircles"]
    mean = grouped.mean()
    std = grouped.std()

    thresholds = mean.index
    plt.errorbar(
        thresholds, mean, yerr=std,
        label=f"BUCB+MI η={eta}",
        color=colors[eta],
        marker=markers[eta],
        capsize=3,
        linewidth=2
    )

plt.xlabel("Tanimoto Threshold")
plt.ylabel("#Circles")
plt.title("BUCB+MI: Circles across thresholds")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save as PDF
out_path = circle_dir / "bucb_mi_circles_eta_comparison.pdf"
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Saved plot → {out_path}")
