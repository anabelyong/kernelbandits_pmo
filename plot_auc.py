#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

IN_DIR = "auc_results"
OUT_PLOT = "auc_results/top5_curves.pdf"

plt.figure(figsize=(8,6))

# Assign colors/styles
colors = {
    "BUCB": "tab:blue",
    "UCB": "tab:orange"
}
linestyles = {
    "0.2": "-",
    "0.4": "--",
    "0.6": "-.",
    "0.8": ":",
    "1.0": (0, (3, 1, 1, 1))  # dash-dot pattern
}

# Load each running csv
for path in sorted(glob(os.path.join(IN_DIR, "*_running_top5.csv"))):
    name = os.path.basename(path).replace("_running_top5.csv", "")
    if name.startswith("BUCB"):
        method = "BUCB"
    elif name.startswith("UCB"):
        method = "UCB"
    else:
        continue
    
    # Extract beta value
    parts = name.split("_")
    beta = parts[-1] if "beta" in parts[-2] else None
    if beta is None:
        continue

    df = pd.read_csv(path)
    x = df["Iteration"].values
    y = df["Top5_running_mean"].values

    plt.plot(
        x, y,
        label=f"{method}, β={beta}",
        color=colors[method],
        linestyle=linestyles.get(beta, "-"),
        linewidth=2.0
    )

plt.xlabel("Number of Evaluations")
plt.ylabel("Running Top-5 Mean (True f)")
plt.title("Top-5 Running Mean Curves (BUCB vs UCB)")
plt.legend(ncol=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PLOT)
plt.show()

print(f"Saved plot: {OUT_PLOT}")
