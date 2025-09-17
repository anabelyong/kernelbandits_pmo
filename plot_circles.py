import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to circle results
circle_dir = Path("circles_results")

# Methods and trials
methods = {
    "ucb": [circle_dir / f"logs_trial{i}_terminal_output_jax_ucb_fex_circles.csv" for i in [1, 2, 3]],
    "ucb_aug": [circle_dir / f"logs_trial{i}_terminal_output_jax_ucb_aug_fex_circles.csv" for i in [1, 2, 3]],
}

# Load all results
all_results = {}
for method, files in methods.items():
    dfs = []
    for f in files:
        if f.exists():
            dfs.append(pd.read_csv(f))
        else:
            print(f"[WARN] Missing file: {f}")
    all_results[method] = dfs

# Compute mean ± std for each threshold
mean_std_results = {}
for method, dfs in all_results.items():
    merged = pd.concat(dfs, keys=range(len(dfs)), names=["trial"])
    grouped = merged.groupby("Threshold")["NumCircles"]
    mean = grouped.mean()
    std = grouped.std()
    mean_std_results[method] = (mean, std)

# Plot
plt.figure(figsize=(8, 6))

colors = {"ucb": "blue", "ucb_aug": "magenta"}
labels = {"ucb": "UCB", "ucb_aug": "UCB+Aug"}

for method, (mean, std) in mean_std_results.items():
    thresholds = mean.index
    plt.errorbar(
        thresholds, mean, yerr=std,
        label=labels[method], color=colors[method],
        marker="o", capsize=3
    )

plt.xlabel("Tanimoto Threshold")
plt.ylabel("#Circles")
plt.title("Fexofenadine: #Circles across thresholds")
plt.legend()
plt.grid(True)
plt.yscale("log")  

plt.savefig("fexofenadine_circles.pdf", format="pdf", bbox_inches="tight")
