#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to circle results
circle_dir = Path("circles_results")

# Filenames for the UCB+eta constant results
trial_files = [
    circle_dir / "logs_trial1_terminal_output_jax_ucb_aug_eta_constant_circles.csv",
    circle_dir / "logs_trial2_terminal_output_jax_ucb_aug_eta_constant_circles.csv",
    circle_dir / "logs_trial3_terminal_output_jax_ucb_aug_eta_constant_circles.csv",
]

plt.figure(figsize=(8, 6))

dfs = []
for f in trial_files:
    if f.exists():
        dfs.append(pd.read_csv(f))
    else:
        print(f"[WARN] Missing file: {f}")

if dfs:
    merged = pd.concat(dfs, keys=range(len(dfs)), names=["trial"])
    grouped = merged.groupby("Threshold")["NumCircles"]
    mean = grouped.mean()
    std = grouped.std()

    thresholds = mean.index
    plt.errorbar(
        thresholds, mean, yerr=std,
        label="UCB+η (constant)",
        color="tab:blue",
        marker="o",
        capsize=3,
        linewidth=2
    )

plt.xlabel("Tanimoto Threshold")
plt.ylabel("#Circles")
plt.title("UCB+η (constant): Circles across thresholds")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save as PDF
out_path = circle_dir / "ucb_eta_constant_circles.pdf"
plt.savefig(out_path, format="pdf", bbox_inches="tight")
plt.show()

print(f"Saved plot → {out_path}")
