import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to circle results
circle_dir = Path("circles_results")

# Define betas and methods
betas = [0.2, 0.4, 0.6, 0.8, 1.0]
methods = ["BUCB", "UCB"]

# Assign colors (different palette for each method)
colors = {
    "BUCB": ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4"],  # shades of blue
    "UCB":  ["#d62728", "#d62728", "#d62728", "#d62728", "#d62728"],  # shades of red
}
markers = ["o", "s", "D", "^", "v"]

plt.figure(figsize=(8, 6))

for method in methods:
    for i, beta in enumerate(betas):
        fname = circle_dir / f"{method}_beta_{beta}_circles.csv"
        if not fname.exists():
            print(f"[WARN] Missing file: {fname}")
            continue

        df = pd.read_csv(fname)
        plt.plot(
            df["Threshold"], df["NumCircles"],
            label=f"{method} β={beta}",
            color=colors[method][i % len(colors[method])],
            marker=markers[i % len(markers)],
            linewidth=2
        )

plt.xlabel("Tanimoto Threshold")
plt.ylabel("NumCircles")
plt.title("#Circles across thresholds (UCB vs BUCB)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save as PDF
plt.savefig("circles_results/circles_across_thresholds.pdf", format="pdf", bbox_inches="tight")
print("Saved plot to circles_results/circles_across_thresholds.pdf")
