import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to results
circle_dir = Path("circles_results")

# Methods to compare
methods = ["BUCB", "BUCB_MI"]

# Colors
colors = {
    "BUCB": "#1f77b4",     # blue
    "BUCB_MI": "#ff7f0e",  # orange
}

plt.figure(figsize=(8, 6))

for method in methods:
    dfs = []
    for trial in [1, 2, 3]:
        fname = circle_dir / f"{method}_beta_0.1_logs_trial{trial}_circles.csv"
        if not fname.exists():
            print(f"[WARN] Missing file: {fname}")
            continue
        df = pd.read_csv(fname)
        dfs.append(df)

    if not dfs:
        continue

    # Concatenate trials
    combined = pd.concat(dfs, keys=[1,2,3], names=["Trial"])
    
    # Group by Threshold and compute mean & std
    grouped = combined.groupby("Threshold")["NumCircles"].agg(["mean", "std"]).reset_index()

    # Plot mean ± std as error bars
    plt.errorbar(
        grouped["Threshold"], grouped["mean"], yerr=grouped["std"],
        label=method.replace("_", "+"),  # nicer legend: "BUCB+MI"
        color=colors[method],
        marker="o",
        capsize=4,
        linewidth=2
    )

plt.xlabel("Tanimoto Threshold")
plt.ylabel("NumCircles")
plt.title("#Circles across thresholds (BUCB vs BUCB+MI, β=0.1)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Save as PDF
plt.savefig(circle_dir / "circles_across_thresholds.pdf", format="pdf", bbox_inches="tight")
print(f"Saved plot to {circle_dir/'circles_across_thresholds.pdf'}")
