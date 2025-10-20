#!/usr/bin/env python
"""
Analyze a single BO results CSV.
Computes running Top-K curve and AUC-Top-K for the first 100 molecules (20 rounds × 5).
Lets you manually specify the CSV path to inspect a single run.
Automatically names outputs by trial ID.
Have UCB as baseline with β=1.0
"""

import os
import numpy as np
import pandas as pd
import re

# ====== CONFIG ======
OUT_DIR = "auc_results_bucb_vary_beta"
K = 5
N_ROUNDS = 20
MOLECULES_PER_ROUND = 5
N_MOLECULES = N_ROUNDS * MOLECULES_PER_ROUND  # = 100
os.makedirs(OUT_DIR, exist_ok=True)


# ====== HELPER FUNCTIONS ======
def load_truef(csv_path):
    """Load 'True f' or 'f_true' column."""
    df = pd.read_csv(csv_path)
    cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df.columns = cols
    if "true_f" in df.columns:
        return np.asarray(df["true_f"], dtype=float)
    elif "f_true" in df.columns:
        return np.asarray(df["f_true"], dtype=float)
    else:
        raise ValueError(f"No 'True f' column found in {csv_path}")


def topk_running_mean(y, k):
    """Compute running mean of top-K values seen so far."""
    running = []
    out = np.empty(len(y), dtype=float)
    for t, val in enumerate(y):
        running.append(val)
        arr = np.sort(running)[::-1]
        out[t] = arr[: min(k, len(arr))].mean()
    return out


def auc_topk(y, k):
    """Mean (area under) the running Top-K curve."""
    curve = topk_running_mean(y, k)
    return float(np.mean(curve))


def final_topk(y, k):
    """Mean of the top-K values at the end."""
    arr = np.sort(y)[::-1]
    return float(arr[: min(k, len(arr))].mean())


def extract_trial_id(csv_path):
    """
    Try to extract the trial ID from the file path.
    Examples:
        logs_bucb_beta_1.0/trial_3/beta_1.00/logs_terminal_output_jax_bucb.csv
        → trial_3
    """
    match = re.search(r"(trial_\d+)", csv_path)
    if match:
        return match.group(1)
    else:
        return "trial_unknown"


def process_csv(csv_path):
    """Compute metrics for one CSV file."""
    y = load_truef(csv_path)

    # Take the first 100 molecules (20 rounds × 5)
    if len(y) >= N_MOLECULES:
        y = y[:N_MOLECULES]

    curve = topk_running_mean(y, K)
    auc_val = auc_topk(y, K)
    final_val = final_topk(y, K)

    return dict(
        file=os.path.basename(csv_path),
        auc_topk=auc_val,
        final_topk=final_val,
        num_evals=len(y),
        curve=curve
    )


# ====== MAIN ======
def main():
    print("\n=== Single CSV Top-K Analysis ===")
    csv_path = input("Enter relative or full path to your CSV file: ").strip()

    # Resolve to absolute path for clarity
    abs_path = os.path.abspath(csv_path)
    print(f"\nResolved path: {abs_path}")

    if not os.path.exists(abs_path):
        print(f"❌ File not found: {abs_path}")
        return

    trial_id = extract_trial_id(abs_path)
    print(f"Detected trial ID: {trial_id}")

    print(f"✅ Found file, processing...\n")

    try:
        res = process_csv(abs_path)
    except Exception as e:
        print(f"❌ Error processing {abs_path}: {e}")
        return

    # Build running curve table
    n = len(res["curve"])
    rounds = np.arange(0, n / MOLECULES_PER_ROUND, 1 / MOLECULES_PER_ROUND)
    df_curve = pd.DataFrame({
        "Molecule_index": np.arange(1, n + 1),
        "Round": rounds[:n],
        f"Top{K}_running_mean": res["curve"]
    })

    # Output filenames based on trial
    base_name = f"logs_terminal_output_jax_ucb_beta_1.0_{trial_id}"
    out_curve_path = os.path.join(OUT_DIR, f"{base_name}_running_top{K}.csv")
    out_summary_path = os.path.join(OUT_DIR, f"{base_name}_summary_top{K}.csv")

    # Save outputs
    df_curve.to_csv(out_curve_path, index=False)
    pd.DataFrame([{
        "Trial": trial_id,
        "File": res["file"],
        f"AUC_Top{K}": res["auc_topk"],
        f"Final_Top{K}": res["final_topk"],
        "Num_evals": res["num_evals"]
    }]).to_csv(out_summary_path, index=False)

    # Print results
    print("✅ Results:")
    print(f"Trial: {trial_id}")
    print(f"File: {res['file']}")
    print(f"Num evaluations: {res['num_evals']}")
    print(f"Final Top-{K}: {res['final_topk']:.4f}")
    print(f"AUC Top-{K}: {res['auc_topk']:.4f}")
    print(f"\nSaved running curve → {out_curve_path}")
    print(f"Saved summary → {out_summary_path}")
    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
