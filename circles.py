#!/usr/bin/env python
"""
Compute Pareto-optimal SMILES and NumCircles metrics
for all BUCB and BUCB+ρ (packing constraint) runs.

Automatically iterates through:
    logs_bucb_beta_1.0/trial_X/beta_1.00/*.csv
    logs_bucb_pack/trial_X/rho_Y/beta_1.00/**/*.csv

Preserves the full directory structure under the output folder.
Cuts off at 100 molecules (20 rounds × 5) per run for consistency.
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List, Any
from rdkit import Chem, DataStructs

# ---- project imports ----
from circles.measures import NCircles
from acquisition_funcs.pareto import pareto_front
from kernel_only_GP.tanimoto_gp import get_fingerprint
from utils.utils import evaluate_fex_objectives


# ---------- Helper functions ----------
def vectorizer(smiles_list: List[str]) -> List[Any]:
    """Convert SMILES to fingerprints."""
    return [get_fingerprint(smi) for smi in tqdm(smiles_list, desc="Vectorizing")]

def sim_matrix(fps_a: List[Any], fps_b: List[Any]) -> List[List[float]]:
    """Compute Tanimoto similarity matrix."""
    return [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a]

def compute_circles(smiles_list: List[str], threshold: float) -> int:
    """Compute the number of circles given SMILES and Tanimoto threshold."""
    ncircle = NCircles(vectorizer=vectorizer, sim_mat_func=sim_matrix, threshold=threshold)
    n_circles, _ = ncircle.measure(smiles_list)
    return n_circles


# ---------- Main ----------
if __name__ == "__main__":
    ROOTS = ["logs_bucb_beta_1.0", "logs_bucb_pack"]
    OUT_ROOT = "circles_results_BUCB_and_pack_trial"
    os.makedirs(OUT_ROOT, exist_ok=True)

    thresholds = np.arange(0.1, 0.91, 0.05)
    summary_rows = []

    N_MOLECULES = 100  # cutoff limit

    for root in ROOTS:
        print(f"\n🔍 Searching in {root} ...")
        csv_paths = glob(os.path.join(root, "**", "*.csv"), recursive=True)
        csv_paths = [p for p in csv_paths if "logs_terminal_output" in p]

        if not csv_paths:
            print(f"⚠️ No matching CSVs found in {root}")
            continue

        for csv_path in sorted(csv_paths):
            print(f"\n== Processing {csv_path} ==")
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[ERROR] Could not read {csv_path}: {e}")
                continue

            # Try to find SMILES column
            smiles_col = None
            for candidate in ["Selected SMILES", "SMILES", "smiles"]:
                if candidate in df.columns:
                    smiles_col = candidate
                    break

            if smiles_col is None:
                print(f"[WARN] No SMILES column in {csv_path}. Skipping.")
                continue

            smiles_list = df[smiles_col].dropna().tolist()
            if len(smiles_list) == 0:
                print(f"[WARN] Empty SMILES list in {csv_path}. Skipping.")
                continue

            # ✅ Cutoff at 100 molecules
            if len(smiles_list) > N_MOLECULES:
                print(f"⚠️ Found {len(smiles_list)} molecules, truncating to first {N_MOLECULES}.")
                smiles_list = smiles_list[:N_MOLECULES]

            # Evaluate objectives
            print(f"Evaluating {len(smiles_list)} SMILES with evaluate_fex_objectives()...")
            try:
                objectives = evaluate_fex_objectives(smiles_list)
            except Exception as e:
                print(f"[ERROR] Objective evaluation failed for {csv_path}: {e}")
                continue

            if not isinstance(objectives, np.ndarray):
                objectives = np.array(objectives)
            if objectives.ndim == 1:
                objectives = objectives.reshape(-1, 1)

            # Pareto front
            mask = pareto_front(objectives, maximize=True)
            pareto_smiles = [smiles_list[i] for i in range(len(smiles_list)) if mask[i]]
            print(f"→ Found {len(pareto_smiles)} Pareto-optimal SMILES out of {len(smiles_list)}")

            if len(pareto_smiles) == 0:
                print(f"[WARN] No Pareto-optimal SMILES in {csv_path}. Skipping.")
                continue

            # Compute Circles
            results = []
            for t in thresholds:
                print(f"Computing NumCircles at threshold {t:.2f}...")
                try:
                    n = compute_circles(pareto_smiles, threshold=t)
                    results.append({"Threshold": t, "NumCircles": n})
                    print(f"  → {n}")
                except Exception as e:
                    print(f"  → Failed at t={t:.2f}: {e}")
                    results.append({"Threshold": t, "NumCircles": np.nan})

            # --- Rebuild output directory hierarchy ---
            rel_path = os.path.relpath(csv_path, start=root)
            save_dir = os.path.join(OUT_ROOT, root, os.path.dirname(rel_path))
            os.makedirs(save_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(csv_path))[0]

            # Save per-run results
            out_path = os.path.join(save_dir, f"{base_name}_pareto_fex_circles.csv")
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"✅ Saved circles metrics to {out_path}")

            # Save Pareto SMILES
            smiles_out = os.path.join(save_dir, f"{base_name}_pareto_smiles.csv")
            pd.DataFrame({"SMILES": pareto_smiles}).to_csv(smiles_out, index=False)
            print(f"✅ Saved Pareto SMILES list to {smiles_out}")

            # --- Log summary ---
            mean_circles = np.nanmean([r["NumCircles"] for r in results])
            trial = None
            rho = None
            if "trial_" in csv_path:
                trial = csv_path.split("trial_")[1].split("/")[0]
            if "rho_" in csv_path:
                rho = csv_path.split("rho_")[1].split("/")[0]

            summary_rows.append({
                "Run": base_name,
                "Source": root,
                "Trial": trial,
                "Rho": rho,
                "NumPareto": len(pareto_smiles),
                "MeanNumCircles": mean_circles
            })

    # --- Save combined summary ---
    if summary_rows:
        summary_path = os.path.join(OUT_ROOT, "summary_all_trials.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f" Master summary written to {summary_path}")
    else:
        print("\n⚠️ No valid runs processed.")
