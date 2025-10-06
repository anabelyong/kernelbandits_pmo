#!/usr/bin/env python
"""
Compute Pareto-optimal SMILES based on fresh evaluate_fex_objectives()
evaluations, then compute NumCircles across Tanimoto thresholds
for all CSVs in csv_BUCB_vs_UCB/.
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

if __name__ == "__main__":
    IN_DIR = "csv_results"       
    OUT_DIR = "circles_results"
    os.makedirs(OUT_DIR, exist_ok=True)

    thresholds = np.arange(0.1, 0.91, 0.05)

    # Iterate over *all* CSVs in the folder
    for csv_path in sorted(glob(os.path.join(IN_DIR, "*.csv"))):
        name = os.path.basename(csv_path).replace(".csv", "")
        print(f"\n== Processing {name} ==")

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Could not read {csv_path}: {e}")
            continue

        if "Selected SMILES" not in df.columns:
            print(f"[WARN] No 'Selected SMILES' column in {csv_path}. Skipping.")
            continue

        smiles_list = df["Selected SMILES"].dropna().tolist()
        if len(smiles_list) == 0:
            print(f"[WARN] Empty SMILES list in {csv_path}. Skipping.")
            continue

        # Evaluate Fex objectives for all SMILES
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

        # Pareto front filtering
        mask = pareto_front(objectives, maximize=True)
        pareto_smiles = [smiles_list[i] for i in range(len(smiles_list)) if mask[i]]
        print(f"→ Found {len(pareto_smiles)} Pareto-optimal SMILES "
              f"from {len(smiles_list)} total entries")

        if len(pareto_smiles) == 0:
            print(f"[WARN] No Pareto-optimal SMILES in {csv_path}. Skipping.")
            continue

        # Compute NumCircles across thresholds
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

        # Save results
        out_path = os.path.join(OUT_DIR, f"{name}_pareto_fex_circles.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved Pareto+Circles results to {out_path}")

        # (Optional) Save Pareto SMILES themselves for inspection
        smiles_out = os.path.join(OUT_DIR, f"{name}_pareto_smiles.csv")
        pd.DataFrame({"SMILES": pareto_smiles}).to_csv(smiles_out, index=False)
        print(f"Saved Pareto SMILES list to {smiles_out}")
