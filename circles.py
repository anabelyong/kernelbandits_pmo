#!/usr/bin/env python
"""
Compute NumCircles from BO CSVs for different Tanimoto thresholds.
Processes both UCB and UCB+aug logs.
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from rdkit import Chem, DataStructs
from tqdm import tqdm
from typing import List, Any

from circles.measures import NCircles
from kernel_only_GP.tanimoto_gp import get_fingerprint

def vectorizer(smiles_list: List[str]) -> List[Any]:
    return [get_fingerprint(smi) for smi in tqdm(smiles_list, desc="Vectorizing")]

def sim_matrix(fps_a: List[Any], fps_b: List[Any]) -> List[List[float]]:
    return [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a]

def compute_circles(smiles_list: List[str], threshold: float) -> int:
    ncircle = NCircles(vectorizer=vectorizer, sim_mat_func=sim_matrix, threshold=threshold)
    n_circles, _ = ncircle.measure(smiles_list)
    return n_circles

if __name__ == "__main__":
    IN_DIR = "csv_results"
    OUT_DIR = "circles_results"
    os.makedirs(OUT_DIR, exist_ok=True)

    thresholds = np.arange(0.1, 0.91, 0.05)

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
        if not smiles_list:
            print(f"[WARN] Empty SMILES list in {csv_path}. Skipping.")
            continue

        results = []
        for t in thresholds:
            print(f"Computing NumCircles at threshold {t:.2f}...")
            try:
                n = compute_circles(smiles_list, threshold=t)
                results.append({"Threshold": t, "NumCircles": n})
                print(f"  → {n}")
            except Exception as e:
                print(f"  → Failed at t={t:.2f}: {e}")
                results.append({"Threshold": t, "NumCircles": np.nan})

        out_path = os.path.join(OUT_DIR, f"{name}_circles.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")
