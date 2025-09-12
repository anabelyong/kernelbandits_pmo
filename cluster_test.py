#!/usr/bin/env python
"""
Compare Tanimoto Gram spectra for random vs. rho-packed batches
using the GuacaMol dataset.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


# ------------------------------
# Fingerprint utilities
# ------------------------------
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)

def smiles_to_fps(smiles_list):
    return [get_fingerprint(s) for s in smiles_list]

def tanimoto_distance(fp1, fp2):
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)

def build_tanimoto_kernel(fps):
    """Construct full Tanimoto kernel Gram matrix for given fingerprints."""
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])


# ------------------------------
# Packing (distance floor) selection
# ------------------------------
def select_packed_batch(smiles_list, batch_size=100, rho=0.9):
    """
    Greedy selection: enforce min Tanimoto distance >= rho.
    """
    fps = smiles_to_fps(smiles_list)
    selected = []
    selected_fps = []

    for fp, smi in zip(fps, smiles_list):
        if not selected:
            selected.append(smi)
            selected_fps.append(fp)
            continue

        # Check distance to all selected
        dists = [tanimoto_distance(fp, sfp) for sfp in selected_fps]
        if min(dists) >= rho:
            selected.append(smi)
            selected_fps.append(fp)

        if len(selected) >= batch_size:
            break

    return selected, selected_fps


# ------------------------------
# Eigenvalue analysis
# ------------------------------
def analyze_spectrum(K):
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.flip(np.sort(eigvals))  # sort descending
    return eigvals


# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    dataset_path = os.path.join("guacamol_dataset", "guacamol_v1_train.smiles")
    n_subset = 5000   # work with manageable subset
    batch_size = 100
    rho = 0.9       # distance floor

    # Load SMILES
    smiles = []
    with open(dataset_path, "r") as f:
        for line in f:
            smiles.append(line.strip())
    smiles = smiles[:n_subset]
    print(f"Loaded {len(smiles)} molecules.")

    # ---- Random batch
    random_batch = random.sample(smiles, batch_size)
    random_fps = smiles_to_fps(random_batch)
    K_random = build_tanimoto_kernel(random_fps)
    eig_random = analyze_spectrum(K_random)

    # ---- Rho-packed batch
    packed_batch, packed_fps = select_packed_batch(smiles, batch_size=batch_size, rho=rho)
    K_packed = build_tanimoto_kernel(packed_fps)
    eig_packed = analyze_spectrum(K_packed)

    print(f"Random batch size: {len(random_batch)}")
    print(f"Packed batch size: {len(packed_batch)}")

    # ---- Plot comparison
    plt.figure(figsize=(7, 5))
    plt.semilogy(eig_random, marker="o", linestyle="None", label="Random batch")
    plt.semilogy(eig_packed, marker="x", linestyle="None", label=f"Packed batch (rho={rho})")
    plt.xlabel("Index (sorted)")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title("Tanimoto Gram Spectra: Random vs. Packed")
    plt.legend()
    plt.tight_layout()

    # Save as PDF instead of showing
    outfile = f"tanimoto_spectra_random_vs_packed_rho{rho}.pdf"
    plt.savefig(outfile, format="pdf")
    print(f"Saved spectrum comparison plot to {outfile}")
