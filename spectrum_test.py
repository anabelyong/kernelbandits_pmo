#!/usr/bin/env python
import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

# ------------------------------
# Fingerprint utilities
# ------------------------------
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)

def build_tanimoto_kernel(fps):
    """
    Construct full Tanimoto kernel Gram matrix.
    """
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

# ------------------------------
# Load dataset
# ------------------------------
def load_smiles_dataset(path: str, n_subset: int = None):
    """
    Load SMILES dataset. Optionally truncate to n_subset.
    """
    smiles = []
    with open(path, "r") as f:
        for line in f:
            smiles.append(line.strip())
    if n_subset:
        smiles = smiles[:n_subset]
    return smiles

# ------------------------------
# Nyström approximation
# ------------------------------
def nystrom_approximation(smiles_list, m=2000, seed=0):
    """
    Nyström approximation of Tanimoto spectrum.
    """
    rng = np.random.default_rng(seed)
    n = len(smiles_list)

    # Subsample m molecules
    chosen_idx = rng.choice(n, size=m, replace=False)
    smiles_sub = [smiles_list[i] for i in chosen_idx]
    fps_sub = [get_fingerprint(s) for s in smiles_sub]

    # Build kernel on subsample
    K_mm = build_tanimoto_kernel(fps_sub)

    # Eigen-decomposition
    eigvals, _ = np.linalg.eigh(K_mm)
    eigvals = np.flip(np.sort(eigvals))

    # Nyström correction
    eigvals_scaled = (n / m) * eigvals
    return eigvals_scaled

# ------------------------------
# Analysis
# ------------------------------
def analyze_effective_dimension(eigvals, lam=1e-4):
    """
    Effective dimension: sum of lambda_i / (lambda_i + lam).
    """
    return np.sum(eigvals / (eigvals + lam))

def plot_spectrum(eigvals, title="Nyström-approximated Tanimoto Spectrum"):
    plt.figure(figsize=(6,4))
    plt.semilogy(eigvals, marker="o", linestyle="None")
    plt.xlabel("Index (sorted)")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main script
# ------------------------------
if __name__ == "__main__":
    # Path to your GuacaMol dataset
    dataset_path = os.path.join("guacamol_dataset", "guacamol_v1_train.smiles")

    # Load all molecules (GuacaMol has ~1.6M, but you may only want 100k)
    smiles_all = load_smiles_dataset(dataset_path, n_subset=100_000)
    print(f"Loaded {len(smiles_all)} molecules.")

    # Nyström approximation with subset m
    eigvals_scaled = nystrom_approximation(smiles_all, m=2000, seed=42)
    print("Top 10 eigenvalues (scaled):", eigvals_scaled[:10])

    # Effective dimension
    d_eff = analyze_effective_dimension(eigvals_scaled, lam=1e-4)
    print("Effective dimension (λ=1e-4):", d_eff)

    # Plot spectrum decay
    plot_spectrum(eigvals_scaled)
