#!/usr/bin/env python
import os
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ------------------------------
# Fingerprint utilities
# ------------------------------
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)

def smiles_to_fps(smiles_list):
    return [get_fingerprint(s) for s in smiles_list]

def build_tanimoto_kernel(fps):
    """
    Construct full Tanimoto kernel Gram matrix.
    """
    return np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])

# ------------------------------
# Load dataset
# ------------------------------
def load_smiles_dataset(path: str, n_subset: int = 5000):
    """
    Load SMILES dataset and take a subset for tractability.
    """
    smiles = []
    with open(path, "r") as f:
        for line in f:
            smiles.append(line.strip())
    smiles = smiles[:n_subset]
    return smiles

# ------------------------------
# Eigenvalue spectrum
# ------------------------------
def analyze_spectrum(K):
    """
    Compute eigenvalues of kernel Gram matrix.
    """
    eigvals = np.linalg.eigvalsh(K)  # symmetric, so eigvalsh is faster
    eigvals = np.flip(np.sort(eigvals))  # sort descending
    return eigvals

def plot_spectrum(eigvals, title="Tanimoto Kernel Spectrum"):
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

    # Load a manageable subset (say 5000 molecules)
    smiles_subset = load_smiles_dataset(dataset_path, n_subset=100000)
    print(f"Loaded {len(smiles_subset)} molecules.")

    # Build fingerprints and kernel
    fps = smiles_to_fps(smiles_subset)
    print("Computing kernel...")
    K = build_tanimoto_kernel(fps)
    print("Kernel shape:", K.shape)

    # Eigenvalue analysis
    eigvals = analyze_spectrum(K)
    print("Top 10 eigenvalues:", eigvals[:10])

    # Plot spectrum decay
    plot_spectrum(eigvals)

