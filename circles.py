import os
import re
import pandas as pd
import numpy as np
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

def extract_initial_smiles(log_text: str) -> list:
    match = re.findall(r"Initial SMILES:\s*\[((?:\s*'[^']+',?\s*)+)\]", log_text, re.DOTALL)
    if not match:
        raise ValueError("No initial SMILES found.")
    return re.findall(r"'([^']+)'", match[0])

def extract_initial_objectives(log_text: str) -> np.ndarray:
    """
    Parses the multi-line initial objective array from the log text.
    Handles formats like:
        Initial objectives:
        [[... ... ...]
         [... ... ...]
         ...
        ]
    """
    lines = log_text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if "Initial Y:" in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Initial objectives block not found in log.")

    objective_lines = []
    for line in lines[start_idx:]:
        if line.strip().startswith("["):
            objective_lines.append(line.strip().lstrip("[").rstrip("]"))
        else:
            break  # end of matrix block

    if not objective_lines:
        raise ValueError("Failed to parse objectives block.")

    # Convert lines into floats
    parsed = []
    for line in objective_lines:
        row = list(map(float, line.split()))
        parsed.append(row)

    return np.array(parsed)


