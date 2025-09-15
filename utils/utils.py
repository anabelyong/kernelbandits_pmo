#!/usr/bin/env python
#creating objective functions SEPARATELY for 3 different MPOs: fexofenadine, ranolazine, and osimertinib objectives
#UPDATED: created separated MPOs for 2 objectives: perindopril and amlodipine MPOs
from pprint import pprint

import numpy as np
import pandas as pd

from tdc_oracles_modified import Oracle

# Load GuacaMol dataset
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
guacamol_dataset_path = os.path.join(current_dir, "../guacamol_dataset/guacamol_v1_train.smiles")

# Confirm the exact path being used
# print("Final path to dataset:", guacamol_dataset_path)
guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]
known_smiles = ALL_SMILES[:10]
#print("Known SMILES:")
#pprint(known_smiles)

# Create "oracles" for FEXOFENADINE objectives
TPSA_ORACLE = Oracle("tpsa_score_single")
LOGP_ORACLE = Oracle("logp_score_single")
FEXOFENADINE_SIM_ORACLE = Oracle("fex_similarity_value_single")

# Create "oracles" for OSIMERTINIB objectives
OSIM_TPSA_ORACLE = Oracle("osimertinib_tpsa_score")
OSIM_LOGP_ORACLE = Oracle("osimertinib_logp_score")
OSIM_SIM_V1_ORACLE = Oracle("osimertinib_similarity_v1_score")
OSIM_SIM_V2_ORACLE = Oracle("osimertinib_similarity_v2_score")

# Create "oracles" for RANOLAZINE objectives
RANOL_TPSA_ORACLE = Oracle("ranolazine_tpsa_score")
RANOL_LOGP_ORACLE = Oracle("ranolazine_logp_score")
RANOL_SIM_ORACLE = Oracle("ranolazine_similarity_value")
RANOL_FLUORINE_ORACLE = Oracle("ranolazine_fluorine_value")

#Create "oracles" for PERINDOPRIL objectives (2 objectives)
PERIN_SIM_ORACLE = Oracle("perindopril_similarity_value")
PERIN_AROM_RINGS_ORACLE = Oracle("perindopril_aromatic_rings")

#Create "oracles" for AMLODIPINE objectives (2 objectives)
AMLO_SIM_ORACLE = Oracle("amlodipine_similarity_value")
AMLO_RINGS_ORACLE = Oracle("amlodipine_num_rings_value")

# 1st MPOs to investigate as baseline
FEXOFENADINE_MPO_ORACLE = Oracle("fexofenadine_mpo")
#2nd and 3rd MPO to investigate as baseline
OSIMERTINIB_MPO_ORACLE = Oracle("osimertinib_mpo")
RANOLAZINE_MPO_ORACLE = Oracle("ranolazine_mpo")

#2D objectives MPO to investigate as baseline 
PERINDOPRIL_MPO_ORACLE = Oracle("perindopril_mpo")
AMLODIPINE_MPO_ORACLE = Oracle("amlodipine_mpo")

def evaluate_perin_objectives(smiles_list: list[str]) -> np.ndarray:
    # Initialize arrays for each objective
    f1 = np.array(PERIN_SIM_ORACLE(smiles_list))
    f2 = np.array(PERIN_AROM_RINGS_ORACLE(smiles_list))
    
    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    
    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1) & ~np.isnan(f2)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2])  
    return out.T  # transpose, Nx2

def evaluate_amlo_objectives(smiles_list: list[str]) -> np.ndarray:
    # Initialize arrays for each objective
    f1 = np.array(AMLO_SIM_ORACLE(smiles_list))
    f2 = np.array(AMLO_RINGS_ORACLE(smiles_list))
    
    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    
    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1) & ~np.isnan(f2)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2])  
    return out.T  # transpose, Nx2

def evaluate_fex_objectives(smiles_list: list[str]) -> np.ndarray:
    # Initialize arrays for each objective
    f1 = np.array(TPSA_ORACLE(smiles_list))
    f2 = np.array(FEXOFENADINE_SIM_ORACLE(smiles_list))
    f3 = np.array(LOGP_ORACLE(smiles_list))
    
    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3])  # 4x3
    return out.T  # transpose, Nx3

def evaluate_osim_objectives(smiles_list: list[str]) -> np.ndarray:
    # Initialize arrays for each objective
    f1 = np.array(OSIM_TPSA_ORACLE(smiles_list))
    f2 = np.array(OSIM_LOGP_ORACLE(smiles_list))
    f3 = np.array(OSIM_SIM_V1_ORACLE(smiles_list))
    f4 = np.array(OSIM_SIM_V2_ORACLE(smiles_list))


    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")
    print(f"f4 shape: {f4.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1) & ~np.isnan(f2) & ~np.isnan(f3) & ~np.isnan(f4)

    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]
    f4 = f4[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3) == len(f4)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3, f4])  
    return out.T  # transpose, Nx4

def evaluate_ranol_objectives(smiles_list: list[str]) -> np.ndarray:
    # Initialize arrays for each objective
    f1 = np.array(RANOL_TPSA_ORACLE(smiles_list))
    f2 = np.array(RANOL_LOGP_ORACLE(smiles_list))
    f3 = np.array(RANOL_SIM_ORACLE(smiles_list))
    f4 = np.array(RANOL_FLUORINE_ORACLE(smiles_list))

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")
    print(f"f3 shape: {f3.shape}")
    print(f"f4 shape: {f4.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1) & ~np.isnan(f2) & ~np.isnan(f3) & ~np.isnan(f4)

    f1 = f1[valid_indices]
    f2 = f2[valid_indices]
    f3 = f3[valid_indices]
    f4 = f4[valid_indices]

    # Ensure all arrays have the same shape
    if not (len(f1) == len(f2) == len(f3) == len(f4)):
        raise ValueError("All input arrays must have the same shape")

    out = np.stack([f1, f2, f3, f4])  
    return out.T  # transpose, Nx4

"""
SINGLE MPO OBJECTIVES <- objectives are not separated here: 
1) fexofenadine MPO 
2) ranolazine MPO 
3) osimertinib MPO 
"""
def evaluate_fex_MPO(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array(FEXOFENADINE_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1

def evaluate_ranol_MPO(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array(RANOLAZINE_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1

def evaluate_osim_MPO(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array(OSIMERTINIB_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1

def evaluate_perin_MPO(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array(PERINDOPRIL_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1

def evaluate_amlo_MPO(smiles_list: list[str]) -> np.ndarray:
    f1 = np.array(AMLODIPINE_MPO_ORACLE(smiles_list))
    print(f"f1 shape: {f1.shape}")

    # Filter out NaN values from f1 and corresponding entries in other arrays
    valid_indices = ~np.isnan(f1)
    f1 = f1[valid_indices]

    out = np.stack([f1])  # Nx1
    return out.T  # transpose, Nx1