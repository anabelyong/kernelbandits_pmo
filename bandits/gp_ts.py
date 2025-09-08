"""
# code modified from https://github.com/AustinT/tanimoto-random-features-neurips23/blob/main/experiment_scripts/bo_with_thompson_sampling.py
# script for exact Thompson Sampling 
# PDTS (https://proceedings.mlr.press/v70/hernandez-lobato17a/hernandez-lobato17a.pdf) forms batches from independent posterior function draws and dispatches a ranked list. 
"""
from __future__ import annotations 
import argparse
import collections
import json
import logging
import random
import pandas as pd 
import time

import numpy as np

logger = logging.getLogger(__name__)

# dataset load 
guacamol_df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header = None, names = ["smiles"])
all_smiles = guacamol_df["smiles".tolist()[:100000]]
random.shuffle(all_smiles)

#class GPTS():


