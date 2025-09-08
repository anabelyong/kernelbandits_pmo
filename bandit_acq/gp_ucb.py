#!/usr/bin/env python
"""
Gaussian Process Upper Confidence Bound (GP-UCB) with Tanimoto kernel.
Implements batch UCB selection for virtual screening.
"""

from __future__ import annotations
import argparse
import json
import logging
import time
import numpy as np
import pandas as pd

from kernel_only_GP.tanimoto_gp import ZeroMeanTanimotoGP, TanimotoGP_Params, get_fingerprint
from tdc_oracles_modified import Oracle  

logger = logging.getLogger(__name__)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SMILES dataset (.smiles).")
    parser.add_argument("--oracle_name", type=str, required=True, help="Which oracle to use (from TDC wrapper).")
    parser.add_argument("--output_json", type=str, required=True, help="Where to save results.")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of molecules to subsample.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for BO.")
    parser.add_argument("--budget", type=int, default=100, help="Total number of molecules to evaluate.")
    parser.add_argument("--beta", type=float, default=2.0, help="UCB exploration parameter.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load dataset
    smiles_all = pd.read_csv(args.dataset_path, header=None)[0].tolist()
    smiles_all = smiles_all[: args.dataset_size]

    # Init oracle
    oracle = Oracle(args.oracle_name)

    # Warm start: pick one random molecule
    init_smiles = [smiles_all[rng.integers(len(smiles_all))]]
    init_y = oracle(init_smiles)

    # GP params (simple fixed init, could optimize MLL here)
    params = TanimotoGP_Params(
        raw_amplitude=np.array(1.0),
        raw_noise=np.array(1e-4),
    )

    # Initialize GP
    gp = ZeroMeanTanimotoGP(get_fingerprint, init_smiles, init_y)

    # Optimization loop
    evaluated = list(init_smiles)
    evaluated_y = list(init_y)
    pool = [s for s in smiles_all if s not in evaluated]

    results = []

    for t in range(0, args.budget, args.batch_size):
        logger.info(f"Iteration {t} / {args.budget}")

        # Predict posterior for pool
        mu, cov = gp.predict_y(params, pool, full_covar=False)
        mu = np.asarray(mu).flatten()
        sigma = np.sqrt(np.asarray(cov))

        # Compute UCB
        ucb_scores = mu + args.beta * sigma

        # Pick top-k molecules
        t_start = time.monotonic()
        topk_idx = np.argsort(-ucb_scores)[: args.batch_size]
        chosen_smiles = [pool[i] for i in topk_idx]
        t_end = time.monotonic()

        # Evaluate oracle
        y_batch = oracle(chosen_smiles)

        # Update GP training data
        evaluated.extend(chosen_smiles)
        evaluated_y.extend(y_batch)
        gp.set_training_data(evaluated, evaluated_y)

        # Remove from pool
        pool = [s for s in pool if s not in chosen_smiles]

        # Save iteration result
        results.append(
            dict(
                method="gp_ucb",
                time=t_end - t_start,
                smiles=chosen_smiles,
                y_values=[float(v) for v in y_batch],
            )
        )

    # Save all results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
