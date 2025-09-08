#!/usr/bin/env python
"""
Gaussian Process Thompson Sampling (GP-TS) with Tanimoto kernel.
Implements exact Thompson Sampling with ranked-list batch selection,
Following Hernández-Lobato et al. (2017) and the Tanimoto Random Features paper by Austin
"""

from __future__ import annotations
import argparse
import collections
import json
import logging
import time
import numpy as np

from kernel_only_GP.tanimoto_gp import ZeroMeanTanimotoGP, TanimotoGP_Params, get_fingerprint
from tdc_oracles_modified import Oracle

logger = logging.getLogger(__name__)


def select_batch_from_thompson_samples(
    samples: np.ndarray,
    batch_size: int,
    forbidden_indices: set[int],
) -> list[int]:
    """Ranked list batch selection from TS samples (avoids duplicates).
    Args:
        samples: array of shape (n_samples, n_points) with function draws.
        batch_size: number of molecules to select.
        forbidden_indices: indices excluded from selection.
    Returns:
        List of selected indices (size = batch_size).
    """
    assert samples.ndim == 2
    samples_argsort = np.argsort(-samples, axis=1)  # descending
    out_set: set[int] = set()
    current_rank = 0
    while len(out_set) < batch_size:
        logger.debug(f"\tTS: Adding rank #{current_rank+1} samples.")
        counter = collections.Counter(list(samples_argsort[:, current_rank]))
        for i, _ in counter.most_common():
            if i not in forbidden_indices and len(out_set) < batch_size:
                out_set.add(i)
        current_rank += 1
    out_list = list(out_set)
    assert len(out_list) == batch_size
    return out_list


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to SMILES dataset (.smiles).")
    parser.add_argument("--oracle_name", type=str, required=True, help="Which oracle to use (from TDC wrapper).")
    parser.add_argument("--output_json", type=str, required=True, help="Where to save results.")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of molecules to subsample.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for BO.")
    parser.add_argument("--budget", type=int, default=100, help="Total number of molecules to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load dataset
    import pandas as pd
    smiles_all = pd.read_csv(args.dataset_path, header=None)[0].tolist()
    smiles_all = smiles_all[: args.dataset_size]

    # Init oracle
    oracle = Oracle(args.oracle_name)

    # Warm start: pick one random molecule
    init_smiles = [smiles_all[rng.integers(len(smiles_all))]]
    init_y = oracle(init_smiles)

    # GP params (random init, could optimize MLL here)
    params = TanimotoGP_Params(
        raw_amplitude=np.array(1.0),  # log-untransformed
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
        mu, cov = gp.predict_y(params, pool, full_covar=True)
        mu = np.asarray(mu).flatten()
        cov = np.asarray(cov)

        # Draw TS samples
        t_start = time.monotonic()
        samples = rng.multivariate_normal(mean=mu, cov=cov, size=args.batch_size)
        t_end = time.monotonic()

        # Ranked-list batch selection
        chosen_idx = select_batch_from_thompson_samples(samples, args.batch_size, forbidden_indices=set())
        chosen_smiles = [pool[i] for i in chosen_idx]

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
                method="gp_ts_exact",
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
