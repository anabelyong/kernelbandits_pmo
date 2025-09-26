#!/usr/bin/env python3.9
"""
GP-BUCB with a Tanimoto kernel GP (JAX implementation you already have).
- Within each round, select a batch of size K *without feedback*.
- After each provisional pick, apply the variance-only (hallucinated) update
  to discourage picking near-duplicates in the same batch.

Outputs:
  csv_results/gp_bucb_fex.csv with columns: iter, idx_in_batch, smiles, ucb, mu, std, true_f
Logging:
  logs_trialX/terminal_output_jax_bucb_fex.log (like your other scripts)
"""

import os, sys, time, random, logging
import numpy as np
import pandas as pd
from pprint import pprint
from rdkit import DataStructs
from typing import Union
from jax.nn import softplus

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO  # your oracle

# ----------------- Config / logging -----------------
TRIAL_TAG = os.environ.get("TRIAL_TAG", "trial1")
LOG_DIR   = f"logs_{TRIAL_TAG}"
CSV_DIR   = "csv_results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "terminal_output_jax_bucb_fex.log")
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

bo_logger = logging.getLogger("bo_loop_logger")
bo_logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_logger.addHandler(h)

# --------------- Helpers: Tanimoto rows, packing mask ---------------
def k_tanimoto_row(sm, pool_smiles, fps_cache):
    """k(x, ·) as a vector over pool, WITHOUT amplitude."""
    return np.array(
        DataStructs.BulkTanimotoSimilarity(fps_cache[sm], [fps_cache[s] for s in pool_smiles]),
        dtype=float,
    )

def apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho):
    """Mask scores for pool points that violate min Tanimoto distance >= rho to any selected."""
    if rho is None or not selected:
        return acq
    ok = np.ones_like(acq, dtype=bool)
    for sb in selected:
        sims = k_tanimoto_row(sb, pool_smiles, fps_cache)  # similarity
        ok &= (1.0 - sims) >= rho                          # distance >= rho
    return np.where(ok, acq, -np.inf)

# --------------- Core: build one BUCB batch ---------------
def build_batch_bucb(
    gp, params,
    pool_smiles, fps_cache,
    K, sqrt_beta,
    rho_floor=None,
):
    """
    Build a batch of K with BUCB:
      - use mu, sigma_f from gp.predict_f
      - after each pick, variance-only update:
            v_new = v_old - (cov(x, x*)^2) / (lambda + v_old[x*])
        where cov(x, x*) = a * Tanimoto(x, x*)
      - optional Tanimoto distance floor rho_floor inside the batch
    """
    # Predict f-mean and f-variance (NO noise) once at the start of the round
    mu_f, var_f = gp.predict_f(params, pool_smiles, full_covar=False)
    mu = np.asarray(mu_f).flatten()
    v_work = np.asarray(var_f).flatten()  # latent variance tilde{sigma}^2

    a = float(softplus(params.raw_amplitude))
    lambda_obs = float(softplus(params.raw_noise))

    selected = []
    idxs_in_pool = []  # indices into pool_smiles

    for j in range(K):
        std = np.sqrt(np.maximum(v_work, 1e-12))
        acq = mu + sqrt_beta * std
        acq = apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho_floor)

        i_star = int(np.argmax(acq))
        x_star = pool_smiles[i_star]
        selected.append(x_star)
        idxs_in_pool.append(i_star)

        # variance-only update everywhere
        k_star = a * k_tanimoto_row(x_star, pool_smiles, fps_cache)  # cov to all pool
        denom = lambda_obs + v_work[i_star]
        v_work = v_work - (k_star * k_star) / max(denom, 1e-12)
        v_work[i_star] = 0.0  # prevent self re-selection

    return selected, idxs_in_pool

# --------------- Main BO loop (batched) ---------------
def run_bucb(
    smiles_all: list[str],
    init_size: int = 10,
    n_rounds: int = 200,          # iterations of "add one molecule" if K=1, else rounds
    batch_size: int = 5,          # K
    sqrt_beta: float = 0.316,     # = sqrt(beta)
    gp_amplitude: float = 1.0,
    gp_noise: float = 1e-4,
    rho_floor: Union[float, None] = None,   # e.g., 0.3 or None
    csv_out: str = os.path.join(CSV_DIR, "gp_bucb_fex.csv"),
):
    rng = random.Random(0)

    # Split init/pool
    smiles = smiles_all.copy()
    rng.shuffle(smiles)
    init_smiles = smiles[:init_size]
    pool_smiles = smiles[init_size:]

    # Cache fingerprints
    fps_cache = {s: get_fingerprint(s) for s in smiles}

    # Warm start labels
    Y0 = evaluate_fex_MPO(init_smiles)  # (n,1)
    bo_logger.info(f"Warm start: {init_size} points")

    # GP init
    gp = ZeroMeanTanimotoGP(lambda s: fps_cache[s], init_smiles, Y0[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude) - 1.0),
        raw_noise=np.log(np.exp(gp_noise) - 1.0),
    )

    chosen = set(init_smiles)

    # CSV writer
    rows = []
    global_iter = 0

    for r in range(n_rounds):
        t0 = time.time()
        pool = [s for s in pool_smiles if s not in chosen]
        if not pool:
            bo_logger.info("Pool exhausted.")
            break

        # Build a BUCB batch
        B = min(batch_size, len(pool))
        batch, idxs = build_batch_bucb(
            gp, params, pool, fps_cache,
            K=B, sqrt_beta=sqrt_beta, rho_floor=rho_floor,
        )

        # Evaluate oracle on the batch
        y_batch = evaluate_fex_MPO(batch)[:, 0].tolist()

        # Log and update GP state
        mu_f, var_f = gp.predict_f(params, batch, full_covar=False)
        std_f = np.sqrt(np.maximum(np.asarray(var_f).flatten(), 1e-12))
        ucb_vals = np.asarray(mu_f).flatten() + sqrt_beta * std_f

        for j, (smi, f_true, mu_j, std_j, ucb_j) in enumerate(zip(batch, y_batch, np.asarray(mu_f).flatten(), std_f, ucb_vals)):
            rows.append(dict(
                iter=global_iter + 1,
                idx_in_batch=j,
                smiles=smi,
                ucb=float(ucb_j),
                mu=float(mu_j),
                std=float(std_j),
                **{"True f": float(f_true)},
            ))
            bo_logger.info(f"[round {r}] pick {j+1}/{B}: {smi} | UCB={ucb_j:.4g} | f={f_true:.4g}")
            chosen.add(smi)
            global_iter += 1

        # Update GP with new labels
        new_X = list(gp._smiles_train) + batch
        new_y = np.concatenate([np.asarray(gp._y_train), np.asarray(y_batch)], axis=0)
        gp.set_training_data(new_X, new_y)

        bo_logger.info(f"Round {r} done in {time.time()-t0:.2f}s; train size={len(new_X)}")

    # Write CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    bo_logger.info(f"Wrote: {csv_out}")

# ----------------- Script entry -----------------
if __name__ == "__main__":
    # Load GuacaMol (first 100k to keep things tractable)
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    smiles_all = df["smiles"].tolist()[:100_000]
    pprint(smiles_all[:5])

    run_bucb(
        smiles_all=smiles_all,
        init_size=10,
        n_rounds=40,         # with K=5 this gives 200 evals total
        batch_size=5,
        sqrt_beta=0.316,
        gp_amplitude=1.0,
        gp_noise=1e-4,
        rho_floor=None,      # set e.g. 0.3 if you want a hard packing floor
        csv_out=os.path.join(CSV_DIR, f"logs_{TRIAL_TAG}_gp_bucb_fex.csv"),
    )

    sys.stdout.close()
