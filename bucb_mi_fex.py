#!/usr/bin/env python
"""
GP-BUCB + tiny MI bonus:
Inside each batch, acquisition = mu + sqrt_beta * sigma + eta_t * zscore(MI),
where MI(x) = 0.5 * log(1 + v_work(x)/lambda_obs).
"""

import os, sys, time, random, logging
import numpy as np
import pandas as pd
from pprint import pprint
from rdkit import DataStructs
from jax.nn import softplus
from typing import Union 

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO

TRIAL_TAG = os.environ.get("TRIAL_TAG", "trial1")
LOG_DIR   = f"logs_{TRIAL_TAG}"
CSV_DIR   = "csv_results"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "terminal_output_jax_bucb_mi_fex.log")
sys.stdout = open(log_file, "w"); sys.stderr = sys.stdout
bo_logger = logging.getLogger("bo_loop_logger")
bo_logger.setLevel(logging.INFO)
h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_logger.addHandler(h)

def k_tanimoto_row(sm, pool_smiles, fps_cache):
    return np.array(
        DataStructs.BulkTanimotoSimilarity(fps_cache[sm], [fps_cache[s] for s in pool_smiles]),
        dtype=float,
    )

def apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho):
    if rho is None or not selected:
        return acq
    ok = np.ones_like(acq, dtype=bool)
    for sb in selected:
        sims = k_tanimoto_row(sb, pool_smiles, fps_cache)
        ok &= (1.0 - sims) >= rho
    return np.where(ok, acq, -np.inf)

def zscore(v, eps=1e-8):
    m = float(np.mean(v)); s = float(np.std(v))
    s = s if s > eps else 1.0
    return (v - m) / s, m, s

def build_batch_bucb_mi(
    gp, params,
    pool_smiles, fps_cache,
    K, sqrt_beta, eta_t,
    rho_floor=None,
):
    mu_f, var_f = gp.predict_f(params, pool_smiles, full_covar=False)
    mu = np.asarray(mu_f).flatten()
    v_work = np.asarray(var_f).flatten()

    a = float(softplus(params.raw_amplitude))
    lambda_obs = float(softplus(params.raw_noise))

    selected, idxs = [], []

    for j in range(K):
        std = np.sqrt(np.maximum(v_work, 1e-12))
        # tiny MI bonus using current hallucinated variance
        mi = 0.5 * np.log1p(np.maximum(v_work, 0.0) / max(lambda_obs, 1e-12))
        mi_z, m_mi, s_mi = zscore(mi)

        acq = mu + sqrt_beta * std + eta_t * mi_z
        acq = apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho_floor)

        i_star = int(np.argmax(acq))
        x_star = pool_smiles[i_star]
        selected.append(x_star); idxs.append(i_star)

        k_star = a * k_tanimoto_row(x_star, pool_smiles, fps_cache)
        denom = lambda_obs + v_work[i_star]
        v_work = v_work - (k_star * k_star) / max(denom, 1e-12)
        v_work[i_star] = 0.0

    return selected, idxs

def run_bucb_mi(
    smiles_all: list[str],
    init_size: int = 10,
    n_rounds: int = 40,
    batch_size: int = 5,
    sqrt_beta: float = 0.316,
    eta0: float = 0.1,         # small starting weight; decays by 1/sqrt(round)
    gp_amplitude: float = 1.0,
    gp_noise: float = 1e-4,
    rho_floor: Union[float, None] = None,
    csv_out: str = os.path.join(CSV_DIR, "gp_bucb_mi_fex.csv"),
):
    rng = random.Random(0)

    smiles = smiles_all.copy(); rng.shuffle(smiles)
    init_smiles = smiles[:init_size]
    pool_smiles = smiles[init_size:]

    fps_cache = {s: get_fingerprint(s) for s in smiles}

    Y0 = evaluate_fex_MPO(init_smiles)
    gp = ZeroMeanTanimotoGP(lambda s: fps_cache[s], init_smiles, Y0[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude) - 1.0),
        raw_noise=np.log(np.exp(gp_noise) - 1.0),
    )
    chosen = set(init_smiles)

    rows = []
    global_iter = 0

    for r in range(n_rounds):
        t0 = time.time()
        pool = [s for s in pool_smiles if s not in chosen]
        if not pool:
            bo_logger.info("Pool exhausted."); break

        eta_t = eta0 / np.sqrt(r + 1.0)

        B = min(batch_size, len(pool))
        batch, idxs = build_batch_bucb_mi(
            gp, params, pool, fps_cache,
            K=B, sqrt_beta=sqrt_beta, eta_t=eta_t, rho_floor=rho_floor,
        )

        y_batch = evaluate_fex_MPO(batch)[:, 0].tolist()

        mu_f, var_f = gp.predict_f(params, batch, full_covar=False)
        std_f = np.sqrt(np.maximum(np.asarray(var_f).flatten(), 1e-12))
        # acq values (for logging) with MI computed on *start-of-batch* stats is fine
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
            bo_logger.info(f"[round {r}] pick {j+1}/{B}: {smi} | UCB+MI | f={f_true:.4g}")
            chosen.add(smi)
            global_iter += 1

        new_X = list(gp._smiles_train) + batch
        new_y = np.concatenate([np.asarray(gp._y_train), np.asarray(y_batch)], axis=0)
        gp.set_training_data(new_X, new_y)

        bo_logger.info(f"Round {r} done in {time.time()-t0:.2f}s; train size={len(new_X)}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    bo_logger.info(f"Wrote: {csv_out}")

if __name__ == "__main__":
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    smiles_all = df["smiles"].tolist()[:100_000]
    pprint(smiles_all[:5])

    run_bucb_mi(
        smiles_all=smiles_all,
        init_size=10,
        n_rounds=40,           # with K=5 → 200 evaluations total
        batch_size=5,
        sqrt_beta=0.316,
        eta0=0.1,
        gp_amplitude=1.0,
        gp_noise=1e-4,
        rho_floor=None,
        csv_out=os.path.join(CSV_DIR, f"logs_{TRIAL_TAG}_gp_bucb_mi_fex.csv"),
    )

    sys.stdout.close()
