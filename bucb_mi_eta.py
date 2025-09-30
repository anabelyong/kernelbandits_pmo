#!/usr/bin/env python
"""
GP-BUCB + MI bonus with multi-trial support.
Each trial logs to its own file but also prints to terminal.
"""

import logging, sys, time, random, os
import numpy as np, pandas as pd
from pprint import pprint
from rdkit import DataStructs
from jax.nn import softplus

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO

# ---- Helpers ----
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
    m, s = float(np.mean(v)), float(np.std(v))
    s = s if s > eps else 1.0
    return (v - m) / s, m, s

def build_batch_bucb_mi(gp, params, pool_smiles, fps_cache, K, sqrt_beta, eta_t, rho_floor=None):
    mu_f, var_f = gp.predict_f(params, pool_smiles, full_covar=False)
    mu = np.asarray(mu_f).flatten()
    v_work = np.asarray(var_f).flatten()

    a = float(softplus(params.raw_amplitude))
    lambda_obs = float(softplus(params.raw_noise))

    selected = []
    for j in range(K):
        std = np.sqrt(np.maximum(v_work, 1e-12))
        mi = 0.5 * np.log1p(np.maximum(v_work, 0.0) / max(lambda_obs, 1e-12))
        mi_z, _, _ = zscore(mi)

        acq = mu + sqrt_beta * std + eta_t * mi_z
        acq = apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho_floor)

        i_star = int(np.argmax(acq))
        x_star = pool_smiles[i_star]
        selected.append(x_star)

        # variance-only hallucination update
        k_star = a * k_tanimoto_row(x_star, pool_smiles, fps_cache)
        denom = lambda_obs + v_work[i_star]
        v_work = v_work - (k_star * k_star) / max(denom, 1e-12)
        v_work[i_star] = 0.0

    return selected

# ---- Main loop ----
def bayesian_optimization_bucb_mi(
    init_smiles,
    pool_smiles,
    init_Y,
    gp_amplitude,
    gp_noise,
    n_rounds=40,
    batch_size=5,
    sqrt_beta=0.316,
    eta0=0.1,
    rho_floor=None,
    trial_id=1
):
    # === Logging setup (per trial and eta0) ===
    eta_str = f"eta_{eta0}"
    log_dir = f"logs_trial{trial_id}/{eta_str}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"terminal_output_jax_bucb_mi_{eta_str}.log")

    bo_loop_logger = logging.getLogger(f"bo_loop_logger_trial{trial_id}_{eta_str}")
    bo_loop_logger.setLevel(logging.INFO)
    bo_loop_logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    bo_loop_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    bo_loop_logger.addHandler(sh)

    # ---- GP + BUCB MI Loop ----
    fps = {s: get_fingerprint(s) for s in (init_smiles + pool_smiles)}

    gp = ZeroMeanTanimotoGP(lambda s: fps[s], init_smiles, init_Y[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude) - 1.0),
        raw_noise=np.log(np.exp(gp_noise) - 1.0),
    )

    chosen = set(init_smiles)

    for r in range(n_rounds):
        bo_loop_logger.info(
            f"\n--- Trial {trial_id}, eta0={eta0}, Round {r} "
            f"(train size={len(gp._smiles_train)}) ---"
        )
        t0 = time.time()

        pool = [s for s in pool_smiles if s not in chosen]
        if not pool:
            bo_loop_logger.info("Pool exhausted.")
            break

        eta_t = eta0 / np.sqrt(r + 1.0)
        B = min(batch_size, len(pool))

        batch = build_batch_bucb_mi(
            gp, params, pool, fps,
            K=B, sqrt_beta=sqrt_beta, eta_t=eta_t, rho_floor=rho_floor
        )
        y_batch = evaluate_fex_MPO(batch)[:, 0].tolist()

        mu_f, var_f = gp.predict_f(params, batch, full_covar=False)
        std_f = np.sqrt(np.maximum(np.asarray(var_f).flatten(), 1e-12))
        ucb_vals = np.asarray(mu_f).flatten() + sqrt_beta * std_f

        for j, (smi, f_true, mu_j, std_j, ucb_j) in enumerate(
            zip(batch, y_batch, np.asarray(mu_f).flatten(), std_f, ucb_vals)
        ):
            bo_loop_logger.info(
                f"[round {r}] pick {j+1}/{B}: {smi} "
                f"| UCB+MI={ucb_j:.4g} | f={f_true:.4g}"
            )

            chosen.add(smi)
            new_X = list(gp._smiles_train) + [smi]
            new_y = np.concatenate([np.asarray(gp._y_train), [f_true]], axis=0)
            gp.set_training_data(new_X, new_y)

        bo_loop_logger.info(
            f"Round {r} done in {time.time()-t0:.2f}s; train size={len(gp._smiles_train)}"
        )


# ---- Entry ----
if __name__ == "__main__":
    NUM_TRIALS = 3   
    ETA_GRID = [0.05, 0.1, 0.25, 0.5]

    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:100000]

    for eta0 in ETA_GRID:
        for trial_id in range(1, NUM_TRIALS + 1):
            random.shuffle(all_sm)

            init_smiles = all_sm[:10]
            pool_smiles = all_sm[10:]

            pprint(init_smiles)
            init_Y = evaluate_fex_MPO(init_smiles)

            bayesian_optimization_bucb_mi(
                init_smiles, pool_smiles, init_Y,
                gp_amplitude=1.0, gp_noise=1e-4,
                n_rounds=40,
                batch_size=5,
                sqrt_beta=0.316,
                eta0=eta0,
                rho_floor=None,
                trial_id=trial_id
            )
