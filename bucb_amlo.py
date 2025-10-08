#!/usr/bin/env python
import logging, sys, time, random, os, csv
import numpy as np, pandas as pd
from pprint import pprint
from rdkit import DataStructs
from jax.nn import softplus

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_amlo_MPO

def k_tanimoto_row(sm, pool_smiles, fps_cache):
    return np.array(
        DataStructs.BulkTanimotoSimilarity(
            fps_cache[sm], [fps_cache[s] for s in pool_smiles]
        ),
        dtype=float,
    )

def apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho):
    """
    Enforce: for any candidate x, min_b d_Tan(x,b) >= rho  (i.e., 1 - sim >= rho).
    If violated, set acquisition to -inf so it won't be chosen.
    """
    if rho is None or not selected:
        return acq
    ok = np.ones_like(acq, dtype=bool)
    for sb in selected:
        sims = k_tanimoto_row(sb, pool_smiles, fps_cache)  # similarity to every pool item
        ok &= (1.0 - sims) >= rho
    return np.where(ok, acq, -np.inf)

def build_batch_bucb(gp, params, pool_smiles, fps_cache, K, sqrt_beta, rho_floor=None):
    """
    Standard BUCB: build a batch of size K with hallucinated variance updates.
    Adds an optional Tanimoto packing floor (rho_floor) inside the batch.
    """
    # Current posterior predictive at pool (f, noise-free)
    mu_f, var_f = gp.predict_f(params, pool_smiles, full_covar=False)
    mu = np.asarray(mu_f).flatten()
    v_work = np.asarray(var_f).flatten()  # this gets updated inside the loop (hallucination)

    a = float(softplus(params.raw_amplitude))
    lambda_obs = float(softplus(params.raw_noise))

    selected = []
    for _ in range(K):
        # UCB on the working variance
        std = np.sqrt(np.maximum(v_work, 1e-12))
        acq = mu + sqrt_beta * std
        # Apply within-batch packing
        acq = apply_packing_floor(acq, selected, pool_smiles, fps_cache, rho_floor)

        i_star = int(np.argmax(acq))
        x_star = pool_smiles[i_star]
        selected.append(x_star)

        # BUCB hallucination: variance-only update for ALL pool points
        # v_new(x) = v_old(x) - (a * k(x, x_star))^2 / (lambda + v_old(x_star))
        k_star = a * k_tanimoto_row(x_star, pool_smiles, fps_cache)
        denom = lambda_obs + v_work[i_star]
        v_work = v_work - (k_star * k_star) / max(denom, 1e-12)
        v_work[i_star] = 0.0

    return selected

def bayesian_optimization_bucb(
    init_smiles,
    pool_smiles,
    init_Y,
    gp_amplitude,
    gp_noise,
    n_rounds=40,
    batch_size=5,
    sqrt_beta=0.7071,   
    rho_floor=None,
    trial_id=1,
    out_root="logs_bucb_pack"
):
    """
    Runs BUCB with an optional Tanimoto packing floor (rho_floor) inside the batch.
    Writes both a terminal log and a CSV (with 'True f') for PMO metrics.
    """
    # ---- IO setup per (trial, rho, beta) ----
    rho_tag = "none" if (rho_floor is None) else f"{rho_floor:.2f}"
    run_dir = os.path.join(out_root, f"trial_{trial_id}", f"rho_{rho_tag}", f"beta_{sqrt_beta**2:.2f}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "terminal_output_jax_bucb_packed.log")
    csv_file = os.path.join(run_dir, "logs_terminal_output_jax_bucb_packed.csv")

    # Mirror to file + stdout
    logger = logging.getLogger(f"bucb_trial{trial_id}_rho{rho_tag}_b{sqrt_beta**2:.2f}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh); logger.addHandler(sh)

    # Prepare CSV
    with open(csv_file, "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["round", "pick_idx", "smiles", "mu", "std", "UCB", "True f"])

    # ---- Fingerprints & GP init ----
    fps = {s: get_fingerprint(s) for s in (init_smiles + pool_smiles)}
    gp = ZeroMeanTanimotoGP(lambda s: fps[s], init_smiles, init_Y[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude) - 1.0),
        raw_noise=np.log(np.exp(gp_noise) - 1.0),
    )

    chosen = set(init_smiles)

    for r in range(n_rounds):
        logger.info(
            f"\n--- Round {r} | trial={trial_id} | sqrt_beta={sqrt_beta:.4f} "
            f"(beta={sqrt_beta**2:.4f}) | rho={rho_tag} | train size={len(gp._smiles_train)} ---"
        )
        t0 = time.time()

        pool = [s for s in pool_smiles if s not in chosen]
        if not pool:
            logger.info("Pool exhausted.")
            break

        B = min(batch_size, len(pool))
        batch = build_batch_bucb(
            gp, params, pool, fps, K=B, sqrt_beta=sqrt_beta, rho_floor=rho_floor
        )

        # Evaluate oracle
        y_batch = evaluate_amlo_MPO(batch)[:, 0].tolist()

        # Log predictive stats for the chosen items
        mu_f, var_f = gp.predict_f(params, batch, full_covar=False)
        mu_vec = np.asarray(mu_f).flatten()
        std_vec = np.sqrt(np.maximum(np.asarray(var_f).flatten(), 1e-12))
        ucb_vals = mu_vec + sqrt_beta * std_vec

        # Write picks to CSV and update GP
        with open(csv_file, "a", newline="") as fcsv:
            wr = csv.writer(fcsv)
            for j, (smi, f_true, mu_j, std_j, ucb_j) in enumerate(
                zip(batch, y_batch, mu_vec, std_vec, ucb_vals)
            ):
                logger.info(
                    f"[round {r}] pick {j+1}/{B}: {smi} | UCB={ucb_j:.4g} | mu={mu_j:.4g} | "
                    f"std={std_j:.4g} | f={f_true:.4g}"
                )
                wr.writerow([r, j+1, smi, float(mu_j), float(std_j), float(ucb_j), float(f_true)])

                chosen.add(smi)
                new_X = list(gp._smiles_train) + [smi]
                new_y = np.concatenate([np.asarray(gp._y_train), [f_true]], axis=0)
                gp.set_training_data(new_X, new_y)

        logger.info(f"Round {r} done in {time.time() - t0:.2f}s; train size={len(gp._smiles_train)}")

    return run_dir, csv_file

# ------------------ Entry ------------------
if __name__ == "__main__":
    RHO_GRID  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]   # packing floors
    BETA_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]                   # UCB beta; we use sqrt(beta) in code

    NUM_TRIALS = 10
    N_ROUNDS   = 40
    BATCH_SIZE = 5

    # Dataset
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:100000]

    for trial_id in range(1, NUM_TRIALS + 1):
        random.shuffle(all_sm)
        init_smiles = all_sm[:10]
        pool_smiles = all_sm[10:]
        pprint(init_smiles)

        init_Y = evaluate_amlo_MPO(init_smiles)  # shape (10, 1)

        for rho in RHO_GRID:
            for beta in BETA_GRID:
                bayesian_optimization_bucb(
                    init_smiles=init_smiles,
                    pool_smiles=pool_smiles,
                    init_Y=init_Y,
                    gp_amplitude=1.0,
                    gp_noise=1e-4,
                    n_rounds=N_ROUNDS,
                    batch_size=BATCH_SIZE,
                    sqrt_beta=np.sqrt(beta),   
                    rho_floor=rho,
                    trial_id=trial_id,
                    out_root="logs_bucb_pack_amlo"
                )
