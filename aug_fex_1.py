#!/usr/bin/env python
import logging, sys, time, random
import numpy as np, pandas as pd
from pprint import pprint
from rdkit import DataStructs  # NEW: for BulkTanimotoSimilarity

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO

# === Logging setup ===
log_file = "logs_trial2/terminal_output_jax_ucb_aug_fex.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_loop_logger.addHandler(h)

# ---- Base UCB (uses predictive variance of y) ----
def ucb(means, vars_, sqrt_beta=0.316):
    """UCB = mu + sqrt_beta * std. (sqrt_beta = sqrt(beta))."""
    std = np.sqrt(np.maximum(vars_, 1e-12))
    return means + sqrt_beta * std

# ---- Tanimoto novelty vs evaluated set S ----
def novelty_against_history(pool_smiles, history_fps, fps_cache):
    """
    Returns novelty(x) = 1 - max_b k_Tan(x, b) for each x in pool_smiles.
    If history is empty, novelty is 1 for all.
    """
    if not history_fps:
        return np.ones(len(pool_smiles), dtype=float)

    nov = np.empty(len(pool_smiles), dtype=float)
    for i, sm in enumerate(pool_smiles):
        sim_list = DataStructs.BulkTanimotoSimilarity(fps_cache[sm], history_fps)
        nov[i] = 1.0 - float(max(sim_list)) if sim_list else 1.0
    return nov

def zscore(arr, eps=1e-8):
    m = float(np.mean(arr))
    s = float(np.std(arr))
    s = s if s > eps else 1.0
    return (arr - m) / s, m, s

def bayesian_optimization_ucb_with_novelty(
    init_smiles,
    pool_smiles,
    init_Y,
    gp_amplitude,
    gp_noise,
    n_iterations=200,
    sqrt_beta=0.316,     # = sqrt(beta); matches "kappa" convention (Tripp code samples this)
    eta0=0.25           # initial weight for novelty; effective eta_t = eta0 / sqrt(t+1)
):
    # 1) fingerprint cache
    all_smiles = init_smiles + pool_smiles
    fps = {s: get_fingerprint(s) for s in all_smiles}

    # 2) GP init
    known_smiles = init_smiles.copy()
    Y_train = init_Y.copy()
    gp = ZeroMeanTanimotoGP(lambda s: fps[s], known_smiles, Y_train[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude) - 1.0),
        raw_noise=np.log(np.exp(gp_noise) - 1.0),
    )

    chosen = set()
    acq_history, best_history = [], []

    # Keep RDKit FP objects of history for fast novelty updates
    history_fps = [fps[s] for s in known_smiles]

    for it in range(n_iterations):
        bo_loop_logger.info(f"\n--- Iter {it} (train size={Y_train.shape[0]}) ---")
        t0 = time.time()

        # 3) predict on pool
        pool = [s for s in pool_smiles if s not in chosen]
        mean_jax, var_jax = gp.predict_y(params, pool, full_covar=False)
        means = np.asarray(mean_jax).flatten()
        vars_ = np.asarray(var_jax).flatten()

        # 4) compute novelty vs evaluated set and z-score it
        novelty_raw = novelty_against_history(pool, history_fps, fps)
        novelty_z, nov_m, nov_s = zscore(novelty_raw)

        # 5) decaying novelty weight
        eta_t = eta0 / np.sqrt(it + 1.0)

        # 6) diversity-aware acquisition
        acq_vals = ucb(means, vars_, sqrt_beta=sqrt_beta) + eta_t * novelty_z

        # Logging diagnostics
        u = ucb(means, vars_, sqrt_beta=sqrt_beta)
        bo_loop_logger.info(
            f"eta_t={eta_t:.4f} | novelty mean={nov_m:.4f}, std={nov_s:.4f} | "
            f"UCB range=({np.min(u):.4f}, {np.max(u):.4f}) | novelty_z range=({np.min(novelty_z):.3f},{np.max(novelty_z):.3f})"
        )

        # 7) select best
        idx = int(np.argmax(acq_vals))
        best_sm = pool[idx]
        best_acq = float(acq_vals[idx])
        bo_loop_logger.info(f"Selected {best_sm} → UCB+nov = {best_acq:.4g}")
        acq_history.append(best_acq)

        # 8) evaluate oracle
        new_y = float(evaluate_fex_MPO([best_sm])[0, 0])
        bo_loop_logger.info(f"  True f = {new_y:.4g}")

        # 9) update GP & caches
        chosen.add(best_sm)
        known_smiles.append(best_sm)
        Y_train = np.vstack([Y_train, [[new_y]]])
        gp.set_training_data(known_smiles, Y_train[:, 0])

        history_fps.append(fps[best_sm])

        # 10) bookkeeping
        best_history.append(float(np.max(Y_train[:, 0])))
        bo_loop_logger.info(f"Iter time: {time.time()-t0:.2f}s")

    return known_smiles, Y_train, acq_history, best_history


if __name__ == "__main__":
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:100000]
    random.shuffle(all_sm)

    init_smiles = all_sm[:10]
    pool_smiles = all_sm[10:]

    pprint(init_smiles)
    init_Y = evaluate_fex_MPO(init_smiles)  # (10,1)
    bo_loop_logger.info(f"Initial Y: {init_Y.flatten().tolist()}")

    bayesian_optimization_ucb_with_novelty(
        init_smiles, pool_smiles, init_Y,
        gp_amplitude=1.0, gp_noise=1e-4,
        n_iterations=200,
        sqrt_beta=0.316,   
        eta0=0.25          
    )

    sys.stdout.close()
