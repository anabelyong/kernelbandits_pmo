#!/usr/bin/env python
import logging, sys, time, random, os
import numpy as np, pandas as pd
from pprint import pprint
from rdkit import DataStructs  

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO


# ---- UCB ----
def ucb(means, vars_, sqrt_beta=0.316):
    std = np.sqrt(np.maximum(vars_, 1e-12))
    return means + sqrt_beta * std


# ---- Novelty vs evaluated set ----
def novelty_against_history(pool_smiles, history_fps, fps_cache):
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
    trial_id=1,
    n_iterations=200,
    sqrt_beta=0.316,
    eta0=0.25
):
    # === Logging setup (per trial) ===
    log_dir = f"logs_trial{trial_id}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "terminal_output_jax_ucb_aug_eta_constant.log")

    logger = logging.getLogger(f"bo_loop_logger_trial{trial_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

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

    history_fps = [fps[s] for s in known_smiles]

    for it in range(n_iterations):
        logger.info(f"\n--- Trial {trial_id}, Iter {it} (train size={Y_train.shape[0]}) ---")
        t0 = time.time()

        pool = [s for s in pool_smiles if s not in chosen]
        mean_jax, var_jax = gp.predict_y(params, pool, full_covar=False)
        means = np.asarray(mean_jax).flatten()
        vars_ = np.asarray(var_jax).flatten()

        novelty_raw = novelty_against_history(pool, history_fps, fps)
        novelty_z, nov_m, nov_s = zscore(novelty_raw)

        eta_t = eta0 / np.sqrt(it + 1.0) + eta0
        acq_vals = ucb(means, vars_, sqrt_beta=sqrt_beta) + eta_t * novelty_z

        u = ucb(means, vars_, sqrt_beta=sqrt_beta)
        logger.info(
            f"eta_t={eta_t:.4f} | novelty mean={nov_m:.4f}, std={nov_s:.4f} | "
            f"UCB range=({np.min(u):.4f}, {np.max(u):.4f}) | novelty_z range=({np.min(novelty_z):.3f},{np.max(novelty_z):.3f})"
        )

        idx = int(np.argmax(acq_vals))
        best_sm = pool[idx]
        best_acq = float(acq_vals[idx])
        logger.info(f"Selected {best_sm} → UCB+nov = {best_acq:.4g}")
        acq_history.append(best_acq)

        new_y = float(evaluate_fex_MPO([best_sm])[0, 0])
        logger.info(f"  True f = {new_y:.4g}")

        chosen.add(best_sm)
        known_smiles.append(best_sm)
        Y_train = np.vstack([Y_train, [[new_y]]])
        gp.set_training_data(known_smiles, Y_train[:, 0])
        history_fps.append(fps[best_sm])

        best_history.append(float(np.max(Y_train[:, 0])))
        logger.info(f"Iter time: {time.time()-t0:.2f}s")

    return known_smiles, Y_train, acq_history, best_history


if __name__ == "__main__":
    NUM_TRIALS = 3   # can increase later
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:100000]

    for trial_id in range(1, NUM_TRIALS + 1):
        random.shuffle(all_sm)
        init_smiles = all_sm[:10]
        pool_smiles = all_sm[10:]

        pprint(init_smiles)
        init_Y = evaluate_fex_MPO(init_smiles)

        bayesian_optimization_ucb_with_novelty(
            init_smiles, pool_smiles, init_Y,
            gp_amplitude=1.0, gp_noise=1e-4,
            trial_id=trial_id,
            n_iterations=200,
            sqrt_beta=0.316,
            eta0=0.25
        )
