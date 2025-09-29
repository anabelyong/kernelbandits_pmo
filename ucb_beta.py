#!/usr/bin/env python
import logging, sys, time, random, os
import numpy as np, pandas as pd
from pprint import pprint

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO

# === Acquisition function ===
def ucb(means, vars_, beta=0.1):
    std = np.sqrt(np.maximum(vars_, 1e-12))
    return means + beta * std

def bayesian_optimization_ucb(
    init_smiles, pool_smiles, init_Y,
    gp_amplitude, gp_noise,
    n_iterations=200, beta=0.1,
    trial_id=1
):
    # --- Logging setup (per trial + beta) ---
    log_dir = f"logs_trial{trial_id}/beta_{beta}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "terminal_output_jax_ucb_fex.log")

    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

    bo_loop_logger = logging.getLogger(f"bo_loop_logger_trial{trial_id}_beta{beta}")
    bo_loop_logger.setLevel(logging.INFO)
    bo_loop_logger.handlers.clear()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    bo_loop_logger.addHandler(h)

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

    for it in range(n_iterations):
        bo_loop_logger.info(f"\n--- Trial {trial_id}, Beta={beta}, Iter {it} (train size={Y_train.shape[0]}) ---")
        t0 = time.time()

        # 3) predict on pool
        pool = [s for s in pool_smiles if s not in chosen]
        mean_jax, var_jax = gp.predict_y(params, pool, full_covar=False)
        means = np.array(mean_jax).flatten()
        vars_ = np.array(var_jax).flatten()

        # 4) UCB acquisition
        acq_vals = ucb(means, vars_, beta=beta)

        # 5) select best
        idx = int(np.argmax(acq_vals))
        best_sm = pool[idx]
        best_acq = float(acq_vals[idx])
        bo_loop_logger.info(f"Selected {best_sm} → UCB = {best_acq:.4g} (beta={beta})")
        acq_history.append(best_acq)

        # 6) evaluate oracle
        new_y = evaluate_fex_MPO([best_sm])[0, 0]
        bo_loop_logger.info(f"  True f = {new_y:.4g}")

        # 7) update GP
        chosen.add(best_sm)
        known_smiles.append(best_sm)
        Y_train = np.vstack([Y_train, [[new_y]]])
        gp.set_training_data(known_smiles, Y_train[:, 0])

        best_history.append(float(np.max(Y_train[:, 0])))
        bo_loop_logger.info(f"Iter time: {time.time()-t0:.2f}s")

    sys.stdout.close()
    return known_smiles, Y_train, acq_history, best_history

# ---- Entry ----
if __name__ == "__main__":
    NUM_TRIALS = 3
    BETA_VALUES = [0.2, 0.4, 0.6, 0.8, 1.0]

    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:100000]

    for trial_id in range(1, NUM_TRIALS + 1):
        random.shuffle(all_sm)  
        init_smiles = all_sm[:10]
        pool_smiles = all_sm[10:]

        pprint(init_smiles)
        init_Y = evaluate_fex_MPO(init_smiles)

        for beta in BETA_VALUES:
            bayesian_optimization_ucb(
                init_smiles, pool_smiles, init_Y,
                gp_amplitude=1.0, gp_noise=1e-4,
                n_iterations=200, beta=beta,
                trial_id=trial_id
            )
