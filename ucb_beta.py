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
def ucb(means, vars_, beta=1.0):
    std = np.sqrt(np.maximum(vars_, 1e-12))
    return means + beta * std


def bayesian_optimization_ucb(
    init_smiles, pool_smiles, init_Y,
    gp_amplitude, gp_noise,
    n_iterations=100, beta=1.0,
    trial_id=1
):
    # --- Logging setup ---
    log_dir = f"logs_ucb/trial_{trial_id}/beta_{beta:.2f}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "logs_terminal_output_jax_ucb.log")
    csv_file = os.path.join(log_dir, "logs_terminal_output_jax_ucb.csv")

    # redirect stdout → log file
    sys.stdout = open(log_file, "w")
    sys.stderr = sys.stdout

    bo_loop_logger = logging.getLogger(f"bo_logger_trial{trial_id}_beta{beta}")
    bo_loop_logger.setLevel(logging.INFO)
    bo_loop_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    bo_loop_logger.addHandler(handler)

    bo_loop_logger.info(f"=== Starting UCB Bayesian Optimization Trial {trial_id} (beta={beta:.2f}) ===")

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
    acq_history, best_history, rows = [], [], []

    # --- BO loop ---
    for it in range(n_iterations):
        bo_loop_logger.info(f"\n--- Trial {trial_id}, Beta={beta}, Iter {it} ---")
        t0 = time.time()

        pool = [s for s in pool_smiles if s not in chosen]
        mean_jax, var_jax = gp.predict_y(params, pool, full_covar=False)
        means = np.array(mean_jax).flatten()
        vars_ = np.array(var_jax).flatten()

        # 4) UCB acquisition
        acq_vals = ucb(means, vars_, beta=beta)
        idx = int(np.argmax(acq_vals))
        best_sm = pool[idx]
        best_acq = float(acq_vals[idx])

        bo_loop_logger.info(f"Selected SMILES: {best_sm}")
        bo_loop_logger.info(f"  UCB value: {best_acq:.4f}")

        # 5) evaluate oracle
        new_y = evaluate_fex_MPO([best_sm])[0, 0]
        bo_loop_logger.info(f"  True f: {new_y:.4f}")

        # 6) update GP
        chosen.add(best_sm)
        known_smiles.append(best_sm)
        Y_train = np.vstack([Y_train, [[new_y]]])
        gp.set_training_data(known_smiles, Y_train[:, 0])

        best_y = float(np.max(Y_train[:, 0]))
        iter_time = time.time() - t0

        bo_loop_logger.info(f"  Best so far: {best_y:.4f}")
        bo_loop_logger.info(f"  Iter time: {iter_time:.2f}s")

        acq_history.append(best_acq)
        best_history.append(best_y)

        # --- save CSV line ---
        rows.append({
            "trial_id": trial_id,
            "iteration": it,
            "beta": beta,
            "selected_smiles": best_sm,
            "ucb_value": best_acq,
            "true_f": new_y,
            "best_so_far": best_y,
            "iter_time_s": iter_time
        })

        # periodic save every 5 iterations
        if it % 5 == 0 or it == n_iterations - 1:
            pd.DataFrame(rows).to_csv(csv_file, index=False)

    # final save
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    bo_loop_logger.info("=== Completed Trial ===")
    sys.stdout.close()
    return known_smiles, Y_train, acq_history, best_history


# ---- Entry ----
if __name__ == "__main__":
    NUM_TRIALS = 3
    BETA_VALUES = [1.0]

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
                n_iterations=100, beta=beta,
                trial_id=trial_id
            )
