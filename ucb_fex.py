#!/usr/bin/env python
import logging, sys, time, random
import numpy as np, pandas as pd
from pprint import pprint

from kernel_only_GP.tanimoto_gp import (
    get_fingerprint,
    ZeroMeanTanimotoGP,
    TanimotoGP_Params,
)
from utils.utils import evaluate_fex_MPO

# === Logging setup ===
log_file = "logs_trial3/terminal_output_jax_ucb_fex.log"
sys.stdout = open(log_file, "w")
sys.stderr = sys.stdout

bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
bo_loop_logger.addHandler(h)

# acquisition function
def ucb(means, vars_, beta=0.1):
    std = np.sqrt(vars_)
    return means + beta * std

def bayesian_optimization_ucb(
    init_smiles, pool_smiles, init_Y,
    gp_amplitude, gp_noise, n_iterations=200, beta=0.1
):
    # 1) fingerprint cache
    all_smiles = init_smiles + pool_smiles
    fps = {s: get_fingerprint(s) for s in all_smiles}

    # 2) GP init
    known_smiles = init_smiles.copy()
    Y_train = init_Y.copy()
    gp = ZeroMeanTanimotoGP(lambda s: fps[s], known_smiles, Y_train[:, 0])
    params = TanimotoGP_Params(
        raw_amplitude=np.log(np.exp(gp_amplitude)-1.0),
        raw_noise    =np.log(np.exp(gp_noise)-1.0),
    )

    chosen = set()
    acq_history, best_history = [], []

    for it in range(n_iterations):
        bo_loop_logger.info(f"\n--- Iter {it} (train size={Y_train.shape[0]}) ---")
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
        bo_loop_logger.info(f"Selected {best_sm} → UCB = {best_acq:.4g}")
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

    bayesian_optimization_ucb(
        init_smiles, pool_smiles, init_Y,
        gp_amplitude=1.0, gp_noise=1e-4,
        n_iterations=200, beta=0.1
    )

    sys.stdout.close()
