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
from utils.utils import evaluate_fex_MPO

# ----------------- Similarity helpers -----------------
def k_tanimoto_row(sm, pool_smiles, fps_cache):
    return np.array(
        DataStructs.BulkTanimotoSimilarity(
            fps_cache[sm], [fps_cache[s] for s in pool_smiles]
        ),
        dtype=float,
    )

def max_tanimoto_to_set(fp_x, fps_set):
    """Return max_b Tanimoto(x,b) for a list of RDKit FPs; 0 if set empty."""
    if not fps_set:
        return 0.0
    sims = DataStructs.BulkTanimotoSimilarity(fp_x, fps_set)
    return float(max(sims)) if sims else 0.0

# ----------------- Packing (diagnostic version) -----------------
def apply_packing_floor_mask(selected_fps, candidate_fps, rho=None, sim_max=None):
    """
    Return boolean mask [N_candidates] that is True for items that PASS the floor.

    Distance floor:   min_b (1 - T) >= rho    <=>   max_b T <= 1 - rho
    Similarity floor: max_b T <= sim_max
    """
    N = len(candidate_fps)
    if (rho is None and sim_max is None) or len(selected_fps) == 0:
        return np.ones(N, dtype=bool)

    if sim_max is None and rho is not None:
        sim_max = 1.0 - float(rho)

    ok = np.ones(N, dtype=bool)
    for fp_sel in selected_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp_sel, candidate_fps), dtype=float)
        ok &= (sims <= sim_max)
    return ok

# ----------------- BUCB with packing + diagnostics -----------------
def build_batch_bucb(
    gp, params, pool_smiles, fps_cache, K, sqrt_beta,
    rho_floor=None,                 # distance floor inside the batch
    pack_against="batch",           # "batch" | "history" | "both"
    history_smiles=None,            # list of already evaluated molecules
    relax_if_infeasible=True,       # adaptively soften rho if batch would stall
    min_feasible=10,                # soften if fewer feasible items than this (or K-j)
    logger=None
):
    """
    Standard BUCB with variance-only hallucination, plus an optional
    Tanimoto packing floor and rich diagnostics.
    """
    # Posterior at pool (f, noise-free)
    mu_f, var_f = gp.predict_f(params, pool_smiles, full_covar=False)
    mu = np.asarray(mu_f).flatten()
    v_work = np.asarray(var_f).flatten()  # updated in-place via hallucination

    a = float(softplus(params.raw_amplitude))
    lambda_obs = float(softplus(params.raw_noise))

    # Precompute FPs for fast masking / diagnostics
    pool_fps = [fps_cache[s] for s in pool_smiles]
    batch_selected, batch_selected_fps = [], []
    history_fps = [fps_cache[s] for s in (history_smiles or [])]

    def feasible_mask(current_rho):
        """Combine masks from chosen-in-batch and/or history according to pack_against."""
        if current_rho is None:
            return np.ones(len(pool_smiles), dtype=bool)

        masks = []
        if pack_against in ("batch", "both") and len(batch_selected_fps) > 0:
            masks.append(apply_packing_floor_mask(batch_selected_fps, pool_fps, rho=current_rho))
        if pack_against in ("history", "both") and len(history_fps) > 0:
            masks.append(apply_packing_floor_mask(history_fps, pool_fps, rho=current_rho))

        if not masks:
            return np.ones(len(pool_smiles), dtype=bool)

        m = masks[0]
        for mm in masks[1:]:
            m &= mm
        return m

    current_rho = rho_floor

    for j in range(K):
        feas = feasible_mask(current_rho)

        # Diagnostics: how many did packing mask?
        if logger is not None and rho_floor is not None:
            logger.info(
                f"[bucb] step {j+1}/{K}: rho={current_rho if current_rho is not None else 'None'} "
                f"-> masked {(~feas).sum()} / {len(feas)} items"
            )

        # Optionally relax the floor if it kills the batch
        need = max(min_feasible, K - j)
        if relax_if_infeasible and feas.sum() < need and current_rho is not None:
            prev = float(current_rho)
            current_rho = prev * 0.9
            feas = feasible_mask(current_rho)
            if logger is not None:
                logger.info(
                    f"[bucb] relaxing rho: {prev:.2f} -> {current_rho:.2f} "
                    f"(feasible {feas.sum()} needed≥{need})"
                )

        # UCB on working variance
        std = np.sqrt(np.maximum(v_work, 1e-12))
        acq = mu + sqrt_beta * std
        acq = np.where(feas, acq, -np.inf)

        i_star = int(np.argmax(acq))
        x_star = pool_smiles[i_star]
        fp_star = pool_fps[i_star]
        batch_selected.append(x_star)
        batch_selected_fps.append(fp_star)

        # Extra diagnostics: min distances
        max_sim_batch   = max_tanimoto_to_set(fp_star, batch_selected_fps[:-1])   # to previous in batch
        max_sim_history = max_tanimoto_to_set(fp_star, history_fps)               # to all history
        if logger is not None:
            logger.info(
                f"[bucb] pick {j+1}/{K}: "
                f"min-d_batch={1.0-max_sim_batch:.3f}, min-d_hist={1.0-max_sim_history:.3f}"
            )

        # Variance-only hallucination update
        k_star = a * np.array(DataStructs.BulkTanimotoSimilarity(fp_star, pool_fps), dtype=float)
        denom = lambda_obs + v_work[i_star]
        v_work = v_work - (k_star * k_star) / max(denom, 1e-12)
        v_work[i_star] = 0.0

    return batch_selected

# ----------------- Main BO loop -----------------
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
    out_root="logs_bucb_pack",
    pack_against="batch",          # "batch" | "history" | "both"
    relax_if_infeasible=True,
    min_feasible=10
):
    """
    BUCB with optional Tanimoto packing floor (rho_floor) and diagnostics.
    Writes both terminal log and a CSV for PMO metrics.
    """
    # ---- IO setup per (trial, rho, beta) ----
    rho_tag = "none" if (rho_floor is None) else f"{rho_floor:.2f}"
    run_dir = os.path.join(out_root, f"trial_{trial_id}", f"rho_{rho_tag}",
                           f"beta_{sqrt_beta**2:.2f}", f"pack_{pack_against}")
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "terminal_output_jax_bucb_packed.log")
    csv_file = os.path.join(run_dir, "logs_terminal_output_jax_bucb_packed.csv")

    # Mirror to file + stdout
    logger = logging.getLogger(
        f"bucb_trial{trial_id}_rho{rho_tag}_b{sqrt_beta**2:.2f}_pack{pack_against}"
    )
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
        wr.writerow(["round", "pick_idx", "smiles", "mu", "std", "UCB", "True f",
                     "min_d_batch", "min_d_hist"])

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
            f"(beta={sqrt_beta**2:.4f}) | rho={rho_tag} | pack={pack_against} "
            f"| train size={len(gp._smiles_train)} ---"
        )
        t0 = time.time()

        pool = [s for s in pool_smiles if s not in chosen]
        if not pool:
            logger.info("Pool exhausted.")
            break

        B = min(batch_size, len(pool))
        batch = build_batch_bucb(
            gp, params, pool, fps, K=B, sqrt_beta=sqrt_beta,
            rho_floor=rho_floor,
            pack_against=pack_against,
            history_smiles=list(gp._smiles_train),
            relax_if_infeasible=relax_if_infeasible,
            min_feasible=min_feasible,
            logger=logger
        )

        # Evaluate oracle
        y_batch = evaluate_fex_MPO(batch)[:, 0].tolist()

        # Predictive stats for chosen
        mu_f, var_f = gp.predict_f(params, batch, full_covar=False)
        mu_vec = np.asarray(mu_f).flatten()
        std_vec = np.sqrt(np.maximum(np.asarray(var_f).flatten(), 1e-12))
        ucb_vals = mu_vec + sqrt_beta * std_vec

        # For distance diagnostics into CSV
        history_fps = [fps[s] for s in gp._smiles_train]

        with open(csv_file, "a", newline="") as fcsv:
            wr = csv.writer(fcsv)
            for j, (smi, f_true, mu_j, std_j, ucb_j) in enumerate(
                zip(batch, y_batch, mu_vec, std_vec, ucb_vals)
            ):
                fp_j = fps[smi]
                # distances at the moment of logging
                max_sim_hist  = max_tanimoto_to_set(fp_j, history_fps)
                # batch min distance (to previously chosen in same round)
                prev_batch_fps = [fps[b] for b in batch[:j]]
                max_sim_batch  = max_tanimoto_to_set(fp_j, prev_batch_fps)

                logger.info(
                    f"[round {r}] pick {j+1}/{B}: {smi} | UCB={ucb_j:.4g} | mu={mu_j:.4g} | "
                    f"std={std_j:.4g} | f={f_true:.4g} | "
                    f"min-d_batch={1.0-max_sim_batch:.3f} | min-d_hist={1.0-max_sim_hist:.3f}"
                )
                wr.writerow([r, j+1, smi, float(mu_j), float(std_j), float(ucb_j),
                             float(f_true), 1.0-max_sim_batch, 1.0-max_sim_hist])

                # GP update
                chosen.add(smi)
                new_X = list(gp._smiles_train) + [smi]
                new_y = np.concatenate([np.asarray(gp._y_train), [f_true]], axis=0)
                gp.set_training_data(new_X, new_y)

        logger.info(f"Round {r} done in {time.time() - t0:.2f}s; train size={len(gp._smiles_train)}")

    return run_dir, csv_file

# ------------------ Entry ------------------
if __name__ == "__main__":
    RHO_GRID  = [0.65, 0.70, 0.75]   
    BETA_GRID = [1.0]                             
    PACK_POLICIES = ["batch", "both"]             

    NUM_TRIALS = 3
    N_ROUNDS   = 200       
    BATCH_SIZE = 5

    MAX_SELECTED = 100    
    
    # Dataset
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:1000000]

    for trial_id in range(2, NUM_TRIALS + 1):
        random.shuffle(all_sm)
        init_smiles = all_sm[:10]
        pool_smiles = all_sm[10:]
        pprint(init_smiles)

        init_Y = evaluate_fex_MPO(init_smiles)  # shape (10, 1)

        for pack_policy in PACK_POLICIES:
            for rho in RHO_GRID:
                for beta in BETA_GRID:
                    # run with internal early-stop after 100 picks
                    run_dir, csv_file = bayesian_optimization_bucb(
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
                        out_root="logs_bucb_pack",
                        pack_against=pack_policy,
                        relax_if_infeasible=True,
                        min_feasible=10
                    )

                    # ---- Enforce early stop (post-run check) ----
                    df_logs = pd.read_csv(csv_file)
                    n_picked = len(df_logs)
                    if n_picked >= MAX_SELECTED:
                        print(f"[Trial {trial_id}] Finished early at {n_picked} picks. "
                              f"Results in {run_dir}")
                        break
                else:
                    continue
                break
            else:
                continue
            break
