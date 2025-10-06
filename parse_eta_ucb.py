#!/usr/bin/env python
import os
import re
import csv

LOG_DIRS = ["logs_trial1", "logs_trial2", "logs_trial3"]
OUT_DIR = "csv_results"

# Regex patterns tuned for your example
ITER_RE       = re.compile(r"--- Trial\s+(\d+),\s*Iter\s+(\d+)")
ETA_RE        = re.compile(r"eta_t=([0-9\.eE\-\+]+)")
NOVELTY_RE    = re.compile(r"novelty mean=([0-9\.eE\-\+]+), std=([0-9\.eE\-\+]+)")
UCB_RANGE_RE  = re.compile(r"UCB range=\(([-0-9\.eE]+),\s*([-0-9\.eE]+)\)")
NOVZ_RANGE_RE = re.compile(r"novelty_z range=\(([-0-9\.eE]+),\s*([-0-9\.eE]+)\)")
SELECT_AUG_RE = re.compile(r"Selected\s+(.+?)\s*→\s*UCB\+nov\s*=\s*([0-9\.eE\-\+]+)")
TRUEF_RE      = re.compile(r"True f\s*=\s*([0-9\.eE\-\+]+)")

def parse_ucb_aug_log(path):
    rows = []
    trial_id = None
    with open(path) as f:
        current = {}
        for line in f:
            # Trial + iteration
            m = ITER_RE.search(line)
            if m:
                if current:
                    rows.append(current)
                trial_id = int(m.group(1))
                current = {"Trial": trial_id, "Iteration": int(m.group(2))}

            # eta_t
            m = ETA_RE.search(line)
            if m:
                current["eta_t"] = float(m.group(1))

            # novelty stats
            m = NOVELTY_RE.search(line)
            if m:
                current["novelty_mean"] = float(m.group(1))
                current["novelty_std"]  = float(m.group(2))

            # UCB range
            m = UCB_RANGE_RE.search(line)
            if m:
                current["ucb_low"]  = float(m.group(1))
                current["ucb_high"] = float(m.group(2))

            # novelty_z range
            m = NOVZ_RANGE_RE.search(line)
            if m:
                current["novz_low"]  = float(m.group(1))
                current["novz_high"] = float(m.group(2))

            # Selected + score
            m = SELECT_AUG_RE.search(line)
            if m:
                current["Selected SMILES"] = m.group(1)
                current["UCB+nov"] = float(m.group(2))

            # True f
            m = TRUEF_RE.search(line)
            if m:
                current["True f"] = float(m.group(1))

        if current:
            rows.append(current)

    headers = [
        "Trial", "Iteration", "Selected SMILES", "True f", "UCB+nov", "eta_t",
        "novelty_mean", "novelty_std", "ucb_low", "ucb_high", "novz_low", "novz_high"
    ]
    return rows, headers


def write_csv(outpath, rows, headers):
    if not rows:
        return
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for log_dir in LOG_DIRS:
        if not os.path.isdir(log_dir):
            continue
        for fn in sorted(os.listdir(log_dir)):
            if not fn.endswith(".log"):
                continue
            path = os.path.join(log_dir, fn)

            rows, headers = parse_ucb_aug_log(path)

            if not rows:
                print(f"[WARN] no rows in {path}")
                continue

            base = fn.replace(".log", "")
            outpath = os.path.join(OUT_DIR, f"{log_dir}_{base}.csv")
            write_csv(outpath, rows, headers)
            print(f"Wrote → {outpath}")

if __name__ == "__main__":
    main()
