import os
import re
import csv

LOG_ROOTS = ["logs_trial1", "logs_trial2", "logs_trial3"]
OUT_DIR = "csv_results"

# BUCB regex
ROUND_RE   = re.compile(r"--- Trial\s+(\d+),\s*sqrt_beta=([0-9\.]+),\s*Round\s+(\d+)")
PICK_RE    = re.compile(r"\[round\s+(\d+)\]\s+pick\s+(\d+)/(\d+):\s+(.+?)\s+\|\s+UCB=([0-9\.eE\-\+]+)\s+\|\s+f=([0-9\.eE\-\+]+)")

# UCB regex
ITER_RE       = re.compile(r"--- Trial\s+(\d+),\s*Beta=([0-9\.]+),\s*Iter\s+(\d+)")
SELECT_FEX_RE = re.compile(r"Selected\s+(.+?)\s*→\s*UCB\s*=\s*([0-9\.eE\-\+]+)")
TRUEF_RE      = re.compile(r"True f\s*=\s*([0-9\.eE\-\+]+)")

def parse_bucb_log(path):
    rows = []
    trial_id = beta = round_id = None
    seen = set()  # for deduplication
    with open(path) as f:
        for line in f:
            if (m := ROUND_RE.search(line)):
                trial_id = int(m.group(1))
                beta     = float(m.group(2))
                round_id = int(m.group(3))

            if (m := PICK_RE.search(line)):
                rnum, pick_idx, pick_total, smiles, ucb, fval = m.groups()
                key = (trial_id, beta, int(rnum), int(pick_idx))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "Trial": trial_id,
                    "Beta": beta,
                    "Round": int(rnum),
                    "Pick": int(pick_idx),
                    "PickTotal": int(pick_total),
                    "Selected SMILES": smiles,
                    "UCB": float(ucb),
                    "True f": float(fval),
                })
    headers = ["Trial","Beta","Round","Pick","PickTotal","Selected SMILES","UCB","True f"]
    return rows, headers

def parse_ucb_log(path):
    rows = []
    trial_id = beta = iteration = None
    current = {}
    with open(path) as f:
        for line in f:
            if (m := ITER_RE.search(line)):
                trial_id = int(m.group(1))
                beta     = float(m.group(2))
                iteration = int(m.group(3))
                if current:
                    rows.append(current)
                current = {"Trial": trial_id, "Beta": beta, "Iteration": iteration}
            if (m := SELECT_FEX_RE.search(line)):
                current["Selected SMILES"] = m.group(1)
                current["UCB"] = float(m.group(2))
            if (m := TRUEF_RE.search(line)):
                current["True f"] = float(m.group(1))
        if current:
            rows.append(current)
    headers = ["Trial","Beta","Iteration","Selected SMILES","UCB","True f"]
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
    for log_root in LOG_ROOTS:
        if not os.path.isdir(log_root):
            continue
        for beta_dir in os.listdir(log_root):
            beta_path = os.path.join(log_root, beta_dir)
            if not os.path.isdir(beta_path):
                continue
            for fn in os.listdir(beta_path):
                path = os.path.join(beta_path, fn)
                if "bucb" in fn:
                    rows, headers = parse_bucb_log(path)
                    outpath = os.path.join(OUT_DIR, f"BUCB_beta_{beta_dir.split('_')[-1]}.csv")
                elif "ucb" in fn:
                    rows, headers = parse_ucb_log(path)
                    outpath = os.path.join(OUT_DIR, f"UCB_beta_{beta_dir.split('_')[-1]}.csv")
                else:
                    continue
                write_csv(outpath, rows, headers)
                print(f"Wrote {outpath} with {len(rows)} rows")

if __name__ == "__main__":
    main()
