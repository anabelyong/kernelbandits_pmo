import os
import re
import csv

LOG_ROOTS = ["logs_trial1", "logs_trial2", "logs_trial3"]
OUT_DIR = "csv_results"

# ---------------- REGEX ----------------
# BUCB
ROUND_RE   = re.compile(r"--- Trial\s+(\d+),\s*sqrt_beta=([0-9\.]+),\s*Round\s+(\d+)")
PICK_BUCB  = re.compile(
    r"\[round\s+(\d+)\]\s+pick\s+(\d+)/(\d+):\s+(.+?)\s+\|\s+UCB=([0-9\.eE\-\+]+)\s+\|\s+f=([0-9\.eE\-\+]+)"
)

# BUCB+MI
ROUND_MI   = re.compile(r"--- Trial\s+(\d+),\s*Round\s+(\d+)")
PICK_MI    = re.compile(
    r"\[round\s+(\d+)\]\s+pick\s+(\d+)/(\d+):\s+(.+?)\s+\|\s+UCB\+MI=([0-9\.eE\-\+]+)\s+\|\s+f=([0-9\.eE\-\+]+)"
)

# ---------------- PARSERS ----------------
def parse_bucb_log(path):
    rows = []
    seen = set()
    trial_id = beta = None
    with open(path) as f:
        for line in f:
            if (m := ROUND_RE.search(line)):
                trial_id = int(m.group(1))
                beta     = float(m.group(2))
            if (m := PICK_BUCB.search(line)):
                rnum, pick_idx, pick_total, smiles, ucb, fval = m.groups()
                key = (trial_id, rnum, pick_idx)
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
                    "Method": "BUCB",
                })
    headers = ["Trial","Beta","Round","Pick","PickTotal","Selected SMILES","UCB","True f","Method"]
    return rows, headers

def parse_bucb_mi_log(path, trial_id):
    rows = []
    seen = set()
    round_id = None
    beta = 0.1  # fixed since you only care about beta=0.1
    with open(path) as f:
        for line in f:
            if (m := ROUND_MI.search(line)):
                trial_id = int(m.group(1))
                round_id = int(m.group(2))
            if (m := PICK_MI.search(line)):
                rnum, pick_idx, pick_total, smiles, ucbmi, fval = m.groups()
                key = (trial_id, rnum, pick_idx)
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
                    "UCB+MI": float(ucbmi),
                    "True f": float(fval),
                    "Method": "BUCB+MI",
                })
    headers = ["Trial","Beta","Round","Pick","PickTotal","Selected SMILES","UCB+MI","True f","Method"]
    return rows, headers

# ---------------- WRITER ----------------
def write_csv(outpath, rows, headers):
    if not rows:
        print(f"[WARN] No rows for {outpath}")
        return
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------------- MAIN ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for log_root in LOG_ROOTS:
        if not os.path.isdir(log_root):
            continue
        trial_name = os.path.basename(log_root)

        for fn in os.listdir(log_root):
            path = os.path.join(log_root, fn)
            if not os.path.isfile(path):
                continue

            if fn == "terminal_output_jax_bucb_fex.log":
                rows, headers = parse_bucb_log(path)
                outpath = os.path.join(OUT_DIR, f"BUCB_beta_0.1_{trial_name}.csv")
                write_csv(outpath, rows, headers)
                print(f"[{trial_name}] Wrote BUCB CSV: {outpath} ({len(rows)} rows)")

            elif fn == "terminal_output_jax_bucb_mi_fex.log":
                rows, headers = parse_bucb_mi_log(path, trial_id=trial_name)
                outpath = os.path.join(OUT_DIR, f"BUCB_MI_beta_0.1_{trial_name}.csv")
                write_csv(outpath, rows, headers)
                print(f"[{trial_name}] Wrote BUCB+MI CSV: {outpath} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
