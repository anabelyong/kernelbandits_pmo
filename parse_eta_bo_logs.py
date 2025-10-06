import os
import re
import csv

LOG_ROOTS = ["logs_trial1", "logs_trial2", "logs_trial3"]
OUT_DIR = "csv_results"

# Regex for BUCB+MI logs
ROUND_RE = re.compile(r"--- Trial\s+(\d+),\s*eta0=([0-9\.]+),\s*Round\s+(\d+)")
PICK_RE  = re.compile(
    r"\[round\s+(\d+)\]\s+pick\s+(\d+)/(\d+):\s+(.+?)\s+\|\s+UCB\+MI=([0-9\.eE\-\+]+)\s+\|\s+f=([0-9\.eE\-\+]+)"
)

def parse_bucb_mi_log(path):
    rows = []
    trial_id = eta = round_id = None
    seen = set()
    with open(path) as f:
        for line in f:
            if (m := ROUND_RE.search(line)):
                trial_id = int(m.group(1))
                eta      = float(m.group(2))
                round_id = int(m.group(3))

            if (m := PICK_RE.search(line)):
                rnum, pick_idx, pick_total, smiles, ucb_mi, fval = m.groups()
                key = (trial_id, eta, int(rnum), int(pick_idx))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "Trial": trial_id,
                    "Eta": eta,
                    "Round": int(rnum),
                    "Pick": int(pick_idx),
                    "PickTotal": int(pick_total),
                    "Selected SMILES": smiles,
                    "UCB+MI": float(ucb_mi),
                    "True f": float(fval),
                })
    headers = ["Trial","Eta","Round","Pick","PickTotal","Selected SMILES","UCB+MI","True f"]
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
        trial_id = log_root.replace("logs_trial", "")

        for eta_dir in os.listdir(log_root):
            eta_path = os.path.join(log_root, eta_dir)
            if not os.path.isdir(eta_path):
                continue

            # eta folder name like "eta_0.25"
            if not eta_dir.startswith("eta_"):
                continue
            eta_val = eta_dir.split("_")[1]

            for fn in os.listdir(eta_path):
                if not fn.endswith(".log"):
                    continue
                path = os.path.join(eta_path, fn)

                if "bucb_mi" in fn:
                    rows, headers = parse_bucb_mi_log(path)
                    outname = f"BUCB_MI_eta_{eta_val}_logs_trial{trial_id}.csv"
                    outpath = os.path.join(OUT_DIR, outname)
                    write_csv(outpath, rows, headers)
                    print(f"Wrote {outpath} with {len(rows)} rows")

if __name__ == "__main__":
    main()
