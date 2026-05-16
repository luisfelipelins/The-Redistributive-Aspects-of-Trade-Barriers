"""
Emergency recovery script: re-runs only the combinations that previously
converged, as listed in successful_runs.log inside the crashed grid_search folder.

Usage
-----
Set CRASHED_FOLDER to the grid_search_* folder from the crashed run, then
run the script. Results are saved to a single pickle file once all runs finish.
"""

import pickle
import multiprocessing as mp
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import LOG_ISOLATED, OUTPUTS
from gmm_grid_search_estimation import _run_combo


# ---------------------------------------------------------------------------
# ← Set this to the crashed grid_search folder
# ---------------------------------------------------------------------------

CRASHED_FOLDER = LOG_ISOLATED / 'grid_search_20260504_230401'


# ---------------------------------------------------------------------------
# Read successful_runs.log to recover the converged combinations
# ---------------------------------------------------------------------------

def get_successful_combos(folder: Path) -> list:
    """Read successful_runs.log and return list of (α, γ, β, w_star, θ, σ, π_LL, π_HH)."""
    log_file = folder / 'successful_runs.log'
    if not log_file.exists():
        raise FileNotFoundError(f"successful_runs.log not found in {folder}")

    df = pd.read_csv(log_file, sep='\t')
    combos = [
        (row['α'], row['γ'], row['β'], row['w_star'],
         row['θ'], row['σ'], row['π_LL'], row['π_HH'])
        for _, row in df.iterrows()
    ]
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_emergency_recovery(crashed_folder=CRASHED_FOLDER, n_workers=None):

    crashed_folder = Path(crashed_folder)
    combos = get_successful_combos(crashed_folder)

    if not combos:
        print(f"No successful runs found in {crashed_folder}")
        return {}

    total = len(combos)
    print(f"Found {total} combinations to re-run from: {crashed_folder.name}")

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir     = LOG_ISOLATED / f'grid_search_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS / f'grid_search_{timestamp}.pkl'

    log_file = log_dir / 'successful_runs.log'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('i\tα\tγ\tβ\tw_star\tθ\tσ\tπ_LL\tπ_HH\n')

    args_list = [
        (α, γ, β, ws, θ, σ, π_LL, π_HH, log_dir, i, total)
        for i, (α, γ, β, ws, θ, σ, π_LL, π_HH) in enumerate(combos, 1)
    ]

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Using {n_workers} worker(s). Output: {output_path}", flush=True)

    results = {}
    n_ok    = 0
    n_done  = 0

    with mp.Pool(processes=n_workers) as pool:
        for key, result in pool.imap_unordered(_run_combo, args_list):
            n_done += 1
            if result is not None:
                n_ok += 1
                results[key] = result
                α, γ, β, w_star, θ, σ, π_LL, π_HH = key
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f'{n_ok}\t{α}\t{γ}\t{β}\t{w_star}\t{θ}\t{σ}\t{π_LL}\t{π_HH}\n')
            print(f"  Progress: {n_done}/{total} done, {n_ok} converged.", flush=True)

    print(f"All runs finished. {n_ok}/{total} recovered. Saving pickle ...", flush=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Log:     {log_file}")
    print(f"Results: {output_path}")
    return results


if __name__ == '__main__':
    mp.freeze_support()
    run_emergency_recovery()
