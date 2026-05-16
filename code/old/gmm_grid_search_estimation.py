"""
Grid search estimation for the one-good general equilibrium model.

Sweeps over combinations of α, γ, β, w_star, θ, σ, (π_LL, π_HH) and
saves results to a single pickle file at the end.

Total combinations: 8×2×6×4×7×3×3 = 24,192
"""

from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from datetime import datetime
from itertools import product
from config import LOG_ISOLATED, OUTPUTS
import multiprocessing as mp
import pickle


# ---------------------------------------------------------------------------
# Fixed calibration / model parameters
# ---------------------------------------------------------------------------

def _make_calib_params():
    return TypeCalibParameters(
        rh_N             = 9    ,
        rh_r             = 2    ,
        rh_c             = 0    ,
        vfi_lb           = 0    ,
        vfi_ubmul        = 15   ,
        vfi_N            = 500  ,
        vfi_eps          = 1e-5 ,
        vfi_howard_steps = 20   ,
        gmc_eps          = 1e-3 ,
        kmc_eps          = 1e-3 ,
        r_init_guess     = 0.03 ,
        inner_loop_eps   = 1e-5 ,
        outer_loop_eps   = 1e-5 ,
        outer_loop_r_lb  = 0.001,
    )


_FIXED_MODEL = dict(
    δ   = 0.960,
    ρ   = 0.950,
    σ_ϵ = 0.210,
    M   = 1.000,
)

# π_LL and π_HH are paired 1-to-1
PI_PAIRS = [(0.85, 0.83), (0.90, 0.88), (0.95, 0.93)]


# ---------------------------------------------------------------------------
# Worker (must be top-level for Windows multiprocessing pickling)
# ---------------------------------------------------------------------------

def _run_combo(args):
    α, γ, β, w_star, θ, σ, π_LL, π_HH, log_dir, i, total = args

    tag     = f'a{α}_g{γ}_b{β}_ws{w_star}_t{θ}_s{σ}_pLL{π_LL}_pHH{π_HH}'
    run_dir = log_dir / tag
    run_dir.mkdir(exist_ok=True)

    print(f"[{i}/{total}] {tag}", flush=True)

    mod_params = TypeModelParameters(
        α      = α,
        γ      = γ,
        β      = β,
        w_star = w_star,
        θ      = θ,
        σ      = σ,
        π_LL   = π_LL,
        π_HH   = π_HH,
        **_FIXED_MODEL,
    )

    try:
        mod = GeneralEquilibriumModel(mod_params, _make_calib_params(), log_dir=run_dir)
        mod.outer_loop_solver()
        if mod.outer_res.fun >= mod.CalibPar.outer_loop_eps:
            print(f"  NO CONV {tag}: outer_resid={mod.outer_res.fun:.4e}", flush=True)
            return (α, γ, β, w_star, θ, σ, π_LL, π_HH), None
        mod.economy_statistics()
        return (α, γ, β, w_star, θ, σ, π_LL, π_HH), mod
    except Exception as e:
        print(f"  FAILED {tag}: {e}", flush=True)
        return (α, γ, β, w_star, θ, σ, π_LL, π_HH), None


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def calibration_grid_search(
    α_vec      = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60],
    γ_vec      = [0.28, 0.33],
    β_vec      = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    w_star_vec = [0.1, 0.3, 0.5, 0.7],
    θ_vec      = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    σ_vec      = [1.5, 2.0, 2.5],
    pi_pairs   = PI_PAIRS,
    output_path = None,
    n_workers   = None,
):
    """Solve the one-good model for every parameter combination in parallel.

    Progress is logged to successful_runs.log after each converged run.
    Results are saved to a single pickle file once all runs are complete.
    Returns a dict keyed by (α, γ, β, w_star, θ, σ, π_LL, π_HH).
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir   = LOG_ISOLATED / f'grid_search_{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = OUTPUTS / f'grid_search_{timestamp}.pkl'

    combos = [
        (α, γ, β, ws, θ, σ, π_LL, π_HH)
        for α, γ, β, ws, θ, σ, (π_LL, π_HH)
        in product(α_vec, γ_vec, β_vec, w_star_vec, θ_vec, σ_vec, pi_pairs)
    ]
    total = len(combos)
    print(f"Starting grid search: {total} combinations.", flush=True)

    args_list = [
        (α, γ, β, ws, θ, σ, π_LL, π_HH, log_dir, i, total)
        for i, (α, γ, β, ws, θ, σ, π_LL, π_HH) in enumerate(combos, 1)
    ]

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"Using {n_workers} worker(s).", flush=True)

    log_file = log_dir / 'successful_runs.log'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('i\tα\tγ\tβ\tw_star\tθ\tσ\tπ_LL\tπ_HH\n')

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

    print(f"All runs finished. {n_ok}/{total} converged. Saving pickle ...", flush=True)
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Log:     {log_file}")
    print(f"Results: {output_path}", flush=True)
    return results


if __name__ == '__main__':
    mp.freeze_support()
    calibration_grid_search()
