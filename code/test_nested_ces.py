# -*- coding: utf-8 -*-
"""Benchmark one GMM evaluation; run with python code/test_nested_ces.py."""
import contextlib
import json
import platform
from time import perf_counter

import numpy as np
from GeneralEquilibriumModel import TypeCalibParameters,TypeModelParameters
from config import DATA_PARAMS,LOG_GMM
from GMM import gmm_objective,DE_KWARGS,DE_MAXITER


def main():
    # Keep these settings and the calibration file identical across machines.
    params    = np.array([1.0,0.5631733145879515,1.1,0.55])  # θ, w_star, β_eff, γ
    n_repeats = 3

    with open(DATA_PARAMS / 'pre_gmm_params_ces.json','r',encoding='utf-8') as f:
        pre_gmm_params = json.load(f)

    CalibPar = TypeCalibParameters(
        rh_N             = 9,
        rh_r             = 2,
        rh_c             = 0,
        vfi_lb           = 0,
        vfi_ubmul        = 60,
        vfi_N            = 500,
        vfi_eps          = 1e-5,
        vfi_howard_steps = 20,
        gmc_eps          = 1e-3,
        kmc_eps          = 1e-3,
        r_init_guess     = 0.03,
        inner_loop_eps   = 1e-5,
        outer_loop_eps   = 1e-5,
        outer_loop_r_lb  = 0.001)

    ModelPar = TypeModelParameters(
        α            = pre_gmm_params['parameters']['α'],
        γ            = np.nan,
        ψ            = pre_gmm_params['parameters']['ψ'],
        χ            = pre_gmm_params['parameters']['χ'],
        β_eff        = np.nan,
        τ            = pre_gmm_params['parameters']['τ'],
        w_star       = np.nan,
        θ            = np.nan,
        σ            = pre_gmm_params['parameters']['σ'],
        δ            = pre_gmm_params['parameters']['δ'],
        ϱ            = pre_gmm_params['parameters']['ϱ'],
        σ_ϵ          = pre_gmm_params['parameters']['σ_ϵ'],
        π_LL         = pre_gmm_params['parameters']['π_LL'],
        π_HH         = pre_gmm_params['parameters']['π_HH'],
        M            = pre_gmm_params['parameters']['M'],
        ξ            = pre_gmm_params['parameters']['ξ'],
        rebate_share = pre_gmm_params['parameters']['rebate_share'])

    data_moments = {
        'skill_premium': pre_gmm_params['moments']['skill_premium'],
        'w_to_wstar': pre_gmm_params['moments']['w_to_wstar'],
        'I': pre_gmm_params['moments']['I'],
        'low_skill_share': pre_gmm_params['moments']['LS_share']}

    log_dir = LOG_GMM / 'benchmark_nested_ces'
    times   = []

    log_dir.mkdir(exist_ok=True)
    print(f'Machine: {platform.node()} | {platform.platform()} | Python {platform.python_version()}')
    print(f'Parameters [theta, w_star, beta_eff, gamma]: {params}')
    print(f'Grid: rh_N={CalibPar.rh_N}, vfi_N={CalibPar.vfi_N}')
    print(f'Model output: {log_dir}',flush=True)

    # Each call builds a fresh model; only compiled code is reused.
    for repetition in range(n_repeats + 1):
        print(f'Running evaluation {repetition + 1}/{n_repeats + 1}...',flush=True)
        with open(log_dir / f'console_{repetition}.log','w',encoding='utf-8') as output:
            with contextlib.redirect_stdout(output):
                start = perf_counter()
                objective = gmm_objective(
                    params           = params,
                    ModelPar         = ModelPar,
                    CalibPar         = CalibPar,
                    data_moments     = data_moments,
                    W                = np.eye(4),
                    log_dir          = log_dir,
                    log_summary_name = f'summary_{repetition}.log')
                elapsed = perf_counter() - start
        times.append(elapsed)
        label = 'First call (includes JIT/cache loading)' if repetition == 0 else 'Warm call'
        print(f'{label}: {elapsed:.2f} s | objective = {objective:.8e}',flush=True)

    median_time = float(np.median(times[1:]))
    population  = DE_KWARGS['popsize'] * len(params)
    evaluations = (DE_MAXITER + 1) * population

    print(f'\nWarm median: {median_time:.2f} s/evaluation')
    print(f'Warm range: {min(times[1:]):.2f} to {max(times[1:]):.2f} s')
    print(f'Projected generation ({population} evaluations): {population * median_time / 60:.2f} min')
    print(f'Projected {DE_MAXITER} generations + initial population: {evaluations * median_time / 3600:.2f} h')
    print('Projection assumes serial evaluations, no early stopping and excludes polishing.')
    print('Other parameter vectors can take different times; this is not a full GMM run.')
    print('PC/VM warm-median ratio measures the VM speedup for this workload.')


if __name__ == '__main__':
    main()
