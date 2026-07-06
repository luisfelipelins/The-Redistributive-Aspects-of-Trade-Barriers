# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 21:15:31 2026

@author: lfval
"""

import numpy as np
from datetime import datetime
from scipy.optimize import differential_evolution, direct, dual_annealing, minimize
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from config import LOG_GMM

MOMENT_NAMES  = ['high_skill_share', 'skill_premium', 'w_to_wstar', 'I']

SEP = '-' * 65


def _write_eval_log(eval_dir, params, g, obj, data_moments):
    α, ws, θ, β = params
    with open(eval_dir / 'gmm_eval_res.log', 'w', encoding='utf-8') as f:
        f.write(f"GMM Evaluation  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{SEP}\n")
        f.write(f"Parameters:\n")
        f.write(f"  α={α:.6f}  "
                f"  θ={θ:.6f}  β={β:.6f}  w_star={ws:.6f}\n")
        f.write(f"{SEP}\n")
        f.write(f"Moment distances (model - data):\n")
        for name, val in zip(MOMENT_NAMES, g):
            model_val = data_moments[name] * (1 + val)
            f.write(f"  {name:<22}: {val:+.6f} [{model_val:.6f}]\n")
        f.write(f"{SEP}\n")
        f.write(f"Objective: {obj:.8e}\n")


def _append_run_log(run_log_path, eval_n, params, obj):
    α, ws, θ, β = params
    with open(run_log_path, 'a', encoding='utf-8') as f:
        f.write(
            f"{eval_n:>6}  α={α:.4f}  θ={θ:.4f}  β={β:.4f} w_star={ws:.6f} obj={obj:.6e}\n"
        )


def _write_final_log(gmm_run_dir, params, g, obj, success, data_moments):
    α, ws, θ, β = params
    with open(gmm_run_dir / 'gmm_final_res.log', 'w', encoding='utf-8') as f:
        f.write(f"GMM Final Result  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Converged: {success}\n")
        f.write(f"{SEP}\n")
        f.write(f"Final parameters:\n")
        f.write(f"  α={α:.6f}  "
                f"  θ={θ:.6f}  β={β:.6f}  w_star={ws:.6f}\n")
        f.write(f"{SEP}\n")
        f.write(f"Moment distances (model - data):\n")
        for name, val in zip(MOMENT_NAMES, g):
            model_val = data_moments[name] * (1 + val)
            f.write(f"  {name:<22}: {val:+.6f} [{model_val:.6f}]\n")
        f.write(f"{SEP}\n")
        f.write(f"Objective: {obj:.8e}\n")


def gmm_model_moments(params, ModelPar, CalibPar, data_moments, log_dir, log_summary_name, t_form='exponential'):
    α, ws, θ, β = params

    modpar = TypeModelParameters(
        α      = α       ,
        γ      = ModelPar.γ    ,
        β      = β       ,
        w_star = ws      ,
        θ      = θ       ,
        σ      = ModelPar.σ   ,
        δ      = ModelPar.δ   ,
        ρ      = ModelPar.ρ   ,
        σ_ϵ    = ModelPar.σ_ϵ   ,
        π_LL   = ModelPar.π_LL    ,
        π_HH   = ModelPar.π_HH    ,
        M      = ModelPar.M   ,
        t_form = t_form  ,
    )

    model = GeneralEquilibriumModel(modpar, CalibPar, log_dir=log_dir, log_inner=False, log_summary_name=log_summary_name)
    model.outer_loop_solver()
    moments = model.economy_statistics()

    if model.outer_res['fun'] > CalibPar.outer_loop_eps:
        raise ValueError(f"Outer loop did not converge!")

    moments_vec = np.array([moments['high_skill_share'],
                            moments['skill_premium'],
                            moments['w_to_wstar'],
                            moments['I']])
    data_vec    = np.array([data_moments['high_skill_share'],
                            data_moments['skill_premium'],
                            data_moments['w_to_wstar'],
                            data_moments['I']])

    return (moments_vec - data_vec) / data_vec


def gmm_objective(params, ModelPar, CalibPar, data_moments, W, log_dir, log_summary_name='run_summary.log'):
    g   = gmm_model_moments(params=params, ModelPar=ModelPar, CalibPar=CalibPar, data_moments=data_moments, log_dir=log_dir, log_summary_name=log_summary_name)
    obj = float(g @ W @ g)
    return obj


def run_gmm(ModelPar, CalibPar, data_moments, W, bounds=None, x0=None, algorithm='differential_evolution', t_form='quadratic'):
    _bounds_based = ('differential_evolution', 'crs', 'simulated_annealing')
    _point_based  = ('nelder_mead', 'powell')
    _valid = _bounds_based + _point_based
    if algorithm not in _valid:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Choose from: {_valid}.")
    if algorithm in _bounds_based and bounds is None:
        raise ValueError(f"Algorithm '{algorithm}' requires 'bounds'.")
    if algorithm in _point_based and x0 is None:
        raise ValueError(f"Algorithm '{algorithm}' requires 'x0'.")

    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    gmm_run_dir = LOG_GMM / f'gmm_run_{timestamp}'
    gmm_run_dir.mkdir()

    param_names = ['α', 'w_star', 'θ', 'β']

    run_log_path = gmm_run_dir / 'gmm_run.log'
    with open(run_log_path, 'w', encoding='utf-8') as f:
        f.write(f"GMM Run  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"{SEP}\n")
        f.write(f"Data moments:\n")
        for name, val in data_moments.items():
            f.write(f"  {name:<22}: {val}\n")
        f.write(f"{SEP}\n")
        f.write(f"Weights (W diagonal):\n")
        w_diag = np.diag(W)
        for name, w in zip(MOMENT_NAMES, w_diag):
            f.write(f"  {name:<22}: {w}\n")
        f.write(f"{SEP}\n")
        if bounds is not None:
            f.write(f"Bounds:\n")
            for name, (lb, ub) in zip(param_names, bounds):
                f.write(f"  {name:<6}: [{lb}, {ub}]\n")
        else:
            f.write(f"Initial point (x0):\n")
            for name, val in zip(param_names, x0):
                f.write(f"  {name:<6}: {val}\n")
        f.write(f"{SEP}\n")
        f.write(f"{'eval':>6}  {'α':>8}  {'θ':>8}  {'β':>8}  {'w_star':>8}  {'obj':>14}\n")

    eval_counter = [0]
    best         = {'obj': np.inf, 'g': None, 'params': None}

    def objective_wrapper(params):
        eval_counter[0] += 1
        n = eval_counter[0]

        try:
            g   = gmm_model_moments(params=params, ModelPar=ModelPar, CalibPar=CalibPar, data_moments=data_moments,
                                    log_dir=gmm_run_dir, log_summary_name=f'run_summary_{n:05d}.log',
                                    t_form=t_form)
            obj = float(g @ W @ g)

            if obj < best['obj']:
                best['obj']    = obj
                best['g']      = g.copy()
                best['params'] = params.copy()

            _append_run_log(run_log_path, n, params, obj)

        except Exception:
            obj = 1e10 + float(np.sum(np.square(params)))
            _append_run_log(run_log_path, n, params, obj)

        return obj

    if algorithm == 'differential_evolution':
        result = differential_evolution(objective_wrapper,
                                        bounds  = bounds,
                                        popsize = 15,
                                        maxiter = 1000,
                                        tol     = 0,
                                        atol    = 1e-6,
                                        seed    = 13051905,
                                        disp    = True,
                                        polish  = True,
                                        workers = 1)
    elif algorithm == 'crs':
        result = direct(objective_wrapper,
                        bounds         = bounds,
                        maxfun         = 15_000,
                        eps            = 1e-4,
                        locally_biased = False)
    elif algorithm == 'simulated_annealing':
        result = dual_annealing(objective_wrapper,
                                bounds  = bounds,
                                maxiter = 10_000,
                                maxfun  = 15_000,
                                seed    = 13051905)
    elif algorithm == 'nelder_mead':
        result = minimize(objective_wrapper,
                          x0      = x0,
                          method  = 'Nelder-Mead',
                          options = dict(maxiter=10_000, maxfev=15_000, xatol=1e-6, fatol=1e-8, disp=True))
    else:  # powell
        result = minimize(objective_wrapper,
                          x0      = x0,
                          method  = 'Powell',
                          options = dict(maxiter=10_000, maxfev=15_000, xtol=1e-6, ftol=1e-8, disp=True))

    if best['params'] is not None:
        _write_final_log(gmm_run_dir, best['params'], best['g'], best['obj'], result.success, data_moments)
    else:
        with open(gmm_run_dir / 'gmm_final_res.log', 'w', encoding='utf-8') as f:
            f.write("GMM Final Result: all evaluations failed — no valid solution found.\n")

    return result, best

