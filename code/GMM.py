# -*- coding: utf-8 -*-
"""
Created on Mon Mar  27 21:15:31 2026

@author: lfval
"""

import re
import numpy as np
from datetime import datetime
from scipy.optimize import differential_evolution,direct,dual_annealing,minimize
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from GeneralEquilibriumModel import TypeModelParameters,TypeCalibParameters,GeneralEquilibriumModel
from config import LOG_GMM

MOMENT_NAMES = ['I','w_to_wstar','skill_premium']
PARAM_NAMES  = ['θ','w_star','β_eff']

SEP = '-' * 65

FAIL_PENALTY = 1e10

DE_SEED    = 13051905
DE_MAXITER = 1000
DE_KWARGS = dict(popsize=15,tol=0,atol=1e-6,polish=True,workers=1,init='latinhypercube',updating='immediate')

_RUNLOG_PAT = re.compile(r'^\s*(\d+)\s+θ=.*\bobj=([-\d.eE+]+)\s*$')
_SUMMARY_PAT = re.compile(r'α=(\S+)\s+γ=\S+\s+β_eff=(\S+)\s+w_star=(\S+)\s*\n' r'\s*θ=(\S+)\s')


def _write_eval_log(eval_dir,params,g,obj,data_moments):
    θ,ws,β_eff = params

    with open(eval_dir / 'gmm_eval_res.log','w',encoding='utf-8') as f:
        f.write(f"GMM Evaluation  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{SEP}\n")
        f.write(f"Parameters:\n")
        f.write(f"  " f"  θ={θ:.6f}  β_eff={β_eff:.6f}  w_star={ws:.6f}\n")
        f.write(f"{SEP}\n")
        f.write(f"Relative moment distances (data - model) / data:\n")

        for name,val in zip(MOMENT_NAMES,g):
            model_val = data_moments[name] * (1 - val)
            f.write(f"  {name:<22}: {val:+.6f} [{model_val:.6f}]\n")

        f.write(f"{SEP}\n")
        f.write(f"Objective: {obj:.8e}\n")


def _append_run_log(run_log_path,eval_n,params,obj):
    θ,ws,β_eff = params

    with open(run_log_path,'a',encoding='utf-8') as f:
        f.write(f"{eval_n:>6}  θ={θ:.4f}  β_eff={β_eff:.4f} w_star={ws:.6f} obj={obj:.6e}\n")


def _write_final_log(gmm_run_dir,params,g,obj,success,data_moments):
    θ,ws,β_eff = params

    with open(gmm_run_dir / 'gmm_final_res.log','w',encoding='utf-8') as f:
        f.write(f"GMM Final Result  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Converged: {success}\n")
        f.write(f"{SEP}\n")
        f.write(f"Final parameters:\n")
        f.write(f"  " f"  θ={θ:.6f}  β_eff={β_eff:.6f}  w_star={ws:.6f}\n")
        f.write(f"{SEP}\n")

        if g is None:
            f.write(f"Moment distances unavailable: best point comes from replayed history.\n")
        else:
            f.write(f"Relative moment distances (data - model) / data:\n")

            for name,val in zip(MOMENT_NAMES,g):
                model_val = data_moments[name] * (1 - val)
                f.write(f"  {name:<22}: {val:+.6f} [{model_val:.6f}]\n")

        f.write(f"{SEP}\n")
        f.write(f"Objective: {obj:.8e}\n")


def gmm_model_moments(params,ModelPar,CalibPar,data_moments,log_dir,log_summary_name):
    θ,ws,β_eff = params
    if not np.all(np.isfinite(params)) or θ <= 0 or ws <= 0 or β_eff < 1:
        raise ValueError('Require θ > 0, w_star > 0 and β_eff >= 1.')

    modpar = TypeModelParameters(
        α            = ModelPar.α,
        γ            = ModelPar.γ,
        ψ            = ModelPar.ψ,
        χ            = ModelPar.χ,
        β_eff        = β_eff,
        τ            = ModelPar.τ,
        w_star       = ws,
        θ            = θ,
        σ            = ModelPar.σ,
        δ            = ModelPar.δ,
        ϱ            = ModelPar.ϱ,
        σ_ϵ          = ModelPar.σ_ϵ,
        π_LL         = ModelPar.π_LL,
        π_HH         = ModelPar.π_HH,
        M            = ModelPar.M,
        ξ            = ModelPar.ξ,
        rebate_share = ModelPar.rebate_share,
    )

    model = GeneralEquilibriumModel(modpar,CalibPar,log_dir=log_dir,log_inner=False,log_summary_name=log_summary_name)
    model.outer_loop_solver()
    if not np.isfinite(model.outer_res['fun']) or model.outer_res['fun'] > CalibPar.outer_loop_eps:
        raise ValueError(f"Outer loop did not converge!")

    moments = model.economy_statistics()

    moments_vec = np.array([moments[name] for name in MOMENT_NAMES])
    data_vec    = np.array([data_moments[name] for name in MOMENT_NAMES])
    if not np.all(np.isfinite(moments_vec)) or np.any(data_vec <= 0):
        raise ValueError('Require finite model moments and positive data targets.')

    return (data_vec - moments_vec) / data_vec


def gmm_objective(params,ModelPar,CalibPar,data_moments,W,log_dir,log_summary_name='run_summary.log'):
    g   = gmm_model_moments(params=params,ModelPar=ModelPar,CalibPar=CalibPar,data_moments=data_moments,log_dir=log_dir,log_summary_name=log_summary_name)
    obj = float(g @ W @ g)

    return obj


def _parse_run_history(run_dir):
    """
    Rebuilds the evaluation history of a previous DE run from its logs.

    Objectives come from gmm_run.log, which rounds them to six significant digits.
    Parameters come from the run_summary_NNNNN.log files, which store them at full
    precision. Evaluations logged at the failure penalty are recomputed exactly from
    those parameters, since gmm_run.log collapses all of them to 1.000000e+10.

    Returns
    -------
    P : ndarray, shape (n, 3)
        Parameter vectors [θ, w_star, β_eff] in evaluation order.
    E : ndarray, shape (n,)
        Objective values in evaluation order.
    """

    if 'Production: nested_ces' not in (run_dir / 'gmm_run.log').read_text(encoding='utf-8'):
        raise ValueError('Cannot resume a Cobb-Douglas run with the nested-CES estimator.')

    eval_ns,objs = [],[]

    with open(run_dir / 'gmm_run.log','r',encoding='utf-8') as f:
        for line in f:
            m = _RUNLOG_PAT.match(line)

            if m:
                eval_ns.append(int(m.group(1)))
                objs.append(float(m.group(2)))

    if eval_ns != list(range(1,len(eval_ns) + 1)):
        raise ValueError(f"gmm_run.log in {run_dir} has gaps or out-of-order evaluations.")

    P = np.empty((len(eval_ns),3))

    for i,n in enumerate(eval_ns):
        summary = run_dir / f'run_summary_{n:05d}.log'

        if not summary.exists():
            raise FileNotFoundError(f"Missing {summary.name}; cannot recover parameters for evaluation {n}.")

        m = _SUMMARY_PAT.search(summary.read_text(encoding='utf-8'))

        if m is None:
            raise ValueError(f"Could not parse parameters from {summary.name}. Runs logged "
                             f"before β was split into (1+τ)·β_eff cannot be resumed.")

        α,β_eff,ws,θ = (float(v) for v in m.groups())
        P[i]         = [θ,ws,β_eff]

    E         = np.asarray(objs)
    failed    = E >= 1e9
    E[failed] = FAIL_PENALTY + np.sum(np.square(P[failed]),axis=1)

    return P,E


def _rebuild_de_solver(func,bounds,run_dir,disp=True):
    """
    Restores a DifferentialEvolutionSolver to the state a previous run had reached.

    The solver is deterministic given DE_SEED: its path depends only on the initial
    latin-hypercube population and on the accept/reject decisions, which are driven by
    the objective values. Replaying the recorded objectives in evaluation order therefore
    reproduces the population, the energies and the RNG state exactly, without solving the
    model once. Every trial the solver generates is checked against the recorded parameters,
    so any divergence is caught at the evaluation where it happens rather than silently
    continuing from a wrong state.

    A trailing partial generation in the log is discarded — the replay only advances in
    whole generations.

    Returns
    -------
    solver : DifferentialEvolutionSolver
        Restored to the end of the last complete generation.
    n_evals : int
        Number of evaluations consumed by the replay.
    n_gen : int
        Number of complete generations replayed after the initial population.
    """

    P,E = _parse_run_history(run_dir)

    solver = DifferentialEvolutionSolver(func,bounds,rng=DE_SEED,maxiter=DE_MAXITER,disp=disp,**DE_KWARGS)

    pop_size = solver.num_population_members
    n_gen    = (len(E) - pop_size) // pop_size

    if n_gen < 0:
        raise ValueError(f"{run_dir.name} holds {len(E)} evaluations, fewer than the " f"{pop_size} needed for the initial population.")

    n_evals = pop_size * (1 + n_gen)

    cursor = [0]

    def replay(x):
        i = cursor[0]
        cursor[0] += 1
        drift = np.abs(np.asarray(x) - P[i]).max()

        if drift > 1e-9:
            raise ValueError(f"Replay diverged from {run_dir.name} at evaluation {i + 1}: " f"generated parameters differ by {drift:.3e}.")

        return E[i]

    real_func   = solver.func
    solver.func = replay

    try:
        solver.feasible,solver.constraint_violation = (
            solver._calculate_population_feasibilities(solver.population))
        solver.population_energies[solver.feasible] = (
            solver._calculate_population_energies(solver.population[solver.feasible]))
        solver._promote_lowest_energy()

        for _ in range(n_gen):
            next(solver)
    finally:
        solver.func = real_func

    if cursor[0] != n_evals:
        raise ValueError(f"Replay consumed {cursor[0]} evaluations, expected {n_evals}.")

    solver._nfev = n_evals

    return solver,n_evals,n_gen


def run_gmm(ModelPar,CalibPar,data_moments,W,bounds=None,x0=None,algorithm='differential_evolution',resume_from=None):
    if np.shape(W) != (3,3) or (bounds is not None and len(bounds) != 3) or (x0 is not None and len(x0) != 3):
        raise ValueError('Nested-CES GMM requires three parameters and a 3-by-3 weight matrix.')

    _bounds_based = ('differential_evolution','crs','simulated_annealing')
    _point_based  = ('nelder_mead','powell')
    _valid        = _bounds_based + _point_based

    if algorithm not in _valid:
        raise ValueError(f"Unknown algorithm: '{algorithm}'. Choose from: {_valid}.")
    if algorithm in _bounds_based and bounds is None:
        raise ValueError(f"Algorithm '{algorithm}' requires 'bounds'.")
    if algorithm in _point_based and x0 is None:
        raise ValueError(f"Algorithm '{algorithm}' requires 'x0'.")
    if resume_from is not None and algorithm != 'differential_evolution':
        raise ValueError(f"'resume_from' is only supported for differential_evolution, got '{algorithm}'.")

    if resume_from is not None:
        gmm_run_dir = LOG_GMM / resume_from if isinstance(resume_from,str) else resume_from

        if not (gmm_run_dir / 'gmm_run.log').exists():
            raise FileNotFoundError(f"No gmm_run.log in {gmm_run_dir}; nothing to resume from.")
        if 'Production: nested_ces' not in (gmm_run_dir / 'gmm_run.log').read_text(encoding='utf-8'):
            raise ValueError('Cannot resume a Cobb-Douglas run with the nested-CES estimator.')
    else:
        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        gmm_run_dir = LOG_GMM / f'gmm_run_{timestamp}'
        gmm_run_dir.mkdir()

    param_names = PARAM_NAMES

    run_log_path = gmm_run_dir / 'gmm_run.log'

    with open(run_log_path,'a' if resume_from is not None else 'w',encoding='utf-8') as f:
        f.write(f"GMM Run  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write("Production: nested_ces\n")
        f.write(f"Calibrated production: α={ModelPar.α} γ={ModelPar.γ} ψ={ModelPar.ψ} χ={ModelPar.χ}\n")
        f.write(f"{SEP}\n")
        f.write(f"Data moments:\n")

        for name,val in data_moments.items():
            f.write(f"  {name:<22}: {val}\n")

        f.write(f"{SEP}\n")
        f.write(f"Weights (W diagonal):\n")
        w_diag = np.diag(W)

        for name,w in zip(MOMENT_NAMES,w_diag):
            f.write(f"  {name:<22}: {w}\n")

        f.write(f"{SEP}\n")

        if bounds is not None:
            f.write(f"Bounds:\n")

            for name,(lb,ub) in zip(param_names,bounds):
                f.write(f"  {name:<6}: [{lb}, {ub}]\n")
        else:
            f.write(f"Initial point (x0):\n")

            for name,val in zip(param_names,x0):
                f.write(f"  {name:<6}: {val}\n")

        f.write(f"{SEP}\n")
        f.write(f"Policy parameters (fixed outside the estimation):\n")
        f.write(f"  τ={ModelPar.τ}  ξ={ModelPar.ξ}  rebate_share={ModelPar.rebate_share}\n")
        f.write(f"{SEP}\n")
        f.write(f"{'eval':>6}  {'θ':>8}  {'β_eff':>8}  {'w_star':>8}  {'obj':>14}\n")

    eval_counter = [0]
    best         = {'obj': np.inf,'g': None,'params': None}

    def objective_wrapper(params):
        eval_counter[0] += 1
        n = eval_counter[0]

        try:
            g = gmm_model_moments(
                params=params,
                ModelPar=ModelPar,
                CalibPar=CalibPar,
                data_moments=data_moments,
                log_dir=gmm_run_dir,
                log_summary_name=f'run_summary_{n:05d}.log')
            obj = float(g @ W @ g)

            if obj < best['obj']:
                best['obj']    = obj
                best['g']      = g.copy()
                best['params'] = params.copy()

            _append_run_log(run_log_path,n,params,obj)

        except Exception as error:
            with open(gmm_run_dir / 'gmm_errors.log','a',encoding='utf-8') as f:
                f.write(f'{n}: {type(error).__name__}: {error}\n')
            obj = FAIL_PENALTY + float(np.sum(np.square(params)))
            _append_run_log(run_log_path,n,params,obj)

        return obj

    if algorithm == 'differential_evolution' and resume_from is not None:
        print(f"Rebuilding solver state from {gmm_run_dir.name} ...")

        solver,n_evals,n_gen = _rebuild_de_solver(objective_wrapper,bounds,gmm_run_dir)
        eval_counter[0]      = n_evals

        remaining = DE_MAXITER - n_gen

        if remaining <= 0:
            raise ValueError(f"{gmm_run_dir.name} already completed {n_gen} of {DE_MAXITER} generations.")

        solver.maxiter = remaining

        best['obj']    = float(solver.population_energies[0])
        best['params'] = solver._scale_parameters(solver.population[0])

        print(f"Replayed {n_evals} evaluations over {n_gen} generations. " f"Best objective so far: {best['obj']:.8e}")
        print(f"Resuming for up to {remaining} further generations.\n")

        result = solver.solve()

    elif algorithm == 'differential_evolution':
        result = differential_evolution(objective_wrapper,bounds=bounds,maxiter=DE_MAXITER,seed=DE_SEED,disp=True,**DE_KWARGS)
    elif algorithm == 'crs':
        result = direct(objective_wrapper,bounds=bounds,maxfun=15_000,eps=1e-4,locally_biased=False)
    elif algorithm == 'simulated_annealing':
        result = dual_annealing(objective_wrapper,bounds=bounds,maxiter=10_000,maxfun=15_000,seed=13051905)
    elif algorithm == 'nelder_mead':
        result = minimize(objective_wrapper,
                          x0      = x0,
                          method  = 'Nelder-Mead',
                          options = dict(maxiter=10_000,maxfev=15_000,xatol=1e-6,fatol=1e-8,disp=True))
    else:  # powell
        result = minimize(objective_wrapper,
                          x0      = x0,
                          method  = 'Powell',
                          options = dict(maxiter=10_000,maxfev=15_000,xtol=1e-6,ftol=1e-8,disp=True))

    if best['params'] is not None:
        _write_final_log(gmm_run_dir,best['params'],best['g'],best['obj'],result.success,data_moments)
    else:
        with open(gmm_run_dir / 'gmm_final_res.log','w',encoding='utf-8') as f:
            f.write("GMM Final Result: all evaluations failed — no valid solution found.\n")

    return result,best
