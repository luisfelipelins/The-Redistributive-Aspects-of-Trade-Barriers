# -*- coding: utf-8 -*-
"""
Machinery for the deterministic transition between two stationary equilibria.

Steady-state code lives in functions.py and is not modified by this module: the
dependency runs one way only, functions_transition -> functions.

@author: Luis Felipe de Oliveira Valadares Lins
"""

import time
import numpy as np
import numba as nb
from scipy.optimize import brentq

import functions
from functions import t, Omega

def solve_transition_system(K_t, I_t, β_t, H, L, ModelPar):
    '''
    Solves at time period t the system determining the aggregate variables given K_t
    and a candidate marginal task I_t.

    Given I_t, K_t and β_t, find:
      (1) w_t from the MT condition: w = w*·β_t·t(I)          -> (TA)
      (2) Y_t from the L market clearing: L = (1-I)·κ·Y/(w·Ω) -> (TB)
      (3) s_t from the H market clearing: H = α·Y/s           -> (TC)
      (4) r_t from the K market clearing: K = γ·Y/r           -> (TD)

    The zero-profit condition is not imposed here: it is the residual rooted in I_t by
    solve_transition_system_wrapper.

    Part of: solve_transition_system -> transition_system_residual
             -> solve_transition_system_wrapper

    Parameters
    ----------
    K_t : float
        Aggregate capital at period t.
    I_t : float
        Guess for the marginal task to be rooted.
    β_t : float
        Value of β at period t.
    H : float
        Aggregate high-skill labour supply.
    L : float
        Aggregate low-skill labour supply.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    dict
        Solution dictionary with keys: w, s, r, Y, I, K, Ω.

    '''

    κ  : float = 1 - ModelPar.α - ModelPar.γ

    w_t: float = ModelPar.w_star * β_t * t(I_t, ModelPar)
    Ω_t: float = Omega(I_t, ModelPar)
    Y_t: float = (L * w_t * Ω_t) / ((1 - I_t) * κ)
    s_t: float = ModelPar.α * Y_t / H
    r_t: float = ModelPar.γ * Y_t / K_t

    return {'w': w_t,
            's': s_t,
            'r': r_t,
            'Y': Y_t,
            'I': I_t,
            'K': K_t,
            'Ω': Ω_t}

def transition_system_residual(I_t, K_t, β_t, H, L, ModelPar):
    '''
    Log zero-profit residual for a given guess of the marginal task at period t.

    R(I) = κ·log(w(I)·Ω(I)/κ) + α·log(s(I)/α) + γ·log(r(I)/γ)

    with w, s and r taken from solve_transition_system. At equilibrium R(I_t) = 0.

    Parameters
    ----------
    I_t : float
        Guess for the marginal task.
    K_t : float
        Aggregate capital at period t.
    β_t : float
        Value of β at period t.
    H : float
        Aggregate high-skill labour supply.
    L : float
        Aggregate low-skill labour supply.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    float
        Log zero-profit residual.

    '''

    sol: dict  = solve_transition_system(K_t=K_t, I_t=I_t, β_t=β_t, H=H, L=L, ModelPar=ModelPar)
    κ  : float = 1 - ModelPar.α - ModelPar.γ

    return (κ * np.log(sol['w'] * sol['Ω'] / κ)
            + ModelPar.α * np.log(sol['s'] / ModelPar.α)
            + ModelPar.γ * np.log(sol['r'] / ModelPar.γ))

def solve_transition_system_wrapper(K_t, β_t, H, L, ModelPar, tol=1e-12, I_lb=1e-9, I_ub=1-1e-9):
    '''
    Solves the period-t production block given aggregate capital K_t and offshoring
    cost β_t.

    The system is reduced to a scalar root-finding problem in I_t: every candidate
    I ∈ (0,1) pins down (w, Y, s, r) through (TA)-(TD), and the equilibrium I_t is the
    one at which the log zero-profit residual vanishes.

    R(I) → +∞ as I → 1 because Y diverges with the L market clearing condition, so a
    sign change over the bracket is enough to guarantee a unique interior root.

    Parameters
    ----------
    K_t : float
        Aggregate capital at period t.
    β_t : float
        Value of β at period t.
    H : float
        Aggregate high-skill labour supply.
    L : float
        Aggregate low-skill labour supply.
    ModelPar : TypeModelParameters
        Model parameters.
    tol : float, optional
        Root-finding tolerance. The default is 1e-12.
    I_lb : float, optional
        Lower bound of the bracket for I_t. The default is 1e-9.
    I_ub : float, optional
        Upper bound of the bracket for I_t. The default is 1-1e-9.

    Returns
    -------
    dict
        Solution dictionary with keys: w, s, r, Y, I, K, Ω.

    '''

    args   : tuple = (K_t, β_t, H, L, ModelPar)

    R_lb   : float = transition_system_residual(I_lb, *args)
    R_ub   : float = transition_system_residual(I_ub, *args)

    if R_lb * R_ub > 0:
        raise ValueError(f'No interior I_t ∈ ({I_lb}, {I_ub}) satisfies the zero-profit '
                         f'condition for K_t={K_t}, β_t={β_t}: '
                         f'R({I_lb})={R_lb}, R({I_ub})={R_ub}.')

    root: float = brentq(f=transition_system_residual, a=I_lb, b=I_ub, args=args, xtol=tol)

    return solve_transition_system(K_t=K_t, I_t=root, β_t=β_t, H=H, L=L, ModelPar=ModelPar)

def production_path(K_path, β_path, H, L, ModelPar):
    '''
    Solves the static production block at every date of a candidate capital path.

    Step 3 of the transition algorithm: the guessed capital path, together with the
    post-shock offshoring costs, generates the complete candidate path of contemporaneous
    prices and quantities.

    A period whose zero-profit condition has no interior root is a bad capital guess
    rather than a corner to be accommodated, so the underlying ValueError is re-raised
    with the offending date attached.

    Parameters
    ----------
    K_path : array_like
        Candidate aggregate capital for t = 0, ..., T.
    β_path : array_like
        Offshoring cost for t = 0, ..., T. Must be the same length as K_path.
    H : float
        Aggregate high-skill labour supply.
    L : float
        Aggregate low-skill labour supply.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    dict
        Keys I, w, s, r, Y, Ω, each a numpy array indexed by t.

    '''

    if len(K_path) != len(β_path):
        raise ValueError(f'K_path has length {len(K_path)} but β_path has length {len(β_path)}.')

    n_periods: int  = len(K_path)
    keys     : list = ['I', 'w', 's', 'r', 'Y', 'Ω']
    out      : dict = {k: np.empty(n_periods) for k in keys}

    for period in range(n_periods):
        try:
            sol: dict = solve_transition_system_wrapper(K_t      = K_path[period],
                                                        β_t      = β_path[period],
                                                        H        = H,
                                                        L        = L,
                                                        ModelPar = ModelPar)
        except ValueError as exc:
            raise ValueError(f'Production block has no interior solution at t={period} '
                             f'(K_t={K_path[period]}, β_t={β_path[period]}). '
                             f'Underlying: {exc}') from exc

        for k in keys:
            out[k][period] = sol[k]

    return out

@nb.njit(parallel=True, cache=True)
def _bellman_step_core(V_next, inc, a_arr, joint_trans, σ, δ):
    """
    Numba-compiled single application of the Bellman operator.

    Mirrors the policy-improvement block of functions._vfi_core exactly, including the
    -1e25 infeasibility penalty, the 1e-10 consumption floor and the strict '>' tie-break
    that sends ties to the lowest asset index. The flow utility is evaluated inline rather
    than read from a precomputed (n_states, n_assets, n_assets) array, which would cost
    tens of MB per period along the transition.

    The inner loop breaks as soon as consumption turns non-positive: a_arr is increasing,
    so every later a' is infeasible too, and a feasible choice always exists at j=0.

    Parameters
    ----------
    V_next : numpy.ndarray
        (n_states, n_assets) continuation value, V_{t+1}.
    inc : numpy.ndarray
        (n_states, n_assets) income at each state and current asset holding.
    a_arr : numpy.ndarray
        (n_assets,) asset grid, increasing.
    joint_trans : numpy.ndarray
        (n_states, n_states) Markov transition matrix.
    σ : float
        CRRA coefficient.
    δ : float
        Discount factor.

    Returns
    -------
    V : numpy.ndarray
        (n_states, n_assets) value at t.
    pol : numpy.ndarray
        (n_states, n_assets) index into a_arr of the optimal next asset.

    """

    n_states = inc.shape[0]
    n_assets = inc.shape[1]

    V   = np.empty((n_states, n_assets))
    pol = np.zeros((n_states, n_assets), dtype=np.int64)

    for i in nb.prange(n_states):
        exp_V = np.dot(joint_trans[i], V_next)
        for a in range(n_assets):
            inc_ia = inc[i, a]
            best_v = -1e300
            best_j = 0
            for j in range(n_assets):
                c = inc_ia - a_arr[j]
                if c <= 0:
                    break
                if c < 1e-10:
                    c = 1e-10
                v = c ** (1 - σ) / (1 - σ) + δ * exp_V[j]
                if v > best_v:
                    best_v = v
                    best_j = j
            V[i, a]   = best_v
            pol[i, a] = best_j

    return V, pol

def build_income_matrix(w, s, r, a_arr, state_grid):
    '''
    Builds the (n_states, n_assets) income matrix for one period's prices.

    Reproduces the per-state income vector that functions.model_vfi computes internally,
    so that the transition and the steady state face identical budget sets.

    Parameters
    ----------
    w : float
        Low-skill wage.
    s : float
        High-skill wage.
    r : float
        Interest rate.
    a_arr : numpy.ndarray
        Asset grid.
    state_grid : list
        List of (skill_type, log_z) tuples.

    Returns
    -------
    inc : numpy.ndarray
        (n_states, n_assets) income matrix.

    '''

    inc: np.ndarray = np.empty((len(state_grid), len(a_arr)))

    for idx, (f, z) in enumerate(state_grid):
        L        : int = 1 if f == 'L' else 0
        inc[idx]       = functions.income_func(r=r, a=a_arr, z=np.exp(z), w=w, s=s, L=L)

    return inc

def bellman_step(V_next, w, s, r, a_arr, state_grid, joint_trans, ModelPar):
    '''
    One backward step of the household problem at period-t prices.

    Parameters
    ----------
    V_next : numpy.ndarray
        (n_states, n_assets) continuation value, V_{t+1}.
    w : float
        Low-skill wage at t.
    s : float
        High-skill wage at t.
    r : float
        Interest rate at t.
    a_arr : numpy.ndarray
        Asset grid.
    state_grid : list
        List of (skill_type, log_z) tuples.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    tuple
        (V_t, pol_t) as returned by _bellman_step_core.

    '''

    inc: np.ndarray = build_income_matrix(w=w, s=s, r=r, a_arr=a_arr, state_grid=state_grid)

    return _bellman_step_core(V_next, inc, a_arr, joint_trans, ModelPar.σ, ModelPar.δ)

def solve_households_backward(price_path, V_terminal, a_arr, state_grid, joint_trans,
                              ModelPar, store_V=False):
    '''
    Step 4 of the transition algorithm: solves the household problem backward along a
    candidate price path.

    Starting from the terminal continuation value V_{T+1} = V¹, applies the Bellman
    operator once per period for t = T, T-1, ..., 0. Households therefore choose current
    savings while anticipating the entire future candidate price path.

    Parameters
    ----------
    price_path : dict
        Contemporaneous prices as returned by production_path; keys w, s and r must each
        be an array indexed by t = 0, ..., T.
    V_terminal : numpy.ndarray
        (n_states, n_assets) terminal continuation value V_{T+1}, normally the value
        function of the post-shock stationary equilibrium.
    a_arr : numpy.ndarray
        Common asset grid.
    state_grid : list
        List of (skill_type, log_z) tuples.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix.
    ModelPar : TypeModelParameters
        Model parameters.
    store_V : bool, optional
        Keep the whole value path as well as the policies. The default is False, which
        returns only V_0.

    Returns
    -------
    pol_path : numpy.ndarray
        (T+1, n_states, n_assets) savings policies as asset-grid indices, indexed by t.
    V_out : numpy.ndarray
        (T+1, n_states, n_assets) value path if store_V, otherwise the (n_states,
        n_assets) value at t=0.

    '''

    n_periods: int = len(price_path['w'])

    for key in ['w', 's', 'r']:
        if len(price_path[key]) != n_periods:
            raise ValueError(f"price_path['{key}'] has length {len(price_path[key])}, "
                             f"expected {n_periods}.")

    n_states, n_assets = V_terminal.shape

    pol_path: np.ndarray = np.empty((n_periods, n_states, n_assets), dtype=np.int64)
    V_path  : np.ndarray = np.empty((n_periods, n_states, n_assets)) if store_V else None

    V_next  : np.ndarray = V_terminal

    for period in range(n_periods - 1, -1, -1):
        V_t, pol_t = bellman_step(V_next      = V_next,
                                  w           = price_path['w'][period],
                                  s           = price_path['s'][period],
                                  r           = price_path['r'][period],
                                  a_arr       = a_arr,
                                  state_grid  = state_grid,
                                  joint_trans = joint_trans,
                                  ModelPar    = ModelPar)

        pol_path[period] = pol_t
        if store_V:
            V_path[period] = V_t
        V_next = V_t

    return pol_path, (V_path if store_V else V_next)

@nb.njit(cache=True)
def _push_distribution_core(dens, pol, joint_trans):
    """
    Numba-compiled single forward step of the distribution: μ_{t+1} = T(g_t, Q)' μ_t.

    Reproduces the endogenous transition implied by the sparse matrix that
    functions.calculate_stationary_distribution_endog builds, but applies it once instead
    of power-iterating to stationarity, and never materialises the matrix. Savings are
    grid indices, so the asset move is deterministic given the state and no interpolation
    is involved.

    The loop is serial: several (i, a) cells write into the same destination cell, so
    parallelising over the source index would race.

    Parameters
    ----------
    dens : numpy.ndarray
        (n_states, n_assets) current distribution.
    pol : numpy.ndarray
        (n_states, n_assets) index into the asset grid of next-period assets.
    joint_trans : numpy.ndarray
        (n_states, n_states) Markov transition matrix.

    Returns
    -------
    out : numpy.ndarray
        (n_states, n_assets) next-period distribution.

    """

    n_states = dens.shape[0]
    n_assets = dens.shape[1]

    out = np.zeros((n_states, n_assets))

    for i in range(n_states):
        for a in range(n_assets):
            mass = dens[i, a]
            if mass == 0.0:
                continue
            j_next = pol[i, a]
            for jj in range(n_states):
                out[jj, j_next] += mass * joint_trans[i, jj]

    return out

def push_distribution(dens, pol, joint_trans):
    '''
    One period of forward iteration on the distribution of households.

    Parameters
    ----------
    dens : numpy.ndarray
        (n_states, n_assets) current distribution.
    pol : numpy.ndarray
        (n_states, n_assets) savings policy as asset-grid indices.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix over (skill, z) states.

    Returns
    -------
    numpy.ndarray
        (n_states, n_assets) next-period distribution.

    '''

    return _push_distribution_core(dens, pol, joint_trans)

def aggregate_capital(dens, a_arr):
    '''
    Aggregate capital implied by a distribution over (state, asset) cells.

    Parameters
    ----------
    dens : numpy.ndarray
        (n_states, n_assets) distribution.
    a_arr : numpy.ndarray
        (n_assets,) asset grid.

    Returns
    -------
    float
        Σ a·dens, matching the K_supply definition in functions.outer_loop_residual.

    '''

    return float((dens * a_arr[None, :]).sum())

def simulate_distribution_forward(pol_path, dens_0, a_arr, joint_trans, store_dens=False):
    '''
    Step 5 of the transition algorithm: propagates the distribution forward and reads off
    the capital implied by household decisions.

    Starting from μ_0 = μ⁰, applies μ_{t+1} = T(g_t, Q) μ_t for t = 0, ..., T-1. Aggregate
    capital at t is Σ a·μ_t, which equals ∫ g_{t-1} dμ_{t-1} because savings land exactly
    on grid points.

    K_imp[0] is the predetermined initial capital and is never a free variable: the
    residuals of step 6 are only meaningful for t = 1, ..., T.

    Parameters
    ----------
    pol_path : numpy.ndarray
        (T+1, n_states, n_assets) savings policies from solve_households_backward. Only
        entries 0 .. T-1 are used; g_T would determine μ_{T+1}, which is outside the
        horizon.
    dens_0 : numpy.ndarray
        (n_states, n_assets) initial distribution μ⁰.
    a_arr : numpy.ndarray
        Common asset grid.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix.
    store_dens : bool, optional
        Keep the whole distribution path. The default is False, which returns only the
        terminal distribution.

    Returns
    -------
    K_imp : numpy.ndarray
        (T+1,) implied aggregate capital, indexed by t.
    dens_out : numpy.ndarray
        (T+1, n_states, n_assets) distribution path if store_dens, otherwise the
        (n_states, n_assets) terminal distribution μ_T.

    '''

    n_periods: int = pol_path.shape[0]

    if pol_path.shape[1:] != dens_0.shape:
        raise ValueError(f'pol_path implies states {pol_path.shape[1:]} but dens_0 has '
                         f'shape {dens_0.shape}.')

    K_imp    : np.ndarray = np.empty(n_periods)
    dens_path: np.ndarray = np.empty((n_periods,) + dens_0.shape) if store_dens else None

    dens     : np.ndarray = dens_0
    K_imp[0]              = aggregate_capital(dens_0, a_arr)
    if store_dens:
        dens_path[0] = dens_0

    for period in range(n_periods - 1):
        dens                  = push_distribution(dens, pol_path[period], joint_trans)
        K_imp[period + 1]     = aggregate_capital(dens, a_arr)
        if store_dens:
            dens_path[period + 1] = dens

    return K_imp, (dens_path if store_dens else dens)

def initial_capital_guess(K_0, K_terminal, n_periods):
    '''
    Smooth interpolation between the initial and final capital stocks, used to start the
    shooting algorithm.

    Nothing here imposes monotonicity on the equilibrium transition: this is only a
    starting point, and the damped updates are free to move the interior of the path in
    either direction.

    Parameters
    ----------
    K_0 : float
        Predetermined initial capital, ∫a dμ⁰.
    K_terminal : float
        Capital of the post-shock stationary equilibrium.
    n_periods : int
        Length of the path, T+1.

    Returns
    -------
    numpy.ndarray
        (T+1,) capital guess with K[0] = K_0.

    '''

    return np.linspace(K_0, K_terminal, n_periods)

def solve_transition_path(K_0, β_path, H, L, V_terminal, dens_0, a_arr, state_grid,
                          joint_trans, ModelPar, K_terminal=None, K_guess=None, ξ=0.3,
                          tol=2e-4, max_iter=300, max_seconds=None, stall_window=15,
                          stall_tol=0.05, verbose=True, log_path=None, store_paths=False):
    '''
    Steps 2 to 8 of the transition algorithm: shooting on the aggregate capital path.

    Iterates guess K → production block → households backward → distribution forward →
    implied K → damped update, until

        max_{1<=t<=T} |K_imp_t / K_t - 1| < tol.

    K_0 is predetermined by μ⁰ and is never updated, so its residual is identically zero
    and is excluded from the convergence criterion.

    The loop always terminates: it stops on convergence, on max_iter, or on max_seconds,
    and reports which in stop_reason. Per-iteration timings are printed and optionally
    appended to a log file so a long run can be inspected while it is still going.

    Parameters
    ----------
    K_0 : float
        Predetermined initial capital.
    β_path : array_like
        Offshoring cost for t = 0, ..., T.
    H : float
        Aggregate high-skill labour supply.
    L : float
        Aggregate low-skill labour supply.
    V_terminal : numpy.ndarray
        Terminal continuation value V_{T+1}, the post-shock steady-state value function.
    dens_0 : numpy.ndarray
        Initial distribution μ⁰.
    a_arr : numpy.ndarray
        Common asset grid.
    state_grid : list
        List of (skill_type, log_z) tuples.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix.
    ModelPar : TypeModelParameters
        Model parameters.
    K_terminal : float or None, optional
        Capital of the final stationary equilibrium, used only to build the default guess.
        Required unless K_guess is supplied.
    K_guess : array_like or None, optional
        Explicit starting path. Overrides K_terminal. Its first entry is replaced by K_0.
    ξ : float, optional
        Damping weight on the implied path, 0 < ξ < 1. The default is 0.3.
    tol : float, optional
        Convergence tolerance on the maximum relative residual. The default is 2e-4.
        Savings policies are grid indices, so K_imp is a step function of the guessed K
        and the residual cannot be driven below the granularity of the asset grid: for
        vfi_N=500 that floor is around 5e-5. Asking for much less than that produces a
        limit cycle rather than convergence.
    max_iter : int, optional
        Iteration cap. The default is 300.
    max_seconds : float or None, optional
        Wall-clock budget. Checked after each iteration, so the loop can overrun by at
        most one iteration. None means no limit.
    stall_window : int or None, optional
        Number of iterations over which to test for a stall. The loop stops when the best
        residual of the last stall_window iterations is no better than the best of the
        stall_window before it. None disables the check. The default is 15.
    stall_tol : float, optional
        Relative improvement required to count as progress. The default is 0.05.
    verbose : bool, optional
        Print per-iteration progress. The default is True.
    log_path : str or None, optional
        File to append the per-iteration log to.
    store_paths : bool, optional
        Also return the value and distribution paths of the final iteration.

    Returns
    -------
    dict
        Keys: K, I, w, s, r, Y, Ω, K_imp, residual, n_iter, converged, stop_reason,
        resid_history, iter_seconds, and V_path / dens_path when store_paths.

    '''

    if not 0 < ξ < 1:
        raise ValueError(f'ξ must lie strictly between 0 and 1, got {ξ}.')

    n_periods: int = len(β_path)

    if K_guess is None:
        if K_terminal is None:
            raise ValueError('Supply either K_terminal or an explicit K_guess.')
        K_path: np.ndarray = initial_capital_guess(K_0, K_terminal, n_periods)
    else:
        K_path: np.ndarray = np.asarray(K_guess, dtype=float).copy()
        if len(K_path) != n_periods:
            raise ValueError(f'K_guess has length {len(K_path)} but β_path has {n_periods}.')

    K_path[0] = K_0

    def emit(line):
        if verbose:
            print(line, flush=True)
        if log_path is not None:
            with open(log_path, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')

    resid_history: list       = []
    iter_seconds : list       = []
    converged    : bool       = False
    stop_reason  : str        = 'max_iter'
    prices       : dict       = None
    K_imp        : np.ndarray = None
    residual     : float      = np.inf
    extras       : dict       = {}

    emit(f'transition: T={n_periods-1}, ξ={ξ}, tol={tol:.1e}, max_iter={max_iter}, '
         f'max_seconds={max_seconds}')

    t_start: float = time.time()

    for n_iter in range(1, max_iter + 1):

        t_iter: float = time.time()

        try:
            prices = production_path(K_path=K_path, β_path=β_path, H=H, L=L, ModelPar=ModelPar)
        except ValueError as exc:
            raise ValueError(f'Capital guess at iteration {n_iter} left the region where '
                             f'the production block has an interior solution. {exc}') from exc

        pol_path, V_out = solve_households_backward(
            price_path = prices, V_terminal = V_terminal, a_arr = a_arr,
            state_grid = state_grid, joint_trans = joint_trans, ModelPar = ModelPar,
            store_V    = store_paths)

        K_imp, dens_out = simulate_distribution_forward(
            pol_path = pol_path, dens_0 = dens_0, a_arr = a_arr,
            joint_trans = joint_trans, store_dens = store_paths)

        rel_resid: np.ndarray = np.abs(K_imp[1:] / K_path[1:] - 1)
        residual : float      = float(rel_resid.max())

        elapsed_iter : float = time.time() - t_iter
        elapsed_total: float = time.time() - t_start
        resid_history.append(residual)
        iter_seconds.append(elapsed_iter)

        emit(f'  iter {n_iter:>4}  resid {residual:.4e}  at t={int(rel_resid.argmax())+1:<4} '
             f'K[1]={K_path[1]:.5f} K[-1]={K_path[-1]:.5f}  '
             f'{elapsed_iter:.2f}s (total {elapsed_total:.1f}s)')

        if store_paths:
            extras = {'V_path': V_out, 'dens_path': dens_out}

        if residual < tol:
            converged   = True
            stop_reason = 'converged'
            break

        if max_seconds is not None and elapsed_total > max_seconds:
            stop_reason = 'max_seconds'
            break

        if stall_window is not None and len(resid_history) >= 2 * stall_window:
            recent: np.ndarray = np.array(resid_history[-stall_window:])
            earlier: np.ndarray = np.array(resid_history[-2*stall_window:-stall_window])
            if recent.min() >= earlier.min() * (1 - stall_tol):
                stop_reason = 'stalled'
                emit(f'  stalled: best residual over the last {stall_window} iterations '
                     f'({recent.min():.4e}) is no better than the {stall_window} before '
                     f'({earlier.min():.4e}). The discrete asset grid puts a floor on how '
                     f'closely K_imp can track K; lower vfi_N raises that floor.')
                break

        if n_iter < max_iter:
            K_path[1:] = (1 - ξ) * K_path[1:] + ξ * K_imp[1:]

    emit(f'  stopped: {stop_reason} after {n_iter} iterations, residual {residual:.4e}, '
         f'{time.time()-t_start:.1f}s total')

    out: dict = {'K'            : K_path,
                 'K_imp'        : K_imp,
                 'residual'     : residual,
                 'n_iter'       : n_iter,
                 'converged'    : converged,
                 'stop_reason'  : stop_reason,
                 'resid_history': np.array(resid_history),
                 'iter_seconds' : np.array(iter_seconds)}
    out.update(prices)
    out.update(extras)

    return out

def tighten_steady_state(model, a_arr, state_grid, joint_trans, tol_V=1e-12, tol_dens=1e-14,
                         max_iter=300_000):
    '''
    Re-solves a converged model's household side on the common transition grid, iterating
    the transition's own operators to machine precision.

    functions.model_vfi stops at vfi_eps and functions.calculate_stationary_distribution_endog
    at 1e-10, which is ample for a steady state but leaves the terminal condition and the
    initial distribution of a transition less converged than the operators that will act on
    them. Prices are held at the model's equilibrium values; only the household problem and
    the distribution are recomputed.

    Parameters
    ----------
    model : GeneralEquilibriumModel
        A model whose outer loop has already converged.
    a_arr : numpy.ndarray
        Common asset grid.
    state_grid : list
        List of (skill_type, log_z) tuples.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix.
    tol_V : float, optional
        Convergence tolerance on the value function. The default is 1e-12.
    tol_dens : float, optional
        Convergence tolerance on the distribution. The default is 1e-14.
    max_iter : int, optional
        Iteration cap for each of the two loops.

    Returns
    -------
    dict
        Keys V, pol, dens, K, w, s, r.

    '''

    V  : np.ndarray = np.zeros((len(state_grid), len(a_arr)))
    pol: np.ndarray = None

    for _ in range(max_iter):
        V_new, pol = bellman_step(V_next = V, w = model.w, s = model.s, r = model.r,
                                  a_arr = a_arr, state_grid = state_grid,
                                  joint_trans = joint_trans, ModelPar = model.ModelPar)
        gap: float = np.max(np.abs(V_new - V))
        V          = V_new
        if gap < tol_V:
            break

    dens: np.ndarray = np.ones_like(V) / V.size

    for _ in range(max_iter):
        dens_new: np.ndarray = push_distribution(dens, pol, joint_trans)
        gap     : float      = np.max(np.abs(dens_new - dens))
        dens                 = dens_new
        if gap < tol_dens:
            break

    dens = dens * model.ModelPar.M

    return {'V'   : V,
            'pol' : pol,
            'dens': dens,
            'K'   : aggregate_capital(dens, a_arr),
            'w'   : model.w,
            's'   : model.s,
            'r'   : model.r}

def build_common_asset_grid(models, CalibPar, ub_margin=1.0):
    '''
    Builds the single asset grid shared by both stationary equilibria and every period
    of the transition.

    model_vfi sizes its own grid from max(w, s)·vfi_ubmul, so each steady state and each
    transition period would otherwise sit on a different grid and the distribution could
    not be carried across them. The common upper bound is taken from the highest wage
    observed across the supplied models.

    Parameters
    ----------
    models : list
        Solved GeneralEquilibriumModel instances (typically the pre- and post-shock ones).
    CalibPar : TypeCalibParameters
        Calibration parameters.
    ub_margin : float, optional
        Multiplies the upper bound to leave headroom for off-steady-state prices along
        the transition. The default is 1.0.

    Returns
    -------
    a_grid : list
        Common asset grid.

    '''

    max_wage: float = max(max(mod.w, mod.s) for mod in models)
    ub      : float = max_wage * CalibPar.vfi_ubmul * ub_margin
    dist    : float = (ub - CalibPar.vfi_lb) / CalibPar.vfi_N

    a_grid  : list  = [CalibPar.vfi_lb + (i * dist) for i in range(CalibPar.vfi_N + 1)]

    return a_grid

def resolve_on_common_grid(model, a_grid):
    '''
    Re-solves a converged model's household side on a supplied asset grid, holding its
    equilibrium prices fixed.

    Used to place both stationary equilibria on the common transition grid so that μ⁰ and
    V¹ are directly comparable. Prices are not re-optimised: only the household problem is
    recomputed on the new grid.

    Parameters
    ----------
    model : GeneralEquilibriumModel
        A model whose outer loop has already converged.
    a_grid : list
        Common asset grid.

    Returns
    -------
    dict
        Keys: V, pol, a_grid, state_grid, joint_trans, dens, K, mod_res.

    '''

    z_grid, trans_z         = functions.rouwenhorst_trans_matrix(ModelPar=model.ModelPar,
                                                                 CalibPar=model.CalibPar)
    trans_f    : np.ndarray = functions.create_LH_skill_mat(ModelPar=model.ModelPar)
    joint_trans: np.ndarray = np.kron(trans_f, trans_z)
    state_grid : list       = [(f, z) for f in ['L', 'H'] for z in z_grid]

    pol_func, pol_idx, a_grid_out, df_val_func, V_arr, _ = functions.model_vfi(
        w                 = model.w,
        s                 = model.s,
        r                 = model.r,
        income_func       = functions.income_func,
        state_grid        = state_grid,
        joint_trans       = joint_trans,
        ModelPar          = model.ModelPar,
        CalibPar          = model.CalibPar,
        print_convergence = False,
        a_grid            = a_grid)

    stat_dist = functions.calculate_stationary_distribution_endog(
        pol_idx     = pol_idx,
        state_grid  = state_grid,
        a_grid      = a_grid_out,
        joint_trans = joint_trans)

    mod_res = functions.full_model_result(
        pol_func    = pol_func,
        state_grid  = state_grid,
        stat_dist   = stat_dist,
        df_val_func = df_val_func,
        ModelPar    = model.ModelPar)

    pol_mat : np.ndarray = np.array([pol_idx[st] for st in state_grid], dtype=np.int64)
    n_assets: int        = len(a_grid_out)
    dens    : np.ndarray = stat_dist['dens'].to_numpy().reshape(len(state_grid), n_assets)
    dens                 = dens * model.ModelPar.M

    return {'V'          : V_arr,
            'pol'        : pol_mat,
            'a_grid'     : np.array(a_grid_out),
            'state_grid' : state_grid,
            'joint_trans': joint_trans,
            'dens'       : dens,
            'K'          : float((dens * np.array(a_grid_out)[None, :]).sum()),
            'mod_res'    : mod_res}

