# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:51:07 2026

@author: Luis Felipe de Oliveira Valadares Lins
"""

import time
import numpy as np
import pandas as pd
import numba as nb
from scipy.sparse import csr_matrix
from scipy.optimize import brentq, bisect

def calculate_stationary_distribution_eigenvector(trans_matrix):
    '''
    Given a Markov chain transition matrix, it calculates the stationary distribution
    of states in the chain using the eigenvalue method.

    Parameters
    ----------
    trans_matrix : numpy.ndarray
        Array with the transition matrix of the Markov chain.

    Returns
    -------
    v : numpy.ndarray
        Vector with the stationary distribution of states.

    '''

    eigvals, eigvecs = np.linalg.eig(trans_matrix.T)

    idx: int        = np.argmin(np.abs(eigvals - 1))
    v  : np.ndarray = np.real(eigvecs[:, idx])
    v               = v / v.sum()

    v               = np.where(v<0,0,v)
    v               = v / v.sum()

    return v

def calculate_stationary_distribution(trans_matrix, tol=1e-10, max_iter=100_000):
    '''
    Given a Markov chain transition matrix, it calculates the stationary distribution
    of states in the chain using the iterative method.

    Parameters
    ----------
    trans_matrix : numpy.ndarray
        Array with the transition matrix of the Markov chain.
    tol : float, optional
        Convergence tolerance. The default is 1e-10.
    max_iter : int, optional
        Maximum number of iterations. The default is 100_000.

    Returns
    -------
    v : numpy.ndarray
        Vector with the stationary distribution of states.

    '''

    v: np.ndarray = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

    for _ in range(max_iter):
        v_new: np.ndarray = v @ trans_matrix
        if np.max(np.abs(v_new - v)) < tol:
            break
        v = v_new

    v = np.where(v < 0, 0, v)
    v = v / v.sum()

    return v

def calculate_stationary_distribution_endog(pol_func, pol_idx, state_grid, a_grid, joint_trans):
    '''
    Finds the endogenous stationary distribution of states after the VFI.

    Parameters
    ----------
    pol_func    : dict
        Dictionary containing the a' and c policy functions for each state of skill type, productivity level z and current asset holdings.
    pol_idx     : dict
        Dictionary containing the indices of the optimal grid point in the policy function.
    state_grid  : list
        List of tuples containing the possible states for skill type and z.
    a_grid      : list
        List of the possible asset holdings states.
    joint_trans : numpy.ndarray
        Array with the transition matrix between the states described in state_grid.

    Returns
    -------
    stat_dist   : pandas.DataFrame
        Data Frame with the endogenous stationary distribution of states (skill type, productivity, assets)

    '''

    n_s     : int = len(state_grid)
    n_assets: int = len(a_grid)
    n       : int = n_s * n_assets

    # pol_mat[i, a] = index of optimal next asset for state i at current asset a
    pol_mat: np.ndarray = np.array([pol_idx[st] for st in state_grid], dtype=np.int64)  # (n_s, n_assets)

    # Build sparse COO transition matrix — no Python loops.
    SI: np.ndarray = np.arange(n_s,      dtype=np.int64)[:, None, None]  # (n_s, 1,       1)
    AI: np.ndarray = np.arange(n_assets, dtype=np.int64)[None, :, None]  # (1,   n_assets, 1)
    SJ: np.ndarray = np.arange(n_s,      dtype=np.int64)[None, None, :]  # (1,   1,        n_s)

    row_idx: np.ndarray = np.broadcast_to(SI * n_assets + AI,         (n_s, n_assets, n_s))
    col_idx: np.ndarray = SJ * n_assets + pol_mat[:, :, None]         # (n_s, n_assets, n_s)
    vals   : np.ndarray = np.broadcast_to(joint_trans[:, None, :],    (n_s, n_assets, n_s))

    end_T_mat: csr_matrix = csr_matrix(
        (vals.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(n, n)
    )

    # Sparse power iteration
    v: np.ndarray = np.ones(n) / n
    for _ in range(100_000):
        v_new: np.ndarray = v @ end_T_mat
        if np.max(np.abs(v_new - v)) < 1e-10:
            break
        v = v_new
    v = np.maximum(v, 0)
    v /= v.sum()

    # Build stationary distribution dataframe
    stat_dist: pd.DataFrame = pd.DataFrame({
        'skill_type': np.repeat([st[0]         for st in state_grid], n_assets),
        'z'         : np.repeat([np.exp(st[1]) for st in state_grid], n_assets),
        'a_0'       : np.tile(a_grid, n_s),
        'dens'      : v
    })

    return stat_dist

def Omega(I, ModelPar):
    '''
    Defines function Ω(I) = (1-I) + I/(θ+1).

    Parameters
    ----------
    I : float
        Marginal task index.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Value of Ω(I).

    '''

    value: float = (1-I) + (I / (ModelPar.θ + 1))

    return value

def t(i, ModelPar):
    '''
    Defines function t(i) = i^θ, the offshoring cost function for task i.

    Parameters
    ----------
    i : float
        Task index.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Value of t(i).

    '''

    value: float = i ** ModelPar.θ

    return value

def create_LH_skill_mat(ModelPar):
    '''
    Creates the 2x2 transition matrix for low/high skill types.

    Parameters
    ----------
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    mat : numpy.ndarray
        Low-high skill Markov chain transition matrix.

    '''

    mat: np.ndarray = np.array([[ModelPar.π_LL  , 1-ModelPar.π_LL],
                                [1-ModelPar.π_HH, ModelPar.π_HH]])

    return mat

def rouwenhorst_trans_matrix(ModelPar, CalibPar):
    '''
    Function applies Rouwenhorst's method of approximating an AR(1) by a Markov Chain.

    Parameters
    ----------
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.

    Returns
    -------
    y_grid : numpy.ndarray
        Array with the possible discretized values of the AR(1).
    Pi_0 : numpy.ndarray
        Transition matrix of the discretized Markov Chain.

    '''

    # AR(1) settings
    mean   : float = CalibPar.rh_c / (1 - ModelPar.ρ)
    sigma_y: float = ModelPar.σ_ϵ / (np.sqrt(1 - ModelPar.ρ**2))
    y1     : float = mean - CalibPar.rh_r * sigma_y
    yN     : float = mean + CalibPar.rh_r * sigma_y
    d      : float = (2 * CalibPar.rh_r * sigma_y) / (CalibPar.rh_N - 1)
    p      : float = (1 + ModelPar.ρ) / 2
    q      : float = (1 + ModelPar.ρ) / 2
    y_grid : np.ndarray = np.zeros(shape=(CalibPar.rh_N,))
    Pi_0   : np.ndarray = np.zeros(shape=(2,2))

    # Initializing Rouwenhorst's Method
    y_grid[0]               = y1
    y_grid[CalibPar.rh_N-1] = yN

    for i in range(1, CalibPar.rh_N-1):
        y_grid[i] = y_grid[i-1] + d

    Pi_0[(0,)] = np.array([p, 1-p])
    Pi_0[(1,)] = np.array([1-q, q])

    # Implementing Rouwenhorst's Method
    for s in range(2, CalibPar.rh_N):
        Pi_1_a: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_b: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_c: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_d: np.ndarray = np.zeros(shape=(s+1,s+1))

        Pi_1_a[:s,:s]  = p     * Pi_0
        Pi_1_b[:s,-s:] = (1-p) * Pi_0
        Pi_1_c[-s:,:s] = (1-q) * Pi_0
        Pi_1_d[-s:,-s:]= q     * Pi_0

        Pi_1: np.ndarray = Pi_1_a + Pi_1_b + Pi_1_c + Pi_1_d

        Pi_1[1:s, :] /= 2

        Pi_0 = Pi_1.copy()

    if CalibPar.rh_N == 1:
        y_grid = np.array([0])
        Pi_0   = np.array([1])

    return y_grid, Pi_0

def solve_representative_household(I_0, ModelPar):
    '''
    For a given I_0, solves the one-good representative household model.

    Given I_0:
      (1) r from Euler equation
      (2) w from MT condition: w = w*·β·I^θ
      (3) s from ZCP (one good, closed form)
      (4) Y from H market clearing: α·Y/s = H = b[1]
      (5) K_d from K market clearing: γ·Y/r
      (6) K_s from SS budget constraint: Y = r·K_s + b[0]·w + b[1]·s

    Part of: solve_representative_household -> representative_household_residual
             -> representative_household_wrapper

    Parameters
    ----------
    I_0 : float
        Guess for the marginal task.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    dict
        Solution dictionary with keys: w, s, r, Y, I, K_d, K_s.

    '''

    r      : float = (1 - ModelPar.δ) / ModelPar.δ
    w_0    : float = ModelPar.w_star * ModelPar.β * t(I_0, ModelPar)
    omega0 : float = Omega(I_0, ModelPar)
    κ      : float = 1 - ModelPar.α - ModelPar.γ

    log_s_0: float = (np.log(ModelPar.α)
                      - (κ / ModelPar.α) * np.log(w_0 * omega0 / κ)
                      - (ModelPar.γ / ModelPar.α) * np.log(r / ModelPar.γ))
    s_0    : float = np.exp(log_s_0)

    b  : np.ndarray = calculate_stationary_distribution_eigenvector(create_LH_skill_mat(ModelPar))
    Y_0: float      = s_0 * b[1] / ModelPar.α
    K_d: float      = ModelPar.γ * Y_0 / r
    K_s: float      = (Y_0 - b[0] * w_0 - b[1] * s_0) / r

    return {'w'  : w_0,
            's'  : s_0,
            'r'  : r,
            'Y'  : Y_0,
            'I'  : I_0,
            'K_d': K_d,
            'K_s': K_s}

def representative_household_residual(I_0, ModelPar):
    '''
    Returns K_d - K_s for a given guess of I. Used in bisection to find the
    equilibrium I.

    Parameters
    ----------
    I_0 : float
        Guess for the marginal task.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    float
        Difference between capital demand and supply.

    '''

    sol = solve_representative_household(I_0=I_0, ModelPar=ModelPar)

    return sol['K_d'] - sol['K_s']

def representative_household_wrapper(ModelPar):
    '''
    Solves the analytical representative household equilibrium.

    Finds I by bisecting representative_household_residual, then computes
    individual consumption, lifelong utilities, and income Gini.

    Parameters
    ----------
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    equilibrium : dict
        Full equilibrium solution with utilities, Gini, and factor prices.

    '''

    root       : float      = bisect(f=representative_household_residual, a=1e-9, b=1-1e-9, args=(ModelPar,))
    equilibrium: dict       = solve_representative_household(I_0=root, ModelPar=ModelPar)
    b          : np.ndarray = calculate_stationary_distribution_eigenvector(create_LH_skill_mat(ModelPar))

    a_L: float = equilibrium['K_s'] * equilibrium['w'] / (b[0]*equilibrium['w'] + b[1]*equilibrium['s'])
    a_H: float = equilibrium['K_s'] * equilibrium['s'] / (b[0]*equilibrium['w'] + b[1]*equilibrium['s'])

    C_L: float = equilibrium['r'] * a_L + equilibrium['w']
    C_H: float = equilibrium['r'] * a_H + equilibrium['s']

    U_L: float = (C_L ** (1 - ModelPar.σ)) / ((1 - ModelPar.σ) * (1 - ModelPar.δ))
    U_H: float = (C_H ** (1 - ModelPar.σ)) / ((1 - ModelPar.σ) * (1 - ModelPar.δ))

    equilibrium['C_L'] = C_L
    equilibrium['C_H'] = C_H
    equilibrium['U_L'] = U_L
    equilibrium['U_H'] = U_H

    # Income Gini
    type_flag    : int   = np.argmin([C_L, C_H])
    b_poor       : float = b[type_flag]
    L_inc_share  : float = b_poor * min(C_L, C_H) / (b[0]*C_L + b[1]*C_H)
    full_triangle: float = 0.5

    small_triangle: float = (b_poor * L_inc_share) / 2
    upper_triangle: float = ((1 - b_poor) * (1 - L_inc_share)) / 2
    rectangle     : float = (1 - b_poor) * L_inc_share
    B             : float = small_triangle + upper_triangle + rectangle

    equilibrium['income_gini'] = 1 - (B / full_triangle)

    return equilibrium

def Theta(I, ModelPar):

    return ModelPar.θ * (1 - I)

def hat_algebra_sysmem(I, ModelPar):
    '''
    Hat-algebra expression for w_hat/beta_hat as a function of I (one-good model).
    From PDF Section 6: w_hat = beta_hat * I * (α - κΘ) / (α(Θ²+Θ+I) + κΘ(Θ+1-I))
    '''

    Θ: float = Theta(I, ModelPar)
    κ: float = 1 - ModelPar.α - ModelPar.γ

    numerator  : float = ModelPar.α - κ * Θ
    denominator: float = ModelPar.α * (Θ**2 + Θ + I) + κ * Θ * (Θ + 1 - I)

    return I * (numerator / denominator)

def solve_wages_given_I(I_0, r, ModelPar):
    '''
    Given marginal task I_0 and interest rate r, returns (w, s) using the
    MT condition and the one-good ZCP.

    Used by solve_representative_household and for verification.

    Parameters
    ----------
    I_0 : float
        Marginal task index.
    r : float
        Interest rate.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    list
        [w, s] — low-skill and high-skill wages.

    '''

    w_0    : float = ModelPar.w_star * ModelPar.β * t(I_0, ModelPar)
    omega0 : float = Omega(I_0, ModelPar)
    κ      : float = 1 - ModelPar.α - ModelPar.γ
    log_s_0: float = (np.log(ModelPar.α)
                      - (κ / ModelPar.α) * np.log(w_0 * omega0 / κ)
                      - (ModelPar.γ / ModelPar.α) * np.log(r / ModelPar.γ))
    s_0    : float = np.exp(log_s_0)

    return [w_0, s_0]

def solve_firm_side_one_good(w, r, ModelPar):
    '''
    Given domestic low-skill wage w and interest rate r, computes the firm-side
    equilibrium analytically for the one-good model.

    Steps:
      (1) I = (w / (w*·β))^(1/θ)       from MT condition
      (2) Ω = (1-I) + I/(θ+1)           offshoring cost index
      (3) s from one-good ZCP (closed form)

    Parameters
    ----------
    w : float
        Low-skill wage (inner loop guess).
    r : float
        Interest rate (outer loop guess).
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    tuple
        (I, Ω, s)

    '''

    I    : float = (w / (ModelPar.w_star * ModelPar.β)) ** (1.0 / ModelPar.θ)
    Ω    : float = Omega(I, ModelPar)
    κ    : float = 1 - ModelPar.α - ModelPar.γ
    log_s: float = (np.log(ModelPar.α)
                    - (κ / ModelPar.α) * np.log(w * Ω / κ)
                    - (ModelPar.γ / ModelPar.α) * np.log(r / ModelPar.γ))
    s    : float = np.exp(log_s)

    return I, Ω, s

def utility_func(inc, a_1, ModelPar):
    '''
    CRRA utility for a household with income inc choosing next-period assets a_1.
    One-good model: U(c) = c^(1-σ)/(1-σ).

    Parameters
    ----------
    inc : float
        Household income.
    a_1 : float
        Next period asset choice.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    float
        Utility value (-1e25 if c ≤ 0).

    '''

    c: float = inc - a_1

    if c <= 0:
        return -1e25

    return c ** (1 - ModelPar.σ) / (1 - ModelPar.σ)

def income_func(r, a, z, w, s, L):
    '''
    Calculates household income: (1+r)·a + z·(L·w + (1-L)·s).

    Parameters
    ----------
    r : float
        Interest rate.
    a : float
        Current asset holdings.
    z : float
        Productivity level.
    w : float
        Low-skill wage.
    s : float
        High-skill wage.
    L : float
        1 if low-skill, 0 if high-skill.

    Returns
    -------
    float
        Household income.

    '''

    value: float = (1+r) * a + z * (L*w + (1-L)*s)

    return value

@nb.njit(parallel=True, cache=True)
def _vfi_core(V, U, joint_trans, delta, eps, howard_steps):
    '''
    Numba-compiled VFI core: Policy Improvement + Howard's acceleration.

    V           : (n_states, n_assets)           initial value function
    U           : (n_states, n_assets, n_assets)  flow utility array
    joint_trans : (n_states, n_states)            Markov transition matrix
    '''
    n_states = V.shape[0]
    n_assets = V.shape[1]
    pol = np.zeros((n_states, n_assets), dtype=np.int64)

    cond     = True
    it_count = 0

    while cond:
        it_count += 1
        V_old = V.copy()

        # Policy Improvement — parallel over states
        for i in nb.prange(n_states):
            exp_V = np.dot(joint_trans[i], V_old)
            for a in range(n_assets):
                best_v = -1e300
                best_j = 0
                for j in range(n_assets):
                    v = U[i, a, j] + delta * exp_V[j]
                    if v > best_v:
                        best_v = v
                        best_j = j
                V[i, a]   = best_v
                pol[i, a] = best_j

        # Convergence check
        max_diff = 0.0
        for i in range(n_states):
            for a in range(n_assets):
                d = abs(V[i, a] - V_old[i, a])
                if d > max_diff:
                    max_diff = d
        cond = max_diff > eps

        # Howard's Policy Evaluation
        if cond:
            for _ in range(howard_steps):
                V_old = V.copy()
                for i in nb.prange(n_states):
                    exp_V = np.dot(joint_trans[i], V_old)
                    for a in range(n_assets):
                        j       = pol[i, a]
                        V[i, a] = U[i, a, j] + delta * exp_V[j]

    return V, pol, it_count


def model_vfi(w, s, r, income_func, state_grid, joint_trans, ModelPar, CalibPar,
              print_convergence=True, V_init=None):
    '''
    Value Function Iteration for the one-good household problem.

    Utility: U(c) = c^(1-σ)/(1-σ). No price index needed.

    Parameters
    ----------
    w : float
        Low-skill wage.
    s : float
        High-skill wage.
    r : float
        Interest rate.
    income_func : callable
        Income function.
    state_grid : list
        List of (skill_type, log_z) tuples.
    joint_trans : numpy.ndarray
        Joint Markov transition matrix over (skill, z) states.
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.
    print_convergence : bool, optional
        Print timing and iteration count. Default True.
    V_init : numpy.ndarray or None, optional
        Warm-start initial value function.

    Returns
    -------
    pol_func : dict
    pol_idx  : dict
    a_grid   : list
    df_val_func : pandas.DataFrame
    V_arr    : numpy.ndarray
    it_count : int

    '''

    start   : float = time.time()
    ub      : float = max(w, s) * CalibPar.vfi_ubmul
    dist    : float = (ub - CalibPar.vfi_lb) / CalibPar.vfi_N
    a_grid  : list  = [CalibPar.vfi_lb + (i * dist) for i in range(CalibPar.vfi_N + 1)]
    a_arr   : np.ndarray = np.array(a_grid)
    n_assets: int        = len(a_grid)
    n_states: int        = len(state_grid)

    # U[i, a, j] = u(income_a - a'_j) for state i, current asset index a, next asset index j
    U_arr: np.ndarray = np.empty((n_states, n_assets, n_assets))
    for idx, (f, z) in enumerate(state_grid):
        L      : int        = 1 if f == 'L' else 0
        inc_vec: np.ndarray = income_func(r=r, a=a_arr, z=np.exp(z), w=w, s=s, L=L)
        c_grid : np.ndarray = inc_vec[None, :] - a_arr[:, None]
        c_safe : np.ndarray = np.maximum(c_grid, 1e-10)
        u_grid : np.ndarray = np.where(c_grid > 0,
                                        c_safe ** (1 - ModelPar.σ) / (1 - ModelPar.σ),
                                        -1e25)
        U_arr[idx] = u_grid.T  # (a_current, a_prime)

    V_arr, pol_arr, it_count = _vfi_core(
        V_init.copy() if V_init is not None else np.zeros((n_states, n_assets)),
        U_arr,
        joint_trans.astype(np.float64),
        float(ModelPar.δ),
        float(CalibPar.vfi_eps),
        int(CalibPar.vfi_howard_steps)
    )

    df_val_func: pd.DataFrame = pd.concat([
        pd.DataFrame({'a_0': a_grid, 'skill_type': state[0], 'z': np.exp(state[1]), 'V': V_arr[i]})
        for i, state in enumerate(state_grid)
    ]).reset_index(drop=True)

    pol_idx: dict = {state: pol_arr[i].copy() for i, state in enumerate(state_grid)}

    pol_func: dict = {}
    for i, state in enumerate(state_grid):
        f, z   = state
        L: int = 1 if f == 'L' else 0
        idxs   = pol_arr[i]
        df              = pd.DataFrame({'a_0': a_grid, 'a_1': a_arr[idxs]})
        df['c']         = income_func(r=r, a=df['a_0'].values, z=np.exp(z), w=w, s=s, L=L) - df['a_1'].values
        pol_func[state] = df

    end: float = time.time() - start
    if print_convergence:
        print(f'Convergence after {end:.2f}s and {it_count} iterations')

    return pol_func, pol_idx, a_grid, df_val_func, V_arr, it_count

def full_model_result(pol_func, state_grid, stat_dist, df_val_func, ModelPar):
    '''
    Assembles the full model result dataframe (one-good version).

    For each state (skill_type, z, a_0) provides:
      a_1, c (consumption of Y), dens (stationary distribution), V (value function).

    Parameters
    ----------
    pol_func : dict
    state_grid : list
    stat_dist : pandas.DataFrame
    df_val_func : pandas.DataFrame
    ModelPar : TypeModelParameters

    Returns
    -------
    pandas.DataFrame

    '''

    df_pol_func: pd.DataFrame = pd.concat([
        pol_func[state].assign(skill_type=state[0], z=np.exp(state[1]))
        for state in state_grid
    ]).reset_index(drop=True)

    df_pol_func = df_pol_func.set_index(['a_0', 'z', 'skill_type'])
    stat_dist   = stat_dist.set_index(['a_0', 'z', 'skill_type']) * ModelPar.M
    df_val_func = df_val_func.set_index(['a_0', 'z', 'skill_type'])

    result: pd.DataFrame = pd.concat([df_pol_func, stat_dist, df_val_func], axis=1).reset_index()

    return result

def inner_loop_residual(w, r, H, L, ModelPar):
    '''
    L-market clearing residual for the inner loop (one-good model).

    Given w (inner loop guess) and r (outer loop), computes the equilibrium
    firm-side analytically and returns the excess demand for L-tasks.

    Residual = κ·Y/(w·Ω) - L/(1-I)
    where Y = s·H/α (from H market clearing, H precomputed from exogenous processes).

    At equilibrium (RC): residual = 0.

    Parameters
    ----------
    w : float
        Low-skill wage (inner loop variable).
    r : float
        Interest rate (outer loop variable).
    H : float
        Aggregate high-skill labour supply (from exogenous processes).
    L : float
        Aggregate low-skill labour supply (from exogenous processes).
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    float
        L-market clearing residual.

    '''

    I, Ω, s = solve_firm_side_one_good(w, r, ModelPar)
    Y       : float = s * H / ModelPar.α
    κ       : float = 1 - ModelPar.α - ModelPar.γ

    return κ * Y / (w * Ω) - L / (1 - I)

def outer_loop_residual(r, Y, mod_res, ModelPar, CalibPar):
    '''
    Capital market clearing residual for the outer loop (one-good model).

    K_supply = Σ a·dens   (from VFI stationary distribution)
    K_demand = γ·Y / r    (from firm-side K market clearing condition RE)

    Parameters
    ----------
    r : float
        Interest rate.
    Y : float
        Output of good Y (from inner loop solution).
    mod_res : pandas.DataFrame
        Full model result dataframe with 'a_0' and 'dens' columns.
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.

    Returns
    -------
    float
        K_supply - K_demand.

    '''

    K_supply: float = (mod_res['dens'] * mod_res['a_0']).sum()
    K_demand: float = ModelPar.γ * Y / r

    check: bool = abs(K_supply - K_demand) < CalibPar.kmc_eps

    if check:
        print('Capital markets clear!')
        return 0.0

    return K_supply - K_demand

def weighted_gini(x, weights):
    '''
    Calculates the Gini index of a vector with different weights.

    Parameters
    ----------
    x : pd.Series
        Values of the variable.
    weights : pd.Series
        Weights for each value.

    Returns
    -------
    gini : float
        Weighted Gini Index.

    '''

    sorted_idx: pd.Series = np.argsort(x)
    x_sorted  : pd.Series = x[sorted_idx]
    w_sorted  : pd.Series = weights[sorted_idx]

    cum_pop   : pd.Series = np.cumsum(w_sorted) / np.sum(w_sorted)
    cum_income: pd.Series = np.cumsum(w_sorted * x_sorted) / np.sum(w_sorted * x_sorted)

    cum_pop   : np.ndarray = np.concatenate([[0], cum_pop])
    cum_income: np.ndarray = np.concatenate([[0], cum_income])

    area: float = np.sum((cum_pop[1:] - cum_pop[:-1]) * (cum_income[1:] + cum_income[:-1]) / 2)
    gini: float = 1 - 2 * area

    return gini

def weighted_percentile(x, weights, q):
    '''
    Calculates the share of x and the value of x at percentile q.

    Parameters
    ----------
    x : pd.Series
        Values.
    weights : pd.Series
        Weights.
    q : int/float
        Percentile.

    Returns
    -------
    share : float
        Share of x held above percentile q.
    perc_income : float
        Value of x at percentile q.

    '''

    sorted_idx: pd.Series = np.argsort(x)
    x_sorted  : pd.Series = x[sorted_idx]
    w_sorted  : pd.Series = weights[sorted_idx]

    cum_w: pd.Series = np.cumsum(w_sorted)
    cum_w: pd.Series = cum_w / cum_w.iloc[-1]
    mask : pd.Series = cum_w >= q / 100

    share      : float = np.sum(x_sorted[mask] * w_sorted[mask]) / np.sum(x_sorted * w_sorted)
    idx        : int   = np.searchsorted(cum_w, q / 100)
    perc_income: float = x_sorted[min(idx, len(x_sorted) - 1)]

    return share, perc_income
