# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:51:07 2026

@author: Luis Felipe de Oliveira Valadares Lins
"""

import time
import warnings
import numpy as np
import pandas as pd
import numba as nb
from scipy.sparse import csr_matrix
from scipy.optimize import brentq,bisect

def calculate_stationary_distribution_eigenvector(trans_matrix):
    '''
    Given a Markov chain transition matrix, it calculates the stationary distribution
    of states in the chain using the eigenvalue method.
    
    Currently using the iterative method instead of this.

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
    # For each triple (current_state i, current_asset a, next_state j):
    #   row = i*n_assets + a
    #   col = j*n_assets + pol_mat[i, a]   (asset tomorrow is deterministic given policy)
    #   val = joint_trans[i, j]
    # All broadcast to shape (n_s, n_assets, n_s).
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

def Omega(I,ModelPar):
    '''
    Defines function Ω(I).

    Parameters
    ----------
    I : float
        Index of task I whose cost is going to be calculated in the function.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Value of Ω(I).

    '''
    
    value: float = (1-I) + (I / (ModelPar.θ + 1))
    
    return value

def t(i,ModelPar):
    '''
    Defines function t(i), the offshoring cost function for task i

    Parameters
    ----------
    i : float
        Index of task i whose cost is going to be calculated in the function.
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
    Creates the 2x2 transition matrix from low to high skill.

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

def rouwenhorst_trans_matrix(ModelPar,CalibPar):
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
        Array with the possible discretized values of the AR(1) in the approximation Markov Chain.
    Pi_0 : numpy.ndarray
        Transition matrix of the discretized values of the AR(1) in the approximation Markov Chain.

    '''
    
    # AR(1) settings
    mean   : float = CalibPar.rh_c / (1 - ModelPar.ρ)
    sigma_y: float = ModelPar.σ_ϵ / (np.sqrt(1 - ModelPar.ρ**2))
    y1     : float = mean - CalibPar.rh_r * sigma_y
    yN     : float = mean + CalibPar.rh_r * sigma_y
    d      : float = (2 * CalibPar.rh_r * sigma_y) / (CalibPar.rh_N - 1)
    p      : float  = (1 + ModelPar.ρ) / 2
    q      : float = (1 + ModelPar.ρ) / 2
    y_grid : np.ndarray = np.zeros(shape=(CalibPar.rh_N,))
    Pi_0   : np.ndarray= np.zeros(shape=(2,2))
    
    # Initializing Rouwenhorst's Method
    y_grid[0]               = y1
    y_grid[CalibPar.rh_N-1] = yN
    
    for i in range(1,CalibPar.rh_N-1):
        y_grid[i] = y_grid[i-1] + d
    
    Pi_0[(0,)] = np.array([p,1-p])
    Pi_0[(1,)] = np.array([1-q,q])
    
    # Implementing Rouwenhorst's Method
    for s in range(2,CalibPar.rh_N):
        Pi_1_a: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_b: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_c: np.ndarray = np.zeros(shape=(s+1,s+1))
        Pi_1_d: np.ndarray = np.zeros(shape=(s+1,s+1))
        
        Pi_1_a[:s,:s]    = p     * Pi_0
        Pi_1_b[:s,-s:]   = (1-p) * Pi_0
        Pi_1_c[-s:,:s]   = (1-q) * Pi_0
        Pi_1_d[-s:,-s:]  = q     * Pi_0
        
        Pi_1: np.ndarray = Pi_1_a + Pi_1_b + Pi_1_c + Pi_1_d
        
        Pi_1[1:s, :] /= 2
        
        Pi_0 = Pi_1.copy()
    
    if CalibPar.rh_N==1:
        y_grid = np.array([0])
        Pi_0   = np.array([1])
    
    return y_grid,Pi_0

def solve_representative_household(I_0,ModelPar):
    '''
    For a given I_0, it solves the representative household problem.
    
    Part of the solver of the representative household problem. Functions run in
    the following order:
        
    solve_representative_household -> representative_household_residual -> representative_household_wrapper

    Parameters
    ----------
    I_0 : float
        Guess for I.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    dict
        Dictionary containing the solution to the representative household problem.

    '''
    
    # Find r from the Euler Equation
    r = (1-ModelPar.δ)/(ModelPar.δ)
    
    # Find w from the marginal task (MT)
    w_0: float = ModelPar.w_star * ModelPar.β * t(I_0, ModelPar)
    
    # Find s from good X's ZPC
    omega0 : float = Omega(I_0, ModelPar)
    theta_x: float = 1 - ModelPar.α_x - ModelPar.γ
    theta_y: float = 1 - ModelPar.α_y - ModelPar.γ
    
    log_s_0: float = (np.log(1)
                      - (theta_x) * np.log((w_0 * omega0) / (theta_x))
                      - ModelPar.γ * np.log(r / ModelPar.γ)) / ModelPar.α_x + np.log(ModelPar.α_x)
    s_0    : float = np.exp(log_s_0)
    
    # Find p from good Y's ZPC
    t1 = ((w_0*omega0)/(theta_y)     )**(theta_y)
    t2 = ((s_0)       /(ModelPar.α_y))**(ModelPar.α_y)
    t3 = ((r)         /(ModelPar.γ)  )**(ModelPar.γ)
    
    p_0 : float = t1 * t2 * t3
    
    # Find X and Y from L and H LMC - system Ax=b
    A      : np.ndarray = np.array([[theta_x / (w_0 * omega0), (p_0 * theta_y) / (w_0 * omega0)],
                                    [ModelPar.α_x / s_0,        (p_0 * ModelPar.α_y) / s_0]])
    A_minus: np.ndarray = np.linalg.inv(A)
    b      : np.ndarray = calculate_stationary_distribution_eigenvector(create_LH_skill_mat(ModelPar))
    b_rhs  : np.ndarray = np.array([b[0] / (1 - I_0), b[1]])
    
    X_0, Y_0 = tuple(np.dot(A_minus, b_rhs))
    
    # Use K market clearing to find K^d
    K_d: float = (ModelPar.γ*X_0+p_0*ModelPar.γ*Y_0)/r
    
    # Use representative HH budget constraint in SS to find K^s
    K_s: float = (X_0/ModelPar.η - w_0*b[0] - s_0*b[1]) / r
    
    return {'w'  :w_0,
            's'  :s_0,
            'p'  :p_0,
            'r'  :r  , 
            'X'  :X_0,
            'Y'  :Y_0,
            'I'  :I_0,
            'K_d':K_d,
            'K_s':K_s}

def representative_household_residual(I_0,ModelPar):
    '''
    Given a guess for I, it calculates the difference between capital supply in both goods demand.
    
    Used to adjust the guess for I based on this residual. Part of the solution
    of the full representative household problem:

    solve_representative_household -> representative_household_residual -> representative_household_wrapper

    Parameters
    ----------
    I_0 : float
        Guess for I.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    float
        Difference between capital supply and demand.

    '''
    
    sol = solve_representative_household(I_0=I_0, ModelPar=ModelPar)
    
    return sol['K_d'] - sol['K_s']

def representative_household_wrapper(ModelPar):
    '''
    Wrapper to solve the analytical problem of the representative household. 
    Runs in the following order:

    solve_representative_household -> representative_household_residual -> representative_household_wrapper        

    Parameters
    ----------
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    equilibrium : dict
        Dictionary containing the solution to the representative household problem.

    '''
    
    root       : float      = bisect(f=representative_household_residual, a=1e-9, b=1-1e-9,args=(ModelPar))
    equilibrium: dict       = solve_representative_household(I_0=root,ModelPar=ModelPar)
    b          : np.ndarray = calculate_stationary_distribution_eigenvector(create_LH_skill_mat(ModelPar))
    a_L        : float      = equilibrium['K_s'] * equilibrium['w'] / (b[0]*equilibrium['w'] + b[1]*equilibrium['s'])
    a_H        : float      = equilibrium['K_s'] * equilibrium['s'] / (b[0]*equilibrium['w'] + b[1]*equilibrium['s'])
    
    # Calculating lifelong utilities
    psi_p: float = psi(p=equilibrium['p'],ModelPar=ModelPar)
    C_L  : float = equilibrium['r']*a_L+equilibrium['w']
    C_H  : float = equilibrium['r']*a_H+equilibrium['s']
    D_L  : float = psi_p * C_L
    D_H  : float = psi_p * C_H
    
    U_L  : float = (D_L**(1-ModelPar.σ))/((1-ModelPar.σ)*(1-ModelPar.δ))
    U_H  : float = (D_H**(1-ModelPar.σ))/((1-ModelPar.σ)*(1-ModelPar.δ))
    
    equilibrium['C_L'] = C_L
    equilibrium['C_H'] = C_H
    equilibrium['U_L'] = U_L
    equilibrium['U_H'] = U_H
    
    # Calculating income Gini
    type_flag    : int        = np.argmin([C_L, C_H])
    b            : np.ndarray = calculate_stationary_distribution_eigenvector(create_LH_skill_mat(ModelPar))
    b_poor       : float      = b[type_flag]
    L_inc_share  : float      = b_poor * min(C_L, C_H) / (b[0]*C_L + b[1]*C_H)
    full_triangle: float      = 0.5
    
    small_triangle: float = (b_poor * L_inc_share) / 2
    upper_triangle: float = ((1 - b_poor) * (1 - L_inc_share)) / 2
    rectangle     : float = (1 - b_poor) * L_inc_share
    B             : float = small_triangle + upper_triangle + rectangle
    
    gini: float = 1 - (B / full_triangle)
    
    equilibrium['income_gini'] = gini
    
    return equilibrium

def Theta(I,ModelPar):
    
    return ModelPar.θ*(1-I)

def hat_algebra_sysmem(I,ModelPar):
    
    Θ = Theta(I,ModelPar)
    
    numerator = ModelPar.α_x - (1-ModelPar.α_x-ModelPar.γ)*Θ
    denominator = ModelPar.α_x*(Θ**2+Θ+I)+(1-ModelPar.α_x-ModelPar.γ)*Θ*(Θ+1-I)
    
    prod = I * (numerator/denominator)
    
    return prod

def solve_wages_given_I(I_0,r,ModelPar):
    '''
    Given an interest rate r and a guess for marginal task I_0, it solves for
    the wage rates for low-skilled (w) and high-skilled (s) labour.
    
    This function is used in the wrapper for solving the firm-side equilibrium.
    In zpc_y_residual, it's used to calculate, given the guess for I_0 and a price
    of good Y (p), the residual in the last equation defining the firm-side 
    equilibrium. The order of the functions used in the wrapper are:
    
    solve_wages_given_I -> zpc_y_residual -> solve_firm_side

    Parameters
    ----------
    I_0 : float
        Guess for the marginal task.
    r : float
        Interest rate.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    list
        List containing the resulting low-skilled (w) and high-skilled (s) wage rates.

    '''
    
    # Find w from the marginal task (MT)
    w_0: float = ModelPar.w_star * ModelPar.β * t(I_0, ModelPar)
    
    # Find s from good X's ZPC
    omega0 : float = Omega(I_0, ModelPar)
    log_s_0: float = (np.log(1)
                      - (1 - ModelPar.α_x - ModelPar.γ) * np.log((w_0 * omega0) / (1 - ModelPar.α_x - ModelPar.γ))
                      - ModelPar.γ * np.log(r / ModelPar.γ)) / ModelPar.α_x + np.log(ModelPar.α_x)
    s_0    : float = np.exp(log_s_0)
    
    return [w_0,s_0]

def zpc_y_residual(I, p, r, ModelPar):
    '''
    Given a guess for I, we know the marginal task and the ZPC condition for good X
    hold. In this function, for a given price of good Y (p), we check the residual
    in the ZPC condition for good Y, to check how accurate I was in the first place.
    
    This function is used in the wrapper for solving the firm-side equilibrium.
    In the function solve_firm_side, we find the root of this function, to check 
    for a given r and p, which I solves the firm-side equilibrium:
        
    solve_wages_given_I -> zpc_y_residual -> solve_firm_side

    Parameters
    ----------
    I : float
        Guess for the marginal task.
    p : float
        Inner loop's good Y's price.
    r : float
        Outer loop's interest rate.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Residual from good Y's ZPC.

    '''
    
    w, s   = solve_wages_given_I(I_0=I, r=r, ModelPar=ModelPar)
    
    eq3_t1: float = ((w * Omega(I,ModelPar)) / (1 - ModelPar.α_y - ModelPar.γ)) ** (1 - ModelPar.α_y - ModelPar.γ)
    eq3_t2: float = (s / ModelPar.α_y) ** (ModelPar.α_y)
    eqs_t3: float = (r / ModelPar.γ)   ** (ModelPar.γ)
    
    value : float = p - eq3_t1 * eq3_t2 * eqs_t3
    
    return value

def check_mt_zpc_system(w,s,I,p,r,ModelPar):
    '''
    Function used in the solve_firm_side to check that the solution given by finding
    the root of solve_wages_given_I indeed zeros the system of equations.

    Parameters
    ----------
    w : float
        Low-skill wage rate. Endogenously determined in solve_wages_given_I.
    s : float
        High-skill wage rate. Endogenously determined in solve_wages_given_I.
    I : float
        Marginal task. Endogenously determined in solve_wages_given_I.
    p : float
        Good Y's price. Determined in the inner loop.
    r : float
        Interest rate. Determined in the outer loop
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    list
        Residual from the marginal task and the ZP conditions.

    '''
    
    # Eq 1 = marginal task; eq 2 = ZPC for good X; eq 3 = ZPC for good Y
    eq1  : float = w - (ModelPar.w_star * ModelPar.β * t(i=I, ModelPar=ModelPar))
    lhs_x: float = ((w * Omega(I,ModelPar)) / (1 - ModelPar.α_x - ModelPar.γ)) ** (1 - ModelPar.α_x - ModelPar.γ) \
                     * (s / ModelPar.α_x) ** ModelPar.α_x \
                     * (r / ModelPar.γ)   ** ModelPar.γ
    eq2  : float = 1 - lhs_x
    lhs_y: float = ((w * Omega(I,ModelPar)) / (1 - ModelPar.α_y - ModelPar.γ)) ** (1 - ModelPar.α_y - ModelPar.γ) \
                     * (s / ModelPar.α_y) ** ModelPar.α_y \
                     * (r / ModelPar.γ)   ** ModelPar.γ
    eq3  : float = p - lhs_y
    
    return [eq1, eq2, eq3]

def solve_firm_side(p, r, ModelPar, CalibPar):
    '''
    Function used to find, for the inner and outer loop's values of p and r, the
    the wage rates w and s, and the marginal task I which solves the marginal task
    and the ZPC condition for both goods. This function is a wrapper for solving 
    the firm-side equilibrium:
        
    solve_wages_given_I -> zpc_y_residual -> solve_firm_side
    
    Parameters
    ----------
    p : float
        Good Y's price. Determined in the inner loop.
    r : float
        Interest rate. Determined in the outer loop
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.

    Raises
    ------
    Warning
        If there is no interior solution for the given values of p and r,
        i.e., the residual function does not change sign on (0, 1).
    ValueError
        If the MT/ZPC system does not converge within the tolerance
        defined by ``CalibPar.sup_side_eq_eps``.

    Returns
    -------
    dict
        Dictionary containing the w, s, and I solutions for the supply side equilibrium.

    '''
    
    obj_func = lambda I: zpc_y_residual(I, p=p, r=r, ModelPar=ModelPar)
            
    at_zero: float = obj_func(1e-16)
    at_one : float = obj_func(1-1e-16)
    
    if at_zero * at_one > 0:
        warnings.warn(f'No interior solution for p={p:.4f}, r={r:.4f}. Trying corner solution.')
        if at_zero > 0:
            I_sol = 1.0
        else:
            I_sol = 0.0
    else:
        I_sol = brentq(obj_func, 1e-16, 1-1e-16)
    
    w_sol, s_sol = solve_wages_given_I(I_0=I_sol, r=r, ModelPar=ModelPar)
    
    check: list = check_mt_zpc_system(w=w_sol,s=s_sol,I=I_sol,p=p,r=r,ModelPar=ModelPar)
    check: bool = (abs(check[0])>CalibPar.sup_side_eq_eps) | (abs(check[1])>CalibPar.sup_side_eq_eps) | (abs(check[2])>CalibPar.sup_side_eq_eps)
    
    if check: raise ValueError("MT/ZPC system didn't converge.")
    
    return {'w_sol':w_sol,
            's_sol':s_sol,
            'I_sol':I_sol}

def psi(p, ModelPar):
    '''
    Calculates the ψ(p) function.

    Parameters
    ----------
    p : float
        Good Y's price.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Value of ψ(p).

    '''
    
    value: float = (ModelPar.η ** ModelPar.η) * (((1-ModelPar.η)/(p)) ** (1-ModelPar.η))
    
    return value

def utility_func(inc,a_1,p,ModelPar):
    '''
    For a given income (calculated in income_func) and a' assets level, calculates
    the utility of a household given a good Y price of p. 

    Parameters
    ----------
    inc : float
        Income of the household.
    a_1 : float
        Next period asset level a'.
    p   : float
        Price of good Y.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    value : float
        Utility level of the household.

    '''
    
    c: float = inc - a_1
    
    # If consumption is negative, sets a large negative value to utility to denote unfeasibility
    if c <= 0:
        return -1e25
    
    psi_p: float = psi(p, ModelPar)
    value: float = (psi_p * c) ** (1 - ModelPar.σ) / (1 - ModelPar.σ)
    
    return value
    
def income_func(r,a,z,w,s,L):
    '''
    Calculates the income for a household in a given state defined by asset holdings
    a, labour productivity z and labour type L (L=1 indicates low-skilled sector), 
    where the economy's state is defined by an interest rate r and wage rates w 
    (low-skilled) and s (high-skilled).

    Parameters
    ----------
    r : float
        Interest rate.
    a : float
        Current asset holdings.
    z : float
        Productivity level.
    w : float
        Low-skill wage rate.
    s : float
        High-skill wage rate.
    L : float
        Indicator whether the household is in a low-skill state (L=1 if that's true)

    Returns
    -------
    value : float
        Income of the household in that period.

    '''
    
    value: float = (1+r) * a + z * (L*w + (1-L)*s)
    
    return value

@nb.njit(parallel=True, cache=True)
def _vfi_core(V, U, joint_trans, delta, eps, howard_steps):
    '''
    Numba-compiled VFI core: Policy Improvement + Howard's acceleration. This core has the basic
    structure of the VFI process I wrote myself. Later, I asked Claude Code to help me 
    parallalize it using Numba. 

    V           : (n_states, n_assets)        initial value function, modified in place
    U           : (n_states, n_assets, n_assets)  U[i, a, j] = flow utility for state i,
                                                   current asset index a, next asset index j
    joint_trans : (n_states, n_states)         Markov transition matrix
    '''
    n_states = V.shape[0]
    n_assets = V.shape[1]
    pol = np.zeros((n_states, n_assets), dtype=np.int64)

    cond     = True
    it_count = 0

    while cond:
        it_count += 1
        V_old = V.copy()

        # Policy Improvement — parallel over states (each state writes to its own row)
        for i in nb.prange(n_states):
            exp_V = np.dot(joint_trans[i], V_old)   # (n_assets,): E[V(s',a')] for each a'
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

        # Convergence check (cheap serial loop)
        max_diff = 0.0
        for i in range(n_states):
            for a in range(n_assets):
                d = abs(V[i, a] - V_old[i, a])
                if d > max_diff:
                    max_diff = d
        cond = max_diff > eps

        # Howard's Policy Evaluation — parallel over states
        if cond:
            for _ in range(howard_steps):
                V_old = V.copy()
                for i in nb.prange(n_states):
                    exp_V = np.dot(joint_trans[i], V_old)
                    for a in range(n_assets):
                        j        = pol[i, a]
                        V[i, a]  = U[i, a, j] + delta * exp_V[j]

    return V, pol, it_count


def model_vfi(w,s,r,p,income_func,state_grid,joint_trans,ModelPar,CalibPar,print_convergence=True,V_init=None):
    '''
    Value Function Iteration of the household's problem solution. Given an economy state defined by
    w,s,r,p derived from the supply-side equilibrium, it solves the household problem using as
    inputs the income function generator, the grid of states, the transition matrix between states.

    Parameters
    ----------
    w : float
        Low-skill wage rate.
    s : float
        High-skill wage rate.
    r : float
        Interest rate.
    p : float
        Price of good Y.
    income_func : callable
        Function defining the income function for any given household's state.
    state_grid : list
        List of tuples containing the possible states for skill type and z.
    joint_trans : numpy.ndarray
        Array with the transition matrix between the states described in state_grid.
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.
    print_convergence : bool, optional
        Flag indicating whether to print the time and iteration count after convergence. The default is True.

    Returns
    -------
    pol_func : dict
        Dictionary containing the a' and c policy functions for each state of skill type, productivity level z and current asset holdings.
    pol_idx  : dict
        Dictionary containing the indices of the optimal grid point in the policy function.
    a_grid   : list
        List of the possible asset holdings states.
    df_val_func   : pandas.DataFrame
        Data Frame mapping the state (skill type, current assets, productivity) to the utility of being in that state

    '''
    
    start: float = time.time()

    # Asset grid
    ub      : float      = max(w, s) * CalibPar.vfi_ubmul
    dist    : float      = (ub - CalibPar.vfi_lb) / CalibPar.vfi_N
    a_grid  : list       = [CalibPar.vfi_lb + (i * dist) for i in range(CalibPar.vfi_N + 1)]
    a_arr   : np.ndarray = np.array(a_grid)
    n_assets: int        = len(a_grid)
    n_states: int        = len(state_grid)
    psi_p   : float      = psi(p, ModelPar)

    # Build U array: U_arr[i, a, j] = flow utility for state i, current asset a, next asset j
    # u_grid has shape (n_assets_prime, n_assets_current) so we transpose to (n_assets_current, n_assets_prime)
    U_arr: np.ndarray = np.empty((n_states, n_assets, n_assets))
    for idx, (f, z) in enumerate(state_grid):
        L      : int        = 1 if f == 'L' else 0
        inc_vec: np.ndarray = income_func(r=r, a=a_arr, z=np.exp(z), w=w, s=s, L=L)
        c_grid : np.ndarray = inc_vec[None, :] - a_arr[:, None]
        c_safe : np.ndarray = np.maximum(c_grid, 1e-10)
        u_grid : np.ndarray = np.where(c_grid > 0, (psi_p * c_safe) ** (1 - ModelPar.σ) / (1 - ModelPar.σ), -1e25)
        U_arr[idx] = u_grid.T  # (a_current, a_prime)

    # Run parallelized VFI core (warm-start from previous solution if available)
    V_arr, pol_arr, it_count = _vfi_core(
        V_init.copy() if V_init is not None else np.zeros((n_states, n_assets)),
        U_arr,
        joint_trans.astype(np.float64),
        float(ModelPar.δ),
        float(CalibPar.vfi_eps),
        int(CalibPar.vfi_howard_steps)
    )

    # Value function DataFrame
    df_val_func: pd.DataFrame = pd.concat([
        pd.DataFrame({'a_0': a_grid, 'skill_type': state[0], 'z': np.exp(state[1]), 'V': V_arr[i]})
        for i, state in enumerate(state_grid)
    ]).reset_index(drop=True)

    # Policy indices dict
    pol_idx: dict = {state: pol_arr[i].copy() for i, state in enumerate(state_grid)}

    # Policy function DataFrames (vectorized consumption — avoids slow iterrows)
    pol_func: dict = {}
    for i, state in enumerate(state_grid):
        f, z   = state
        L: int = 1 if f == 'L' else 0
        idxs   = pol_arr[i]
        df                  = pd.DataFrame({'a_0': a_grid, 'a_1': a_arr[idxs]})
        df['c']             = income_func(r=r, a=df['a_0'].values, z=np.exp(z), w=w, s=s, L=L) - df['a_1'].values
        pol_func[state]     = df

    end: float = time.time() - start
    if print_convergence: print(f'Convergence after {end:.2f}s and {it_count} iterations')

    return pol_func, pol_idx, a_grid, df_val_func, V_arr, it_count

def full_model_result(pol_func, state_grid, stat_dist, df_val_func, p, ModelPar):
    '''
    Creates a dataframe describing the full model result. For a given household state described by:
        Current assets: a_0
        Productivity: z, already exponentialized
        Skill type: sill_type
    It gives the starionary distribution, value functions and policy functions:
        Future assets: a_1
        Consumption: c
        X good consumption: c_x
        Y good consumption: c_y
        Stat. Distribution: dens
        Value Function/Utility: V
        
    Parameters
    ----------
    pol_func : dict
        Policy function dictionary.
    state_grid : list
        List of tuples containing the possible states for skill type and z.
    stat_dist : pandas.DataFrame
        Data Frame with the stationary distribution of states.
    df_val_func : pandas.DataFrame
        Data Frame with the value function/utility in each state.
    p : float
        Price of good Y.
    ModelPar : TypeModelParameters
        Model parameters.

    Returns
    -------
    full_model_result : pandas.DataFrame
        Data Frame with the full results from the model, as in the description.

    '''
    
    # Creating the policy function dataframe
    df_pol_func: pd.DataFrame = pd.concat([pol_func[state].assign(skill_type=state[0], z=np.exp(state[1])) for state in state_grid]).reset_index(drop=True)
    
    df_pol_func['c_x'] = df_pol_func['c'] * ModelPar.η
    df_pol_func['c_y'] = (df_pol_func['c']/p) * (1-ModelPar.η)
    
    # Setting indices to concat the dataframes
    df_pol_func = df_pol_func.set_index(['a_0','z','skill_type'])
    stat_dist   = stat_dist.set_index(['a_0','z','skill_type']) * ModelPar.M
    df_val_func = df_val_func.set_index(['a_0','z','skill_type'])
    
    full_model_result = pd.concat([df_pol_func,stat_dist,df_val_func], axis=1).reset_index()
    
    return full_model_result

def inner_loop_residual(p,w,s,I,mod_res,ModelPar,CalibPar):
    '''
    Calculates the labour type market clearing which determines the inner loop residual
    relative to the price of good Y. Adjustments are based on excess demand for low-skilled labour

    Parameters
    ----------
    p : float
        Price of good Y.
    w : float
        Low-skilled wage.
    s : float
        High-skilled wage.
    I : float
        Marginal task index.
    mod_res : pandas.DataFrame
        Data Frame with the full results from the model.
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.

    Returns
    -------
    ret : float
        Excess demand for low-skilled labour.

    '''
    
    # Goods demand
    x_D: float = (mod_res['c_x'] * mod_res['dens']).sum()
    y_D: float = (mod_res['c_y'] * mod_res['dens']).sum()
    
    # Low-skilled labour supply
    temp_res: pd.DataFrame = mod_res.loc[mod_res['skill_type']=='L']
    L_supply: float        = (temp_res['dens'] * temp_res['z']).sum()/(1-I)
    del(temp_res)

    # Low-skilled labour demand
    numerator  : float = (1-ModelPar.α_x-ModelPar.γ) * x_D + p * (1-ModelPar.α_y-ModelPar.γ) * y_D
    denominator: float = w * Omega(I,ModelPar=ModelPar)
    L_demand   : float = numerator/denominator

    # High-skilled labour supply
    temp_res: pd.DataFrame = mod_res.loc[mod_res['skill_type']=='H']
    H_supply: float        = (temp_res['dens'] * temp_res['z']).sum()
    del(temp_res)
    
    # High-skilled labour demand
    numerator  : float = (ModelPar.α_x) * x_D + p * (ModelPar.α_y) * y_D
    denominator: float = s
    H_demand   : float = numerator/denominator
    
    # Check if demand for labour is clearing
    check: bool = (abs(L_supply-L_demand) < CalibPar.gmc_eps) & (abs(H_supply-H_demand) < CalibPar.gmc_eps)

    if check:
        print('Good markets clear!')
        ret: float = 0.0
        
    else:
        ret: float = L_supply-L_demand
    
    return ret

def outer_loop_residual(p,r,mod_res,ModelPar,CalibPar):
    '''
    Calculates the capital market clearing which determines the outer loop residual
    relative to the interest rate r.

    Parameters
    ----------
    p : float
        Price of good Y.
    r : float
        Interest rate.
    mod_res : pandas.DataFrame
        Data Frame with the full results from the model.
    ModelPar : TypeModelParameters
        Model parameters.
    CalibPar : TypeCalibParameters
        Calibration parameters.

    Returns
    -------
    ret : float
        Excess demand for capital.

    '''
    
    # Goods demand
    x_D: float = (mod_res['c_x'] * mod_res['dens']).sum()
    y_D: float = (mod_res['c_y'] * mod_res['dens']).sum()
    
    # Capital supply
    K_supply: float        = (mod_res['dens'] * mod_res['a_0']).sum()

    # Capital demand
    numerator  : float = (ModelPar.γ) * x_D + p * (ModelPar.γ) * y_D
    denominator: float = r
    K_demand   : float = numerator/denominator
    
    # Check if demand for capital is clearing
    check: bool = (abs(K_supply-K_demand) < CalibPar.kmc_eps)

    if check:
        print('Capital markets clear!')
        ret: float = 0.0
        
    else:
        ret: float = K_supply-K_demand
    
    return ret

def weighted_gini(x, weights):
    '''
    Calculates the Gini index of a vector with different weights.

    Parameters
    ----------
    x : pd.Series
        Series with the values of the variable whose Gini is being calculated.
    weights : pd.Series
        Series with the weights of each value in x.

    Returns
    -------
    gini : float
        Weighted Gini Index.

    '''
    
    sorted_idx: pd.Series = np.argsort(x)
    x_sorted  : pd.Series = x[sorted_idx]
    w_sorted  : pd.Series = weights[sorted_idx]
    
    # X-axis and Y-axis cumulative shares
    cum_pop   : pd.Series = np.cumsum(w_sorted) / np.sum(w_sorted)
    cum_income: pd.Series = np.cumsum(w_sorted * x_sorted) / np.sum(w_sorted * x_sorted)
    
    # Adding the origin to the cumulative shares
    cum_pop   : np.ndarray = np.concatenate([[0], cum_pop])
    cum_income: np.ndarray = np.concatenate([[0], cum_income])
    
    # Approximates the curve using trapezes and finds the Gini coefficient
    area: float = np.sum((cum_pop[1:] - cum_pop[:-1]) * (cum_income[1:] + cum_income[:-1]) / 2)
    gini: float = 1 - 2 * area
    
    return gini

def weighted_percentile(x, weights, q):
    '''
    Calculates the share of x and the value of x relative to percentile q when
    x is a vector with different weights.

    Parameters
    ----------
    x : pd.Series
        Series with the values of the variable whose Gini is being calculated.
    weights : pd.Series
        Series with the weights of each value in x.
    q : int/float
        Percentile of x we're after.

    Returns
    -------
    share : float
        Share of x which is held by percentile q.
    perc_income : float
        Value of x relative to percentile q.

    '''
    
    sorted_idx: pd.Series = np.argsort(x)
    x_sorted  : pd.Series = x[sorted_idx]
    w_sorted  : pd.Series = weights[sorted_idx]
    
    cum_w: pd.Series = np.cumsum(w_sorted)
    cum_w: pd.Series = cum_w / cum_w.iloc[-1]
    mask : pd.Series = cum_w >= q / 100
    
    share      : float = np.sum(x_sorted[mask] * w_sorted[mask]) / np.sum(x_sorted * w_sorted)
    idx        : int = np.searchsorted(cum_w, q / 100)
    perc_income: float = x_sorted[min(idx, len(x_sorted) - 1)]
    
    return share, perc_income

