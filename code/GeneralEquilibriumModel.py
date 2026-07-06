# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import os
import time
import numpy as np
import pandas as pd
import config
import functions
from dataclasses import dataclass
from datetime import datetime
from scipy.optimize import brentq, minimize_scalar

class NoEquilibriumError(Exception):
    '''Raised when no w ∈ (w*, w*β) clears the L-market for a given r.'''
    pass

@dataclass
class TypeModelParameters:
    α     : float   # high-skill labour share in production
    γ     : float   # capital share in production  (κ = 1-α-γ is low-skill share)
    β     : float   # offshoring cost scale (β > 1 required)
    w_star: float   # foreign wage
    θ     : float   # offshoring cost convexity
    σ     : float   # CRRA coefficient
    δ     : float   # discount factor
    ρ     : float   # AR(1) persistence for log productivity z
    σ_ϵ   : float   # AR(1) innovation std for log productivity z
    π_LL  : float   # prob(stay L | L)
    π_HH  : float   # prob(stay H | H)
    M     : float   # mass of households
    t_form: str = 'exponential'  # 'quadratic' → i^θ | 'exponential' → e^(θi)

@dataclass
class TypeCalibParameters:
    rh_N             : float
    rh_r             : float
    rh_c             : float
    vfi_lb           : float
    vfi_ubmul        : float
    vfi_N            : float
    vfi_eps          : float
    vfi_howard_steps : float
    gmc_eps          : float
    kmc_eps          : float
    r_init_guess     : float
    inner_loop_eps   : float
    outer_loop_eps   : float
    outer_loop_r_lb  : float

class GeneralEquilibriumModel:

    def __init__(self, ModelPar, CalibPar, log_dir="auto", log_inner=True, log_summary_name='run_summary.log'):

        self.ModelPar          = ModelPar
        self.CalibPar          = CalibPar
        self._log_dir_arg      = log_dir
        self._log_inner        = log_inner
        self._log_summary_name = log_summary_name

        self.w            = None
        self.s            = None
        self.I            = None
        self.r            = None
        self.Y            = None
        self.mod_res      = None
        self._vfi_V_cache = None

        # Calculates steady-state H and L from exogenous stochastic process
        self.H, self.L = self._compute_HL_supply()

        # Logging state
        self._log_dir           = None
        self._summary_log_path  = None
        self._current_inner_log = None
        self._outer_eval_count  = 0
        self._inner_eval_count  = 0
        self._last_vfi_iters    = None
        self._inner_w_residual  = None

    def _compute_HL_supply(self):
        """
        Computes H and L supplies in steady-state from the exogenous stochastic process

        Returns
        -------
        H : float
            Mass of high-skilled households in steady-state
        L : float
            Mass of low-skilled households in steady-state
        
        """

        z_grid, trans_z = functions.rouwenhorst_trans_matrix(ModelPar=self.ModelPar, CalibPar=self.CalibPar)
        z_stat_dist     = functions.calculate_stationary_distribution(trans_matrix=trans_z)
        trans_skill     = functions.create_LH_skill_mat(self.ModelPar)
        skill_stat_dist = functions.calculate_stationary_distribution_eigenvector(trans_matrix=trans_skill)

        mean_z: float = float(np.sum(np.exp(z_grid) * z_stat_dist))
        H     : float = float(skill_stat_dist[1] * mean_z)
        L     : float = float(skill_stat_dist[0] * mean_z)

        return H, L

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _init_logging(self):
        self._outer_eval_count = 0

        if self._log_dir_arg is None:
            return

        if self._log_dir_arg == "auto":
            timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._log_dir = config.LOG_ISOLATED / f'run_{timestamp}'
            self._log_dir.mkdir(exist_ok=True)
        else:
            self._log_dir = self._log_dir_arg

        self._summary_log_path = os.path.join(self._log_dir, self._log_summary_name)

        SEP = '-' * 87
        par = self.ModelPar
        with open(self._summary_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{SEP}\n")
            f.write(f"Parameters:\n")
            f.write(f"  α={par.α}  γ={par.γ}  β={par.β}  w_star={par.w_star}\n")
            f.write(f"  θ={par.θ}  σ={par.σ}  δ={par.δ}  ρ={par.ρ}  σ_ϵ={par.σ_ϵ}\n")
            f.write(f"  π_LL={par.π_LL}  π_HH={par.π_HH}  M={par.M}\n")
            f.write(f"  H={self.H:.6f}  L={self.L:.6f}\n")
            f.write(f"{SEP}\n")
            f.write(f"{'trial':>7}  {'r':>10}  {'w_conv':>10}  {'inner_resid':>12}  {'K_resid':>12}  {'VFI_iters':>10}  {'time(s)':>9}\n")

    def _log_inner_trial(self, w, residual):
        if self._current_inner_log is None:
            return
        with open(self._current_inner_log, 'a', encoding='utf-8') as f:
            f.write(f"{self._inner_eval_count:>7}  {w:>12.6f}  {residual:>+14.6e}\n")

    def _log_outer_eval(self, r, outer_residual, total_elapsed):
        if self._log_dir is None:
            return

        SEP = '-' * 87
        inner_ok  = self._inner_w_residual is not None and abs(self._inner_w_residual) < self.CalibPar.inner_loop_eps
        conv_str  = "successful" if inner_ok else "unsuccessful"
        w_val     = self.w                if self.w                is not None else float('nan')
        resid_val = self._inner_w_residual if self._inner_w_residual is not None else float('nan')

        if self._current_inner_log is not None:
            with open(self._current_inner_log, 'a', encoding='utf-8') as f:
                f.write(f"{SEP}\n")
                f.write(
                    f"Inner loop {conv_str} for w={w_val:.6f} in {total_elapsed:.2f}s."
                    f"  Outer residual: {outer_residual:+.4e}.\n"
                )

        with open(self._summary_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{self._outer_eval_count:>7}  {r:>10.6f}  {w_val:>10.6f}"
                f"  {resid_val:>12.4e}  {outer_residual:>+12.4e}"
                f"  {self._last_vfi_iters:>10}  {total_elapsed:>9.2f}\n"
            )

    # ------------------------------------------------------------------
    # Model solution methods
    # ------------------------------------------------------------------

    def solve_representative_household(self):
        """
        Solves the analytical representative household equilibrium.
        Sets self.rep_hh_res.
        """

        rep_hh_res = functions.representative_household_wrapper(ModelPar=self.ModelPar)
        self.rep_hh_res = rep_hh_res

    def solve_firm_side(self, w, r):
        """
        Computes the firm-side equilibrium analytically given w and r.
        Sets self.I and self.s.

        Parameters
        ----------
        w : float
            Low-skilled wages
        r : float
            Interest rate
        """
        
        I, Ω, s     = functions.solve_firm_side(w=w, r=r, ModelPar=self.ModelPar)
        self.I: float = I
        self.s: float = s

    def _inner_loop_trial(self, w, r):
        """
        Evaluates the L-market clearing residual at candidate w. Logs the trial.
        """

        self._inner_eval_count += 1
        residual = functions.inner_loop_residual(w        = w, 
                                                 r        = r, 
                                                 H        = self.H, 
                                                 L        = self.L,
                                                 ModelPar = self.ModelPar)
        self._log_inner_trial(w=w, residual=residual)
        return residual

    def inner_loop_solver(self, r):
        """
        Finds w that clears the L-market for a given r.

        Bracket depends on t_form:
          quadratic:   w ∈ (w*, w*β)       — I ∈ ((1/β)^(1/θ), 1)
          exponential: w ∈ (w*β, w*β·e^θ)  — I ∈ (0, 1)

        Uses brentq after verifying a sign change at the bracket endpoints.
        Raises NoEquilibriumError if no root exists in the bracket.

        Sets self.w, self.I, self.s, self.Y on success.

        Parameters
        ----------
        r : float
            Interest rate
        """

        self._inner_eval_count = 0

        if self.ModelPar.t_form == 'exponential':
            w_lb: float = self.ModelPar.w_star * self.ModelPar.β
            w_ub: float = self.ModelPar.w_star * self.ModelPar.β * np.exp(self.ModelPar.θ)
        else:  # 'quadratic'
            w_lb: float = self.ModelPar.w_star
            w_ub: float = self.ModelPar.w_star * self.ModelPar.β

        obj_func = lambda w: self._inner_loop_trial(w, r)

        f_lb: float = obj_func(w_lb + 1e-10)
        f_ub: float = obj_func(w_ub - 1e-10)

        print(f"  Bracket check: f(w_lb)={f_lb:+.4e}, f(w_ub)={f_ub:+.4e}")

        if f_lb * f_ub > 0:
            raise NoEquilibriumError(f"No L-market clearing w for r={r:.4f}. "\
                                     f"Residual signs: f(w_lb)={f_lb:+.4e}, f(w_ub)={f_ub:+.4e}")

        w_sol: float = brentq(obj_func, w_lb + 1e-10, w_ub - 1e-10, xtol=self.CalibPar.inner_loop_eps)

        I_sol, Ω_sol, s_sol = functions.solve_firm_side(w=w_sol, r=r, ModelPar=self.ModelPar)
        Y_sol               = s_sol * self.H / self.ModelPar.α

        self.w = w_sol
        self.I = I_sol
        self.s = s_sol
        self.Y = Y_sol

        self._inner_w_residual = abs(functions.inner_loop_residual(w        = w_sol, 
                                                                   r        = r, 
                                                                   H        = self.H, 
                                                                   L        = self.L, 
                                                                   ModelPar = self.ModelPar))

        print(f"  Inner loop solved: w={self.w:.4f}, I={self.I:.4f}, s={self.s:.4f}, "
              f"Y={self.Y:.4f}, |resid|={self._inner_w_residual:.2e}")
    
    def solve_household_side(self, r):
        """
        Runs VFI with current prices (self.w, self.s, r) to obtain the stationary asset distribution. 
        
        Sets self.mod_res.

        Parameters
        ----------
        r : float
            Interest rate
        """

        z_grid, trans_z     = functions.rouwenhorst_trans_matrix(ModelPar=self.ModelPar,
                                                                 CalibPar=self.CalibPar)
        trans_f    : np.ndarray = functions.create_LH_skill_mat(ModelPar=self.ModelPar)
        joint_trans: np.ndarray = np.kron(trans_f, trans_z)
        state_grid : list       = [(f, z) for f in ['L', 'H'] for z in z_grid]

        pol_func, pol_idx, a_grid, utility, V_arr, vfi_iters = functions.model_vfi(
            w               = self.w,
            s               = self.s,
            r               = r,
            income_func     = functions.income_func,
            state_grid      = state_grid,
            joint_trans     = joint_trans,
            CalibPar        = self.CalibPar,
            ModelPar        = self.ModelPar,
            V_init          = self._vfi_V_cache)

        self._vfi_V_cache    = V_arr
        self._last_vfi_iters = vfi_iters

        stat_dist: pd.DataFrame = functions.calculate_stationary_distribution_endog(
            pol_idx     = pol_idx,
            state_grid  = state_grid,
            a_grid      = a_grid,
            joint_trans = joint_trans)

        self.mod_res = functions.full_model_result(
            pol_func    = pol_func,
            state_grid  = state_grid,
            stat_dist   = stat_dist,
            df_val_func = utility,
            ModelPar    = self.ModelPar)

    def outer_solution_wrapper(self, r):
        """
        Wrapper for capital market clearing given r.

        For a given r:
          1. Find w analytically (inner loop, no VFI).
          2. Run VFI with (w, s, r) to get stationary asset distribution.
          3. Check K market clearing: K_supply vs γ·Y/r.
        """

        self._outer_eval_count += 1
        self._inner_eval_count  = 0

        if self._log_dir is not None and self._log_inner:
            inner_log_name          = f'inner_r{self._outer_eval_count:03d}_r{r:.6f}.log'
            self._current_inner_log = os.path.join(self._log_dir, inner_log_name)
            SEP = '-' * 60
            with open(self._current_inner_log, 'w') as f:
                f.write(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  r = {r:.6f}\n")
                f.write(f"{SEP}\n")
                f.write(f"{'trial':>7}  {'w':>12}  {'L_resid':>14}\n")
        else:
            self._current_inner_log = None

        print(f"\nTrying r={r:.4f}.")
        step_start = time.time()

        # Step 1: inner loop (analytical, no VFI)
        try:
            self.inner_loop_solver(r=r)
        except NoEquilibriumError as e:
            print(f"  NoEquilibriumError: {e}")
            self._log_outer_eval(r=r, outer_residual=1e10,
                                 total_elapsed=time.time()-step_start)
            return 1e10

        # Step 2: VFI (runs once per outer iteration)
        print(f"  Running VFI for r={r:.4f}, w={self.w:.4f}, s={self.s:.4f} ...")
        self.solve_household_side(r=r)

        # Step 3: K market clearing residual
        residual = functions.outer_loop_residual(
            r        = r,
            Y        = self.Y,
            mod_res  = self.mod_res,
            ModelPar = self.ModelPar,
            CalibPar = self.CalibPar)

        elapsed = time.time() - step_start
        self._log_outer_eval(r=r, outer_residual=residual, total_elapsed=elapsed)

        return abs(residual)

    def outer_loop_solver(self):
        """
        Wrapper to solve the full model. 
        
        Sets self.r and self.outer_res
        """

        self._init_logging()
        run_start = time.time()

        obj_func = lambda r: self.outer_solution_wrapper(r)
        r_ub = 1 / self.ModelPar.δ - 1

        res = minimize_scalar(obj_func,
                              bounds =(self.CalibPar.outer_loop_r_lb, r_ub),
                              method ='bounded',
                              tol    = self.CalibPar.outer_loop_eps)

        self.r        : float = res.x
        self.outer_res = res

        total_elapsed = time.time() - run_start
        success = res.fun < self.CalibPar.outer_loop_eps

        if success: print(f"Outer loop successful for r={self.r:.4f}.")
        else      : print("Outer loop unsuccessful for current parameters.")

        if self._summary_log_path is None:
            return

        SEP = '-' * 87
        conv_str = "successful" if success else "unsuccessful"
        with open(self._summary_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{SEP}\n")
            f.write(f"Outer loop {conv_str} for r={self.r:.6f} in {total_elapsed:.2f}s.\n")
            f.write(f"Final: I={self.I:.4f}  w={self.w:.4f}  s={self.s:.4f}  "
                    f"Y={self.Y:.6f}  r={self.r:.6f}\n")
            f.write(f"Outer residual: {self.outer_res.fun:.4e}  |  "
                    f"Inner residual: {self._inner_w_residual:.4e}\n")

        print(f"Model solved in {total_elapsed/60:.2f}m.")
        print("\nResults:")
        print(f"I: {self.I} | Y: {self.Y} | r: {self.r} | s: {self.s} | w: {self.w}")
        print(f"Outer residual: {self.outer_res.fun}.")
        print(f"Inner residual: {self._inner_w_residual}.")

    def weighted_percentile_aggregation(self, x_var, y_var, n_bins=50):
        x      : np.ndarray = np.array(self.mod_res[x_var])
        y      : np.ndarray = np.array(self.mod_res[y_var])
        weights: np.ndarray = np.array(self.mod_res['dens'])

        sorted_idx: np.ndarray = np.argsort(x)
        x_sorted  : np.ndarray = x[sorted_idx]
        y_sorted  : np.ndarray = y[sorted_idx]
        w_sorted  : np.ndarray = weights[sorted_idx]

        cum_w: np.ndarray = np.cumsum(w_sorted)
        cum_w: np.ndarray = cum_w / cum_w[-1]

        rows: list = []

        for i in range(n_bins):
            q_low : float = i / n_bins
            q_high: float = (i + 1) / n_bins

            if i == 0:
                mask: np.ndarray = cum_w <= q_high
            elif i == n_bins - 1:
                mask: np.ndarray = cum_w > q_low
            else:
                mask: np.ndarray = (cum_w > q_low) & (cum_w <= q_high)

            w_group: np.ndarray = w_sorted[mask]

            if w_group.sum() > 0:
                y_mean: float = np.sum(y_sorted[mask] * w_group) / w_group.sum()
                x_mean: float = np.sum(x_sorted[mask] * w_group) / w_group.sum()
            else:
                y_mean: float = np.nan
                x_mean: float = np.nan

            rows.append({
                'bin'      : i + 1,
                'x_low'    : x_sorted[mask][0] if mask.any() else np.nan,
                'x_high'   : x_sorted[mask][-1] if mask.any() else np.nan,
                'x_mean'   : x_mean,
                'y_mean'   : y_mean,
                'pop_share': w_group.sum()
            })

        return pd.DataFrame(rows)

    def economy_statistics(self, iq_top=90, iq_bottom=10):

        if self.mod_res is None:
            raise ValueError('Model results are non-existent. Run the solver first.')

        mod_res: pd.DataFrame = self.mod_res.copy()

        # Income and savings
        mod_res['y'] = (mod_res['a_0'] * self.r) + (
            mod_res['z'] * np.where(mod_res['skill_type'] == 'L', self.w, self.s))
        mod_res['labour_inc_share'] = 1 - (mod_res['a_0'] * self.r) / mod_res['y']
        mod_res['s_rate']           = (mod_res['a_1'] - mod_res['a_0']) / mod_res['y']

        # CEV: V = C_eq^(1-σ)/((1-σ)(1-δ)) → C_eq = (V·(1-σ)·(1-δ))^(1/(1-σ))
        numerator    : pd.Series = mod_res['V'] * (1 - self.ModelPar.σ) * (1 - self.ModelPar.δ)
        exp_         : float     = 1 / (1 - self.ModelPar.σ)
        mod_res['c_eq'] = numerator ** exp_

        # Aggregates
        aggregate_K  : float = (mod_res['a_0'] * mod_res['dens']).sum()
        aggregate_C  : float = (mod_res['c']   * mod_res['dens']).sum()
        aggregate_ceq: float = (mod_res['c_eq'] * mod_res['dens']).sum()
        aggregate_V  : float = (mod_res['V']    * mod_res['dens']).sum()

        # Factor income shares
        L_df   : pd.DataFrame = mod_res.loc[mod_res['skill_type'] == 'L']
        mean_V_L: float       = (L_df['V'] * L_df['dens']).sum() / L_df['dens'].sum()
        L_agg  : float        = (L_df['z'] * L_df['dens']).sum()

        H_df   : pd.DataFrame = mod_res.loc[mod_res['skill_type'] == 'H']
        mean_V_H: float       = (H_df['V'] * H_df['dens']).sum() / H_df['dens'].sum()
        H_agg  : float        = (H_df['z'] * H_df['dens']).sum()

        ls_share: float = self.w * L_agg
        hs_share: float = self.s * H_agg
        k_share : float = self.r * aggregate_K
        nom_gdp : float = ls_share + hs_share + k_share
        ls_share        = ls_share / nom_gdp
        hs_share        = hs_share / nom_gdp
        k_share         = k_share  / nom_gdp

        # Real GDP = nominal GDP (one good, p=1)
        real_gdp  : float = nom_gdp
        K_to_Y    : float = aggregate_K / real_gdp
        skill_prem: float = self.s / self.w
        w_to_wstar: float = self.w / self.ModelPar.w_star

        # Inequality
        income_gini : float = functions.weighted_gini(x=mod_res['y'],   weights=mod_res['dens'])
        ceq_gini    : float = functions.weighted_gini(x=mod_res['c_eq'], weights=mod_res['dens'])

        income_top  : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_top)[1]
        income_bot  : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_bottom)[1]
        income_ratio: float = income_top / income_bot
        income_conc : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_top)[0]

        wealth_top  : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_top)[1]
        wealth_bot  : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_bottom)[1]
        wealth_ratio: float = wealth_top / wealth_bot
        wealth_conc : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_top)[0]

        ret: dict = {
            'K'                   : aggregate_K,
            'mean_V'              : aggregate_V,
            'mean_V_L'            : mean_V_L,
            'mean_V_H'            : mean_V_H,
            'C'                   : aggregate_C,
            'mean_c_eq'           : aggregate_ceq,
            'low_skill_share'     : ls_share,
            'high_skill_share'    : hs_share,
            'k_share'             : k_share,
            'real_gdp'            : real_gdp,
            'K/Y'                 : K_to_Y,
            'skill_premium'       : skill_prem,
            'w_to_wstar'          : w_to_wstar,
            'income_gini'         : income_gini,
            'ceq_gini'            : ceq_gini,
            'income_ratio'        : income_ratio,
            'wealth_ratio'        : wealth_ratio,
            'income_concentration': income_conc,
            'wealth_concentration': wealth_conc,
            'I'                   : self.I,
            'w'                   : self.w,
            's'                   : self.s,
            'Y'                   : self.Y,
            'r'                   : self.r
        }

        self.economy_stats = ret

        return ret
