# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import time
import numpy as np
import pandas as pd
import functions
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

@dataclass
class TypeModelParameters:
    α_x   : float
    α_y   : float
    γ     : float
    β     : float
    w_star: float
    θ     : float
    η     : float
    σ     : float
    δ     : float
    ρ     : float
    σ_ϵ   : float
    π_LL  : float
    π_HH  : float
    M     : float

@dataclass
class TypeCalibParameters:
    rh_N             : float
    rh_r             : float
    rh_c             : float
    sup_side_eq_eps  : float
    vfi_lb           : float
    vfi_ubmul        : float
    vfi_N            : float
    vfi_eps          : float
    vfi_howard_steps : float
    gmc_eps          : float
    kmc_eps          : float
    p_init_guess     : float
    r_init_guess     : float
    inner_loop_eps   : float
    inner_loop_p_lb  : float
    inner_loop_p_ub  : float
    inner_loop_marg  : float
    outer_loop_eps   : float
    outer_loop_r_lb  : float

class GeneralEquilibriumModel:
    
    def __init__(self, ModelPar, CalibPar):
        
        self.ModelPar = ModelPar
        self.CalibPar = CalibPar
        
        self.w        = None
        self.s        = None
        self.I        = None
        self.r        = None
        self.p        = None
        self.mod_res  = None
        
    def solve_firm_side(self,p,r):
        
        sol: dict = functions.solve_firm_side(p        = p,
                                              r        = r,
                                              ModelPar = self.ModelPar,
                                              CalibPar = self.CalibPar)
        
        self.w: float = sol['w_sol']
        self.s: float = sol['s_sol']
        self.I: float = sol['I_sol']
        
    def solve_household_side(self,p,r):
        
        z_grid,trans_z      = functions.rouwenhorst_trans_matrix(ModelPar = self.ModelPar,
                                                                 CalibPar = self.CalibPar)
        trans_f: np.ndarray = functions.create_LH_skill_mat(ModelPar = self.ModelPar)

        joint_trans: np.ndarray = np.kron(trans_f, trans_z)
        state_grid : list       = [(f, z) for f in ['L', 'H'] for z in z_grid]
        stat_dist  : np.ndarray = functions.calculate_stationary_distribution(joint_trans)

        pol_func, pol_idx, a_grid, utility = functions.model_vfi(w           = self.w,
                                                                 s           = self.s,
                                                                 r           = r,
                                                                 p           = p,
                                                                 income_func = functions.income_func,
                                                                 state_grid  = state_grid,
                                                                 joint_trans = joint_trans,
                                                                 CalibPar    = self.CalibPar,
                                                                 ModelPar    = self.ModelPar)

        stat_dist: pd.DataFrame = functions.endog_stationary_distribution(pol_func    = pol_func,
                                                                          pol_idx     = pol_idx,
                                                                          state_grid  = state_grid,
                                                                          a_grid      = a_grid,
                                                                          joint_trans = joint_trans)
        
        mod_res  : pd.DataFrame = functions.full_model_result(pol_func    = pol_func,
                                                              state_grid  = state_grid,
                                                              stat_dist   = stat_dist,
                                                              df_val_func = utility,
                                                              p           = p,
                                                              ModelPar    = self.ModelPar)
        self.mod_res = mod_res
        
    def solve_representative_household(self):
        
        rep_hh_res = functions.representative_household_wrapper(ModelPar = self.ModelPar)
        
        self.rep_hh_res = rep_hh_res
        
    def inner_solution_wrapper(self,p,r):
        
        print("")
        print(f"Trying p={p:.4f}.")
        
        self.solve_firm_side(p=p,r=r)
        self.solve_household_side(p=p,r=r)
        
        residual = functions.inner_loop_residual(p        = p,
                                                 w        = self.w,
                                                 s        = self.s,
                                                 I        = self.I,
                                                 mod_res  = self.mod_res,
                                                 ModelPar = self.ModelPar,
                                                 CalibPar = self.CalibPar)
        
        return abs(residual)
    
    def outer_solution_wrapper(self,r):
        
        print("")
        print(f"Trying r={r:.4f}.")
        
        self.inner_loop_solver(r=r)
        
        residual = functions.outer_loop_residual(p        = self.p, 
                                                 r        = r, 
                                                 mod_res  = self.mod_res, 
                                                 ModelPar = self.ModelPar, 
                                                 CalibPar = self.CalibPar)
        return abs(residual)
        
    def inner_loop_solver(self, r):
    
        obj_func = lambda p: self.inner_solution_wrapper(p, r=r)
        
        if hasattr(self, 'p') and self.p is not None:
            lb = max(self.CalibPar.inner_loop_p_lb, self.p - self.CalibPar.inner_loop_marg)
            ub = min(self.CalibPar.inner_loop_p_ub, self.p + self.CalibPar.inner_loop_marg)
        else:
            lb = self.CalibPar.inner_loop_p_lb
            ub = self.CalibPar.inner_loop_p_ub
        
        res = minimize_scalar(obj_func, 
                              bounds =(lb, ub), 
                              method ='bounded',
                              tol    = self.CalibPar.inner_loop_eps)
        
        self.p        : float = res.x
        self.inner_res = res
        
        if res.fun<self.CalibPar.inner_loop_eps: print(f"Inner loop successfull for p={self.p:.4f}.")
        else                                   : print(f"Inner loop unsuccessfull for r={r:.4f}. ")
        
    def outer_loop_solver(self):
        
        start = time.time()
        
        obj_func = lambda r: self.outer_solution_wrapper(r)
        r_ub = 1/self.ModelPar.δ-1
        
        res = minimize_scalar(obj_func, 
                              bounds =(self.CalibPar.outer_loop_r_lb, r_ub), 
                              method ='bounded',
                              tol    = self.CalibPar.outer_loop_eps)
        
        self.r        : float = res.x
        self.outer_res = res
        
        if res.fun<self.CalibPar.outer_loop_eps: print(f"Outer loop successfull for r={self.r:.4f}.")
        else                                   : print("Outer loop unsuccessfull for current parameters.")
        
        end = time.time()
        
        print(f"Model solved in {(end-start)/60}m.")
        print("")
        print("Results:")
        print(f"I: {self.I} | p: {self.p} | r: {self.r} | s: {self.s} | w: {self.w}")
        print(f"Outer residual: {self.outer_res.fun}.")
        print(f"Inner residual: {self.inner_res.fun}.")
        
    def weighted_percentile_aggregation(self,x_var, y_var, n_bins=50):
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
                'bin'       : i + 1,
                'x_low'     : x_sorted[mask][0] if mask.any() else np.nan,
                'x_high'    : x_sorted[mask][-1] if mask.any() else np.nan,
                'x_mean'    : x_mean,
                'y_mean'    : y_mean,
                'pop_share' : w_group.sum()
            })
        
        df: pd.DataFrame = pd.DataFrame(rows)
        
        return df
        
    def economy_statistics(self,iq_top=90,iq_bottom=10):
        
        if type(self.mod_res) == type(None):
            raise ValueError('Model results are non-existent. Please solve the model once to run the economy statistics.')
        
        mod_res: pd.DataFrame = self.mod_res.copy()
        
        # Income, income decomposition & savings rate
        mod_res['y']        = (mod_res['a_0']*self.r) + (mod_res['z'] * (np.where(mod_res['skill_type'] == 'L', self.w, self.s)))
        mod_res['y_decomp'] = (mod_res['a_0']*self.r)/mod_res['y']
        mod_res['s']        = (mod_res['a_1']-mod_res['a_0'])/mod_res['y']
        
        # CEV
        psi_p         : float     = functions.psi(p=self.p,ModelPar=self.ModelPar)
        numerator     : pd.Series = mod_res['V']*(1-self.ModelPar.σ)*(1-self.ModelPar.δ)
        exp           : float     = 1/(1-self.ModelPar.σ)
        
        mod_res['c_eq'] = (numerator**exp)/psi_p
        
        # Aggregate variables
        aggregate_K  : float = (mod_res['a_0'] * mod_res['dens']).sum()
        aggregate_C  : float = (mod_res['c'] * mod_res['dens']).sum()
        aggregate_C_x: float = (mod_res['c_x'] * mod_res['dens']).sum()
        aggregate_C_y: float = (mod_res['c_y'] * mod_res['dens']).sum()
        aggregate_ceq: float = (mod_res['c_eq'] * mod_res['dens']).sum()
        aggregate_V  : float = (mod_res['V'] * mod_res['dens']).sum()
        
        # Income Share GDP
        L       : pd.DataFrame = mod_res.loc[mod_res['skill_type']=='L']
        mean_V_L: float        = (L['V'] * L['dens']).sum()/L['dens'].sum()
        L       : float        = (L['z']*L['dens']).sum()
        
        H       : pd.DataFrame = mod_res.loc[mod_res['skill_type']=='H']
        mean_V_H: float        = (H['V'] * H['dens']).sum()/H['dens'].sum()
        H       : float        = (H['z']*H['dens']).sum()
        
        ls_share: float = (self.w*L)
        hs_share: float = (self.s*H)
        k_share : float = (self.r*aggregate_K)
        ss_gdp  : float = ls_share+hs_share+k_share
        ls_share: float = (self.w*L)/ss_gdp
        hs_share: float = (self.s*H)/ss_gdp
        k_share : float = (self.r*aggregate_K)/ss_gdp
        
        # Skill premium
        skill_prem: float = self.s/self.w
        
        # Income inequality
        income_gini : float = functions.weighted_gini(x=mod_res['y'], weights=mod_res['dens'])
        wealth_gini : float = functions.weighted_gini(x=mod_res['a_0'], weights=mod_res['dens'])
        ceq_gini    : float = functions.weighted_gini(x=mod_res['c_eq'], weights=mod_res['dens'])
        c_gini      : float = functions.weighted_gini(x=mod_res['c'], weights=mod_res['dens'])
        
        income_top  : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_top)[1]
        income_bot  : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_bottom)[1]
        income_ratio: float = income_top/income_bot
        income_conc : float = functions.weighted_percentile(x=mod_res['y'], weights=mod_res['dens'], q=iq_top)[0]
        
        wealth_top  : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_top)[1]
        wealth_bot  : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_bottom)[1]
        wealth_ratio: float = wealth_top/wealth_bot
        wealth_conc : float = functions.weighted_percentile(x=mod_res['a_0'], weights=mod_res['dens'], q=iq_top)[0]
        
        ret: dict ={'K'                   :aggregate_K,
                    'mean_V'              :aggregate_V,
                    'mean_V_L'            :mean_V_L,
                    'mean_V_H'            :mean_V_H,
                    'C'                   :aggregate_C,
                    'C_x'                 :aggregate_C_x,
                    'C_y'                 :aggregate_C_y,
                    'mean_c_eq'           :aggregate_ceq,
                    'low_skill_share'     :ls_share,
                    'high_skill_share'    :hs_share,
                    'k_share'             :k_share,
                    'skill_premium'       :skill_prem,
                    'income_gini'         :income_gini,
                    'wealth_gini'         :wealth_gini,
                    'ceq_gini'            :ceq_gini,
                    'c_gini'              :c_gini,
                    'income_ratio'        :income_ratio,
                    'wealth_ratio'        :wealth_ratio,
                    'income_concentration':income_conc,
                    'wealth_concentration':wealth_conc,
                    'I'                   :self.I,
                    'w'                   :self.w,
                    's'                   :self.s,
                    'p'                   :self.p,
                    'r'                   :self.r}
        
        self.economy_stats = ret
        
        return ret
