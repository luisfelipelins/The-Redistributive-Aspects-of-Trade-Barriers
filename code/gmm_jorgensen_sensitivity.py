# -*- coding: utf-8 -*-
import numpy as np
import json
from GeneralEquilibriumModel import TypeCalibParameters,TypeModelParameters,GeneralEquilibriumModel
from config import *
from GMM import gmm_model_moments,MOMENT_NAMES,PARAM_NAMES
from dataclasses import replace
from ces_production import firm_at_interest
import functions
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from config import OUTPUTS

with open(DATA_PARAMS / 'pre_gmm_params_ces.json','r',encoding='utf-8') as f:
    pre_gmm = json.load(f)
with open(DATA_PARAMS / 'post_gmm_params_ces.json','r',encoding='utf-8') as f:
    post_gmm = json.load(f)

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

p = post_gmm['parameters']

ModelPar = TypeModelParameters(
    α            = p['α'],
    γ            = p['γ'],
    ψ            = p['ψ'],
    χ            = p['χ'],
    β_eff        = np.nan,
    w_star       = np.nan,
    θ            = np.nan,
    σ            = p['σ'],
    δ            = p['δ'],
    ϱ            = p['ϱ'],
    σ_ϵ          = p['σ_ϵ'],
    π_LL         = p['π_LL'],
    π_HH         = p['π_HH'],
    M            = p['M'],
    τ            = p['τ'],
    ξ            = p['ξ'],
    rebate_share = p['rebate_share'])

data_moments = {
    'skill_premium': pre_gmm['moments']['skill_premium'],
    'w_to_wstar': pre_gmm['moments']['w_to_wstar'],
    'I': pre_gmm['moments']['I']}

estimated_values: np.ndarray = np.array([p[name] for name in PARAM_NAMES])
param_names : list       = PARAM_NAMES
moment_names             = MOMENT_NAMES

log_dir = LOG_GMM / 'jacobian_ces'
log_dir.mkdir(exist_ok=True)

h = 1e-3  # Perturbations must exceed equilibrium-solver noise.

#############################################################################################
### --- Jorgensen (2023) Sensitivity of Estimated Parameters to Calibrated Parameters --- ###
#############################################################################################

calibrated_names  = ['σ','δ','ϱ','σ_ϵ','α','γ','ψ','χ','M','π_LL','π_HH']
calibrated_values = np.array([p[k] for k in calibrated_names])
calibrated_base   = {k: p[k] for k in calibrated_names}


def _make_model_par(gv: dict) -> TypeModelParameters:
    """
    Construct a calibrated-parameter perturbation with tariff policy held fixed.

    Parameters
    ----------
    gv : dict
        Perturbed externally calibrated parameters, including both CES nests.

    Returns
    -------
    TypeModelParameters
        Parameters with baseline tariff, progressivity and rebate share.
    """

    return TypeModelParameters(α            = gv['α'],
                               γ            = gv['γ'],
                               ψ            = gv['ψ'],
                               χ            = gv['χ'],
                               β_eff        = np.nan,
                               w_star       = np.nan,
                               θ            = np.nan,
                               σ            = gv['σ'],
                               δ            = gv['δ'],
                               ϱ            = gv['ϱ'],
                               σ_ϵ          = gv['σ_ϵ'],
                               π_LL         = gv['π_LL'],
                               π_HH         = gv['π_HH'],
                               M            = gv['M'],
                               τ            = p['τ'],
                               ξ            = p['ξ'],
                               rebate_share = p['rebate_share'])


D_n = np.zeros((len(moment_names),len(calibrated_values)))

for l,(key,val) in enumerate(zip(calibrated_names,calibrated_values)):
    step_size = h * abs(val)

    gf      = dict(calibrated_base)
    gf[key] = val + step_size

    gb      = dict(calibrated_base)
    gb[key] = val - step_size

    g_fwd    = gmm_model_moments(estimated_values,_make_model_par(gf),CalibPar,data_moments,log_dir,f'sens_fwd_{l}.log')
    g_bwd    = gmm_model_moments(estimated_values,_make_model_par(gb),CalibPar,data_moments,log_dir,f'sens_bwd_{l}.log')
    D_n[:,l] = (g_fwd - g_bwd) / (2.0 * step_size)

G_n = np.zeros((len(moment_names),len(estimated_values)))

for k,(name_k,val_k) in enumerate(zip(param_names,estimated_values)):
    step_size = h * abs(val_k)

    tf    = estimated_values.copy()
    tf[k] = val_k + step_size

    tb    = estimated_values.copy()
    tb[k] = val_k - step_size

    t_fwd    = gmm_model_moments(tf,ModelPar,CalibPar,data_moments,log_dir,f'jac_fwd_{k}.log')
    t_bwd    = gmm_model_moments(tb,ModelPar,CalibPar,data_moments,log_dir,f'jac_bwd_{k}.log')
    G_n[:,k] = (t_fwd - t_bwd) / (2.0 * step_size)

# S matrix - Approximation according to Jorgensen (2023)'s Corollary 1

S_hat = -np.linalg.solve(G_n,D_n)

# E matrix (elasticity)
E_hat = S_hat * calibrated_values[np.newaxis,:] / estimated_values[:,np.newaxis]
E_df  = pd.DataFrame(E_hat,index=param_names,columns=calibrated_names).drop(columns=['π_LL','π_HH'])

abs_max = np.abs(E_df.values).max()
fig,ax  = plt.subplots(figsize=(6,6))

sns.heatmap(E_df,annot=True,fmt='.2f',cmap='RdBu_r',center=0,vmin=-abs_max,vmax=abs_max,linewidths=0.5,linecolor='white',ax=ax)
ax.set_xlabel('Calibrated parameters')
ax.set_ylabel('Estimated parameters ($\\hat{\\vartheta}$)')

plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'jorgensenelasticity.pdf')
plt.close()

######################################################################
### --- Checking narrative-hold regions given sensitivity of γ --- ###
######################################################################

gamma_p_base = p['γ']
idx_gamma_p  = calibrated_names.index('γ')
idx_theta    = param_names.index('θ')
theta_grid   = np.linspace(0.5,27,600)
I_grid       = np.linspace(0.01,0.99,600)
tt,ii        = np.meshgrid(theta_grid,I_grid)
G_grid       = 1 / (tt * (1 - ii)) + 1 / (-np.expm1(-tt * ii))
skill_dist   = functions.calculate_stationary_distribution_eigenvector(functions.create_LH_skill_mat(ModelPar))
r_rep        = 1 / ModelPar.δ - 1
scenarios    = [(gamma_p_base,'#163A63'),(gamma_p_base - 0.01,'#B23A48'),(gamma_p_base + 0.01,'#176B47')]

for filename,selected in [('sensboundaries.pdf',scenarios),('sensboundaries_baseline.pdf',scenarios[:1])]:
    fig,ax = plt.subplots(figsize=(6,6))
    handles = []

    for gamma_value,color in selected:
        par   = replace(ModelPar,γ=gamma_value)
        theta = estimated_values[idx_theta] + S_hat[idx_theta,idx_gamma_p] * (gamma_value - gamma_p_base)
        shares = [firm_at_interest(I,r_rep,skill_dist[1],skill_dist[0],par) for I in I_grid]
        omega_L = np.array([sol['omega_L'] for sol in shares])
        omega_X = np.array([sol['omega_X'] for sol in shares])
        eta_H   = np.array([sol['eta_H'] for sol in shares])
        eta_K   = np.array([sol['eta_K'] for sol in shares])
        Lambda  = ((1 - par.ψ) * omega_X * (1 - par.χ) * eta_H
                   / ((1 - par.χ) * eta_H + (1 - par.ψ) * omega_L * eta_K))

        ax.contour(theta_grid,I_grid,G_grid - 1 / Lambda[:,None],levels=[0],colors=[color],linewidths=2)
        ax.axvline(theta,color=color,ls='--',lw=1.2)
        handles.append(Line2D([0],[0],color=color,lw=2,label=f'γ = {gamma_value:.2f}'))

    ax.axhline(data_moments['I'],color='#555555',ls=':',lw=1.2)
    ax.set_xlabel(r'$\theta$',fontsize=14)
    ax.set_ylabel(r'$I$',fontsize=14)
    ax.legend(handles=handles,loc=0)
    ax.grid(linestyle='--',alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUTS_GMM / filename)
    plt.close(fig)
