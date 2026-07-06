# -*- coding: utf-8 -*-
import numpy as np
import json
from GeneralEquilibriumModel import TypeCalibParameters, TypeModelParameters, GeneralEquilibriumModel
from config import *
from GMM import gmm_model_moments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from config import OUTPUTS

with open(DATA_PARAMS / 'pre_gmm_params.json', 'r', encoding='utf-8') as f:
    pre_gmm = json.load(f)
with open(DATA_PARAMS / 'post_gmm_params.json', 'r', encoding='utf-8') as f:
    post_gmm = json.load(f)

CalibPar = TypeCalibParameters(
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
        outer_loop_r_lb  = 0.001)

p = post_gmm['parameters']

ModelPar = TypeModelParameters(
        α      = np.nan  ,
        γ      = p['γ']  ,
        β      = np.nan  ,
        w_star = np.nan  ,
        θ      = np.nan  ,
        σ      = p['σ']  ,
        δ      = p['δ']  ,
        ρ      = p['ρ']  ,
        σ_ϵ    = p['σ_ϵ'],
        π_LL   = p['π_LL'],
        π_HH   = p['π_HH'],
        M      = p['M']  ,
        t_form = 'exponential')

data_moments = {
    'high_skill_share': pre_gmm['moments']['HS_share'],
    'skill_premium'   : pre_gmm['moments']['skill_premium'],
    'w_to_wstar'      : pre_gmm['moments']['w_to_wstar'],
    'I'               : pre_gmm['moments']['I']}

theta_values = np.array([p['α'], p['w_star'], p['θ'], p['β']])
param_names  = ['α', 'w_star', 'θ', 'β']
moment_names = ['high_skill_share', 'w_to_wstar', 'I', 'skill_premium']

log_dir = LOG_GMM / 'jacobian'
log_dir.mkdir(exist_ok=True)

h = 1e-9

#############################################################################################
### --- Jorgensen (2023) Sensitivity of Estimated Parameters to Calibrated Parameters --- ###
#############################################################################################

gamma_names  = ['σ', 'δ', 'ρ', 'σ_ϵ', 'γ', 'M', 'π_LL', 'π_HH']
gamma_values = np.array([p[k] for k in gamma_names])
gamma_base   = {k: p[k] for k in gamma_names}

def _make_model_par(gv):
    return TypeModelParameters(α      = np.nan    , 
                               γ      = gv['γ']   , 
                               β      = np.nan    ,
                               w_star = np.nan    , 
                               θ      = np.nan    , 
                               σ      = gv['σ']   ,
                               δ      = gv['δ']   , 
                               ρ      = gv['ρ']   , 
                               σ_ϵ    = gv['σ_ϵ'] ,
                               π_LL   = gv['π_LL'], 
                               π_HH   = gv['π_HH'], 
                               M      = gv['M']   ,
                               t_form = 'exponential')

D_n = np.zeros((len(moment_names), len(gamma_values)))

for l, (key, val) in enumerate(zip(gamma_names, gamma_values)):
    step_size = h * abs(val)

    gf     = dict(gamma_base)
    gf[key] = val + step_size

    gb     = dict(gamma_base)
    gb[key] = val - step_size

    g_fwd  = gmm_model_moments(theta_values, _make_model_par(gf), CalibPar, data_moments, log_dir, f'sens_fwd_{l}.log')
    g_bwd  = gmm_model_moments(theta_values, _make_model_par(gb), CalibPar, data_moments, log_dir, f'sens_bwd_{l}.log')
    D_n[:, l] = (g_fwd - g_bwd) / (2.0 * step_size)

G_n = np.zeros((len(moment_names), len(theta_values)))

for k, (name_k, val_k) in enumerate(zip(param_names, theta_values)):
    step_size = h * abs(val_k)

    tf    = theta_values.copy()
    tf[k] = val_k + step_size

    tb    = theta_values.copy()
    tb[k] = val_k - step_size

    t_fwd = gmm_model_moments(tf, ModelPar, CalibPar, data_moments, log_dir, f'jac_fwd_{k}.log')
    t_bwd = gmm_model_moments(tb, ModelPar, CalibPar, data_moments, log_dir, f'jac_bwd_{k}.log')
    G_n[:, k] = (t_fwd - t_bwd) / (2.0 * step_size)

# S matrix - Approximation according to Jorgensen (2023)'s Corollary 1
S_hat = -np.dot(np.linalg.inv(G_n),D_n)

# E matrix (elasticity)
E_hat = S_hat * gamma_values[np.newaxis, :] / theta_values[:, np.newaxis]
E_df  = pd.DataFrame(E_hat, index=param_names, columns=gamma_names)

abs_max = np.abs(E_hat).max()
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    E_df, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, vmin=-abs_max, vmax=abs_max,
    linewidths=0.5, linecolor='white', ax=ax
)
ax.set_title(r'Elasticity of Estimated Parameters to Calibrated Parameters')
ax.set_xlabel('Calibrated parameters ($\\gamma$)')
ax.set_ylabel('Estimated parameters ($\\hat{\\theta}$)')
plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'jorgensenelasticity.pdf')
plt.close()

######################################################################
### --- Checking narrative-hold regions given sensitivity of δ --- ###
######################################################################

delta_base      = p['δ']
delta_scenarios = {'δ=0.94': 0.94, 'δ=0.96': 0.96}
idx_delta       = gamma_names.index('δ')
s_delta         = S_hat[:, idx_delta]

theta_grid = np.linspace(0.5, 27, 600)
I_grid     = np.linspace(0.01, 0.99, 600)
tt, ii     = np.meshgrid(theta_grid, I_grid)
G_grid     = 1 / (tt * (1 - ii)) + 1 / (1 - np.exp(-tt * ii))

gamma_val = p['γ']
idx_alpha = param_names.index('α')
idx_theta = param_names.index('θ')

delta_plot_scenarios = [
    ('δ=0.95 (base)', delta_base, 'tab:blue',   '-',  '-'),
    ('δ=0.94',        0.94,       'tab:orange',  '--', ':'),
    ('δ=0.96',        0.96,       'tab:green',   '--', ':'),
]

fig, ax = plt.subplots(figsize=(6,6))
legend_elements = []

for label, delta_val, color, ls_contour, ls_vline in delta_plot_scenarios:
    dd     = delta_val - delta_base
    alpha  = theta_values[idx_alpha] + S_hat[idx_alpha, idx_delta] * dd
    theta  = theta_values[idx_theta] + S_hat[idx_theta, idx_delta] * dd
    thr    = (1 - gamma_val) / alpha

    ax.contour(theta_grid, I_grid, G_grid, levels=[thr],
               colors=[color], linewidths=2, linestyles=[ls_contour])
    ax.axvline(theta, color=color, ls=ls_vline, lw=1.2)

    legend_elements.append(Line2D([0], [0], color=color, lw=2, ls=ls_contour,
        label=f'{label}: (1−γ)/α={thr:.3f}'))
    legend_elements.append(Line2D([0], [0], color=color, lw=1.2, ls=ls_vline,
        label=f'{label}: θ̂={theta:.4f}'))

ax.annotate(r'$\hat{w}/\hat{\beta} > 0$', xy=(2, 0.8), fontsize=10, color='gray', ha='center')
ax.annotate(r'$\hat{w}/\hat{\beta} < 0$', xy=(14,  0.5), fontsize=10, color='gray', ha='center')
ax.set_xlabel(r'$\theta$', fontsize=14)
ax.set_ylabel(r'$I$', fontsize=14)
ax.set_title(r'Narrative-hold region w/ representative agent', fontsize=15)
ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'sensboundaries.pdf')
plt.close()
