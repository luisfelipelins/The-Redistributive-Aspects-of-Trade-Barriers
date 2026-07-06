# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product, combinations
from GeneralEquilibriumModel import (TypeModelParameters, TypeCalibParameters,GeneralEquilibriumModel)
from config import *

class NoEquilibriumError(Exception):
    '''Raised when capital markets are not clearing.'''
    pass

rerun_estimation = False   # True -> re-run even if grid_search.csv already exists

α_vec      = [0.1, 0.2, 0.3, 0.4, 0.5]
θ_vec      = [1, 5, 10, 15, 20, 25, 30, 35]
w_star_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
β_vec      = [0.001, 0.005, 0.02, 0.03, 0.1, 0.3, 0.5, 0.8, 1.0]

RESULTS_FILE = DATA_INT / 'grid_search.csv'

params  = ['α', 'θ', 'w_star', 'β']

##########################
### --- Estimation --- ###
##########################

if rerun_estimation or not RESULTS_FILE.exists():

    with open(DATA_PARAMS / 'pre_gmm_params.json', 'r') as f:
        calib = json.load(f)['parameters']

    CalibPar = TypeCalibParameters(rh_N=9, 
                                   rh_r=2, 
                                   rh_c=0,
                                   vfi_lb=0, 
                                   vfi_ubmul=15, 
                                   vfi_N=500, 
                                   vfi_eps=1e-5, 
                                   vfi_howard_steps=20,
                                   gmc_eps=1e-3, 
                                   kmc_eps=1e-3,
                                   r_init_guess=0.03, 
                                   inner_loop_eps=1e-7, 
                                   outer_loop_eps=1e-7,
                                   outer_loop_r_lb=0.001)

    combos = list(product(α_vec, θ_vec, w_star_vec, β_vec))
    total  = len(combos)
    print(f"Running {total} combinations "
          f"({len(α_vec)}α × {len(θ_vec)}θ × {len(w_star_vec)}w* × {len(β_vec)}β) ...")

    rows = []
    n_ok = 0
    for i, (α, θ, w_star, β) in enumerate(combos, 1):
        row = {'α': α, 'θ': θ, 'w_star': w_star, 'β': β, 'converged': 0}
        try:
            ModelPar = TypeModelParameters(α=α, 
                                           γ=calib['γ'], 
                                           β=β, 
                                           w_star=w_star, 
                                           θ=θ,
                                           σ=calib['σ'], 
                                           δ=calib['δ'], 
                                           ρ=calib['ρ'], 
                                           σ_ϵ=calib['σ_ϵ'],
                                           π_LL=calib['π_LL'], 
                                           π_HH=calib['π_HH'], 
                                           M=calib['M'],
                                           t_form='exponential')
            mod = GeneralEquilibriumModel(ModelPar=ModelPar, CalibPar=CalibPar,log_dir=None)
            mod.outer_loop_solver()

            if mod.outer_res.fun>mod.CalibPar.outer_loop_eps:
                raise NoEquilibriumError()

            stats = mod.economy_statistics()
            row.update({
                'converged'           : 1,
                'high_skill_share'    : stats['high_skill_share'],
                'low_skill_share'     : stats['low_skill_share'],
                'K/Y'                 : stats['K/Y'],
                'skill_premium'       : stats['skill_premium'],
                'w_to_wstar'          : stats['w_to_wstar'],
                'income_gini'         : stats['income_gini'],
                'wealth_ratio'        : stats['wealth_ratio'],
                'income_ratio'        : stats['income_ratio'],
                'wealth_concentration': stats['wealth_concentration'],
                'income_concentration': stats['income_concentration'],
                'I'                   : stats['I'],
            })
            n_ok += 1
        except Exception as e:
            print(f"  [{i:4d}/{total}] failed."
                  f"α={α}, θ={θ:2g}, w*={w_star}, β={β}: {e}")

        rows.append(row)
        if i % 200 == 0:
            print(f"  {i}/{total} done  ({n_ok} converged so far)")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"\nDone — {n_ok}/{total} converged.  Saved to {RESULTS_FILE}")

else:
    results_df = pd.read_csv(RESULTS_FILE)
    n_ok  = results_df['converged'].sum()
    total = len(results_df)
    print(f"Loaded {total} rows from {RESULTS_FILE}  ({n_ok} converged)")

# Converged subset used for moment charts and excluding skillpremia and w_to_star higher than 10
conv_df = results_df[results_df['converged'] == 1].copy()

###################################
### --- Sensitivity Heatmap --- ###
###################################
moments = ['high_skill_share','skill_premium','w_to_wstar','I']

sens = pd.DataFrame(index=params, columns=moments, dtype=float)
for par in params:
    for mom in moments:
        group_means = conv_df.groupby(par)[mom].mean()
        total_std   = conv_df[mom].std()
        sens.loc[par, mom] = (group_means.quantile(q=0.95) - group_means.quantile(q=0.05)) / total_std

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(sens.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_xlabel('Moments')
ax.set_ylabel('Parameters')
plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'grid_search_sensitivity.pdf')
plt.close()

##########################################
### --- Achieveble Moments Boxplot --- ###
##########################################
moments = ['high_skill_share','skill_premium','w_to_wstar','I']

conv_df = conv_df.loc[conv_df['skill_premium']<10]
conv_df = conv_df.loc[conv_df['w_to_wstar']<10]

fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(10, 5))

for n,c,cn in zip(moments,[(0,0),(0,1),(1,0),(1,1),(0,2),(1,2)],['H-skill Inc. Share','Skill Premium',r'$w$ to $w^*$','Task-offshore Share','Capital to output ratio','Income Gini']):
    ax[c].boxplot(x=conv_df[n])
    ax[c].set_xlabel(cn)

plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'achievable_moments.pdf')
plt.close()

#################################
### --- Main Effects Grid --- ###
#################################

moments = ['high_skill_share','skill_premium','w_to_wstar','I']

moments_norm = conv_df.copy()
for mom in moments:
    lo, hi = conv_df[mom].min(), conv_df[mom].max()
    if hi > lo:
        moments_norm[mom] = (conv_df[mom] - lo) / (hi - lo)

fig, axes = plt.subplots(len(params), len(moments),
                          figsize=(6,6),
                          sharey=True)

for i, par in enumerate(params):
    for j, mom in enumerate(moments):
        ax = axes[i, j]
        group = moments_norm.groupby(par)[mom]
        ax.plot(group.mean().index, group.mean().values,
                marker='o', markersize=3, linewidth=1.2)
        ax.fill_between(group.mean().index,
                        group.mean() - group.std(),
                        group.mean() + group.std(),
                        alpha=0.2)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_title(mom, fontsize=10, pad=4)
        if j == 0:
            ax.set_ylabel(par, fontsize=10)
        ax.tick_params(labelsize=8)

fig.suptitle('Main effects: group mean ± 1 std  (moments normalised to [0,1])',
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUTPUTS_GMM / 'main_effects_grid.pdf')
plt.close()