# -*- coding: utf-8 -*-
"""
Nested-CES GMM runner: estimate θ, w_star and β_eff by differential evolution.
Save the best valid result as post_gmm_params_ces.json.
"""

import numpy as np
import json
from GeneralEquilibriumModel import TypeCalibParameters,TypeModelParameters
from config import DATA_PARAMS
from GMM import run_gmm

with open(DATA_PARAMS / 'pre_gmm_params_ces.json','r',encoding='utf-8') as f:
    pre_gmm_params = json.load(f)

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

ModelPar = TypeModelParameters(
    α            = pre_gmm_params['parameters']['α'],
    γ            = pre_gmm_params['parameters']['γ'],
    ψ            = pre_gmm_params['parameters']['ψ'],
    χ            = pre_gmm_params['parameters']['χ'],
    β_eff        = np.nan,
    τ            = pre_gmm_params['parameters']['τ'],
    w_star       = np.nan,
    θ            = np.nan,
    σ            = pre_gmm_params['parameters']['σ'],
    δ            = pre_gmm_params['parameters']['δ'],
    ϱ            = pre_gmm_params['parameters']['ϱ'],
    σ_ϵ          = pre_gmm_params['parameters']['σ_ϵ'],
    π_LL         = pre_gmm_params['parameters']['π_LL'],
    π_HH         = pre_gmm_params['parameters']['π_HH'],
    M            = pre_gmm_params['parameters']['M'],
    ξ            = pre_gmm_params['parameters']['ξ'],
    rebate_share = pre_gmm_params['parameters']['rebate_share'])

data_moments = {
    'skill_premium': pre_gmm_params['moments']['skill_premium'],
    'w_to_wstar': pre_gmm_params['moments']['w_to_wstar'],
    'I': pre_gmm_params['moments']['I']}

W      = np.eye(3)
bounds = [(1e-6,40),(0.05,0.7),(1.0,7.0)]  # θ, w_star, β_eff

# --- Stage 1: Differential Evolution ---

print(f"\n{'='*65}")
print(f"  Starting GMM estimation — algorithm: differential_evolution")
print(f"{'='*65}\n")

_,best_de = run_gmm(ModelPar=ModelPar,CalibPar=CalibPar,data_moments=data_moments,W=W,bounds=bounds,algorithm='differential_evolution')
if best_de['params'] is None:
    raise RuntimeError('No valid nested-CES equilibrium found; parameters were not saved.')

all_bests = {'differential_evolution': best_de}

# --- Select winner and save ---
best_algo    = min(all_bests,key=lambda k: all_bests[k]['obj'])
best_overall = all_bests[best_algo]

print(f"\nBest algorithm: {best_algo}  (obj={best_overall['obj']:.8e})")

θ,w_star,β_eff = best_overall['params']

post_gmm = {
    'production': 'nested_ces',
    'parameters': {
        'σ': pre_gmm_params['parameters']['σ'],
        'δ': pre_gmm_params['parameters']['δ'],
        'ϱ': pre_gmm_params['parameters']['ϱ'],
        'σ_ϵ': pre_gmm_params['parameters']['σ_ϵ'],
        'γ': pre_gmm_params['parameters']['γ'],
        'M': pre_gmm_params['parameters']['M'],
        'π_LL': pre_gmm_params['parameters']['π_LL'],
        'π_HH': pre_gmm_params['parameters']['π_HH'],
        'α': pre_gmm_params['parameters']['α'],
        'ψ': pre_gmm_params['parameters']['ψ'],
        'χ': pre_gmm_params['parameters']['χ'],
        'w_star': float(w_star),
        'θ': float(θ),
        'β_eff': float(β_eff),
        'τ': float(pre_gmm_params['parameters']['τ']),
        'ξ': float(pre_gmm_params['parameters']['ξ']),
        'rebate_share': float(pre_gmm_params['parameters']['rebate_share']),
    }
}

with open(DATA_PARAMS / 'post_gmm_params_ces.json','w',encoding='utf-8') as f:
    json.dump(post_gmm,f,indent=4,ensure_ascii=False)

print(f"Saved post_gmm_params_ces.json")
