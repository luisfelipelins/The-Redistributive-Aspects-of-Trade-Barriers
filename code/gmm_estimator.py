# -*- coding: utf-8 -*-
"""
GMM estimation runner — differential evolution followed by Nelder-Mead and Powell fine-tuning.
The best result across all three methods is saved as post_gmm_params.json.
"""

import numpy as np
import json
from GeneralEquilibriumModel import TypeCalibParameters, TypeModelParameters
from config import DATA_PARAMS
from GMM import run_gmm

with open(DATA_PARAMS/'pre_gmm_params.json', 'r') as f:
    pre_gmm_params = json.load(f)

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

ModelPar = TypeModelParameters(
        α      = np.nan                              ,
        γ      = pre_gmm_params['parameters']['γ']   ,
        β      = np.nan                              ,
        w_star = np.nan                              ,
        θ      = np.nan                              ,
        σ      = pre_gmm_params['parameters']['σ']   ,
        δ      = pre_gmm_params['parameters']['δ']   ,
        ρ      = pre_gmm_params['parameters']['ρ']   ,
        σ_ϵ    = pre_gmm_params['parameters']['σ_ϵ'] ,
        π_LL   = pre_gmm_params['parameters']['π_LL'],
        π_HH   = pre_gmm_params['parameters']['π_HH'],
        M      = pre_gmm_params['parameters']['M']   ,
        t_form = 'exponential'                       )

data_moments = {
    'high_skill_share': pre_gmm_params['moments']['HS_share'],
    'skill_premium'   : pre_gmm_params['moments']['skill_premium'],
    'w_to_wstar'      : pre_gmm_params['moments']['w_to_wstar'],
    'I'               : pre_gmm_params['moments']['I']}

W      = np.diag([1.0, 1.0, 1.0, 1.0])
bounds = [(0.1, 0.55), (0.05, 0.7), (1e-6, 99), (1e-6, 7.0)]  # α, w_star, θ, β

# --- Stage 1: Differential Evolution ---
print(f"\n{'='*65}")
print(f"  Starting GMM estimation — algorithm: differential_evolution")
print(f"{'='*65}\n")
_, best_de = run_gmm(ModelPar=ModelPar, CalibPar=CalibPar, data_moments=data_moments, W=W,
                     bounds=bounds, algorithm='differential_evolution', t_form='exponential')

# --- Stage 2: Fine-tuning from DE best ---
x0 = best_de['params'].tolist()

all_bests = {'differential_evolution': best_de}

for algo in ['nelder_mead', 'powell']:
    print(f"\n{'='*65}")
    print(f"  Starting GMM estimation — algorithm: {algo}")
    print(f"{'='*65}\n")
    _, best = run_gmm(ModelPar=ModelPar, CalibPar=CalibPar, data_moments=data_moments, W=W,
                      x0=x0, algorithm=algo, t_form='exponential')
    all_bests[algo] = best

# --- Select winner and save ---
best_algo    = min(all_bests, key=lambda k: all_bests[k]['obj'])
best_overall = all_bests[best_algo]
print(f"\nBest algorithm: {best_algo}  (obj={best_overall['obj']:.8e})")

α, w_star, θ, β = best_overall['params']

post_gmm = {
    'parameters': {
        'σ'    : pre_gmm_params['parameters']['σ']   ,
        'δ'    : pre_gmm_params['parameters']['δ']   ,
        'ρ'    : pre_gmm_params['parameters']['ρ']   ,
        'σ_ϵ'  : pre_gmm_params['parameters']['σ_ϵ'] ,
        'γ'    : pre_gmm_params['parameters']['γ']   ,
        'M'    : pre_gmm_params['parameters']['M']   ,
        'π_LL' : pre_gmm_params['parameters']['π_LL'],
        'π_HH' : pre_gmm_params['parameters']['π_HH'],
        'α'    : float(α)                            ,
        'w_star': float(w_star)                      ,
        'θ'    : float(θ)                            ,
        'β'    : float(β)                            ,
    }
}

with open(DATA_PARAMS / 'post_gmm_params.json', 'w', encoding='utf-8') as f:
    json.dump(post_gmm, f, indent=4, ensure_ascii=False)

print(f"Saved post_gmm_params.json")