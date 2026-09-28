# -*- coding: utf-8 -*-
import numpy as np
import json
from GeneralEquilibriumModel import TypeCalibParameters,TypeModelParameters
from config import DATA_PARAMS,LOG_GMM
from GMM import run_gmm

with open(DATA_PARAMS / 'pre_gmm_params.json','r') as f:
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
    α            = np.nan,
    γ            = pre_gmm_params['parameters']['γ'],
    β_eff        = np.nan,
    w_star       = np.nan,
    θ            = np.nan,
    σ            = pre_gmm_params['parameters']['σ'],
    δ            = pre_gmm_params['parameters']['δ'],
    ρ            = pre_gmm_params['parameters']['ρ'],
    σ_ϵ          = pre_gmm_params['parameters']['σ_ϵ'],
    π_LL         = pre_gmm_params['parameters']['π_LL'],
    π_HH         = pre_gmm_params['parameters']['π_HH'],
    M            = pre_gmm_params['parameters']['M'],
    τ            = pre_gmm_params['parameters']['τ'],
    ξ            = pre_gmm_params['parameters']['ξ'],
    rebate_share = pre_gmm_params['parameters']['rebate_share'])

data_moments = {
    'high_skill_share': pre_gmm_params['moments']['HS_share'],
    'skill_premium': pre_gmm_params['moments']['skill_premium'],
    'w_to_wstar': pre_gmm_params['moments']['w_to_wstar'],
    'I': pre_gmm_params['moments']['I']}

W      = np.diag([1.0,1.0,1.0,1.0])
bounds = [(0.1,0.55),(0.05,0.7),(1e-6,40),(1.0,7.0)]  # α, w_star, θ, β_eff

resume_dir = sorted(LOG_GMM.glob('gmm_run_*'))[-1]

print(f"\n{'='*65}")
print(f"  Resuming GMM estimation from {resume_dir.name}")
print(f"{'='*65}\n")

_,best_de = run_gmm(
    ModelPar=ModelPar,
    CalibPar=CalibPar,
    data_moments=data_moments,
    W=W,
    bounds=bounds,
    algorithm='differential_evolution',
    resume_from=resume_dir)

α,w_star,θ,β_eff = best_de['params']

post_gmm = {
    'parameters': {
        'σ': pre_gmm_params['parameters']['σ'],
        'δ': pre_gmm_params['parameters']['δ'],
        'ρ': pre_gmm_params['parameters']['ρ'],
        'σ_ϵ': pre_gmm_params['parameters']['σ_ϵ'],
        'γ': pre_gmm_params['parameters']['γ'],
        'M': pre_gmm_params['parameters']['M'],
        'π_LL': pre_gmm_params['parameters']['π_LL'],
        'π_HH': pre_gmm_params['parameters']['π_HH'],
        'α': float(α),
        'w_star': float(w_star),
        'θ': float(θ),
        'β_eff': float(β_eff),
        'τ': float(pre_gmm_params['parameters']['τ']),
        'ξ': float(pre_gmm_params['parameters']['ξ']),
        'rebate_share': float(pre_gmm_params['parameters']['rebate_share']),
    }
}

with open(DATA_PARAMS / 'post_gmm_params.json','w',encoding='utf-8') as f:
    json.dump(post_gmm,f,indent=4,ensure_ascii=False)

print(f"Saved post_gmm_params.json")
