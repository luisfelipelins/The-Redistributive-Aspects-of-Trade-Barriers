# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import pickle
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel

if __name__ == '__main__':

    CalibPar = TypeCalibParameters(rh_N             = 7    ,
                                    rh_r             = 2    , 
                                    rh_c             = 0    ,
                                    sup_side_eq_eps  = 1e-5 ,
                                    vfi_lb           = 0    ,
                                    vfi_ubmul        = 15    ,
                                    vfi_N            = 250  ,
                                    vfi_eps          = 1e-5 ,
                                    vfi_howard_steps = 20   ,
                                    gmc_eps          = 1e-3 ,
                                    kmc_eps          = 1e-3 ,
                                    p_init_guess     = 1,
                                    r_init_guess     = 0.01,
                                    inner_loop_eps   = 1e-3,
                                    inner_loop_p_lb  = 0.1,
                                    inner_loop_p_ub  = 1.2,
                                    inner_loop_marg  = 0.3,
                                    outer_loop_eps   = 1e-3,
                                    outer_loop_r_lb  = 0.001)
    
    results = {}
    
    for b in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]:
        ModelPar = TypeModelParameters(α_x    = 0.400,
                                       α_y    = 0.250,
                                       γ      = 0.200,
                                       β      = b,
                                       w_star = 1.000,
                                       θ      = 0.500,
                                       η      = 0.500,
                                       σ      = 2.000,
                                       δ      = 0.960,
                                       ρ      = 0.900,
                                       σ_ϵ    = 0.150,
                                       π_LL   = 0.700,
                                       π_HH   = 0.800,
                                       M      = 1.000)
        mod = GeneralEquilibriumModel(ModelPar = ModelPar,
                                      CalibPar = CalibPar)
    
        try:
            mod.outer_loop_solver()
            results[b] = mod
        except Exception:
            pass
        
    with open('res_wt.p','wb') as p:
        pickle.dump(results,p,pickle.HIGHEST_PROTOCOL)
        
    for b in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]:
        ModelPar = TypeModelParameters(α_x    = 0.400,
                                       α_y    = 0.250,
                                       γ      = 0.200,
                                       β      = b,
                                       w_star = 1.000,
                                       θ      = 0.500,
                                       η      = 0.500,
                                       σ      = 2.000,
                                       δ      = 0.960,
                                       ρ      = 0.900,
                                       σ_ϵ    = 0.150,
                                       π_LL   = 1.000,
                                       π_HH   = 1.000,
                                       M      = 1.000)
        mod = GeneralEquilibriumModel(ModelPar = ModelPar,
                                      CalibPar = CalibPar)
    
        try:
            mod.outer_loop_solver()
            results[b] = mod
        except Exception:
            pass
        
    with open('res_nt.p','wb') as p:
        pickle.dump(results,p,pickle.HIGHEST_PROTOCOL)