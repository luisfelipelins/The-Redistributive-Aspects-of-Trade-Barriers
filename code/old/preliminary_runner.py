# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

import pickle
from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from config import *

CalibPar = TypeCalibParameters(rh_N              = 11   ,
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

ModelPar = TypeModelParameters(α      = 0.400,
                               γ      = 0.330,
                               β      = 4    ,
                               w_star = 0.300,
                               θ      = 0.500,
                               σ      = 2.000,
                               δ      = 0.960,
                               ρ      = 0.950,
                               σ_ϵ    = 0.210,
                               π_LL   = 0.95  ,
                               π_HH   = 0.93  ,
                               M      = 1.000)

self = GeneralEquilibriumModel(ModelPar, CalibPar, log_dir="auto")
# self = GeneralEquilibriumModel(ModelPar, CalibPar, log_dir= LOG_GMM / f"eval_0001")


self.outer_loop_solver()

