# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:09:02 2026

@author: lfval
"""

from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from config import *

CalibPar = TypeCalibParameters(rh_N             = 11   ,
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

ModelPar = TypeModelParameters(α      = 0.385675,
                               γ      = 0.280000,
                               β      = 2.199644,
                               w_star = 0.259112,
                               θ      = 0.001608,
                               σ      = 2.000000,
                               δ      = 0.960000,
                               ρ      = 0.950000,
                               σ_ϵ    = 0.210000,
                               π_LL   = 0.900000,
                               π_HH   = 0.880000,
                               M      = 1.000)

self = GeneralEquilibriumModel(ModelPar, CalibPar, log_dir="auto")

self.outer_loop_solver()

