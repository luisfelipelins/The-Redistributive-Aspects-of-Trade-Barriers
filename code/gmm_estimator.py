# -*- coding: utf-8 -*-
"""
GMM estimation runner — estimates the model using three global optimizers:
  1. Differential Evolution (baseline)
  2. CRS via DIRECT
  3. Simulated Annealing via dual_annealing
"""

import numpy as np
from GMM import run_gmm, _make_calib_params

data_moments = {
    'high_skill_share': 0.386,
    'skill_premium'   : 1.85 ,
    'w_to_wstar'      : 2.20 ,
    'I'               : 0.22 ,
}

W       = np.diag([1.0, 1.0, 1.0, 1.0])
CalibPar = _make_calib_params()

# α, w_star, θ, β
bounds = [(0.35, 0.5), (0.05, 0.7), (1e-6, 1.3), (1.1, 7.0)]

algorithms = ['differential_evolution', 'crs', 'simulated_annealing']

for algo in algorithms:
    print(f"\n{'='*65}")
    print(f"  Starting GMM estimation — algorithm: {algo}")
    print(f"{'='*65}\n")
    run_gmm(CalibPar=CalibPar, data_moments=data_moments, W=W,
            bounds=bounds, algorithm=algo)
