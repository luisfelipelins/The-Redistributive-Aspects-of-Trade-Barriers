# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:00:52 2026

@author: lfval
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR           = ROOT_DIR / "data"
DATA_RAW           = DATA_DIR / "raw"
DATA_FINAL         = DATA_DIR / "final"
DATA_PARAMS        = DATA_DIR / "parameters"
DATA_INT           = DATA_DIR / "intermediate"
OUTPUTS            = ROOT_DIR / "outputs"
OUTPUTS_MOTIVATION = OUTPUTS / 'motivation'
OUTPUTS_GMM        = OUTPUTS / 'gmm'
LOG                = ROOT_DIR / "log"
LOG_ISOLATED       = LOG / "isolated_runs"
LOG_GMM            = LOG / "gmm"
CONFIG             = ROOT_DIR / 'config'

for folder in [DATA_RAW, DATA_FINAL, DATA_PARAMS, DATA_INT, OUTPUTS, OUTPUTS_MOTIVATION, OUTPUTS_GMM, LOG_ISOLATED, LOG_GMM, CONFIG]:
    folder.mkdir(parents=True, exist_ok=True)