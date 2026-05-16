# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:00:52 2026

@author: lfval
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR   = ROOT_DIR / "data"
DATA_RAW   = DATA_DIR / "raw"
DATA_FINAL = DATA_DIR / "final"
OUTPUTS    = ROOT_DIR / "outputs"
LOG          = ROOT_DIR / "log"
LOG_ISOLATED = LOG / "isolated_runs"
LOG_GMM      = LOG / "gmm"

for folder in [DATA_RAW, DATA_FINAL, OUTPUTS, LOG_ISOLATED, LOG_GMM]:
    folder.mkdir(parents=True, exist_ok=True)