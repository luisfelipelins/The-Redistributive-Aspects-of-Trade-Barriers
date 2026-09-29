run_motivation  = True
run_ext_calib   = True
run_gmm         = False
run_sensitivity = True
run_exercises   = True

import subprocess
import sys
from pathlib import Path

code_dir = Path(__file__).resolve().parent
scripts  = ['config.py']

if run_motivation:
    scripts.append('motivation.py')
if run_ext_calib:
    scripts.append('external_calibration.py')
if run_gmm:
    scripts.append('gmm_estimator.py')
if run_sensitivity:
    scripts.append('gmm_jorgensen_sensitivity.py')
if run_exercises:
    scripts.extend(['quant_exercises.py','transition.py'])

for script in scripts:
    print(f'Running {script}',flush=True)
    subprocess.run([sys.executable,'-X','utf8',str(code_dir / script)],cwd=code_dir,check=True)
