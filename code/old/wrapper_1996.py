run_ext_calib = False
run_gmm       = False
run_exercises = True

import subprocess
import sys
from pathlib import Path

old_dir  = Path(__file__).resolve().parent
code_dir = old_dir.parent
scripts  = [code_dir / 'config.py']

if run_ext_calib:
    scripts.append(old_dir / 'external_calibration_1996.py')
if run_gmm:
    scripts.append(old_dir / 'gmm_estimator_1996.py')
if run_exercises:
    scripts.append(old_dir / 'quant_exercises_1996.py')

for script in scripts:
    print(f'Running {script.name}',flush=True)
    subprocess.run([sys.executable,'-X','utf8',str(script)],cwd=code_dir,check=True)
