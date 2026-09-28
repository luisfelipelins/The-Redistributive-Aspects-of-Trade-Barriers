# The Redistributive Aspects of Trade Barriers

This project studies how tariffs on offshored tasks affect wages, capital accumulation, inequality, and welfare in an incomplete-markets economy. Households differ in skill, productivity, and asset holdings. They save subject to a borrowing constraint and face idiosyncratic productivity and skill transitions.

Production combines low-skill tasks with a composite of high-skill labour and capital. Firms can offshore low-skill tasks, while tariff revenue is returned to households through a redistribution rule. The code estimates the model using US data, solves stationary equilibria and transition paths, and evaluates the distributional effects of tariff policy.

## Model and estimation

The nested-CES technology is

$$
X_{HK}=\left[(1-\alpha)H^\chi+\alpha K^\chi\right]^{1/\chi},
\qquad
Y=\left[(1-\gamma)S_L^\psi+\gamma X_{HK}^\psi\right]^{1/\psi}.
$$

For an interior offshoring cutoff $I$, domestic low-skill labour satisfies $S_L=L/(1-I)$. Firms face the total offshoring cost $\beta=(1+\tau)\beta^{\mathrm{eff}}$, where $\tau$ is the tariff. Factor prices and income shares are endogenous.

The GMM parameter vector is ordered as follows:

```python
['θ', 'w_star', 'β_eff', 'γ']
```

The four jointly targeted moments are:

| Model statistic | Empirical target |
| --- | --- |
| `I` | Share of low-skill tasks offshored |
| `w_to_wstar` | Domestic-to-foreign low-skill wage ratio |
| `skill_premium` | High-skill-to-low-skill wage ratio |
| `low_skill_share` | Low-skill labour income share, compared with `w * L / Y` in the model |

The empirical low-skill income share is constructed as

$$
\mathrm{LS\_share}=\frac{1-\mathrm{capital\_income\_share}}{1+H\_to\_L},
\qquad
H\_to\_L=\mathrm{skill\_premium}\frac{1-\mathrm{ls\_emp\_share}}{\mathrm{ls\_emp\_share}}.
$$

The estimator minimizes the sum of squared relative moment errors, using an identity weight matrix. The current runner uses differential evolution. Its bounds are $\theta\in[10^{-6},40]$, $w^*\in[0.05,0.7]$, $\beta^{\mathrm{eff}}\in[1,7]$, and $\gamma\in[0.01,0.99]$.

The remaining production parameters are externally calibrated: $\alpha=0.81$, $\psi=0.40$, and $\chi=-0.49$. Productivity persistence is denoted by $\varrho$ (`ϱ` in the source). Calibration values and data construction are defined in `code/external_calibration.py`.

The redistribution rule is proportional to labour income raised to $-\xi$, with its scale determined by the available tariff revenue. Thus, $\xi=0$ gives equal lump-sum transfers, $\xi>0$ is progressive, and $\xi<0$ is regressive.

## Repository structure

```text
code/                  Model, data preparation and execution scripts
  old/                 Historical scripts, outside the active pipeline
config/                Local configuration, including the private FRED key
data/
  raw/                 Local input datasets
  intermediate/        Intermediate data
  final/               Processed data
  parameters/          Pre- and post-estimation JSON files
outputs/
  motivation/          Descriptive figures
  gmm/                 Estimation sensitivity figures
  quant_exercises/     Stationary and transition figures and LaTeX tables
log/                   Estimation and model-solution logs
requirements.txt       Exact versions of project dependencies
```

`config.py` creates the working directories and resolves paths relative to the repository. The `data/`, `outputs/`, and `log/` directories are excluded from Git. A fresh clone therefore does not contain the local datasets or saved estimates.

## What each script does

| File | Role |
| --- | --- |
| [`wrapper.py`](code/wrapper.py) | Runs the pipeline in order, with three flags controlling the estimation and analysis stages. |
| [`config.py`](code/config.py) | Defines project paths and creates working directories. |
| [`motivation.py`](code/motivation.py) | Builds descriptive figures on manufacturing, foreign investment, wages, and inequality using online series and local data. |
| [`external_calibration.py`](code/external_calibration.py) | Constructs empirical moments and externally calibrated parameters; writes `pre_gmm_params_ces.json`. |
| [`gmm_estimator.py`](code/gmm_estimator.py) | Configures and runs the four-parameter estimation; writes `post_gmm_params_ces.json`. |
| [`GMM.py`](code/GMM.py) | Evaluates model moments and the objective, runs optimizers, records failures and results, and contains estimation-history utilities. |
| [`GeneralEquilibriumModel.py`](code/GeneralEquilibriumModel.py) | Defines parameter containers and coordinates the stationary equilibrium, household problem, and aggregate statistics. |
| [`ces_production.py`](code/ces_production.py) | Implements nested-CES output, marginal products, factor demands, and endogenous cost shares. |
| [`functions.py`](code/functions.py) | Provides productivity and skill processes, household value-function iteration, stationary distributions, offshoring and transfer calculations, and inequality measures. |
| [`functions_transition.py`](code/functions_transition.py) | Solves production along a capital path, household decisions backward, and distributions forward; updates the transition path. |
| [`gmm_jorgensen_sensitivity.py`](code/gmm_jorgensen_sensitivity.py) | Computes local sensitivity of the four estimates to calibrated parameters and plots representative-agent narrative conditions. |
| [`quant_exercises.py`](code/quant_exercises.py) | Runs stationary tariff and redistribution counterfactuals, welfare decompositions, social-weight exercises, and household tariff-preference comparisons. |
| [`transition.py`](code/transition.py) | Runs transition-path exercises and produces aggregate, inequality, and welfare results. |
| [`test_nested_ces.py`](code/test_nested_ces.py) | Checks production identities, equilibrium consistency, welfare decompositions, and GMM parameter/moment handling without running a full estimation. |

Scripts under `code/old/`, including the historical grid-search routines, are not called by the wrapper and may depend on earlier model specifications.

## Installation

The development environment uses **Python 3.10.9**. `requirements.txt` pins the project dependencies to the exact versions reported by `pip list` in that environment. It is a list of project dependencies, not a complete lockfile for all transitive packages. A fresh Linux installation has not been validated end to end.

Clone the repository and change to its root directory. On Linux, with Python 3.10 and its `venv` module already installed:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, using a Python 3.10 interpreter:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use the activated environment for all commands below. The wrapper launches its subprocesses with that same Python interpreter.

## Required data and configuration

Create `config/fred_key.txt` containing your FRED API key as plain text. This file is excluded from Git; supply it separately on the machine where the project runs.

Copy these files into `data/raw/` before running the full pipeline:

| File | Used by |
| --- | --- |
| `2002_detailed_industry.dat` | Manufacturing establishment comparisons in `motivation.py` |
| `2022.dat` | Manufacturing establishment comparisons in `motivation.py` |
| `bea_bop_direct_investment_position.xlsx` | Foreign direct investment position figures |
| `bea_bop_direct_investment_income.xlsx` | Foreign direct investment income figures |
| `ilccompensationtimeseries_2016.xlsx` | Domestic-to-foreign wage target in `external_calibration.py` |

These are the local input files expected by the scripts, including their existing column and sheet layouts. The code does not download these five files automatically. The motivation and calibration stages also retrieve data from FRED, BLS, the Federal Reserve's SHED, the Census Bureau, and OECD TiVA. Those stages require internet access, and data revisions can change the resulting calibration.

The parameter-file dependency is:

```text
external_calibration.py
    -> data/parameters/pre_gmm_params_ces.json
gmm_estimator.py
    -> data/parameters/post_gmm_params_ces.json
gmm_jorgensen_sensitivity.py / quant_exercises.py / transition.py
    -> read both parameter files
```

When reusing estimates, copy the corresponding pre- and post-estimation files together. Historical Cobb–Douglas estimates and earlier fixed-γ estimation histories are not interchangeable with the current four-parameter specification.

## Running the project

Edit the flags at the top of `code/wrapper.py`:

```python
run_gmm         = True
run_sensitivity = True
run_exercises   = True
```

Then run, from the repository root:

```bash
python -X utf8 code/wrapper.py
```

The execution order is:

1. `config.py`
2. `motivation.py`
3. `external_calibration.py`
4. `gmm_estimator.py`, if `run_gmm` is `True`
5. `gmm_jorgensen_sensitivity.py`, if `run_sensitivity` is `True`
6. `quant_exercises.py`, if `run_exercises` is `True`
7. `transition.py`, if `run_exercises` is `True`

Each script runs in a separate process. A nonzero exit status stops the wrapper. The first three scripts always run, even when all flags are `False`.

Setting `run_gmm=False` skips estimation; it does not resume an optimizer. Later stages require an existing `post_gmm_params_ces.json`. The wrapper still rebuilds external calibration, so ensure it is consistent with the saved estimates. To reuse an unchanged calibration and run only a particular analysis, call the script directly:

```bash
python -X utf8 code/quant_exercises.py
python -X utf8 code/transition.py
```

The other entry-point scripts can be run in the same way, following their parameter-file dependencies.

## Running on an AWS Linux machine

On an existing Linux instance, clone the repository, create the Python environment, and provide the local data and FRED key as described above. No AWS-specific Python package is required by this project.

Use a non-interactive plotting backend and unbuffered console output:

```bash
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
mkdir -p log
nohup python -X utf8 code/wrapper.py > log/wrapper.log 2>&1 &
echo $! > log/wrapper.pid
```

This launches the pipeline in the background and redirects its console output to a log. Monitor progress with:

```bash
tail -f log/wrapper.log
```

The wrapper runs stages sequentially, and the current differential-evolution configuration uses `workers=1`. Additional instance cores do not automatically parallelize GMM evaluations. Runtime and memory requirements depend on the household grids, optimizer settings, and transition horizon; the repository does not provide a benchmark for selecting an instance size.

## Results and numerical checks

The main outputs are figures in PDF and tables in LaTeX. Numerical counterfactual results are also held in memory while the relevant script runs. Those in-memory objects are not shared between wrapper subprocesses.

Two stationary welfare exercises in `quant_exercises.py` are:

- `sec_B_redistribution_welfare.pdf`: varies ξ from −2 to 2 at the implemented tariff, keeping structural estimates fixed, and reports mean stationary welfare changes relative to ξ = 0. Failed equilibria are reported and appear as missing points.
- `sec_B_weighted_welfare.pdf`: searches for the smallest pro-poor exponential decile-weight tilt that makes the implemented tariff a welfare maximum among the evaluated tariffs. Ties are allowed; infeasibility is reported explicitly. This is a grid-based stationary comparison, not a proof of optimality over a continuous tariff interval or a transition-welfare calculation.

GMM records evaluation failures in `gmm_errors.log` within its run directory. A saved best candidate is not, by itself, evidence that optimization converged or all moments fit well; inspect the final objective, moment distances, and convergence status. Counterfactual and sensitivity calculations also depend on equilibrium convergence and numerical resolution.

Run the available short tests before starting a long computation:

```bash
python -X utf8 -m unittest discover -s code -p "test_*.py"
```

The tests do not perform a full estimation or regenerate all empirical results. Existing figure and parameter filenames may be overwritten when their generating stages are rerun.
