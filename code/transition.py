from GeneralEquilibriumModel import TypeModelParameters,TypeCalibParameters,GeneralEquilibriumModel
from config import DATA_PARAMS,OUTPUTS_QUANT_EX
import functions
import functions_transition
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import scipy.interpolate
import scipy.stats
import numpy as np
import copy
import json
from dataclasses import replace

# Creating the baseline model

with open(DATA_PARAMS / 'pre_gmm_params_ces.json','r',encoding='utf-8') as f:
    pre_gmm = json.load(f)
with open(DATA_PARAMS / 'post_gmm_params_ces.json','r',encoding='utf-8') as f:
    post_gmm = json.load(f)

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

p = post_gmm['parameters']

# Tariff shock with technology held fixed
τ_0 = p['τ']
τ_1 = 0.096

T0_ModelPar = TypeModelParameters(
    α            = p['α'],
    γ            = p['γ'],
    ψ            = p['ψ'],
    χ            = p['χ'],
    β_eff        = p['β_eff'],
    τ            = τ_0,
    w_star       = p['w_star'],
    θ            = p['θ'],
    σ            = p['σ'],
    δ            = p['δ'],
    ϱ            = p['ϱ'],
    σ_ϵ          = p['σ_ϵ'],
    π_LL         = p['π_LL'],
    π_HH         = p['π_HH'],
    M            = p['M'],
    ξ            = p['ξ'],
    rebate_share = p['rebate_share'])

T0_model = GeneralEquilibriumModel(T0_ModelPar,CalibPar,log_dir=None,log_inner=False)
T0_model.outer_loop_solver()
T0_stats = T0_model.economy_statistics()
T0_sol   = T0_model.mod_res

# Creating post-shock economy
T1_ModelPar = TypeModelParameters(
    α            = p['α'],
    γ            = p['γ'],
    ψ            = p['ψ'],
    χ            = p['χ'],
    β_eff        = p['β_eff'],
    τ            = τ_1,
    w_star       = p['w_star'],
    θ            = p['θ'],
    σ            = p['σ'],
    δ            = p['δ'],
    ϱ            = p['ϱ'],
    σ_ϵ          = p['σ_ϵ'],
    π_LL         = p['π_LL'],
    π_HH         = p['π_HH'],
    M            = p['M'],
    ξ            = p['ξ'],
    rebate_share = p['rebate_share'])

T1_model = GeneralEquilibriumModel(T1_ModelPar,CalibPar,log_dir=None,log_inner=False)
T1_model.outer_loop_solver()
T1_stats = T1_model.economy_statistics()

########################################
### Section A - Transition between Steady-States ###
########################################

T      = 350
τ_path = np.full(T + 1,τ_1)

z_grid,trans_z = functions.rouwenhorst_trans_matrix(ModelPar=T0_ModelPar,CalibPar=CalibPar)
joint_trans    = np.kron(functions.create_LH_skill_mat(ModelPar=T0_ModelPar),trans_z)
state_grid     = [(f,z) for f in ['L','H'] for z in z_grid]

a_grid = functions_transition.build_common_asset_grid([T0_model,T1_model],CalibPar)
a_arr  = np.array(a_grid)

E0 = functions_transition.tighten_steady_state(T0_model,a_arr,state_grid,joint_trans)
E1 = functions_transition.tighten_steady_state(T1_model,a_arr,state_grid,joint_trans)

transition = functions_transition.solve_transition_path(
    K_0         = E0['K'],
    τ_path      = τ_path,
    H           = T0_model.H,
    L           = T0_model.L,
    state_probs = T0_model.state_probs,
    V_terminal  = E1['V'],
    dens_0      = E0['dens'],
    a_arr       = a_arr,
    state_grid  = state_grid,
    joint_trans = joint_trans,
    ModelPar    = T1_ModelPar,
    K_terminal  = E1['K'],
    damp        = 0.3,
    store_paths = True)

periods = np.arange(T + 1)

# Leave space before the pre-shock point to show the impact jump.
X_LEFT = -1 - max(3.0,T * 0.03)


def plot_with_baseline(axis,pre_shock_value,path,**kwargs):
    '''
    Draws a transition path with its pre-shock steady state attached at t = -1 as the first
    point of the same line, so the impact effect reads as a jump in the series rather than
    as a separate marker or a reference line.
    '''

    return axis.plot(np.r_[-1,periods],np.r_[pre_shock_value,path],**kwargs)

### --- A. Aggregate transition paths --- ###


panel = [('K','Aggregate capital'),('Y','Output'),('I','Marginal task'),
         ('r','Interest rate'),('w','Low-skill wage'),('s','High-skill wage')]

ss0_lvl = {'K': E0['K'],'Y': T0_model.Y,'I': T0_model.I,
           'r': T0_model.r,'w': T0_model.w,'s': T0_model.s,
           'τ': τ_0,'R': T0_model.R}
ss1_lvl = {'K': E1['K'],'Y': T1_model.Y,'I': T1_model.I,
           'r': T1_model.r,'w': T1_model.w,'s': T1_model.s,
           'τ': τ_1,'R': T1_model.R}

fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(14,7))

for axis,(key,title) in zip(ax.flat,panel):

    # Plot interest rates in levels and other aggregates relative to baseline.
    in_levels: bool = key == 'r'

    if in_levels:
        plot_with_baseline(axis,ss0_lvl[key],transition[key],color='C0',lw=1.8,zorder=3)
        axis.axhline(ss1_lvl[key],color='C3',ls=':',lw=1.2,zorder=2)
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=2))
        axis.set_ylabel('Level')
    else:
        plot_with_baseline(axis,1.0,transition[key] / ss0_lvl[key],color='C0',lw=1.8,zorder=3)
        axis.axhline(ss1_lvl[key] / ss0_lvl[key],color='C3',ls=':',lw=1.2,zorder=2)
        axis.set_ylabel('Relative to pre-shock')

    axis.set_title(title,fontsize=10)
    axis.set_xlabel('Periods since the shock')
    axis.set_xlim(X_LEFT,T)
    axis.grid(linestyle='--',alpha=0.5)
    axis.set_axisbelow(True)

ax.flat[0].legend(
    handles=[Line2D([],[],color='C0',lw=1.8,label='Transition'), Line2D([],[],color='C3',ls=':',lw=1.2,label='Final steady state')],
    fontsize=8,
    frameon=False)

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_C_transition_aggregates.pdf')
plt.close()

### --- D. Welfare by initial state --- ###


def consumption_equivalent(V,ModelPar):
    '''
    Per-period consumption equivalent of a value function, matching the c_eq definition
    used in GeneralEquilibriumModel.economy_statistics.
    '''

    return (V * (1 - ModelPar.σ) * (1 - ModelPar.δ)) ** (1 / (1 - ModelPar.σ))


ceq_0 : np.ndarray = consumption_equivalent(E0['V'],T0_ModelPar)
ceq_tr: np.ndarray = consumption_equivalent(transition['V_path'][0],T1_ModelPar)
cev   : np.ndarray = ceq_tr / ceq_0 - 1

dens_0: np.ndarray = E0['dens']
n_z   : int        = len(z_grid)

aggregate_cev: float = float((cev * dens_0).sum() / dens_0.sum())
share_gaining: float = float(dens_0[cev > 0].sum() / dens_0.sum())

total_mass: float = float(dens_0.sum())

print(f'Average consumption-equivalent welfare change: {100*aggregate_cev:+.4f}%')
print(f'Share of households better off             : {100*share_gaining:.2f}%')

for idx,label in enumerate(['Low-skill','High-skill']):
    rows: slice      = slice(idx * n_z,(idx + 1) * n_z)
    wgt : np.ndarray = dens_0[rows]

    print(f'  {label:<11}: mean CEV {100*float((cev[rows]*wgt).sum()/wgt.sum()):+.4f}%, '
          f'share gaining {100*float(wgt[cev[rows] > 0].sum()/wgt.sum()):.2f}%')

### --- E. Inequality over the transition --- ###

dens_path : np.ndarray = transition['dens_path']
skill_is_L: np.ndarray = np.array([1.0 if f == 'L' else 0.0 for f,_ in state_grid])
z_levels  : np.ndarray = np.array([np.exp(z) for _,z in state_grid])


def inequality_stats(dens_t,V_t,w_t,s_t,r_t,transfer_t,ModelPar):
    '''
    Cross-sectional inequality and welfare measures for one date.

    Income follows the definition used in GeneralEquilibriumModel.economy_statistics:
    capital income plus effective labour income plus the rebate, excluding the asset
    principal. The pre-rebate Gini is reported alongside it, so the redistribution the
    tariff itself performs can be separated from the price movements it causes.

    Welfare is the density-weighted mean of V by skill type, as in the sec_A
    disaggregation table. No Gini is taken over V: with σ > 1 the value function is
    negative everywhere, and a Gini index is undefined on negative values.
    '''

    labour : np.ndarray = z_levels * (skill_is_L * w_t + (1 - skill_is_L) * s_t)
    inc_pre: np.ndarray = a_arr[None,:] * r_t + labour[:,None]
    income : np.ndarray = inc_pre + transfer_t[:,None]

    w_flat: pd.Series = pd.Series(dens_t.ravel())

    out: dict = {'income_gini': functions.weighted_gini(pd.Series(income.ravel()),w_flat),
                 'income_gini_pre': functions.weighted_gini(pd.Series(inc_pre.ravel()),w_flat)}

    for idx,label in enumerate(['V_low_skill','V_high_skill']):
        rows      : slice      = slice(idx * n_z,(idx + 1) * n_z)
        weights   : np.ndarray = dens_t[rows]
        out[label]             = float((V_t[rows] * weights).sum() / weights.sum())

    return out


keys : list = ['income_gini','income_gini_pre','V_low_skill','V_high_skill']
stats: dict = {k: np.empty(T + 1) for k in keys}

# Use pre-shock prices and distribution as the reference, before the impact jump.
pre_shock: dict = inequality_stats(dens_0,E0['V'],T0_model.w,T0_model.s,T0_model.r,T0_model.transfer,T0_ModelPar)

for t in periods:
    period_stats: dict = inequality_stats(
        dens_path[t],
        transition['V_path'][t],
        transition['w'][t],
        transition['s'][t],
        transition['r'][t],
        transition['transfer'][t],
        T1_ModelPar)

    for k in keys:
        stats[k][t] = period_stats[k]

fig,ax = plt.subplots(figsize=(7,4.5))

plot_with_baseline(ax,pre_shock['income_gini'],stats['income_gini'],color='C0',lw=1.8,label='Post-rebate')
plot_with_baseline(ax,pre_shock['income_gini_pre'],stats['income_gini_pre'],color='C3',lw=1.4,ls='--',label='Pre-rebate')
ax.set_xlabel('Periods since the shock')
ax.set_xlim(X_LEFT,T)
ax.grid(linestyle='--',alpha=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_C_transition_inequality.pdf')
plt.close()

### --- D. Welfare over the transition --- ###

# Reverse proportional utility changes because baseline CRRA utility is negative.

fig,ax = plt.subplots(figsize=(7,4.5))

notes: list = []

for idx,(key,label,colour) in enumerate([('V_low_skill','Low-skill','C0'), ('V_high_skill','High-skill','C3')]):
    change: np.ndarray = -100 * (stats[key] / pre_shock[key] - 1)
    plot_with_baseline(ax,0.0,change,color=colour,lw=1.8,label=label)

    rows: slice      = slice(idx * n_z,(idx + 1) * n_z)
    wgt : np.ndarray = dens_0[rows]
    notes.append(f'{label.lower()} {100*float(wgt[cev[rows] > 0].sum()/wgt.sum()):.1f}%')

ax.set_xlabel('Periods since the shock')
ax.set_ylabel('%')
ax.set_xlim(X_LEFT,T)
ax.grid(linestyle='--',alpha=0.5)
ax.set_axisbelow(True)

ax.legend(frameon=False,loc='lower left')

ax.text(
    0.98,
    0.97,
    'Share of households with higher welfare at t=0:\n' f'All {100*share_gaining:.1f}%   ·   ' + '   ·   '.join(notes),
    transform=ax.transAxes,
    fontsize=8,
    va='top',
    ha='right',
    bbox=dict(boxstyle='round',facecolor='white',edgecolor='lightgrey',alpha=0.9))

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_C_transition_welfare.pdf')
plt.close()

### --- Welfare Change over the State Space --- ###

# State-level consumption equivalents include transition costs.
cev_grid: np.ndarray = 100 * cev.reshape(2,n_z,len(a_arr))

# Limit wealth to the baseline 99.9th percentile.
cum_wealth: np.ndarray = dens_0.sum(axis=0).cumsum() / dens_0.sum()
a_cut     : float      = float(a_arr[np.searchsorted(cum_wealth,0.999)])
keep      : np.ndarray = a_arr <= a_cut

norm = TwoSlopeNorm(vmin=min(cev_grid.min(),-1e-9),vcenter=0.0,vmax=max(cev_grid.max(),1e-9))

fig,ax = plt.subplots(ncols=2,figsize=(12,5),sharey=True)

X,Y = np.meshgrid(np.exp(z_grid),a_arr[keep])

for idx,(axis,title) in enumerate(zip(ax,['Low-skill','High-skill'])):
    axis.contourf(X,Y,cev_grid[idx][:,keep].T,levels=25,cmap='RdYlGn',norm=norm)
    axis.contour(X,Y,cev_grid[idx][:,keep].T,levels=[0],colors='black',linewidths=1.2)
    axis.set_title(title)
    axis.set_xlabel('Productivity z')
    axis.set_axisbelow(True)
    axis.grid(linestyle='--',alpha=0.3)

ax[0].set_ylabel(r'Wealth $a_0$')

fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='RdYlGn'),ax=ax,label='Welfare change over the transition (%)')

plt.savefig(OUTPUTS_QUANT_EX / 'sec_C_welfare_state_space.pdf',bbox_inches='tight')
plt.close()

print(f'\nWelfare over the state space: ' f'[{cev_grid.min():+.3f}%, {cev_grid.max():+.3f}%]  (cut at assets = {a_cut:.1f})')

PATH_KEYS = ['K','Y','I','r','w','s','τ','R']

transition_summary = pd.DataFrame({'t': periods, **{k: transition[k] for k in PATH_KEYS}, **stats})

# Add the pre-shock steady state at t = -1.
baseline = pd.DataFrame([{'t': -1,**{k: ss0_lvl[k] for k in PATH_KEYS},**pre_shock}])

transition_summary = pd.concat([baseline,transition_summary],ignore_index=True)

for label,key in [('post-rebate','income_gini'),('pre-rebate','income_gini_pre')]:
    print(f'\nIncome Gini ({label}): pre-shock -> t=0 (impact) -> t=T')
    print(f'  {pre_shock[key]:.5f} -> {stats[key][0]:.5f} -> {stats[key][-1]:.5f}   '
          f'(impact {stats[key][0]-pre_shock[key]:+.5f}, '
          f'total {stats[key][-1]-pre_shock[key]:+.5f})')

print('\nMean welfare V by skill type, change from pre-shock (+ is better off)')

for key,label in [('V_low_skill','Low-skill'),('V_high_skill','High-skill')]:
    print(f'  {label:<11} impact {-100*(stats[key][0]/pre_shock[key]-1):+.4f}%, ' f't={T} {-100*(stats[key][-1]/pre_shock[key]-1):+.4f}%')


########################################################################
### Section B - Condorcet Voting with Transition Utility at t=0 ###
########################################################################

τ_grid: np.ndarray = 0.75 * np.linspace(0,1,100) ** 2
preference_tariffs: np.ndarray = np.unique(np.r_[τ_grid,τ_0,τ_1])
terminal_models: dict = {τ_0: T0_model,τ_1: T1_model}
failed_tariffs : dict = {}

# Preparing terminal economies

for tariff in preference_tariffs:
    if tariff in terminal_models:
        continue

    print(f'Preparing terminal economy: tariff {tariff:.4%}')

    try:
        terminal_parameters = replace(T0_ModelPar,τ=float(tariff))
        terminal_model = GeneralEquilibriumModel(terminal_parameters,CalibPar,log_dir=None,log_inner=False)
        terminal_model.outer_loop_solver()
        terminal_models[tariff] = terminal_model

    except (RuntimeError,ValueError,FloatingPointError) as error:
        failed_tariffs[tariff] = f'Terminal economy: {error}'

        print(f'Skipping tariff {tariff:.4%}: {error}')


# Use one asset grid and baseline electorate for every tariff.

election_assets = np.array(functions_transition.build_common_asset_grid(list(terminal_models.values()),CalibPar))
election_E0 = functions_transition.tighten_steady_state(T0_model,election_assets,state_grid,joint_trans)
transition_state_values: np.ndarray = np.full((election_E0['V'].size,len(preference_tariffs)),np.nan)
transition_diagnostics: dict = {}

for tariff_idx,tariff in enumerate(preference_tariffs):
    if tariff == τ_0:
        transition_state_values[:,tariff_idx] = election_E0['V'].ravel()
        continue

    if tariff in failed_tariffs:
        continue

    print(f'Solving voting transition {tariff_idx + 1}/{len(preference_tariffs)}: tariff {tariff:.4%}')

    try:
        terminal_model = terminal_models[tariff]
        election_E1 = functions_transition.tighten_steady_state(terminal_model,election_assets,state_grid,joint_trans)
        candidate_path = np.full(T + 1,tariff)

        candidate_transition = functions_transition.solve_transition_path(
            K_0         = election_E0['K'],
            τ_path      = candidate_path,
            H           = T0_model.H,
            L           = T0_model.L,
            state_probs = T0_model.state_probs,
            V_terminal  = election_E1['V'],
            dens_0      = election_E0['dens'],
            a_arr       = election_assets,
            state_grid  = state_grid,
            joint_trans = joint_trans,
            ModelPar    = terminal_model.ModelPar,
            K_terminal  = election_E1['K'],
            damp        = 0.3,
            store_paths = False)

        transition_diagnostics[tariff] = {key: candidate_transition[key] for key in ('converged','residual','n_iter','stop_reason')}

        if not candidate_transition['converged']:
            raise RuntimeError(f'Transition at tariff {tariff:.4%} did not converge; see transition_diagnostics.')

        candidate_values = candidate_transition['V_0'].ravel()

        if not np.all(np.isfinite(candidate_values)):
            raise ValueError('Non-finite t=0 utility.')

        transition_state_values[:,tariff_idx] = candidate_values

        del candidate_transition,candidate_values,election_E1,candidate_path

    except (RuntimeError,ValueError,FloatingPointError) as error:
        failed_tariffs[tariff] = f'Transition: {error}'

        print(f'Skipping tariff {tariff:.4%}: {error}')


successful_tariffs: np.ndarray = np.all(np.isfinite(transition_state_values),axis=0)

transition_state_values = transition_state_values[:,successful_tariffs]
preference_tariffs     = preference_tariffs[successful_tariffs]

print(f'Voting over {len(preference_tariffs)} successfully solved tariffs; {len(failed_tariffs)} candidates excluded.')

if len(preference_tariffs) < 2:
    raise RuntimeError('At least two successful candidates are needed for a Condorcet comparison.')

### --- Condorcet Winner under Baseline Population Voting --- ###

voter_values : np.ndarray = transition_state_values
voter_weights: np.ndarray = election_E0['dens'].ravel() / election_E0['dens'].sum()
tariff_count : int        = len(preference_tariffs)
support      : np.ndarray = np.zeros((tariff_count,tariff_count))

# Entry (i,j) is the population share strictly preferring tariff i to tariff j.

for i in range(tariff_count):
    for j in range(i + 1,tariff_count):
        support[i,j] = voter_weights[voter_values[:,i] > voter_values[:,j]].sum()
        support[j,i] = voter_weights[voter_values[:,j] > voter_values[:,i]].sum()

pairwise_support: pd.DataFrame = pd.DataFrame(support,index=preference_tariffs,columns=preference_tariffs)

pairwise_support.index.name   = 'candidate_tariff'
pairwise_support.columns.name = 'opponent_tariff'

pairwise_indifference: pd.DataFrame = 1 - pairwise_support - pairwise_support.T
pairwise_margins     : pd.DataFrame = pairwise_support - pairwise_support.T

# Indifferent households abstain. Treat vote margins within rounding error as ties.

vote_tolerance   : float        = 1e-12
pairwise_wins    : np.ndarray   = pairwise_margins.to_numpy() > vote_tolerance
win_counts       : np.ndarray   = pairwise_wins.sum(axis=1)
loss_counts      : np.ndarray   = pairwise_wins.sum(axis=0)
condorcet_winners: np.ndarray   = preference_tariffs[win_counts == tariff_count - 1]
condorcet_summary: pd.DataFrame = pd.DataFrame(index=pd.Index(preference_tariffs,name='tariff'))

condorcet_summary['wins']   = win_counts
condorcet_summary['losses'] = loss_counts
condorcet_summary['ties']   = tariff_count - 1 - win_counts - loss_counts

print('\nCondorcet election among successful candidates: t=0 utilities, fixed baseline electorate; exact utility ties abstain.')

if condorcet_winners.size:
    for tariff in condorcet_winners:
        winning_margins = pairwise_margins.loc[tariff].drop(tariff)
        closest_margin  = winning_margins.min()

        print(f'Condorcet winner: {tariff:.4%}; smallest head-to-head winning margin: {closest_margin:.4%} of covered households.')

        del winning_margins,closest_margin
else:
    print('No strict Condorcet winner on the evaluated tariff grid. Inspect condorcet_summary and pairwise_margins for defeats and ties.')

del voter_values,voter_weights,tariff_count,support,pairwise_wins,win_counts,loss_counts


### --- Median Preferred Tariff --- ###

preference_values : np.ndarray  = transition_state_values
preference_states: pd.DataFrame = pd.DataFrame({
    'skill_type': np.repeat([state[0] for state in state_grid],len(election_assets)),
    'z': np.repeat(np.exp([state[1] for state in state_grid]),len(election_assets)),
    'a_0': np.tile(election_assets,len(state_grid)),
    'dens': election_E0['dens'].ravel()})
peak_indices: np.ndarray = np.argmax(preference_values,axis=1)
covered_mass: float      = float(preference_states['dens'].sum())

# Exact utility ties select the lower tariff; the median weights households by baseline mass.
preference_states['preferred_tariff'] = preference_tariffs[peak_indices]

preferred_tariff_distribution: pd.Series = preference_states.groupby('preferred_tariff')['dens'].sum() / covered_mass
preference_cdf               : pd.Series = preferred_tariff_distribution.cumsum()
median_index                : int       = int(np.searchsorted(preference_cdf.to_numpy(),0.5))
median_preferred_tariff      : float     = float(preference_cdf.index[median_index])
coverage_share              : float     = covered_mass / election_E0['dens'].sum()

median_rows = [
    ['Median preferred tariff',f'{median_preferred_tariff:.4%}'.replace('%',r'\%')],
    ['Baseline population covered',f'{coverage_share:.4%}'.replace('%',r'\%')],
    ['Lowest evaluated tariff',f'{preference_tariffs.min():.2%}'.replace('%',r'\%')],
    ['Highest evaluated tariff',f'{preference_tariffs.max():.2%}'.replace('%',r'\%')]]
median_rows += [['Successful candidate tariffs',str(len(preference_tariffs))],['Excluded candidate tariffs',str(len(failed_tariffs))]]
median_summary = pd.DataFrame(median_rows,columns=['Statistic','Value'])
median_table   = median_summary.to_latex(
    index   = False,
    escape  = False,
    caption = 'Median preferred tariff at t=0 among successful transitions, weighted by baseline population',
    label   = 'tab:transition_median_preferred_tariff')

with open(OUTPUTS_QUANT_EX / 'sec_B_transition_median_preferred_tariff.tex','w',encoding='utf-8') as f:
    f.write(median_table)

print(f'Median preferred tariff at t=0 among successful candidates: {median_preferred_tariff:.4%}; baseline population covered: {coverage_share:.4%}.')

del median_index,median_rows,median_summary,median_table

### --- Utility Profiles and Single-Peakedness --- ###

utility_steps : np.ndarray = np.diff(preference_values,axis=1)
left_of_peak : np.ndarray = np.arange(utility_steps.shape[1])[None,:] < peak_indices[:,None]
single_peaked: np.ndarray = np.all(np.where(left_of_peak,utility_steps > 0,utility_steps < 0),axis=1)
peak_values  : np.ndarray = preference_values[np.arange(len(peak_indices)),peak_indices]

# A positive state-specific scale preserves rankings and makes utility gaps comparable visually.
if np.any(peak_values == 0):
    raise ValueError('Utility profiles cannot be normalized by zero peak utility.')

utility_profiles: np.ndarray = (preference_values - peak_values[:,None]) / np.abs(peak_values[:,None])
wrong_way_steps : np.ndarray = np.where(left_of_peak,-utility_steps,utility_steps)
largest_reversal: np.ndarray = np.maximum(wrong_way_steps.max(axis=1),0) / np.abs(peak_values)

preference_states['strictly_single_peaked'] = single_peaked
preference_states['largest_reversal']      = largest_reversal

single_peaked_share: float = float(preference_states['dens'].to_numpy() @ single_peaked / covered_mass)

print(f'Strictly single-peaked: {single_peaked.mean():.2%} of covered states; {single_peaked_share:.2%} of covered households.')
print(f'Largest adjacent wrong-way utility change: {largest_reversal.max():.4%} of the corresponding state\'s absolute peak utility.')

# Every covered state is plotted, including zero-mass states. Ties fail the strict test.
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12,8),sharex=True,sharey=True,layout='constrained')

for col,(skill,title) in enumerate([('L','Low-skill'),('H','High-skill')]):
    for row,(passes,label,color) in enumerate([(True,'Single-peaked','#1baf7a'),(False,'Not single-peaked','#e34948')]):
        selected = (preference_states['skill_type'].to_numpy() == skill) & (single_peaked == passes)
        axis     = ax[row,col]
        opacity  = float(np.clip(3 / np.sqrt(max(selected.sum(),1)),0.08,0.85))

        axis.plot(preference_tariffs,utility_profiles[selected].T,color=color,linewidth=0.6,alpha=opacity,rasterized=True)
        axis.axhline(0,color='black',linestyle=':',linewidth=1)
        axis.set_title(f'{title}: {label} ({selected.sum():,} states)',fontsize=12)
        axis.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        axis.tick_params(labelsize=11)
        axis.set_axisbelow(True)
        axis.grid(linestyle='--',alpha=0.3)

        if not selected.any():
            axis.text(0.5,0.5,'No states',transform=axis.transAxes,ha='center',va='center',fontsize=12)

fig.supxlabel(r'Tariff rate ($\tau$)',fontsize=12)
fig.supylabel('t=0 utility gap relative to each state\'s absolute peak utility',fontsize=12)
fig.savefig(OUTPUTS_QUANT_EX / 'sec_B_transition_utility_profiles.pdf')
plt.close(fig)
