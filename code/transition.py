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
    vfi_N            = 2000,
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

####################################################
### Section A - Transition between Steady-States ###
####################################################

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
