# -*- coding: utf-8 -*-
"""
Created on Fri Jul  10 14:36:26 2026

@author: lfval
"""

from GeneralEquilibriumModel import TypeModelParameters,TypeCalibParameters,GeneralEquilibriumModel
from config import DATA_PARAMS,OUTPUTS_QUANT_EX
import functions
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import scipy.interpolate
import scipy.stats
import numpy as np
import copy
from dataclasses import replace
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
    vfi_N            = 500,
    vfi_eps          = 1e-5,
    vfi_howard_steps = 20,
    gmc_eps          = 1e-3,
    kmc_eps          = 1e-3,
    r_init_guess     = 0.03,
    inner_loop_eps   = 1e-5,
    outer_loop_eps   = 1e-5,
    outer_loop_r_lb  = 0.001)

p  : dict  = post_gmm['parameters']
τ_0: float = p['τ']
τ_1: float = 0.096

T0_ModelPar: TypeModelParameters = TypeModelParameters(
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

###########################################################################
### Section A - Alternative Economy after the 2025 tariff shock ###
###########################################################################

# Creating post-shock economy
T1_ModelPar: TypeModelParameters = TypeModelParameters(
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

T1_no_rebate_ModelPar: TypeModelParameters = replace(T1_ModelPar,rebate_share=0.0)
T1_no_rebate_model: GeneralEquilibriumModel = GeneralEquilibriumModel(T1_no_rebate_ModelPar,CalibPar,log_dir=None,log_inner=False)
T1_no_rebate_model.outer_loop_solver()
T1_no_rebate_stats: dict = T1_no_rebate_model.economy_statistics()

### --- Aggregates Comparison --- ###
T0_stats['Omega'] = functions.Omega(I=T0_model.I,ModelPar=T0_model.ModelPar)
T1_stats['Omega'] = functions.Omega(I=T1_model.I,ModelPar=T1_model.ModelPar)

comp_table               = pd.DataFrame([T0_stats,T1_stats],index=['T0','T1']).T
comp_table['pct change'] = comp_table.pct_change(axis=1).dropna(axis=1)['T1']

actual_what = comp_table.loc['w']['pct change']

tariff0_label: str = f"$\\tau={τ_0 * 100:.1f}\\%$"
tariff1_label: str = f"$\\tau={τ_1 * 100:.1f}\\%$"


def _latex_tabular(header: list,row_groups: list,label_pad: int = 0) -> str:
    """
    Format grouped rows as a booktabs tabular environment.

    Parameters
    ----------
    header : list
        Column labels.
    row_groups : list
        Groups of formatted table rows.
    label_pad : int
        Extra padding for row labels.

    Returns
    -------
    str
        LaTeX tabular source.
    """

    n_cols   = len(header) - 1
    all_rows = [row for group in row_groups for row in group]
    label_w  = max(len(row[0]) for row in all_rows) + label_pad
    col_w    = [max(len(row[j + 1]) for row in all_rows) for j in range(n_cols)]

    lines = [f"\\begin{{tabular}}{{l{'c' * n_cols}}}",
             '\\toprule',
             ' & '.join(header) + ' \\\\',
             '\\midrule']

    for pos,group in enumerate(row_groups):
        if pos:
            lines.append('\\addlinespace')
        for row in group:
            cells = [row[0].ljust(label_w)] + [row[j + 1].ljust(col_w[j]) for j in range(n_cols)]
            lines.append(' & '.join(cells) + ' \\\\')

    lines += ['\\bottomrule','\\end{tabular}']

    return '\n'.join(lines)


def _latex_table(tabulars: list,caption: str,label: str) -> str:
    """
    Wrap tabular environments in a captioned LaTeX table.

    Parameters
    ----------
    tabulars : list
        Tabular source strings.
    caption : str
        Table caption.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    str
        Complete LaTeX table source.
    """

    body = '\n\n\\vspace{0.4cm}\n\n'.join(tabulars)

    return (f"\\begin{{table}}[htbp]\n\\small\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\n"
            f"{body}\n\\end{{table}}\n")


AGG_SPEC = [
    ('I','$I$','pp',2),
    ('w','$w$','lvl',2),
    ('s','$s$','lvl',2),
    ('r','$r$','pp',2),
    ('tariff_share','Tariff Rev./$Y$','pp',2),
    ('Omega','$\\Omega(I)$','lvl',3),
    ('real_gdp','$Y$','lvl',2),
    ('K','$K$','lvl',2),
    ('income_gini','Inc. Gini','x100',1),
    ('mean_V','Disc. Utility','flip',2),
]


def _agg_cells(key: str,kind: str,decimals: int) -> list:
    """
    Format baseline, tariff-shock and change cells for one statistic.

    Parameters
    ----------
    key : str
        Economy-statistics key.
    kind : str
        Level, percentage-point, Gini or utility formatting rule.
    decimals : int
        Displayed decimal places.

    Returns
    -------
    list
        Formatted levels and change.
    """

    v0,v1 = T0_stats[key],T1_stats[key]

    if kind == 'pp':
        return [f"{v0*100:.{decimals}f}\\%",
                f"{v1*100:.{decimals}f}\\%",
                f"{(v1 - v0)*100:+.2f}pp"]

    if kind == 'x100':
        levels = [f"{v0*100:,.{decimals}f}",f"{v1*100:,.{decimals}f}"]
    else:
        levels = [f"{v0:,.{decimals}f}",f"{v1:,.{decimals}f}"]

    change = v1 / v0 - 1

    if kind == 'flip':
        change = -change

    return levels + [f"{change*100:+.2f}\\%"]


def _agg_tabular(spec: list) -> str:
    """
    Format an aggregate comparison using the requested statistics.

    Parameters
    ----------
    spec : list
        Statistic keys, labels and formatting rules.

    Returns
    -------
    str
        LaTeX tabular source.
    """

    rows = [['Pre-shock'],['Post-shock'],['$\\Delta\\%$']]

    for key,_,kind,decimals in spec:
        for row,cell in zip(rows,_agg_cells(key,kind,decimals)):
            row.append(cell)

    return _latex_tabular([''] + [label for _,label,_,_ in spec],[rows])


comp_table_latex = _latex_table(
    [_agg_tabular(AGG_SPEC[:5]),_agg_tabular(AGG_SPEC[5:])],
    caption = 'Aggregate effects of the tariff shock ($\\Delta\\tau$)',
    label   = 'tab:agg_comparison')

with open(OUTPUTS_QUANT_EX / 'sec_A_agg_comparison.tex','w',encoding='utf-8') as f:
    f.write(comp_table_latex)

REBATE_SPEC: list = [
    ('w','$w$'),
    ('I','$I$'),
    ('income_gini','Income Gini'),
    ('income_gini_pre','Pre-rebate Income Gini'),
    ('mean_c_eq','Mean Consumption Equivalent'),
    ('mean_V','Mean Discounted Utility'),
]
rebate_rows: list = [
    [label] + [f"{(stats[key] / T0_stats[key] - 1) * (-100 if key == 'mean_V' else 100):+.2f}\\%"
               for stats in (T1_stats,T1_no_rebate_stats)]
    for key,label in REBATE_SPEC
]
rebate_comparison_latex: str = _latex_table(
    [_latex_tabular(['Variable','$\\Delta\\%$ (rebate)','$\\Delta\\%$ (no rebate)'],[rebate_rows])],
    caption = (f'Tariff and rebate comparison: T0 tariff {τ_0 * 100:.1f}'
             + r'\%, T1 tariff ' + f'{τ_1 * 100:.1f}'
             + r'\%. The no-rebate economy discards all tariff revenue.'),
    label   = 'tab:rebate_comparison')

with open(OUTPUTS_QUANT_EX / 'sec_A_rebate_comparison.tex','w',encoding='utf-8') as f:
    f.write(rebate_comparison_latex)

### --- Decomposing Wage Change --- ###

prod_effect = -(functions.Omega(I=T1_model.I,ModelPar=T0_model.ModelPar) / functions.Omega(I=T0_model.I,ModelPar=T0_model.ModelPar) - 1)
base_shares = functions.firm_at_interest(T0_model.I,T0_model.r,T0_model.H,T0_model.L,T0_model.ModelPar)
wage_ls     = (1 - T0_model.ModelPar.ψ) * base_shares['omega_X']
wage_k      = wage_ls * base_shares['eta_K']
ls_effect   = -wage_ls * (T1_model.I - T0_model.I) / (1 - T0_model.I)
k_effect    = wage_k * (T1_model.economy_stats['K'] / T0_model.economy_stats['K'] - 1)

hatw     = prod_effect + ls_effect + k_effect
residual = actual_what - hatw

decomp_table = pd.Series({'Total Wage Change': actual_what,'Productivity': prod_effect,'Labour-supply': ls_effect,'Capital': k_effect,'GE 2nd-order': residual},name='$\\Delta\\%$')

decomp_table_latex = decomp_table.map(lambda x: f"{x*100:+.2f}\\%").to_frame().to_latex(
    escape  = False,
    caption = 'Decomposition of the low-skill wage change ($\\Delta\\tau$)',
    label   = 'tab:wage_decomp')

with open(OUTPUTS_QUANT_EX / 'sec_A_wage_decomp.tex','w',encoding='utf-8') as f:
    f.write(decomp_table_latex)

### --- Labour/Capital Income Gini --- ###

income_gini_table = pd.DataFrame({
    'Labour Income Gini': {
        tariff0_label: functions.weighted_gini(x=T0_model.mod_res['labour_inc'],weights=T0_model.mod_res['dens']),
        tariff1_label: functions.weighted_gini(x=T1_model.mod_res['labour_inc'],weights=T1_model.mod_res['dens'])},
    'Capital Income Gini': {
        tariff0_label: functions.weighted_gini(x=T0_model.mod_res['capital_inc'],weights=T0_model.mod_res['dens']),
        tariff1_label: functions.weighted_gini(x=T1_model.mod_res['capital_inc'],weights=T1_model.mod_res['dens'])},
}).T

income_gini_table_latex = income_gini_table.map(lambda x: f"{x:.3f}").to_latex(
    escape  = False,
    caption = 'Labour and capital income Gini indices',
    label   = 'tab:income_gini_split')

with open(OUTPUTS_QUANT_EX / 'sec_A_income_gini_split.tex','w',encoding='utf-8') as f:
    f.write(income_gini_table_latex)

### --- Income Distribtuion --- ###

T0_data,T0_w = T0_model.mod_res['y'].to_numpy(),T0_model.mod_res['dens'].to_numpy()
T1_data,T1_w = T1_model.mod_res['y'].to_numpy(),T1_model.mod_res['dens'].to_numpy()

T0_kde = scipy.stats.gaussian_kde(T0_data,weights=T0_w)
T1_kde = scipy.stats.gaussian_kde(T1_data,weights=T1_w)

x_grid = np.linspace(0,max(T0_data.max(),T1_data.max()),500)

fig,ax = plt.subplots(figsize=(7,4))

ax.plot(x_grid,T0_kde(x_grid),color='#fa003f',linewidth=1.5,label=tariff0_label)
ax.fill_between(x_grid,T0_kde(x_grid),color='#9ec5f4',alpha=0.3)
ax.plot(x_grid,T1_kde(x_grid),color='#184f95',linewidth=1.5,label=tariff1_label)
ax.fill_between(x_grid,T1_kde(x_grid),color='#184f95',alpha=0.3)

ax.set_axisbelow(True)
ax.grid(linestyle='--',alpha=0.5)
ax.set_xlabel('Income')
ax.set_ylabel('Density')
ax.legend(loc='upper right',frameon=True)

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_income_distribution.pdf')
plt.close()

### --- Creating Supply and Demand Curves for Capital in both Models --- ###
ub_diff = 1 / T0_model.ModelPar.δ - 1 - T0_model.r

r_grid = np.linspace(T0_model.r - ub_diff,1 / T0_model.ModelPar.δ - 1 - 1e-4,100)

df = dict()

for r in r_grid:

    T0_copy = GeneralEquilibriumModel(T0_ModelPar,CalibPar,log_dir=None,log_inner=False)
    T1_copy = GeneralEquilibriumModel(T1_ModelPar,CalibPar,log_dir=None,log_inner=False)

    T0_copy.inner_loop_solver(r=r)
    T0_copy.solve_household_side(r=r)

    K0_supply = (T0_copy.mod_res['dens'] * T0_copy.mod_res['a_0']).sum()
    K0_demand = T0_copy.K_demand

    T1_copy.inner_loop_solver(r=r)
    T1_copy.solve_household_side(r=r)

    K1_supply = (T1_copy.mod_res['dens'] * T1_copy.mod_res['a_0']).sum()
    K1_demand = T1_copy.K_demand

    df[r] = pd.Series({'M0_K_d': K0_demand,'M0_K_s': K0_supply,'M1_K_d': K1_demand,'M1_K_s': K1_supply})

capital_curves            = pd.DataFrame.from_dict(df,orient='index')
capital_curves.index.name = 'r'

fig,ax = plt.subplots(figsize=(7,4))

ax.plot(capital_curves.index,capital_curves['M0_K_d'],color='#fa003f',label=r'Pre-shock')
ax.plot(capital_curves.index,capital_curves['M0_K_s'],color='#fa003f',label='_')
ax.plot(capital_curves.index,capital_curves['M1_K_d'],color='#184f95',linestyle='--',label=r'Post-shock')
ax.plot(capital_curves.index,capital_curves['M1_K_s'],color='#184f95',linestyle='--',label='_')
ax.legend(loc=0)
ax.grid(linestyle='--')
ax.set_xlabel('Interest rate ($r$)')
ax.set_ylabel('Capital')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=2))

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_K_demand_supply.pdf')
plt.close()

### --- Disagg. Comparison - Table --- ###

for model in (T0_model,T1_model):
    model.mod_res['u_c']      = model.mod_res['c'] ** (1 - model.ModelPar.σ) / (1 - model.ModelPar.σ)
    model.mod_res['cont_val'] = model.mod_res['V'] - model.mod_res['u_c']

VAR_RENAMER = {'a_0_mean': 'Wealth',
               'y_mean': 'Income',
               'labour_inc_mean': 'Labour Income',
               'capital_inc_mean': 'Capital Income',
               'transfer_mean': 'Transfer Income',
               'V_mean': 'Welfare',
               'u_c_mean': 'Current Welfare',
               'cont_val_mean': 'Continuation Value'}

DISAG_Y_VARS = ('a_0','y','labour_inc','capital_inc','transfer','V','u_c','cont_val')


def group_row(group_var: str,iloc_pos: int,row_name: str) -> pd.Series:
    """
    Compare a household group across the two tariff equilibria.

    Parameters
    ----------
    group_var : str
        Variable defining household groups.
    iloc_pos : int
        Position of the group to report.
    row_name : str
        Display label for the group.

    Returns
    -------
    pandas.Series
        Group levels and proportional changes by income claim.
    """

    T0 = T0_model.weighted_group_table(group_var=group_var,y_vars=DISAG_Y_VARS).iloc[iloc_pos].drop(['group','dens_total']).rename(VAR_RENAMER)
    T1 = T1_model.weighted_group_table(group_var=group_var,y_vars=DISAG_Y_VARS).iloc[iloc_pos].drop(['group','dens_total']).rename(VAR_RENAMER)
    Δ  = T1 / T0 - 1

    values          = {tariff0_label: T0,tariff1_label: T1,'Δ': Δ}
    row             = pd.Series({(var,model): values[model][var] for var in T0.index for model in (tariff0_label,tariff1_label,'Δ')})
    row.index.names = ['Variable','Model']

    return row.rename(row_name)


def disag_comp(group_var: str,L_name: str,H_name: str) -> pd.DataFrame:
    """
    Compare the lowest and highest groups across tariff equilibria.

    Parameters
    ----------
    group_var : str
        Variable defining household groups.
    L_name : str
        Label for the first group.
    H_name : str
        Label for the last group.

    Returns
    -------
    pandas.DataFrame
        Two-group distributional comparison.
    """

    L_row = group_row(group_var,0,L_name)
    H_row = group_row(group_var,-1,H_name)

    return pd.concat([L_row,H_row],axis=1).T


group_vars = ['a_0','y','skill_type']
full_disag_table = pd.concat([
    disag_comp('a_0','Wealth (1st Dec.)','Wealth (10th Dec.)'),
    disag_comp('y','Income (1st Dec.)','Income (10th Dec.)'),
    disag_comp('skill_type','Low-skill','High-skill'),
])

DISAG_SPEC = [
    ('Wealth','Wealth'),
    ('Income','Income'),
    ('Labour Income','Labor Income'),
    ('Capital Income','Capital Income'),
    ('Transfer Income','Transfer Income'),
    ('Welfare','Welfare'),
]

FLIP_SIGN_VARS = ['Welfare','Current Welfare','Continuation Value']


def _format_disag_table_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format distributional changes with utility gains signed positively.

    Parameters
    ----------
    df : pandas.DataFrame
        Grouped levels and changes with a Model column level.

    Returns
    -------
    pandas.DataFrame
        Percentage-change strings for the LaTeX table.
    """

    delta = df.xs('Δ',level='Model',axis=1).copy()

    for col in FLIP_SIGN_VARS:
        delta[col] = -delta[col]

    return delta.map(lambda x: f"${x*100:+.2f}\\%$")


def _disag_tabular(df: pd.DataFrame,spec: list) -> str:
    """
    Format the selected distributional statistics as a LaTeX tabular.

    Parameters
    ----------
    df : pandas.DataFrame
        Grouped distributional comparison.
    spec : list
        Column keys and display labels.

    Returns
    -------
    str
        LaTeX tabular source.
    """

    formatted = _format_disag_table_for_latex(df)
    row_groups = [[[name] + [formatted.loc[name,key] for key,_ in spec]
                   for name in formatted.index[pos:pos + 2]]
                  for pos in range(0,len(formatted.index),2)]

    return _latex_tabular(['Variable'] + [label for _,label in spec],row_groups,label_pad=1)


disag_table_latex = _latex_table(
    [_disag_tabular(full_disag_table,DISAG_SPEC)],
    caption = 'Distributional effects of the tariff shock ($\\Delta\\tau$)',
    label   = 'tab:disagg_comparison')

with open(OUTPUTS_QUANT_EX / 'sec_A_disagg_comparison.tex','w',encoding='utf-8') as f:
    f.write(disag_table_latex)

### --- Disagg. Comparison - Chart by Wealth Groups --- ###

INCOME_SPLIT_VARS = ('a_0','y','labour_inc','capital_inc')
T0_group_table    = T0_model.weighted_group_table(group_var='y',y_vars=INCOME_SPLIT_VARS)
T1_group_table    = T1_model.weighted_group_table(group_var='y',y_vars=INCOME_SPLIT_VARS)
group_pctchange   = T1_group_table / T0_group_table - 1

fig,ax = plt.subplots(ncols=2,nrows=1,figsize=(10,3))

for a,col,name in zip(ax.flat,['a_0_mean','y_mean'],['Wealth','Income']):
    a.set_axisbelow(True)
    a.grid(linestyle='--')
    a.set_title(name)
    a.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    if name == 'Income':
        width = 0.4
        x     = group_pctchange.index
        a.bar(x=x - width / 2,height=group_pctchange['labour_inc_mean'],width=width,color='darkblue',label='Labour Income')
        a.bar(x=x + width / 2,height=group_pctchange['capital_inc_mean'],width=width,color='firebrick',label='Capital Income')
        a.scatter(x,group_pctchange[col],color='black',zorder=3,label='Total Income')
        a.legend(fontsize='small')
    else:
        a.bar(x=group_pctchange.index,height=group_pctchange[col],color='darkblue')

fig.supxlabel('Income Decile')
fig.supylabel(r'Change after tariff shock')

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_disagg_comp_wealth_groups.pdf')
plt.close()

### --- Changes in Policy Function --- ###


def polfunc_creator_comp(model: GeneralEquilibriumModel) -> tuple:
    """
    Average savings policies over productivity within each skill type.

    Parameters
    ----------
    model : GeneralEquilibriumModel
        Solved model with stationary household results.

    Returns
    -------
    tuple
        Low-skill and high-skill interpolators and the averaged policy table.
    """

    res = model.mod_res[['a_0','a_1','skill_type','z','dens']].copy()

    def agg_skill(skill: str) -> pd.Series:
        """
        Average one skill group savings policy over its productivity marginal.

        Parameters
        ----------
        skill : str
            Skill group, L or H.

        Returns
        -------
        pandas.Series
            Expected next-period assets at each current asset level.
        """

        sub  = res.loc[res['skill_type'] == skill]
        pi_z = sub.groupby('z')['dens'].sum()
        pi_z = pi_z / pi_z.sum()

        pivot = sub.pivot(index='a_0',columns='z',values='a_1')

        return pivot.mul(pi_z,axis=1).sum(axis=1)

    res = pd.concat([agg_skill('L').rename('L'),agg_skill('H').rename('H')],axis=1)

    L_interp = scipy.interpolate.interp1d(res.index,res['L'])
    H_interp = scipy.interpolate.interp1d(res.index,res['H'])

    return L_interp,H_interp,res


L_int,H_int,T0_polfunc = polfunc_creator_comp(model=T0_model)
T1_polfunc             = polfunc_creator_comp(model=T1_model)[2]

T0_polfunc['L_new'] = L_int(T1_polfunc.index)
T0_polfunc['H_new'] = H_int(T1_polfunc.index)

T0_polfunc['L_diff'] = T0_polfunc['L_new'] / T0_polfunc['L'] - 1
T0_polfunc['H_diff'] = T0_polfunc['H_new'] / T0_polfunc['H'] - 1

fig,ax = plt.subplots(ncols=2,figsize=(10,4),sharey=True)

for a,col,title in zip(ax,['L_diff','H_diff'],['Low Skill','High Skill']):
    x = T0_polfunc.index
    y = T0_polfunc[col]

    a.axhline(0,color='black',linestyle=':',linewidth=1)
    a.plot(x,y,color='#0b0b0b',linewidth=1.5)
    a.fill_between(x,y,0,where=(y >= 0),color='#2a78d6',alpha=0.3,interpolate=True,label="Higher $a'$")
    a.fill_between(x,y,0,where=(y <= 0),color='#e34948',alpha=0.3,interpolate=True,label="Lower $a'$")

    a.set_title(title)
    a.set_axisbelow(True)
    a.grid(linestyle='--',alpha=0.5)
    a.legend(loc='upper right',frameon=True)

fig.supxlabel(r'Initial Wealth $a_0$')
fig.supylabel('Perc. Change in Policy Function after tariff shock')

plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_delta_pol_func.pdf')
plt.close()

### --- Steady-State Welfare Change over the State Space --- ###
def plot_stationary_welfare(baseline: GeneralEquilibriumModel,
                            alternative: GeneralEquilibriumModel,filename: str) -> None:
    """
    Plot consumption-equivalent gains between stationary value functions.

    Parameters
    ----------
    baseline : GeneralEquilibriumModel
        Reference economy; its wealth distribution sets the 99.9% plot cutoff.
    alternative : GeneralEquilibriumModel
        Counterfactual economy with the same preferences and productivity states.
    filename : str
        PDF filename in OUTPUTS_QUANT_EX.

    Returns
    -------
    None
        Saves skill-specific maps at matching productivity and wealth states.
        This compares steady states, excluding transition costs and social weights.
    """

    wealth_mass = baseline.mod_res.groupby('a_0')['dens'].sum().sort_index()
    wealth_cdf  = wealth_mass.cumsum() / wealth_mass.sum()
    wealth_cut  = wealth_mass.index[np.searchsorted(wealth_cdf.to_numpy(),0.999)]
    gains       = []

    for skill in ('L','H'):
        base = baseline.mod_res.loc[baseline.mod_res['skill_type'] == skill].pivot(index='a_0',columns='z',values='V').sort_index().sort_index(axis=1)
        alt = alternative.mod_res.loc[alternative.mod_res['skill_type'] == skill].pivot(index='a_0',columns='z',values='V').sort_index().reindex(columns=base.columns)

        # Restrict to common asset support so the comparison never extrapolates.
        base = base.loc[(base.index <= wealth_cut) & (base.index >= alt.index.min())
                        & (base.index <= alt.index.max())]

        if len(base) < 2 or base.shape[1] < 2 or alt.isna().any().any():
            raise ValueError('Welfare maps require common productivity states and at least two asset points.')

        alt_values = scipy.interpolate.interp1d(alt.index,alt.to_numpy(),axis=0)(base.index)
        value_ratio = alt_values / base.to_numpy()

        if not np.all(np.isfinite(value_ratio)) or np.any(value_ratio <= 0):
            raise ValueError('Consumption-equivalent welfare requires finite, positive value ratios.')

        gains.append(100 * (value_ratio ** (1 / (1 - baseline.ModelPar.σ)) - 1))

    norm   = TwoSlopeNorm(vmin=min(min(g.min() for g in gains),-1e-9),vcenter=0,vmax=max(max(g.max() for g in gains),1e-9))
    levels = np.linspace(norm.vmin,norm.vmax,26)
    X,Y    = np.meshgrid(base.columns.to_numpy(),base.index.to_numpy())
    fig,ax = plt.subplots(ncols=2,figsize=(12,5),sharey=True,layout='constrained')

    for axis,title,gain in zip(ax,['Low-skill','High-skill'],gains):
        axis.contourf(X,Y,gain,levels=levels,cmap='RdYlGn',norm=norm)

        if gain.min() < 0 < gain.max():
            axis.contour(X,Y,gain,levels=[0],colors='black',linewidths=1.2)

        axis.set_title(title)
        axis.set_xlabel('Productivity z')
        axis.set_axisbelow(True)
        axis.grid(linestyle='--',alpha=0.3)

    ax[0].set_ylabel(r'Wealth $a_0$')
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='RdYlGn'),ax=ax,label='Consumption-equivalent welfare change (%)')

    fig.savefig(OUTPUTS_QUANT_EX / filename,bbox_inches='tight')
    plt.close(fig)


plot_stationary_welfare(T0_model,T1_model,'sec_A_welfare_state_space.pdf')

#################################################################
### Section B - Menu of Economies with Different Tariff Rates ###
#################################################################

# Estimating the different models
τ_grid: np.ndarray = 0.5 * np.linspace(0,1,75) ** 2


def solve_shocked_model(τ: float,warm_start_model: GeneralEquilibriumModel) -> GeneralEquilibriumModel:
    """
    Solve a tariff counterfactual with technological costs held fixed.

    Parameters
    ----------
    τ : float
        Counterfactual tariff rate in levels.
    warm_start_model : GeneralEquilibriumModel
        Previous solved economy supplying the initial value function.

    Returns
    -------
    GeneralEquilibriumModel
        Solved tariff economy with aggregate and household statistics.
    """

    ModelPar: TypeModelParameters = TypeModelParameters(
        α            = p['α'],
        γ            = p['γ'],
        ψ            = p['ψ'],
        χ            = p['χ'],
        β_eff        = p['β_eff'],
        τ            = τ,
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

    model             : GeneralEquilibriumModel = GeneralEquilibriumModel(ModelPar,CalibPar,log_dir=None,log_inner=False)
    model._vfi_V_cache                          = warm_start_model._vfi_V_cache
    model.outer_loop_solver()
    model.economy_statistics()

    return model


tariff_menu    : dict                    = {}
prev_model     : GeneralEquilibriumModel = T0_model
tariff_progress: str                     = ''

try:
    for tariff_idx,τ in enumerate(τ_grid,start=1):
        tariff_progress = f'Solving tariff {tariff_idx}/{len(τ_grid)}: tau = {τ:.2%}'

        print(f'\r{tariff_progress}',end='',flush=True)

        model          = solve_shocked_model(τ,warm_start_model=prev_model)
        tariff_menu[τ] = model
        prev_model     = model
finally:
    print('\r' + ' ' * len(tariff_progress) + '\r',end='',flush=True)

### --- Tariff Laffer Curve --- ###

laffer_τ: np.ndarray = np.array(sorted(tariff_menu),dtype=float)
laffer_revenue: np.ndarray = np.array([tariff_menu[τ].economy_stats['tariff_revenue'] for τ in laffer_τ],dtype=float)

if laffer_τ.size == 0 or not np.all(np.isfinite(laffer_revenue)):
    raise ValueError('The tariff Laffer curve requires a nonempty grid with finite revenues.')

laffer_peak_idx: int   = int(np.argmax(laffer_revenue))
τ_revenue_max  : float = float(laffer_τ[laffer_peak_idx])
revenue_max    : float = float(laffer_revenue[laffer_peak_idx])

print(f'Tariff revenue maximum on the evaluated grid: tau = {τ_revenue_max:.2%}, ' f'revenue = {revenue_max:.6f}')

if laffer_peak_idx in (0,laffer_τ.size - 1):
    print('The maximum is at a grid endpoint; extend the tariff range to locate the peak.')

laffer_fig: plt.Figure = plt.figure(figsize=(10,5))

laffer_ax : plt.Axes   = laffer_fig.add_subplot(111)
laffer_ax.plot(laffer_τ,laffer_revenue,color='#184f95',linewidth=1.5,markersize=3,label='Tariff revenue')
laffer_ax.scatter(τ_revenue_max,revenue_max,color='#e34948',zorder=3,label=rf'Grid maximum: $\tau^*={τ_revenue_max:.2%}$'.replace('%',r'\%'))
laffer_ax.axvline(τ_revenue_max,color='#e34948',linestyle='--',linewidth=1)
laffer_ax.set_xlabel(r'Tariff rate ($\tau$)')
laffer_ax.set_ylabel('Total tariff revenue (model units)')
laffer_ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
laffer_ax.set_axisbelow(True)
laffer_ax.grid(linestyle='--',alpha=0.5)
laffer_ax.legend(loc='best',frameon=True)
laffer_fig.tight_layout()

laffer_fig.savefig(OUTPUTS_QUANT_EX / 'sec_B_tariff_laffer_curve.pdf')
plt.close(laffer_fig)

### --- Aggregates and factor prices for different tariff rates --- ###
shock_A: float = τ_1

w0 = T0_model.w
s0 = T0_model.s

r0 = T0_model.r

shocks_sorted = sorted(tariff_menu.keys())
w_rel         = [tariff_menu[shock].w / w0 for shock in shocks_sorted]
s_rel         = [tariff_menu[shock].s / s0 for shock in shocks_sorted]
r_rel         = [tariff_menu[shock].r / r0 for shock in shocks_sorted]
I_vals        = [tariff_menu[shock].I for shock in shocks_sorted]
gini_vals     = [tariff_menu[shock].economy_stats['income_gini'] for shock in shocks_sorted]
Y_rel         = [tariff_menu[shock].economy_stats['real_gdp'] / T0_model.economy_stats['real_gdp'] for shock in shocks_sorted]
V_rel         = [2 - tariff_menu[shock].economy_stats['mean_V'] / T0_model.economy_stats['mean_V'] for shock in shocks_sorted]
CEV_rel       = [tariff_menu[shock].economy_stats['mean_c_eq'] / T0_model.economy_stats['mean_c_eq'] for shock in shocks_sorted]

fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(14,8))

ax[0,0].axhline(1,color='black',linestyle=':',linewidth=1)
ax[0,0].axvline(shock_A,color='#898781',linestyle='--',linewidth=1,label='Liberation Day Shock')
ax[0,0].plot(shocks_sorted,w_rel,color='#184f95',linewidth=1.5,markersize=3,label='Low-skill wage (w)')
ax[0,0].plot(shocks_sorted,s_rel,color='#9ec5f4',linewidth=1.5,markersize=3,label='High-skill wage (s)')
ax[0,0].plot(shocks_sorted,r_rel,color='#e34948',linewidth=1.5,markersize=3,label='Interest rate (r)')
ax[0,0].set_ylabel('Relative to Baseline (=1)')
ax[0,0].legend(loc='best',frameon=True)

ax[0,1].axvline(shock_A,color='#898781',linestyle='--',linewidth=1,label='Liberation Day Shock')
ax[0,1].plot(shocks_sorted,I_vals,color='#2a78d6',linewidth=1.5,markersize=3)
ax[0,1].set_ylabel('Share of Offshored Tasks')
ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax[0,1].legend(loc='best',frameon=True)

ax[1,0].axvline(shock_A,color='#898781',linestyle='--',linewidth=1,label='Liberation Day Shock')
ax[1,0].plot(shocks_sorted,gini_vals,color='#e34948',linewidth=1.5,markersize=3)
ax[1,0].set_ylabel('Income Gini')
ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax[1,0].legend(loc='best',frameon=True)

ax[1,1].axhline(1,color='black',linestyle=':',linewidth=1)
ax[1,1].axvline(shock_A,color='#898781',linestyle='--',linewidth=1,label='Liberation Day Shock')
ax[1,1].plot(shocks_sorted,Y_rel,color='#1baf7a',linewidth=1.5,markersize=3,label='GDP (Y)')
ax[1,1].plot(shocks_sorted,V_rel,color='#4a3aa7',linewidth=1.5,markersize=3,label='Mean Utility')
ax[1,1].plot(shocks_sorted,CEV_rel,color='#eda100',linewidth=1.5,markersize=3,label='Cons. Equivalent')
ax[1,1].set_ylabel('Relative to Baseline (=1)')
ax[1,1].legend(loc='best',frameon=True)

for a in ax.flat:
    a.set_axisbelow(True)
    a.grid(linestyle='--',alpha=0.5)
    a.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    a.set_xlabel(r'Tariff rate ($\tau$)')

plt.tight_layout()

plt.savefig(OUTPUTS_QUANT_EX / 'sec_B_aggregates_fac_prices.pdf')
plt.close()

### --- Wage Decomposition for the Tariff Grid --- ###
decomp_rows = []

for shock in shocks_sorted:
    model  = tariff_menu[shock]
    actual = model.w / T0_model.w - 1
    prod   = -(functions.Omega(I=model.I,ModelPar=T0_model.ModelPar) / functions.Omega(I=T0_model.I,ModelPar=T0_model.ModelPar) - 1)
    ls     = -wage_ls * (model.I - T0_model.I) / (1 - T0_model.I)
    k      = wage_k * (model.economy_stats['K'] / T0_model.economy_stats['K'] - 1)
    resid  = actual - (prod + ls + k)

    decomp_rows.append({'shock': shock,'Productivity': prod,'Labour-supply': ls,'Capital': k,'GE 2nd-order': resid,'Total': actual})

decomp_grid = pd.DataFrame(decomp_rows).set_index('shock')

components = ['Productivity','Labour-supply','Capital','GE 2nd-order']
colors     = ['#2a78d6','#1baf7a','#eda100','#898781']
spacing    = np.diff(decomp_grid.index.to_numpy())
width      = 0.8 * np.minimum(np.r_[spacing[0],spacing],np.r_[spacing,spacing[-1]]) if len(spacing) else 0.01

bottom_pos = np.zeros(len(decomp_grid))
bottom_neg = np.zeros(len(decomp_grid))

fig,ax = plt.subplots(figsize=(10,5))

for comp,color in zip(components,colors):
    vals = decomp_grid[comp].to_numpy()
    pos  = np.where(vals >= 0,vals,0)
    neg  = np.where(vals < 0,vals,0)

    ax.bar(decomp_grid.index,pos,bottom=bottom_pos,width=width,color=color,label=comp)
    ax.bar(decomp_grid.index,neg,bottom=bottom_neg,width=width,color=color)

    bottom_pos += pos
    bottom_neg += neg

ax.plot(decomp_grid.index,decomp_grid['Total'],color='#0b0b0b',linewidth=1.5,markersize=3,label='Total Wage Change')
ax.axhline(0,color='black',linestyle=':',linewidth=1)
ax.axvline(shock_A,color='#898781',linestyle='--',linewidth=1,label='Liberation Day Shock')

ax.set_axisbelow(True)
ax.grid(linestyle='--',alpha=0.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel(r'Tariff rate ($\tau$)')
ax.set_ylabel(r'Contribution to $\Delta w$')
ax.legend(loc='best',frameon=True)

plt.tight_layout()

plt.savefig(OUTPUTS_QUANT_EX / 'sec_B_wage_disagg.pdf')
plt.close()

### --- Social Welfare Weights by Stationary Income Decile --- ###
def stationary_decile_welfare(model: GeneralEquilibriumModel) -> np.ndarray:
    """
    Average stationary value within exact population deciles of total income.

    Parameters
    ----------
    model : GeneralEquilibriumModel
        Solved economy with income, value and stationary density in mod_res.

    Returns
    -------
    numpy.ndarray
        Ten decile means, ranked within this economy. Tied incomes share
        their pooled mean value; mass crossing a decile boundary is split.
    """

    res       = model.mod_res[['y','V','dens']].copy()
    res['V_mass'] = res['V'] * res['dens']
    income    = res.groupby('y',sort=True)[['dens','V_mass']].sum()
    income    = income.loc[income['dens'] > 0]
    mass      = income['dens'].to_numpy()
    values    = income['V_mass'].to_numpy() / mass
    upper     = np.cumsum(mass / mass.sum())
    lower     = np.r_[0,upper[:-1]]
    edges     = np.linspace(0,1,11)
    overlaps  = np.maximum(0,np.minimum(upper[:,None],edges[1:]) - np.maximum(lower[:,None],edges[:-1]))

    return 10 * (overlaps.T @ values)


# Ranks are recomputed in each stationary economy; the social weights stay fixed.
welfare_models: dict = {**tariff_menu,τ_0: T0_model,τ_1: T1_model}
decile_values: pd.DataFrame = pd.DataFrame({τ: stationary_decile_welfare(model) for τ,model in welfare_models.items()},index=pd.Index(range(1,11),name='income_decile')).T.sort_index()
decile_values.index.name = 'tariff'
decile_delta: np.ndarray = (decile_values.loc[τ_1] - decile_values.loc[τ_0]).to_numpy()
decile_rank : np.ndarray = np.linspace(0,1,10)

# With t = exp(-eta/9), neutrality is the polynomial sum_d delta_V[d] * t**d = 0.
# Checking all real roots avoids assuming that weighted welfare is monotone in eta.

if np.all(decile_delta == 0):
    neutral_eta: float = 0.0
else:
    neutral_roots = np.polynomial.polynomial.polyroots(decile_delta / np.max(np.abs(decile_delta)))
    neutral_t     = [root.real for root in neutral_roots
                     if abs(root.imag) < 1e-8 and 0 < root.real <= 1 + 1e-10]
    neutral_eta   = float(-9 * np.log(min(max(neutral_t),1))) if neutral_t else np.nan

equal_welfare: pd.Series = decile_values.mean(axis=1)
equal_change : pd.Series = (equal_welfare - equal_welfare.loc[τ_0]) / abs(equal_welfare.loc[τ_0])

print(f'Equal-weight stationary welfare change in section A: {equal_change.loc[τ_1]:+.4%}')

if not np.isfinite(neutral_eta):
    print('No finite pro-poor exponential decile weighting makes section A welfare-neutral.')
else:
    social_weights: np.ndarray = np.exp(-neutral_eta * decile_rank)
    social_weights             = social_weights / social_weights.mean()
    weighted_welfare: pd.Series = decile_values @ social_weights / 10
    weighted_change : pd.Series = (weighted_welfare - weighted_welfare.loc[τ_0]) / abs(weighted_welfare.loc[τ_0])

    if abs(weighted_change.loc[τ_1]) > 1e-8:
        raise ValueError('The computed social weights do not satisfy welfare neutrality.')


    welfare_optimal_tariff: float = float(weighted_welfare.idxmax())
    welfare_weights: pd.DataFrame = pd.DataFrame({
        'equal_weight': np.ones(10),
        'neutral_weight': social_weights,
        'social_weight_share': social_weights / 10,
        'T0_mean_V': decile_values.loc[τ_0].to_numpy(),
        'T1_mean_V': decile_values.loc[τ_1].to_numpy(),
        'delta_V': decile_delta,
    },index=pd.Index(range(1,11),name='income_decile'))
    welfare_comparison: pd.DataFrame = pd.DataFrame({
        'equal_welfare': equal_welfare,
        'neutral_weight_welfare': weighted_welfare,
        'equal_change': equal_change,
        'neutral_weight_change': weighted_change,
    })

    print(f'Neutrality eta: {neutral_eta:.6f}; bottom/top decile weight ratio: '
          f'{np.exp(neutral_eta):.4f}; poorest 50% social weight: {social_weights[:5].sum() / 10:.2%}')
    print(f'Welfare-maximizing evaluated tariff: {welfare_optimal_tariff:.2%}; ' f'weighted welfare change: {weighted_change.loc[welfare_optimal_tariff]:+.4%}')
    print(welfare_weights[['neutral_weight','social_weight_share']].to_string())

    # Map rank weights to baseline income, averaging weights at split decile boundaries.

    income_mass = T0_model.mod_res.groupby('y',sort=True)['dens'].sum()
    income_mass = income_mass.loc[income_mass > 0]
    income_prob = income_mass.to_numpy() / income_mass.sum()
    upper       = np.cumsum(income_prob)
    lower       = np.r_[0,upper[:-1]]
    edges       = np.linspace(0,1,11)
    overlaps    = np.maximum(0,np.minimum(upper[:,None],edges[1:]) - np.maximum(lower[:,None],edges[:-1]))
    income_weights = (overlaps @ social_weights) / income_prob

    fig,ax = plt.subplots(ncols=2,figsize=(13,5.5),gridspec_kw={'width_ratios': [1,1.65]},layout='constrained')

    ax[0].axhline(1,color='#898781',linestyle='--',linewidth=1.5,label='Equal weights')
    ax[0].step(income_mass.index,income_weights,where='mid',color='#184f95',linewidth=1.5,label='Welfare-neutral weights')
    ax[0].set_xlabel('Baseline income (including transfers)')
    ax[0].set_ylabel('Social weight per household')
    ax[0].set_xlim(0,4.5)
    ax[0].legend(loc='upper right',fontsize=11)

    ax[1].plot(equal_change.index,equal_change,color='#898781',label='Equal weights')
    ax[1].plot(weighted_change.index,weighted_change,color='#184f95',label='Neutrality weights')
    ax[1].axhline(0,color='black',linestyle=':',linewidth=1)
    ax[1].axvline(τ_0,color='black',linestyle=':',linewidth=1)
    ax[1].axvline(τ_1,color='#898781',linestyle='--',linewidth=1)

    for tariff,label in [(τ_0,f'Baseline tariff ({τ_0:.1%})'),(τ_1,f'Liberation Day tariff ({τ_1:.1%})')]:
        ax[1].annotate(
            label,
            xy=(tariff,0.06),
            xycoords=ax[1].get_xaxis_transform(),
            xytext=(4,0),
            textcoords='offset points',
            rotation=90,
            ha='left',
            va='bottom',
            fontsize=11,
            bbox={'facecolor': 'white','edgecolor': 'none','alpha': 0.85,'pad': 1.5})

    equal_optimal_tariff: float = float(equal_welfare.idxmax())
    ax[1].scatter(equal_optimal_tariff,equal_change.loc[equal_optimal_tariff],color='#1baf7a',marker='o',zorder=3,label='Equal-weight grid maximum')
    ax[1].scatter(welfare_optimal_tariff,weighted_change.loc[welfare_optimal_tariff],color='#e34948',zorder=3,label='Weighted grid maximum')
    ax[1].set_xlabel(r'Tariff rate ($\tau$)')
    ax[1].set_ylabel('Welfare change (% of |baseline welfare|)')
    ax[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax[1].set_xlim(0,0.18)
    ax[1].set_ylim(-0.008,0.002)
    ax[1].legend(loc='upper right',fontsize=11,framealpha=0.95)

    for a in ax:
        a.xaxis.label.set_size(12)
        a.yaxis.label.set_size(12)
        a.tick_params(axis='both',labelsize=11)
        a.set_axisbelow(True)
        a.grid(linestyle='--',alpha=0.5)

    plt.savefig(OUTPUTS_QUANT_EX / 'sec_B_weighted_welfare.pdf')
    plt.close()

    plot_stationary_welfare(T0_model,welfare_models[welfare_optimal_tariff],'sec_B_optimal_welfare_state_space.pdf')

### --- Household Values for Condorcet Voting --- ###
# Compare tariffs at fixed baseline states.

preference_models: dict = {**tariff_menu,τ_0: T0_model,τ_1: T1_model}
preference_tariffs: np.ndarray = np.array(sorted(preference_models))
preferred_tariffs: pd.DataFrame = T0_model.mod_res[['skill_type','z','a_0','dens']].copy().reset_index(drop=True)
state_values: np.ndarray = np.full((len(preferred_tariffs),len(preference_tariffs)),np.nan)

for tariff_idx,tariff in enumerate(preference_tariffs):
    result = preference_models[tariff].mod_res

    for (skill,z),states in preferred_tariffs.groupby(['skill_type','z']):
        values = result.loc[(result['skill_type'] == skill) & (result['z'] == z)].sort_values('a_0')
        value_interp = scipy.interpolate.interp1d(values['a_0'],values['V'],bounds_error=False,fill_value=np.nan)
        state_values[states.index,tariff_idx] = value_interp(states['a_0'])

        del value_interp

# Restrict voting to common asset support; do not extrapolate.

common_states: np.ndarray = np.all(np.isfinite(state_values),axis=1)
covered_mass: float = preferred_tariffs.loc[common_states,'dens'].sum()

if covered_mass <= 0:
    raise ValueError('No baseline population mass lies within the common solved state space.')

### --- Condorcet Winner under Baseline Population Voting --- ###

voter_values : np.ndarray = state_values[common_states]
voter_weights: np.ndarray = preferred_tariffs.loc[common_states,'dens'].to_numpy() / covered_mass
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

print('\nCondorcet election: fixed baseline electorate on common asset support; exact utility ties abstain.')

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

preference_values : np.ndarray = state_values[common_states]
preference_states: pd.DataFrame = preferred_tariffs.loc[common_states].copy()
peak_indices     : np.ndarray = np.argmax(preference_values,axis=1)

# Exact utility ties select the lower tariff; the median weights households by baseline mass.
preference_states['preferred_tariff'] = preference_tariffs[peak_indices]

preferred_tariff_distribution: pd.Series = preference_states.groupby('preferred_tariff')['dens'].sum() / covered_mass
preference_cdf               : pd.Series = preferred_tariff_distribution.cumsum()
median_index                : int       = int(np.searchsorted(preference_cdf.to_numpy(),0.5))
median_preferred_tariff      : float     = float(preference_cdf.index[median_index])
coverage_share              : float     = covered_mass / preferred_tariffs['dens'].sum()

median_rows = [
    ['Median preferred tariff',f'{median_preferred_tariff:.4%}'.replace('%',r'\%')],
    ['Baseline population covered',f'{coverage_share:.4%}'.replace('%',r'\%')],
    ['Lowest evaluated tariff',f'{preference_tariffs.min():.2%}'.replace('%',r'\%')],
    ['Highest evaluated tariff',f'{preference_tariffs.max():.2%}'.replace('%',r'\%')]]
median_tabular = _latex_tabular(['Statistic','Value'],[median_rows])
median_table   = _latex_table([median_tabular],'Median preferred tariff under baseline population weights','tab:median_preferred_tariff')

with open(OUTPUTS_QUANT_EX / 'sec_B_median_preferred_tariff.tex','w',encoding='utf-8') as f:
    f.write(median_table)

print(f'Median preferred tariff: {median_preferred_tariff:.4%}; baseline population covered: {coverage_share:.4%}.')

del median_index,median_rows,median_tabular,median_table

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
fig.supylabel('Utility gap relative to each state\'s absolute peak utility',fontsize=12)
fig.savefig(OUTPUTS_QUANT_EX / 'sec_B_utility_profiles.pdf')
plt.close(fig)
