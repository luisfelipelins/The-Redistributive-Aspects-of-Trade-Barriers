# -*- coding: utf-8 -*-
"""
Created on Fri Jul  10 14:36:26 2026

@author: lfval
"""

from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from config import DATA_PARAMS, OUTPUTS_QUANT_EX
import functions
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import scipy.interpolate
import scipy.stats
import numpy as np
import copy
import json

# Creating the baseline model
with open(DATA_PARAMS / 'pre_gmm_params.json', 'r', encoding='utf-8') as f:
    pre_gmm = json.load(f)
with open(DATA_PARAMS / 'post_gmm_params.json', 'r', encoding='utf-8') as f:
    post_gmm = json.load(f)

CalibPar = TypeCalibParameters(
        rh_N             = 9    ,
        rh_r             = 2    ,
        rh_c             = 0    ,
        vfi_lb           = 0    ,
        vfi_ubmul        = 60   ,
        vfi_N            = 500  ,
        vfi_eps          = 1e-5 ,
        vfi_howard_steps = 20   ,
        gmc_eps          = 1e-3 ,
        kmc_eps          = 1e-3 ,
        r_init_guess     = 0.03 ,
        inner_loop_eps   = 1e-5 ,
        outer_loop_eps   = 1e-5 ,
        outer_loop_r_lb  = 0.001)

p = post_gmm['parameters']

T0_ModelPar = TypeModelParameters(
        α      = p['α']  ,
        γ      = p['γ']  ,
        β      = p['β']  ,
        w_star = p['w_star']  ,
        θ      = p['θ']  ,
        σ      = p['σ']  ,
        δ      = p['δ']  ,
        ρ      = p['ρ']  ,
        σ_ϵ    = p['σ_ϵ'],
        π_LL   = p['π_LL'],
        π_HH   = p['π_HH'],
        M      = p['M']  ,
        t_form = 'exponential')

T0_model = GeneralEquilibriumModel(T0_ModelPar, CalibPar, log_dir=None, log_inner=False)
T0_model.outer_loop_solver()
T0_stats = T0_model.economy_statistics()
T0_sol   = T0_model.mod_res

###########################################################################
### Section A - Alternative Economy with Δβ% equal to 2025 tariff shock ###
###########################################################################

# Creating post-shock economy
Δβ = 1.096/1.024

T1_ModelPar = TypeModelParameters(
        α      = p['α']  ,
        γ      = p['γ']  ,
        β      = p['β'] * Δβ ,
        w_star = p['w_star']  ,
        θ      = p['θ']  ,
        σ      = p['σ']  ,
        δ      = p['δ']  ,
        ρ      = p['ρ']  ,
        σ_ϵ    = p['σ_ϵ'],
        π_LL   = p['π_LL'],
        π_HH   = p['π_HH'],
        M      = p['M']  ,
        t_form = 'exponential')

T1_model = GeneralEquilibriumModel(T1_ModelPar, CalibPar, log_dir=None, log_inner=False)
T1_model.outer_loop_solver()
T1_stats = T1_model.economy_statistics()

for model in (T0_model, T1_model):
    model.mod_res['capital_inc'] = model.mod_res['a_0'] * model.r
    model.mod_res['labour_inc']  = model.mod_res['y'] - model.mod_res['capital_inc']

### --- Aggregates Comparison --- ###
T0_stats['Omega'] = functions.Omega(I=T0_model.I, ModelPar=T0_model.ModelPar)
T1_stats['Omega'] = functions.Omega(I=T1_model.I, ModelPar=T1_model.ModelPar)

comp_table = pd.DataFrame([T0_stats,T1_stats],index=['T0','T1']).T
comp_table['pct change'] = comp_table.pct_change(axis=1).dropna(axis=1)['T1']

actual_what = comp_table.loc['w']['pct change']

beta0_label = f"$\\beta={round(p['β'],2)}$"
beta1_label = f"$\\beta={round(p['β']* Δβ,2)}$"

def _latex_tabular(header, row_groups, label_pad=0):
    n_cols   = len(header) - 1
    all_rows = [row for group in row_groups for row in group]
    label_w  = max(len(row[0]) for row in all_rows) + label_pad
    col_w    = [max(len(row[j + 1]) for row in all_rows) for j in range(n_cols)]

    lines = [f"\\begin{{tabular}}{{l{'c' * n_cols}}}",
             '\\toprule',
             ' & '.join(header) + ' \\\\',
             '\\midrule']

    for pos, group in enumerate(row_groups):
        if pos:
            lines.append('\\addlinespace')
        for row in group:
            cells = [row[0].ljust(label_w)] + [row[j + 1].ljust(col_w[j]) for j in range(n_cols)]
            lines.append(' & '.join(cells) + ' \\\\')

    lines += ['\\bottomrule', '\\end{tabular}']

    return '\n'.join(lines)

def _latex_table(tabulars, caption, label):
    body = '\n\n\\vspace{0.4cm}\n\n'.join(tabulars)
    return (f"\\begin{{table}}[htbp]\n\\small\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\n"
            f"{body}\n\\end{{table}}\n")

AGG_SPEC = [
    ('I'            , '$I$'          , 'pp'  , 2),
    ('w'            , '$w$'          , 'lvl' , 2),
    ('s'            , '$s$'          , 'lvl' , 2),
    ('r'            , '$r$'          , 'pp'  , 2),
    ('skill_premium', 'Skill Prem.'  , 'lvl' , 2),
    ('Omega'        , '$\\Omega(I)$' , 'lvl' , 3),
    ('real_gdp'     , '$Y$'          , 'lvl' , 2),
    ('K'            , '$K$'          , 'lvl' , 2),
    ('income_gini'  , 'Inc. Gini'    , 'x100', 1),
    ('mean_V'       , 'Disc. Utility', 'flip', 2),
]

def _agg_cells(key, kind, decimals):
    v0, v1 = T0_stats[key], T1_stats[key]

    if kind == 'pp':
        return [f"{v0*100:.{decimals}f}\\%",
                f"{v1*100:.{decimals}f}\\%",
                f"{(v1 - v0)*100:+.2f}pp"]

    if kind == 'x100':
        levels = [f"{v0*100:,.{decimals}f}", f"{v1*100:,.{decimals}f}"]
    else:
        levels = [f"{v0:,.{decimals}f}", f"{v1:,.{decimals}f}"]

    change = v1 / v0 - 1
    if kind == 'flip':
        change = -change

    return levels + [f"{change*100:+.2f}\\%"]

def _agg_tabular(spec):
    rows = [['Pre-shock'], ['Post-shock'], ['$\\Delta\\%$']]

    for key, _, kind, decimals in spec:
        for row, cell in zip(rows, _agg_cells(key, kind, decimals)):
            row.append(cell)

    return _latex_tabular([''] + [label for _, label, _, _ in spec], [rows])

comp_table_latex = _latex_table(
    [_agg_tabular(AGG_SPEC[:5]), _agg_tabular(AGG_SPEC[5:])],
    caption='Aggregate effects of the offshoring-cost shock ($\\Delta\\beta$)',
    label='tab:agg_comparison')

with open(OUTPUTS_QUANT_EX / 'sec_A_agg_comparison.tex', 'w', encoding='utf-8') as f:
    f.write(comp_table_latex)

### --- Decomposing Wage Change --- ###
prod_effect = -(functions.Omega(I = T1_model.I, ModelPar = T0_model.ModelPar)/functions.Omega(I = T0_model.I, ModelPar = T0_model.ModelPar) - 1)
ls_effect   = -(T0_model.ModelPar.α+T0_model.ModelPar.γ) * (T1_model.I - T0_model.I) / (1-T0_model.I)
k_effect    = T0_model.ModelPar.γ * (T1_model.economy_stats['K']/T0_model.economy_stats['K']-1)

hatw     = prod_effect + ls_effect + k_effect
residual = actual_what - hatw

decomp_table = pd.Series({'Total Wage Change':actual_what,'Productivity':prod_effect,'Labour-supply':ls_effect,'Capital':k_effect,'GE 2nd-order':residual}, name = '$\\Delta\\%$')

decomp_table_latex = decomp_table.map(lambda x: f"{x*100:+.2f}\\%").to_frame().to_latex(
    escape=False,
    caption='Decomposition of the low-skill wage change ($\\Delta\\beta$)',
    label='tab:wage_decomp')

with open(OUTPUTS_QUANT_EX / 'sec_A_wage_decomp.tex', 'w', encoding='utf-8') as f:
    f.write(decomp_table_latex)

### --- Labour/Capital Income Gini --- ###
income_gini_table = pd.DataFrame({
    'Labour Income Gini' : {
        beta0_label: functions.weighted_gini(x=T0_model.mod_res['labour_inc'],  weights=T0_model.mod_res['dens']),
        beta1_label: functions.weighted_gini(x=T1_model.mod_res['labour_inc'],  weights=T1_model.mod_res['dens'])},
    'Capital Income Gini': {
        beta0_label: functions.weighted_gini(x=T0_model.mod_res['capital_inc'], weights=T0_model.mod_res['dens']),
        beta1_label: functions.weighted_gini(x=T1_model.mod_res['capital_inc'], weights=T1_model.mod_res['dens'])},
}).T

income_gini_table_latex = income_gini_table.map(lambda x: f"{x:.3f}").to_latex(
    escape=False,
    caption='Labour and capital income Gini indices',
    label='tab:income_gini_split')

with open(OUTPUTS_QUANT_EX / 'sec_A_income_gini_split.tex', 'w', encoding='utf-8') as f:
    f.write(income_gini_table_latex)

### --- Income Distribtuion --- ###
T0_data, T0_w = T0_model.mod_res['y'].to_numpy(), T0_model.mod_res['dens'].to_numpy()
T1_data, T1_w = T1_model.mod_res['y'].to_numpy(), T1_model.mod_res['dens'].to_numpy()

T0_kde = scipy.stats.gaussian_kde(T0_data, weights=T0_w)
T1_kde = scipy.stats.gaussian_kde(T1_data, weights=T1_w)

x_grid = np.linspace(0, max(T0_data.max(), T1_data.max()), 500)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x_grid, T0_kde(x_grid), color='#fa003f', linewidth=1.5, label=beta0_label)
ax.fill_between(x_grid, T0_kde(x_grid), color='#9ec5f4', alpha=0.3)
ax.plot(x_grid, T1_kde(x_grid), color='#184f95', linewidth=1.5, label=beta1_label)
ax.fill_between(x_grid, T1_kde(x_grid), color='#184f95', alpha=0.3)

ax.set_axisbelow(True)
ax.grid(linestyle='--', alpha=0.5)
ax.set_xlabel('Income')
ax.set_ylabel('Density')
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_income_distribution.pdf')
plt.close()

### --- Creating Supply and Demand Curves for Capital in both Models --- ###
ub_diff = 1/T0_model.ModelPar.δ - 1 - T0_model.r

r_grid = np.linspace(T0_model.r - ub_diff , 1/T0_model.ModelPar.δ - 1 - 1e-4, 100)

df = dict()

for r in r_grid:

    T0_copy = GeneralEquilibriumModel(T0_ModelPar, CalibPar, log_dir=None, log_inner=False)
    T1_copy = GeneralEquilibriumModel(T1_ModelPar, CalibPar, log_dir=None, log_inner=False)

    T0_copy.inner_loop_solver(r=r)
    T0_copy.solve_household_side(r=r)

    K0_supply = (T0_copy.mod_res['dens'] * T0_copy.mod_res['a_0']).sum()
    K0_demand = T0_copy.ModelPar.γ * T0_copy.Y / r

    T1_copy.inner_loop_solver(r=r)
    T1_copy.solve_household_side(r=r)

    K1_supply = (T1_copy.mod_res['dens'] * T1_copy.mod_res['a_0']).sum()
    K1_demand = T1_copy.ModelPar.γ * T1_copy.Y / r

    df[r] = pd.Series({'M0_K_d':K0_demand,'M0_K_s':K0_supply,'M1_K_d':K1_demand,'M1_K_s':K1_supply})

capital_curves = pd.DataFrame.from_dict(df, orient='index')
capital_curves.index.name = 'r'

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(capital_curves.index, capital_curves['M0_K_d'],color = '#fa003f',label=r'Pre-shock')
ax.plot(capital_curves.index, capital_curves['M0_K_s'],color = '#fa003f',label='_')
ax.plot(capital_curves.index, capital_curves['M1_K_d'],color = '#184f95',linestyle='--',label=r'Post-shock')
ax.plot(capital_curves.index, capital_curves['M1_K_s'],color = '#184f95',linestyle='--',label='_')
ax.legend(loc=0)
ax.grid(linestyle='--')
ax.set_xlabel('Interest rate ($r$)')
ax.set_ylabel('Capital')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_K_demand_supply.pdf')
plt.close()

### --- Disagg. Comparison - Table --- ###
for model in (T0_model, T1_model):
    model.mod_res['u_c']         = model.mod_res['c'] ** (1 - model.ModelPar.σ) / (1 - model.ModelPar.σ)
    model.mod_res['cont_val']    = model.mod_res['V'] - model.mod_res['u_c']

VAR_RENAMER = {'a_0_mean'        :'Wealth'             ,
               'y_mean'          :'Income'             ,
               'labour_inc_mean' :'Labour Income'      ,
               'capital_inc_mean':'Capital Income'     ,
               'V_mean'          :'Welfare'            ,
               'u_c_mean'        :'Current Welfare'    ,
               'cont_val_mean'   :'Continuation Value' }

DISAG_Y_VARS = ('a_0', 'y', 'labour_inc', 'capital_inc', 'V', 'u_c', 'cont_val')

def group_row(group_var, iloc_pos, row_name):

    T0 = T0_model.weighted_group_table(group_var=group_var, y_vars=DISAG_Y_VARS).iloc[iloc_pos].drop(['group','dens_total']).rename(VAR_RENAMER)
    T1 = T1_model.weighted_group_table(group_var=group_var, y_vars=DISAG_Y_VARS).iloc[iloc_pos].drop(['group','dens_total']).rename(VAR_RENAMER)
    Δ  = T1 / T0 - 1

    values = {beta0_label: T0, beta1_label: T1, 'Δ': Δ}
    row    = pd.Series({(var, model): values[model][var] for var in T0.index for model in (beta0_label, beta1_label, 'Δ')})
    row.index.names = ['Variable','Model']

    return row.rename(row_name)

def disag_comp(group_var, L_name, H_name):
    L_row = group_row(group_var, 0,  L_name)
    H_row = group_row(group_var, -1, H_name)

    return pd.concat([L_row, H_row], axis=1).T

group_vars = ['a_0', 'y', 'skill_type']
full_disag_table = pd.concat([
    disag_comp('a_0',        'Wealth (1st Dec.)', 'Wealth (10th Dec.)'),
    disag_comp('y',          'Income (1st Dec.)', 'Income (10th Dec.)'),
    disag_comp('skill_type', 'Low-skill'        , 'High-skill'),
])

DISAG_SPEC = [
    ('Wealth'        , 'Wealth'),
    ('Income'        , 'Income'),
    ('Labour Income' , 'Labor Income'),
    ('Capital Income', 'Capital Income'),
    ('Welfare'       , 'Welfare'),
]

FLIP_SIGN_VARS = ['Welfare', 'Current Welfare', 'Continuation Value']

def _format_disag_table_for_latex(df):
    delta = df.xs('Δ', level='Model', axis=1).copy()
    for col in FLIP_SIGN_VARS:
        delta[col] = -delta[col]
    return delta.map(lambda x: f"${x*100:+.2f}\\%$")

def _disag_tabular(df, spec):
    formatted  = _format_disag_table_for_latex(df)
    row_groups = [[[name] + [formatted.loc[name, key] for key, _ in spec]
                   for name in formatted.index[pos:pos + 2]]
                  for pos in range(0, len(formatted.index), 2)]

    return _latex_tabular(['Variable'] + [label for _, label in spec], row_groups, label_pad=1)

disag_table_latex = _latex_table(
    [_disag_tabular(full_disag_table, DISAG_SPEC)],
    caption='Distributional effects of the offshoring-cost shock ($\\Delta\\beta$)',
    label='tab:disagg_comparison')

with open(OUTPUTS_QUANT_EX / 'sec_A_disagg_comparison.tex', 'w', encoding='utf-8') as f:
    f.write(disag_table_latex)

### --- Disagg. Comparison - Chart by Wealth Groups --- ###
INCOME_SPLIT_VARS = ('a_0', 'y', 'labour_inc', 'capital_inc')
T0_group_table = T0_model.weighted_group_table(group_var='y', y_vars=INCOME_SPLIT_VARS)
T1_group_table = T1_model.weighted_group_table(group_var='y', y_vars=INCOME_SPLIT_VARS)
group_pctchange = T1_group_table / T0_group_table - 1

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 3))

for a, col, name in zip(ax.flat, ['a_0_mean','y_mean'], ['Wealth','Income']):
    a.set_axisbelow(True)
    a.grid(linestyle='--')
    a.set_title(name)
    a.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    if name == 'Income':
        width = 0.4
        x = group_pctchange.index
        a.bar(x=x - width/2, height=group_pctchange['labour_inc_mean'], width=width, color='darkblue', label='Labour Income')
        a.bar(x=x + width/2, height=group_pctchange['capital_inc_mean'], width=width, color='firebrick', label='Capital Income')
        a.scatter(x, group_pctchange[col], color='black', zorder=3, label='Total Income')
        a.legend(fontsize='small')
    else:
        a.bar(x=group_pctchange.index, height=group_pctchange[col], color='darkblue')

fig.supxlabel('Income Decile')
fig.supylabel(r'Change after $\beta$ shock')
plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_disagg_comp_wealth_groups.pdf')
plt.close()

### --- Changes in Policy Function --- ###
def polfunc_creator_comp(model):
    res = model.mod_res[['a_0','a_1','skill_type','z','dens']].copy()

    def agg_skill(skill):
        sub  = res.loc[res['skill_type'] == skill]
        pi_z = sub.groupby('z')['dens'].sum()
        pi_z = pi_z / pi_z.sum()

        pivot = sub.pivot(index='a_0', columns='z', values='a_1')
        return pivot.mul(pi_z, axis=1).sum(axis=1)

    res = pd.concat([agg_skill('L').rename('L'), agg_skill('H').rename('H')], axis=1)

    L_interp = scipy.interpolate.interp1d(res.index,res['L'])
    H_interp = scipy.interpolate.interp1d(res.index,res['H'])

    return L_interp, H_interp, res

L_int,H_int,T0_polfunc = polfunc_creator_comp(model=T0_model)
T1_polfunc             = polfunc_creator_comp(model=T1_model)[2]

T0_polfunc['L_new'] = L_int(T1_polfunc.index)
T0_polfunc['H_new'] = H_int(T1_polfunc.index)

T0_polfunc['L_diff'] = T0_polfunc['L_new'] / T0_polfunc['L'] - 1
T0_polfunc['H_diff'] = T0_polfunc['H_new'] / T0_polfunc['H'] - 1

fig, ax = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

for a, col, title in zip(ax, ['L_diff', 'H_diff'], ['Low Skill', 'High Skill']):
    x = T0_polfunc.index
    y = T0_polfunc[col]

    a.axhline(0, color='black', linestyle=':', linewidth=1)
    a.plot(x, y, color='#0b0b0b', linewidth=1.5)
    a.fill_between(x, y, 0, where=(y >= 0), color='#2a78d6', alpha=0.3, interpolate=True, label="Higher $a'$")
    a.fill_between(x, y, 0, where=(y <= 0), color='#e34948', alpha=0.3, interpolate=True, label="Lower $a'$")

    a.set_title(title)
    a.set_axisbelow(True)
    a.grid(linestyle='--', alpha=0.5)
    a.legend(loc='upper right', frameon=True)

fig.supxlabel(r'Initial Wealth $a_0$')
fig.supylabel('Perc. Change in Policy Function after β shock')
plt.tight_layout()
plt.savefig(OUTPUTS_QUANT_EX / 'sec_A_delta_pol_func.pdf')
plt.close()

##############################################################
### Section B - Menu of Economies with Diffferent β Shocks ###
##############################################################

# Estimating the different models 
beta_shocks = np.linspace(0.01, 0.55, 50)

def solve_shocked_model(shock, warm_start_model):
    ModelPar = TypeModelParameters(
        α      = p['α']  ,
        γ      = p['γ']  ,
        β      = p['β'] * (1 + shock),
        w_star = p['w_star']  ,
        θ      = p['θ']  ,
        σ      = p['σ']  ,
        δ      = p['δ']  ,
        ρ      = p['ρ']  ,
        σ_ϵ    = p['σ_ϵ'],
        π_LL   = p['π_LL'],
        π_HH   = p['π_HH'],
        M      = p['M']  ,
        t_form = 'exponential')

    model = GeneralEquilibriumModel(ModelPar, CalibPar, log_dir=None, log_inner=False)
    model._vfi_V_cache = warm_start_model._vfi_V_cache
    model.outer_loop_solver()
    model.economy_statistics()

    return model

beta_menu  = {}
prev_model = T0_model
for shock in beta_shocks:
    model            = solve_shocked_model(shock, warm_start_model=prev_model)
    beta_menu[shock] = model
    prev_model       = model

### --- Aggregates and factor prices for different β values --- ###
shock_A = Δβ - 1

w0 = T0_model.w
s0 = T0_model.s

r0 = T0_model.r

shocks_sorted = sorted(beta_menu.keys())
w_rel     = [beta_menu[shock].w / w0 for shock in shocks_sorted]
s_rel     = [beta_menu[shock].s / s0 for shock in shocks_sorted]
r_rel     = [beta_menu[shock].r / r0 for shock in shocks_sorted]
I_vals    = [beta_menu[shock].I for shock in shocks_sorted]
gini_vals = [beta_menu[shock].economy_stats['income_gini'] for shock in shocks_sorted]
Y_rel     = [beta_menu[shock].economy_stats['real_gdp'] / T0_model.economy_stats['real_gdp'] for shock in shocks_sorted]
V_rel     = [2 - beta_menu[shock].economy_stats['mean_V'] / T0_model.economy_stats['mean_V'] for shock in shocks_sorted]
CEV_rel   = [beta_menu[shock].economy_stats['mean_c_eq'] / T0_model.economy_stats['mean_c_eq'] for shock in shocks_sorted]

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

ax[0,0].axhline(1, color='black', linestyle=':', linewidth=1)
ax[0,0].axvline(shock_A, color='#898781', linestyle='--', linewidth=1, label='Liberation Day Shock')
ax[0,0].plot(shocks_sorted, w_rel, color='#184f95', linewidth=1.5, marker='o', markersize=3, label='Low-skill wage (w)')
ax[0,0].plot(shocks_sorted, s_rel, color='#9ec5f4', linewidth=1.5, marker='o', markersize=3, label='High-skill wage (s)')
ax[0,0].plot(shocks_sorted, r_rel, color='#e34948', linewidth=1.5, marker='o', markersize=3, label='Interest rate (r)')
ax[0,0].set_ylabel('Relative to Baseline (=1)')
ax[0,0].legend(loc='best', frameon=True)

ax[0,1].axvline(shock_A, color='#898781', linestyle='--', linewidth=1, label='Liberation Day Shock')
ax[0,1].plot(shocks_sorted, I_vals, color='#2a78d6', linewidth=1.5, marker='o', markersize=3)
ax[0,1].set_ylabel('Share of Offshored Tasks')
ax[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax[0,1].legend(loc='best', frameon=True)

ax[1,0].axvline(shock_A, color='#898781', linestyle='--', linewidth=1, label='Liberation Day Shock')
ax[1,0].plot(shocks_sorted, gini_vals, color='#e34948', linewidth=1.5, marker='o', markersize=3)
ax[1,0].set_ylabel('Income Gini')
ax[1,0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax[1,0].legend(loc='best', frameon=True)

ax[1,1].axhline(1, color='black', linestyle=':', linewidth=1)
ax[1,1].axvline(shock_A, color='#898781', linestyle='--', linewidth=1, label='Liberation Day Shock')
ax[1,1].plot(shocks_sorted, Y_rel, color='#1baf7a', linewidth=1.5, marker='o', markersize=3, label='GDP (Y)')
ax[1,1].plot(shocks_sorted, V_rel, color='#4a3aa7', linewidth=1.5, marker='o', markersize=3, label='Mean Utility')
ax[1,1].plot(shocks_sorted, CEV_rel, color='#eda100', linewidth=1.5, marker='o', markersize=3, label='Cons. Equivalent')
ax[1,1].set_ylabel('Relative to Baseline (=1)')
ax[1,1].legend(loc='best', frameon=True)

for a in ax.flat:
    a.set_axisbelow(True)
    a.grid(linestyle='--', alpha=0.5)
    a.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    a.set_xlabel(r'β Shock Size')

plt.tight_layout()

plt.savefig(OUTPUTS_QUANT_EX / 'sec_B_aggregates_fac_prices.pdf')
plt.close()

### --- Wage Decomposition for the β Grid --- ###
decomp_rows = []
for shock in shocks_sorted:
    model  = beta_menu[shock]
    actual = model.w / T0_model.w - 1
    prod   = -(functions.Omega(I=model.I, ModelPar=T0_model.ModelPar) / functions.Omega(I=T0_model.I, ModelPar=T0_model.ModelPar) - 1)
    ls     = -(T0_model.ModelPar.α + T0_model.ModelPar.γ) * (model.I - T0_model.I) / (1 - T0_model.I)
    k      = T0_model.ModelPar.γ * (model.economy_stats['K'] / T0_model.economy_stats['K'] - 1)
    resid  = actual - (prod + ls + k)

    decomp_rows.append({'shock': shock, 'Productivity': prod, 'Labour-supply': ls, 'Capital': k, 'GE 2nd-order': resid, 'Total': actual})

decomp_grid = pd.DataFrame(decomp_rows).set_index('shock')

components = ['Productivity', 'Labour-supply', 'Capital', 'GE 2nd-order']
colors     = ['#2a78d6', '#1baf7a', '#eda100', '#898781']
width      = (decomp_grid.index[1] - decomp_grid.index[0]) * 0.8 if len(decomp_grid) > 1 else 0.01

bottom_pos = np.zeros(len(decomp_grid))
bottom_neg = np.zeros(len(decomp_grid))

fig, ax = plt.subplots(figsize=(10, 5))

for comp, color in zip(components, colors):
    vals = decomp_grid[comp].to_numpy()
    pos  = np.where(vals >= 0, vals, 0)
    neg  = np.where(vals < 0, vals, 0)

    ax.bar(decomp_grid.index, pos, bottom=bottom_pos, width=width, color=color, label=comp)
    ax.bar(decomp_grid.index, neg, bottom=bottom_neg, width=width, color=color)

    bottom_pos += pos
    bottom_neg += neg

ax.plot(decomp_grid.index, decomp_grid['Total'], color='#0b0b0b', linewidth=1.5, marker='o', markersize=3, label='Total Wage Change')
ax.axhline(0, color='black', linestyle=':', linewidth=1)
ax.axvline(shock_A, color='#898781', linestyle='--', linewidth=1, label='Liberation Day Shock')

ax.set_axisbelow(True)
ax.grid(linestyle='--', alpha=0.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel(r'β Shock Size')
ax.set_ylabel(r'Contribution to $\Delta w$')
ax.legend(loc='best', frameon=True)
plt.tight_layout()

plt.savefig(OUTPUTS_QUANT_EX / 'sec_B_wage_disagg.pdf')
plt.close()