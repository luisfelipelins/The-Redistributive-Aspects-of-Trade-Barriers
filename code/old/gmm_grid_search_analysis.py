"""
Created on Mon Mar  27 21:15:31 2026

@author: lfval
"""

from GeneralEquilibriumModel import TypeModelParameters, TypeCalibParameters, GeneralEquilibriumModel
from config import OUTPUTS
import pandas as pd
from itertools import product, combinations
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the grid search results
output_path = OUTPUTS / 'calibration_20260429_090847.pkl'

with open(output_path, 'rb') as f:
    results = pickle.load(f)

# Creating the object
α_x_vec  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
α_y_vec  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
β_vec    = [1.5, 2.0, 2.5, 3.0, 3.5]
θ_vec    = [0.5, 1.0, 1.5, 2.0]
w_star_vec = [0.3, 0.4, 0.5, 0.6]

moments_df = []
for α_x, α_y, β, θ, w_star in product(α_x_vec, α_y_vec, β_vec, θ_vec, w_star_vec):

    if α_x<=α_y: pass
    else:
        index   = (α_x,α_y,β,θ,w_star)
        cur_mod = results[index]

        if cur_mod is None: pass
        else:
            economy_stats = cur_mod.economy_stats

            entry = {'α_x':α_x,
                    'α_y':α_y,
                    'β'  :β,
                    'θ'  :θ,
                    'w_star':w_star,
                    'LS_share':economy_stats['low_skill_share'],
                    'HS_share':economy_stats['high_skill_share'],
                    'K/Y':economy_stats['K/Y'],
                    'skill_premium':economy_stats['skill_premium'],
                    'income_gini':economy_stats['income_gini'],
                    'wealth_ratio':economy_stats['wealth_ratio'],
                    'income_ratio':economy_stats['income_ratio'],
                    'wealth_concentration':economy_stats['wealth_concentration'],
                    'income_concentration':economy_stats['income_concentration'],
                    'I':economy_stats['I']}

            moments_df.append(entry)

moments_df = pd.DataFrame(moments_df)

# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------

params  = ['α_x', 'α_y', 'β', 'θ', 'w_star']
moments = ['LS_share', 'HS_share', 'K/Y', 'skill_premium',
           'income_gini', 'wealth_ratio', 'income_ratio',
           'wealth_concentration', 'income_concentration', 'I']

sens = pd.DataFrame(index=params, columns=moments, dtype=float)
for p in params:
    for m in moments:
        group_means = moments_df.groupby(p)[m].mean()
        total_std   = moments_df[m].std()
        sens.loc[p, m] = (group_means.max() - group_means.min()) / total_std

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(sens.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title('Normalized sensitivity: moment range per unit of total std', pad=12)
ax.set_xlabel('Moments')
ax.set_ylabel('Parameters')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Main effects grid
# ---------------------------------------------------------------------------

moments_norm = moments_df.copy()
for m in moments:
    lo, hi = moments_df[m].min(), moments_df[m].max()
    moments_norm[m] = (moments_df[m] - lo) / (hi - lo)

fig, axes = plt.subplots(len(params), len(moments), figsize=(12, 6), sharey=True)

for i, p in enumerate(params):
    for j, m in enumerate(moments):
        ax = axes[i, j]
        group = moments_norm.groupby(p)[m]
        ax.plot(group.mean().index, group.mean().values, marker='o', markersize=3, linewidth=1.2)
        ax.fill_between(group.mean().index,
                        group.mean() - group.std(),
                        group.mean() + group.std(),
                        alpha=0.2)
        ax.set_ylim(0, 1)
        if i == 0:
            ax.set_title(m, fontsize=7, pad=4)
        if j == 0:
            ax.set_ylabel(p, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('')

fig.suptitle('Main effects: group mean ± 1 std (moments normalized to [0,1])', fontsize=10, y=1.01)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Convergence analysis
# ---------------------------------------------------------------------------

conv_rows = []
for α_x, α_y, β, θ, w_star in product(α_x_vec, α_y_vec, β_vec, θ_vec, w_star_vec):
    if α_x <= α_y:
        continue
    conv_rows.append({
        'α_x'     : α_x,
        'α_y'     : α_y,
        'β'       : β,
        'θ'       : θ,
        'w_star'  : w_star,
        'converged': int(results[(α_x, α_y, β, θ, w_star)] is not None),
    })

conv_df      = pd.DataFrame(conv_rows)
overall_rate = conv_df['converged'].mean()
print(f"Overall convergence rate: {overall_rate:.1%}  "
      f"({conv_df['converged'].sum()} / {len(conv_df)})")

# --- Marginal convergence rates ---
fig, axes = plt.subplots(1, len(params), figsize=(10, 6))
for ax, p in zip(axes, params):
    rates = conv_df.groupby(p)['converged'].mean()
    rates.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white', width=0.6)
    ax.axhline(overall_rate, color='tomato', linestyle='--', linewidth=1.2, label='overall')
    ax.set_title(p, fontsize=10)
    ax.set_ylabel('Convergence rate' if ax is axes[0] else '')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.legend(fontsize=7)

fig.suptitle('Marginal convergence rate per parameter value', fontsize=11)
plt.tight_layout()
plt.show()

# --- Pairwise convergence heatmaps ---
param_pairs = list(combinations(params, 2))

fig, axes = plt.subplots(2, 5, figsize=(14, 8))
axes = axes.flatten()

for ax, (p1, p2) in zip(axes, param_pairs):
    pivot = conv_df.groupby([p1, p2])['converged'].mean().unstack(p2)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='white', cbar=False)
    ax.set_title(f'{p1} × {p2}', fontsize=9)
    ax.tick_params(labelsize=7)

fig.suptitle('Pairwise convergence rates (averaged over remaining parameters)', fontsize=11)
plt.tight_layout()
plt.show()
