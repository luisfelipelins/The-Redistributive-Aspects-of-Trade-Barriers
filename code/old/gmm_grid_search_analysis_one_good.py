"""
Grid-search analysis for the one-good general equilibrium model.

Same diagnostic charts as the old two-good analysis, adapted for the new
parameters (α, γ, β, w_star, θ, σ, π_LL, π_HH) and the extended moment set.
Additional charts are appended at the bottom.
"""

from config import OUTPUTS
from itertools import product, combinations
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------------------------
# Load results  ← change filename here
# ---------------------------------------------------------------------------

output_path = OUTPUTS / 'grid_search_20260505_172259.pkl'

with open(output_path, 'rb') as f:
    results = pickle.load(f)

# ---------------------------------------------------------------------------
# Grid parameter vectors (must match gmm_grid_search_estimation.py)
# ---------------------------------------------------------------------------

alpha_vec  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.60]
gamma_vec  = [0.28, 0.33]
beta_vec   = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
w_star_vec = [0.1, 0.3, 0.5, 0.7]
theta_vec  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
sigma_vec  = [1.5, 2.0, 2.5]
pi_pairs   = [(0.85, 0.83), (0.90, 0.88), (0.95, 0.93)]

params  = ['α', 'γ', 'β', 'w_star', 'θ', 'σ', 'π_LL', 'π_HH']
moments = [
    'LS_share', 'HS_share',
    'K/Y', 'skill_premium', 'w_to_wstar',
    'income_gini', 'ceq_gini',
    'wealth_ratio', 'income_ratio',
    'wealth_concentration', 'income_concentration',
    'I', 'mean_V_L', 'mean_V_H',
]

# ---------------------------------------------------------------------------
# Build moments and convergence dataframes
# ---------------------------------------------------------------------------

moments_rows = []
conv_rows    = []

for α, γ, β, w_star, θ, σ, (π_LL, π_HH) in product(
        alpha_vec, gamma_vec, beta_vec, w_star_vec, theta_vec, sigma_vec, pi_pairs):

    key       = (α, γ, β, w_star, θ, σ, π_LL, π_HH)
    converged = key in results

    conv_rows.append({
        'α': α, 'γ': γ, 'β': β, 'w_star': w_star,
        'θ': θ, 'σ': σ, 'π_LL': π_LL, 'π_HH': π_HH,
        'converged': int(converged),
    })

    if not converged:
        continue

    es = results[key].economy_stats
    moments_rows.append({
        'α': α, 'γ': γ, 'β': β, 'w_star': w_star,
        'θ': θ, 'σ': σ, 'π_LL': π_LL, 'π_HH': π_HH,
        'LS_share'            : es['low_skill_share'],
        'HS_share'            : es['high_skill_share'],
        'k_share'             : es['k_share'],
        'K/Y'                 : es['K/Y'],
        'skill_premium'       : es['skill_premium'],
        'w_to_wstar'          : es['w_to_wstar'],
        'income_gini'         : es['income_gini'],
        'ceq_gini'            : es['ceq_gini'],
        'wealth_ratio'        : es['wealth_ratio'],
        'income_ratio'        : es['income_ratio'],
        'wealth_concentration': es['wealth_concentration'],
        'income_concentration': es['income_concentration'],
        'I'                   : es['I'],
        'mean_V_L'            : es['mean_V_L'],
        'mean_V_H'            : es['mean_V_H'],
    })

moments_df = pd.DataFrame(moments_rows)
conv_df    = pd.DataFrame(conv_rows)

overall_rate = conv_df['converged'].mean()
print(f"Loaded {len(results)} converged runs  "
      f"({overall_rate:.1%} of {len(conv_df)} total combinations)")

# ===========================================================================
# CHART 1 — Sensitivity heatmap
# ===========================================================================

sens = pd.DataFrame(index=params, columns=moments, dtype=float)
for p in params:
    for m in moments:
        group_means = moments_df.groupby(p)[m].mean()
        total_std   = moments_df[m].std()
        sens.loc[p, m] = (group_means.max() - group_means.min()) / total_std

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(sens.astype(float), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, linecolor='white')
ax.set_title('Normalized sensitivity: moment range per unit of total std', pad=12)
ax.set_xlabel('Moments')
ax.set_ylabel('Parameters')
plt.tight_layout()
plt.show()

# ===========================================================================
# CHART 3 — Convergence analysis
# ===========================================================================

print(f"\nOverall convergence rate: {overall_rate:.1%}  "
      f"({conv_df['converged'].sum()} / {len(conv_df)})")

# --- Marginal convergence rates per parameter ---
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
axes = axes.flatten()

for ax, p in zip(axes, params):
    rates = conv_df.groupby(p)['converged'].mean()
    rates.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white', width=0.6)
    ax.axhline(overall_rate, color='tomato', linestyle='--', linewidth=1.2, label='overall')
    ax.set_title(p, fontsize=10)
    ax.set_ylabel('Convergence rate' if ax is axes[0] else '')
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis='x', rotation=0, labelsize=8)
    ax.legend(fontsize=7)

fig.suptitle('Marginal convergence rate per parameter value', fontsize=11)
plt.tight_layout()
plt.show()

# ===========================================================================
# CHART 5 — Key moments by key parameters (4 × 4 grid)
# Rows: HS_share, skill_premium, w_to_wstar, I   Columns: α, β, w_star, θ
# ===========================================================================

chart5_params  = ['α', 'β', 'w_star', 'θ']
chart5_moments = ['HS_share', 'skill_premium', 'w_to_wstar', 'I']
chart5_mlabels = ['High-skill share', 'Skill premium (s/w)', 'w / w*', 'Offshoring activity (I)']

fig, axes = plt.subplots(4, 4, figsize=(8, 6), sharey='row')

for row, (m, ml) in enumerate(zip(chart5_moments, chart5_mlabels)):
    for col, p in enumerate(chart5_params):
        ax    = axes[row, col]
        group = moments_df.groupby(p)[m]
        ax.plot(group.mean().index, group.mean().values,
                marker='o', linewidth=1.5, color='steelblue', markersize=4)
        ax.fill_between(group.mean().index,
                        group.mean() - group.std(),
                        group.mean() + group.std(),
                        alpha=0.2, color='steelblue')
        ax.set_xlabel(p, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', linewidth=0.5, alpha=0.5)
        if row == 0:
            ax.set_title(p, fontsize=9, pad=4)
        if col == 0:
            ax.set_ylabel(ml, fontsize=9)

fig.suptitle('Key moments by key parameters (mean ± 1 std)', fontsize=12)
plt.tight_layout()
plt.show()

# ===========================================================================
# CHART 5.1 — Same as chart 5, conditioned on a fixed α value
# ← change alpha_fixed to any value in alpha_vec
# ===========================================================================

alpha_fixed = 0.40

df51 = moments_df[moments_df['α'] == alpha_fixed]
print(f"Chart 5.1: α = {alpha_fixed}  ({len(df51)} converged runs)")

chart51_params  = ['β', 'w_star', 'θ']
chart51_moments = chart5_moments
chart51_mlabels = chart5_mlabels

fig, axes = plt.subplots(4, 3, figsize=(7, 7), sharey='row')

for row, (m, ml) in enumerate(zip(chart51_moments, chart51_mlabels)):
    for col, p in enumerate(chart51_params):
        ax    = axes[row, col]
        group = df51.groupby(p)[m]
        ax.plot(group.mean().index, group.mean().values,
                marker='o', linewidth=1.5, color='steelblue', markersize=4)
        ax.fill_between(group.mean().index,
                        group.mean() - group.std(),
                        group.mean() + group.std(),
                        alpha=0.2, color='steelblue')
        ax.set_xlabel(p, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis='y', linewidth=0.5, alpha=0.5)
        if row == 0:
            ax.set_title(p, fontsize=9, pad=4)
        if col == 0:
            ax.set_ylabel(ml, fontsize=9)

fig.suptitle(f'Key moments by key parameters (α = {alpha_fixed}, mean ± 1 std)', fontsize=12)
plt.tight_layout()
plt.show()

