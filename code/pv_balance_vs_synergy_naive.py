"""
Portfolio balance min(s, 1-s) vs average naive synergy ratio.
All 28 countries, three tolerance levels.
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

CAPACITY = {
    'AT':{'wind':1981,'solar':404},'BE':{'wind':2172,'solar':3068},
    'BG':{'wind':701,'solar':1041},'CH':{'wind':60,'solar':756},
    'CZ':{'wind':277,'solar':2067},'DE':{'wind':43429,'solar':38411},
    'DK':{'wind':5082,'solar':781},'EE':{'wind':301,'solar':6},
    'ES':{'wind':23003,'solar':6967},'FI':{'wind':1082,'solar':11},
    'FR':{'wind':10312,'solar':6192},'EL':{'wind':1775,'solar':2444},
    'HR':{'wind':384,'solar':44},'HU':{'wind':328,'solar':29},
    'IE':{'wind':2400,'solar':1},'IT':{'wind':8750,'solar':19100},
    'LT':{'wind':290,'solar':69},'LU':{'wind':60,'solar':116},
    'LV':{'wind':70,'solar':2},'NL':{'wind':3641,'solar':1429},
    'NO':{'wind':860,'solar':14},'PL':{'wind':5186,'solar':87},
    'PT':{'wind':4826,'solar':429},'RO':{'wind':2923,'solar':1249},
    'SE':{'wind':3029,'solar':104},'SI':{'wind':3,'solar':263},
    'SK':{'wind':3,'solar':532},'UK':{'wind':13563,'solar':9000},
}

COUNTRIES = list(CAPACITY.keys())

pv_s = {c: v['solar'] / (v['solar'] + v['wind']) for c, v in CAPACITY.items()}
balance = pd.Series({c: min(pv_s[c], 1 - pv_s[c]) for c in COUNTRIES},
                    name='balance')

# ── naive synergy ratios ──────────────────────────────────────────────────────
df = pd.concat(
    [pd.read_csv(f) for f in glob.glob('naive_results/naive_results_*.csv')],
    ignore_index=True,
)
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df = df[df['ratio'] > 0]
mean_ratio = (df.groupby(['country', 'tol'])['ratio']
              .mean()
              .reset_index()
              .rename(columns={'ratio': 'mean_ratio'}))

# ── plot ──────────────────────────────────────────────────────────────────────
tols = sorted(mean_ratio['tol'].unique())

fig, axes = plt.subplots(1, len(tols), figsize=(5 * len(tols), 5),
                         constrained_layout=True)

print(f"{'tol':<6}  {'R2':>6}  {'p':>8}  {'slope':>8}  {'intercept':>10}")

for ax, tol in zip(axes, tols):
    tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')

    rows = []
    for c in COUNTRIES:
        if c in tol_data.index:
            rows.append({'country': c,
                         'x': balance[c],
                         'y': tol_data.loc[c, 'mean_ratio']})
    plot_df = pd.DataFrame(rows).dropna()
    x, y = plot_df['x'].values, plot_df['y'].values

    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2
    x_line = np.linspace(x.min(), x.max(), 200)

    ax.scatter(x, y, color='#1E1E1E', s=45, zorder=3, alpha=0.85)
    for _, pt in plot_df.iterrows():
        ax.annotate(pt['country'], (pt['x'], pt['y']),
                    fontsize=8, xytext=(4, 3),
                    textcoords='offset points', color='#444444')
    ax.plot(x_line, slope * x_line + intercept,
            color='#C0392B', linewidth=1.8, zorder=2)
    ax.text(0.05, 0.95, f'$R^2={r2:.3f}$\n$p={p:.3f}$',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_title(f'$\\varepsilon$ = {tol:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('min(s, 1–s)', fontsize=11)
    ax.set_ylabel('Avg Synergy Ratio (Naive)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    print(f'{tol:<6.2f}  {r2:>6.3f}  {p:>8.4f}  {slope:>8.3f}  {intercept:>10.3f}')

fig.suptitle('Portfolio Balance vs Naive Synergy Ratio\n'
             '(all 28 countries)', fontsize=13)

out = 'results/pv_balance_vs_synergy_naive.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'\nSaved {out}')
