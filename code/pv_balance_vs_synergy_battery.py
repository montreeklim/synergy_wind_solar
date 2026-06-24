"""
Regress storage-enhanced synergy ratio on:
  min(s, 1-s)  — portfolio balance (peaks at 0.5, equal mix)
  max(s, 1-s)  — portfolio dominance (peaks at 1.0, pure one resource)
where s = PV share of installed capacity.

All 28 countries included (no exclusion filter).
"""

import glob, re
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

features = pd.DataFrame({
    'pv_share':  pd.Series(pv_s),
    'balance':   pd.Series({c: min(pv_s[c], 1 - pv_s[c]) for c in COUNTRIES}),
    'dominance': pd.Series({c: max(pv_s[c], 1 - pv_s[c]) for c in COUNTRIES}),
}, index=COUNTRIES)

print('PV share and derived features:')
print(features.sort_values('pv_share').to_string())

# ── battery synergy ratios ────────────────────────────────────────────────────
PAT = re.compile(r'battery_([A-Z]+)_day_(\d+)_tol_([\d.]+)_100_installed_capacity')
dfs = []
for f in glob.glob('battery_results/100_installed_capacity/*.csv'):
    m = PAT.search(f)
    if not m:
        continue
    tmp = pd.read_csv(f)
    tmp['country'] = m.group(1)
    tmp['tol']     = float(m.group(3))
    dfs.append(tmp[['country', 'tol', 'ratio']])

battery_df = pd.concat(dfs, ignore_index=True)
battery_df['ratio'] = pd.to_numeric(battery_df['ratio'], errors='coerce')
battery_df = battery_df[battery_df['ratio'] > 0]
mean_ratio = (battery_df
              .groupby(['country', 'tol'])['ratio']
              .mean()
              .reset_index()
              .rename(columns={'ratio': 'mean_ratio'}))

# ── one separate plot per tolerance level ────────────────────────────────────
pred_col = 'balance'
tols = sorted(mean_ratio['tol'].unique())

print('\nIndividual R²:')
print(f"{'tol':<10}  R²")

for tol in tols:
    tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')

    rows = []
    for c in COUNTRIES:
        if c in tol_data.index:
            rows.append({'country': c,
                         'x': features.loc[c, pred_col],
                         'y': tol_data.loc[c, 'mean_ratio']})
    plot_df = pd.DataFrame(rows).dropna()
    x, y = plot_df['x'].values, plot_df['y'].values

    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2
    x_line = np.linspace(x.min(), x.max(), 200)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(x, y, color='#1E1E1E', s=45, zorder=3, alpha=0.85)
    for _, pt in plot_df.iterrows():
        ax.annotate(pt['country'], (pt['x'], pt['y']),
                    fontsize=8, xytext=(4, 3),
                    textcoords='offset points', color='#444444')
    ax.plot(x_line, slope * x_line + intercept,
            color='#C0392B', linewidth=1.8, zorder=2)
    ax.text(0.05, 0.95, f'$R^2={r2:.3f}$\n$p={p:.3f}$',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_xlabel('Minimum of penetration level', fontsize=10)
    ax.set_ylabel('Synergy ratio', fontsize=10)
    ax.tick_params(labelsize=9)

    tol_tag = f'{tol:.2f}'.replace('.', '')
    out = f'results/pv_balance_vs_battery_tol{tol_tag}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'tol={tol:.2f}  R2={r2:.3f}  saved: {out}')
