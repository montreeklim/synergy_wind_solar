"""
Relate storage-enhanced synergy ratio to:
  - PV share  (solar / (solar + wind) installed capacity)
  - Mean PV generation   (MW·h/yr, capacity factor × installed)
  - Mean wind generation (MW·h/yr, capacity factor × installed)

Individual scatter + OLS per predictor, then a combined multiple regression.
Exclusion: PV share < 0.5% or > 95%.
"""

import glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ── installed capacity ─────────────────────────────────────────────────────────
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

pv_share_raw = {c: v['solar'] / (v['solar'] + v['wind']) for c, v in CAPACITY.items()}
EXCLUDED = {c for c, s in pv_share_raw.items() if s < 0.005 or s > 0.95}
COUNTRIES = [c for c in CAPACITY if c not in EXCLUDED]
print(f'Excluded: {sorted(EXCLUDED)}  |  Using: {len(COUNTRIES)} countries')

# ── mean annual generation from EMHIRES (capacity factor × installed, MW) ─────
pv_cf   = pd.read_csv('data/EMHIRES_PV_2015.csv')[COUNTRIES].mean()   # mean cf
wind_cf = pd.read_csv('data/EMHIRES_wind_2015.csv')[COUNTRIES].mean()

features = pd.DataFrame({
    'pv_share':   pd.Series({c: pv_share_raw[c] for c in COUNTRIES}),
    'pv_gen':     pd.Series({c: pv_cf[c] * CAPACITY[c]['solar'] for c in COUNTRIES}),
    'wind_gen':   pd.Series({c: wind_cf[c] * CAPACITY[c]['wind']  for c in COUNTRIES}),
})
features['total_gen'] = features['pv_gen'] + features['wind_gen']

print('\nFeature summary:')
print(features.describe().round(3))

# ── average battery synergy ratio per (country, tol) ─────────────────────────
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

# ── plot: rows = predictors, cols = tols ──────────────────────────────────────
PREDICTORS = {
    'PV Share':              'pv_share',
    'Mean PV Generation (MW)':   'pv_gen',
    'Mean Wind Generation (MW)': 'wind_gen',
}
tols = sorted(mean_ratio['tol'].unique())

fig, axes = plt.subplots(len(PREDICTORS), len(tols),
                         figsize=(5 * len(tols), 4.5 * len(PREDICTORS)),
                         constrained_layout=True)

r2_table = {}
for row_idx, (pred_label, pred_col) in enumerate(PREDICTORS.items()):
    r2_table[pred_label] = {}
    for col_idx, tol in enumerate(tols):
        ax = axes[row_idx, col_idx]
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
        r2_table[pred_label][tol] = r2
        x_line = np.linspace(x.min(), x.max(), 200)

        ax.scatter(x, y, color='#1E1E1E', s=45, zorder=3, alpha=0.85)
        for _, pt in plot_df.iterrows():
            ax.annotate(pt['country'], (pt['x'], pt['y']),
                        fontsize=7.5, xytext=(4, 3),
                        textcoords='offset points', color='#444444')
        ax.plot(x_line, slope * x_line + intercept,
                color='#C0392B', linewidth=1.8, zorder=2)
        ax.text(0.05, 0.95, f'$R^2={r2:.3f}$\n$p={p:.3f}$',
                transform=ax.transAxes, fontsize=10, va='top')

        if row_idx == 0:
            ax.set_title(f'$\\varepsilon$ = {tol:.2f}', fontsize=13,
                         fontweight='bold')
        ax.set_xlabel(pred_label, fontsize=10)
        if col_idx == 0:
            ax.set_ylabel('Avg Synergy Ratio\n(Storage-Enhanced)', fontsize=10)
        ax.tick_params(labelsize=9)

fig.suptitle('Predictors of Storage-Enhanced Synergy Ratio\n'
             '(1× Installed Capacity, 25 countries)', fontsize=13)
out_scatter = 'results/pv_share_gen_vs_synergy_battery.png'
plt.savefig(out_scatter, dpi=300, bbox_inches='tight')
plt.close()
print(f'\nSaved {out_scatter}')

# ── individual R² table ───────────────────────────────────────────────────────
print('\nIndividual R² per predictor:')
header = f"{'Predictor':<30}" + ''.join(f'  tol={t:.2f}' for t in tols)
print(header)
for pred_label in PREDICTORS:
    row_str = f'{pred_label:<30}' + ''.join(
        f'  {r2_table[pred_label][t]:.3f}' for t in tols)
    print(row_str)

# ── multiple regression per tol ───────────────────────────────────────────────
print('\nMultiple regression (pv_share + pv_gen + wind_gen):')
print(f"{'tol':<6} {'R2':>6}  {'adj_R2':>8}  coefficients (standardised)")

for tol in tols:
    tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')
    rows = []
    for c in COUNTRIES:
        if c in tol_data.index:
            rows.append({**{k: features.loc[c, v] for k, v in PREDICTORS.items()},
                         'y': tol_data.loc[c, 'mean_ratio'],
                         'country': c})
    df_reg = pd.DataFrame(rows).dropna()

    X_raw = df_reg[[k for k in PREDICTORS]].values
    y_arr = df_reg['y'].values
    n, p_vars = X_raw.shape

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    reg = LinearRegression().fit(X_std, y_arr)
    y_pred = reg.predict(X_std)
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
    r2_mult = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2_mult) * (n - 1) / (n - p_vars - 1)

    coef_str = '  '.join(
        f'{k}: {c:+.4f}' for k, c in zip(PREDICTORS.keys(), reg.coef_))
    print(f'{tol:<6.2f} {r2_mult:>6.3f}  {adj_r2:>8.3f}  {coef_str}')
