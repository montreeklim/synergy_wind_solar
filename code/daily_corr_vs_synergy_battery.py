"""
Scatter: per-country daily-mean PV–wind correlation vs average
storage-enhanced synergy ratio (1x installed capacity).

Daily-mean correlation captures day-to-day weather variability that a
single-day battery cannot fully smooth, making it more relevant to the
storage operating regime than instantaneous hourly correlation.

Exclusion: PV share < 0.5% or > 95% of (PV + wind) installed capacity.
"""

import glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── installed capacity & exclusion filter ─────────────────────────────────────
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

pv_share = {c: v['solar'] / (v['solar'] + v['wind']) for c, v in CAPACITY.items()}
EXCLUDED = {c for c, s in pv_share.items() if s < 0.005 or s > 0.95}
COUNTRIES = [c for c in CAPACITY if c not in EXCLUDED]
print(f'Excluded ({len(EXCLUDED)}): {sorted(EXCLUDED)}')
print(f'Countries used: {len(COUNTRIES)}')

# ── daily-mean PV–wind correlation ────────────────────────────────────────────
pv_df   = pd.read_csv('data/EMHIRES_PV_2015.csv')
wind_df = pd.read_csv('data/EMHIRES_wind_2015.csv')

# Group hourly rows by calendar day using the pre-parsed Year/Month/Day columns
pv_daily   = pv_df.groupby(['Year','Month','Day'])[COUNTRIES].mean()   # 365 × 25
wind_daily = wind_df.groupby(['Year','Month','Day'])[COUNTRIES].mean()

daily_corr = {}
for c in COUNTRIES:
    r, _ = stats.pearsonr(pv_daily[c], wind_daily[c])
    daily_corr[c] = r

corr_series = pd.Series(daily_corr, name='daily_corr')

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

# ── plot: 1 row × 3 cols (one per tol) ───────────────────────────────────────
tols = sorted(mean_ratio['tol'].unique())
fig, axes = plt.subplots(1, len(tols), figsize=(5 * len(tols), 5),
                         constrained_layout=True)

print('\nR2 summary (daily-mean correlation):')
print(f"{'tol':<8} {'slope':>8} {'intercept':>10} {'R2':>6} {'p':>8}")

for ax, tol in zip(axes, tols):
    tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')

    rows = []
    for c in COUNTRIES:
        if c in tol_data.index:
            rows.append({'country': c,
                         'corr': corr_series[c],
                         'mean_ratio': tol_data.loc[c, 'mean_ratio']})
    plot_df = pd.DataFrame(rows).dropna()

    x = plot_df['corr'].values
    y = plot_df['mean_ratio'].values

    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    ax.scatter(x, y, color='#1E1E1E', s=50, zorder=3, alpha=0.85)
    for _, row_pt in plot_df.iterrows():
        ax.annotate(row_pt['country'],
                    (row_pt['corr'], row_pt['mean_ratio']),
                    fontsize=8, xytext=(4, 3),
                    textcoords='offset points', color='#444444')
    ax.plot(x_line, y_line, color='#C0392B', linewidth=1.8, zorder=2)
    ax.text(0.05, 0.95, f'$R^2 = {r2:.3f}$\n$p = {p:.3f}$',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top')

    ax.set_title(f'$\\varepsilon$ = {tol:.2f}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Daily-Mean PV–Wind Correlation', fontsize=11)
    ax.set_ylabel('Avg Synergy Ratio (Storage-Enhanced)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    print(f'{tol:<8.2f} {slope:>8.4f} {intercept:>10.4f} {r2:>6.3f} {p:>8.4f}')

fig.suptitle('Daily-Mean PV–Wind Correlation vs Storage-Enhanced Synergy Ratio\n'
             '(1× Installed Capacity, 25 countries)', fontsize=13)

out = 'results/daily_corr_vs_synergy_battery.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'\nSaved {out}')
