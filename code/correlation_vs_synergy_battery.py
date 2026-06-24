"""
Scatter plots: per-country PV-wind correlation (by time period) vs average
storage-enhanced synergy ratio (1x installed capacity), with OLS regression.

Time periods
  early_morning : hours 5-6   (05:00-07:00)
  daytime       : hours 7-15  (07:00-16:00)
  late_afternoon: hours 16-17 (16:00-18:00)

Exclusion: PV share < 0.5 % or > 95 % of (PV + wind) installed capacity.
"""

import glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── installed capacity & PV share ─────────────────────────────────────────────
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
print(f'Excluded countries ({len(EXCLUDED)}): {sorted(EXCLUDED)}')

COUNTRIES_28 = [c for c in CAPACITY if c not in EXCLUDED]

# ── time-period definitions (hour values present in EMHIRES) ──────────────────
PERIODS = {
    'Early Morning\n(05:00–07:00)': list(range(5, 7)),
    'Daytime\n(07:00–16:00)':       list(range(7, 16)),
    'Late Afternoon\n(16:00–18:00)':list(range(16, 18)),
}

# ── compute per-country, per-period PV-wind correlation ───────────────────────
pv_df   = pd.read_csv('data/EMHIRES_PV_2015.csv')
wind_df = pd.read_csv('data/EMHIRES_wind_2015.csv')

corr_records = []
for country in COUNTRIES_28:
    pv_series   = pv_df.set_index('Hour')[country]   # indexed by hour within each row
    wind_series = wind_df.set_index('Hour')[country]

    # rebuild with the full hourly index
    pv_full   = pv_df[['Hour', country]].rename(columns={country: 'pv'})
    wind_full = wind_df[['Hour', country]].rename(columns={country: 'wind'})
    merged = pv_full.merge(wind_full, left_index=True, right_index=True,
                           suffixes=('_pv', '_wind'))
    merged = pd.DataFrame({'hour': pv_df['Hour'].values,
                           'pv':   pv_df[country].values,
                           'wind': wind_df[country].values})

    row = {'country': country}
    for period_name, hours in PERIODS.items():
        subset = merged[merged['hour'].isin(hours)]
        r, _ = stats.pearsonr(subset['pv'], subset['wind'])
        row[period_name] = r
    corr_records.append(row)

corr_df = pd.DataFrame(corr_records).set_index('country')

# ── average synergy ratio per (country, tol) from battery 100% capacity ───────
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

# ── build combined dataset ─────────────────────────────────────────────────────
tols = sorted(mean_ratio['tol'].unique())
period_names = list(PERIODS.keys())

# ── plot: 3 rows (periods) × 3 cols (tols) ────────────────────────────────────
fig, axes = plt.subplots(3, len(tols), figsize=(5 * len(tols), 12),
                         constrained_layout=True)

for col_idx, tol in enumerate(tols):
    tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')

    for row_idx, period in enumerate(period_names):
        ax = axes[row_idx, col_idx]

        rows = []
        for c in COUNTRIES_28:
            if c in tol_data.index and c in corr_df.index:
                rows.append({'country': c,
                             'corr': corr_df.loc[c, period],
                             'mean_ratio': tol_data.loc[c, 'mean_ratio']})
        plot_df = pd.DataFrame(rows).dropna()

        x = plot_df['corr'].values
        y = plot_df['mean_ratio'].values

        # OLS regression
        slope, intercept, r, p, _ = stats.linregress(x, y)
        r2 = r ** 2
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept

        ax.scatter(x, y, color='#1E1E1E', s=40, zorder=3, alpha=0.8)
        for _, row_pt in plot_df.iterrows():
            ax.annotate(row_pt['country'],
                        (row_pt['corr'], row_pt['mean_ratio']),
                        fontsize=7, xytext=(3, 3),
                        textcoords='offset points', color='#444444')
        ax.plot(x_line, y_line, color='#C0392B', linewidth=1.5, zorder=2)
        ax.text(0.05, 0.93, f'$R^2 = {r2:.2f}$',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top')

        if row_idx == 0:
            ax.set_title(f'ε = {tol:.2f}', fontsize=13, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(period.replace('\n', ' ') + '\n\nAvg Synergy Ratio',
                          fontsize=10)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('PV–Wind Correlation', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

fig.suptitle('PV–Wind Correlation vs Storage-Enhanced Synergy Ratio\n'
             '(1× Installed Capacity, 25 countries)', fontsize=14, y=1.01)

out = 'results/correlation_vs_synergy_battery.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved {out}')

# ── print R² summary table ────────────────────────────────────────────────────
print('\nR² summary:')
print(f"{'Period':<30} " + '  '.join(f'tol={t:.2f}' for t in tols))
for period in period_names:
    r2s = []
    for tol in tols:
        tol_data = mean_ratio[mean_ratio['tol'] == tol].set_index('country')
        rows = [{'corr': corr_df.loc[c, period], 'mean_ratio': tol_data.loc[c, 'mean_ratio']}
                for c in COUNTRIES_28 if c in tol_data.index and c in corr_df.index]
        df_tmp = pd.DataFrame(rows).dropna()
        _, _, r, _, _ = stats.linregress(df_tmp['corr'], df_tmp['mean_ratio'])
        r2s.append(f'{r**2:.3f}')
    label = period.replace('\n', ' ')
    print(f'{label:<30} ' + '  '.join(r2s))
