import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REGION_MAP = {
    'IE': 'Atlantic Maritime', 'UK': 'Atlantic Maritime',
    'FR': 'Atlantic Maritime', 'BE': 'Atlantic Maritime',
    'NL': 'Atlantic Maritime',
    'DE': 'Continental',       'PL': 'Continental',
    'CZ': 'Continental',       'HU': 'Continental',
    'AT': 'Continental',       'CH': 'Continental',
    'SK': 'Continental',       'LU': 'Continental',
    'ES': 'Mediterranean',     'IT': 'Mediterranean',
    'EL': 'Mediterranean',     'HR': 'Mediterranean',
    'SI': 'Mediterranean',     'PT': 'Mediterranean',
    'BG': 'Mediterranean',     'RO': 'Mediterranean',
    'NO': 'Nordic & Baltic',   'SE': 'Nordic & Baltic',
    'FI': 'Nordic & Baltic',   'DK': 'Nordic & Baltic',
    'EE': 'Nordic & Baltic',   'LV': 'Nordic & Baltic',
    'LT': 'Nordic & Baltic',
}

REGION_ORDER = ['Continental', 'Atlantic Maritime', 'Mediterranean', 'Nordic & Baltic']

TOL_PALETTE = {
    'ε = 0.01': '#2171B5',
    'ε = 0.05': '#F16913',
    'ε = 0.10': '#238B45',
}
TOL_ORDER = list(TOL_PALETTE.keys())

df = pd.concat(
    [pd.read_csv(f) for f in glob.glob('../naive_results/naive_results_??.csv')],
    ignore_index=True,
)
df['ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
df = df[df['ratio'] > 0].copy()
df['Region'] = df['country'].map(REGION_MAP)
df = df.dropna(subset=['Region'])

TOL_LABEL_MAP = {0.01: 'ε = 0.01', 0.05: 'ε = 0.05', 0.10: 'ε = 0.10'}
df['Tolerance'] = df['tol'].map(TOL_LABEL_MAP)

sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df,
    x='Region', y='ratio',
    hue='Tolerance',
    order=REGION_ORDER,
    hue_order=TOL_ORDER,
    palette=TOL_PALETTE,
    linewidth=1.0,
    flierprops=dict(marker='o', markersize=2.5, linestyle='none', alpha=0.4),
    ax=ax,
)

ax.set_xlabel('Region', fontsize=14)
ax.set_ylabel('Synergy Ratio', fontsize=14)

ax.set_ylim(0, 22)

ax.tick_params(axis='both', labelsize=12)
plt.xticks(rotation=20, ha='right')

ax.legend(title='Tolerance', fontsize=11, title_fontsize=11,
          loc='upper right', framealpha=0.9)

plt.tight_layout()
out = '../results/naive_regional_boxplot_combined.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
