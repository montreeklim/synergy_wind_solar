#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import natural_earth
import pycountry
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as PathEffects

# --- 1) Load & convert CF → MW ----------------------------------------------
df_pv   = pd.read_csv("../data/EMHIRES_PV_2015.csv")
df_wind = pd.read_csv("../data/EMHIRES_wind_2015.csv")

caps = {
    "AT": {"wind":1981,  "solar":404},   "BE": {"wind":2172,  "solar":3068},
    "BG": {"wind":701,   "solar":1041},   "CH": {"wind":60,    "solar":756},
    "CZ": {"wind":277,   "solar":2067},   "DE": {"wind":43429, "solar":38411},
    "DK": {"wind":5082,  "solar":781},    "EE": {"wind":301,   "solar":6},
    "ES": {"wind":23003, "solar":6967},   "FI": {"wind":1082,  "solar":11},
    "FR": {"wind":10312, "solar":6192},   "EL": {"wind":1775,  "solar":2444},
    "HR": {"wind":384,   "solar":44},     "HU": {"wind":328,   "solar":29},
    "IE": {"wind":2400,  "solar":1},      "IT": {"wind":8750,  "solar":19100},
    "LT": {"wind":290,   "solar":69},     "LU": {"wind":60,    "solar":116},
    "LV": {"wind":70,    "solar":2},      "NL": {"wind":3641,  "solar":1429},
    "NO": {"wind":860,   "solar":14},     "PL": {"wind":5186,  "solar":87},
    "PT": {"wind":4826,  "solar":429},    "RO": {"wind":2923,  "solar":1249},
    "SI": {"wind":3,     "solar":263},    "SK": {"wind":3,     "solar":532},
    "SE": {"wind":3029,  "solar":104},    "UK": {"wind":13563, "solar":9000},
}

countries = list(caps.keys())

# apply installed‐capacity scaling
pv_mult   = pd.Series({c: caps[c]['solar'] for c in countries})
wind_mult = pd.Series({c: caps[c]['wind']  for c in countries})
df_pv   [countries] = df_pv   [countries].mul(pv_mult,   axis=1)
df_wind[countries] = df_wind[countries].mul(wind_mult, axis=1)

# restrict to hours 5–17 and valid countries
valid = [c for c in countries if caps[c]['wind']>0 and caps[c]['solar']>0]

# compute the hourly correlations, then average
corrs = []
for h in range(5, 18):
    p = df_pv  [df_pv['Hour']==h][valid]
    w = df_wind[df_wind['Hour']==h][valid]
    corrs.append(p.corrwith(w))
df_corrs   = pd.concat(corrs, axis=1)
mean_corrs = df_corrs.mean(axis=1)

df_map = (
    mean_corrs
    .rename("corr")
    .rename_axis("ISO2")
    .reset_index()
)

# ISO2→ISO3 for Natural Earth
iso2to3 = {c.alpha_2:c.alpha_3 for c in pycountry.countries}
iso2to3.update({
    'UK':'GBR','EL':'GRC','GR':'GRC','CZ':'CZE',
    'BA':'BIH','ME':'MNE','MK':'MKD','RS':'SRB','XK':'KOS'
})
df_map['ISO3'] = df_map['ISO2'].map(iso2to3)

# --- 2) Load Europe geometry & merge ---------------------------------------
shp   = natural_earth('110m','cultural','admin_0_countries')
world = gpd.read_file(shp)

# keep only the countries we have
df_geo = (
    world[world.ADM0_A3.isin(df_map['ISO3'])]
    .merge(df_map, left_on='ADM0_A3', right_on='ISO3')
    .explode(index_parts=False)
)

# mask stray fragments by representative_point in lon/lat
df_geo = df_geo.to_crs(epsg=4326)
rp     = df_geo.geometry.representative_point()
df_geo['lon'], df_geo['lat'] = rp.x, rp.y

fr_mask = (
    (df_geo.ISO2=='FR') &
    ((df_geo.lon< -5.5)|(df_geo.lon> 8.2)|
     (df_geo.lat<41.3)|(df_geo.lat>51.1))
)
no_mask = (df_geo.ISO2=='NO') & (df_geo.lat>72)
df_geo = df_geo[~(fr_mask|no_mask)].drop(columns=['lon','lat'])

# back to WGS84 for PlateCarree plotting
df_geo = df_geo.to_crs(epsg=4326)

# --- 3) Plot on PlateCarree for stable axis extents ------------------------
fig, ax = plt.subplots(
    1, 1,
    figsize=(12, 8),
    subplot_kw={'projection': PlateCarree()}
)

# diverging norm around zero
c = df_geo['corr'].astype(float)
absmax = c.abs().max()
norm   = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=+absmax)
cmap   = 'RdBu_r'

# fill countries
df_geo.plot(
    column='corr',
    cmap=cmap,
    norm=norm,
    edgecolor='black',
    linewidth=0.5,
    ax=ax,
    transform=PlateCarree(),
    legend=False
)

# clean up & zoom into Europe
# ax.set_extent([-10, 35, 35, 72], crs=PlateCarree())
ax.set_facecolor('#f0f0f0')
ax.set_xticks([]); ax.set_yticks([])

# get the map axes position
pos = ax.get_position()  # Bbox(x0, y0, x1, y1)

# create a new axes for the colorbar the same width as the map:
cax = fig.add_axes([
    pos.x0,           # left edge aligned with map
    pos.y0 - 0.05,    # a little below the map
    pos.width,        # same width as the map
    0.02              # height of the colorbar
])

# build the colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
# cbar.set_label('Mean PV–Wind Pearson Correlation (2015, 05–17 h)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

# one label per country (no duplicates)
labels = df_geo.dissolve(by='ISO2').representative_point()
for iso2, pt in labels.geometry.items():
    txt = ax.text(
        pt.x, pt.y, iso2,
        transform=PlateCarree(),
        ha='center', va='center',
        fontsize=7, fontweight='bold'
    )
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=1.5, foreground="white"),
        PathEffects.Normal()
    ])

# title & save
ax.set_title('Average of hourly PV–Wind Correlation\n by Country(2015, 05–17 h)',
             fontsize=16, pad=14)
# plt.tight_layout()
plt.savefig('heatmap_corr_mean.png', dpi=300, bbox_inches='tight')
plt.show()
