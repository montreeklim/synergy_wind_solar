#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
warnings.filterwarnings("ignore", message="Downloading:.*")

import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import natural_earth
import geopandas as gpd
from shapely.geometry import MultiPolygon
from matplotlib.colors import LogNorm
import matplotlib.patheffects as PathEffects

# —————————————————————————————————————————————————————————————
# 1) Load & prep the raw data
pv   = pd.read_csv('../data/EMHIRES_PV_2015.csv',   parse_dates=['Date'])
wind = pd.read_csv('../data/EMHIRES_wind_2015.csv', parse_dates=['Date'])

drop_cols = ['Date','Year','Month','Day','Hour','CY']
pv   = pv.drop(columns=[c for c in drop_cols if c in pv.columns])
wind = wind.drop(columns=[c for c in drop_cols if c in wind.columns])

installed_capacity_2015 = {
    "AT": {"wind":1981,  "solar":404},  "BE":{"wind":2172,"solar":3068},
    "BG": {"wind":701,   "solar":1041}, "CH":{"wind":60,  "solar":756},
    "CZ": {"wind":277,   "solar":2067}, "DE":{"wind":43429,"solar":38411},
    "DK": {"wind":5082,  "solar":781},  "EE":{"wind":301, "solar":6},
    "ES": {"wind":23003, "solar":6967}, "FI":{"wind":1082,"solar":11},
    "FR": {"wind":10312, "solar":6192}, "EL":{"wind":1775,"solar":2444},
    "HR": {"wind":384,   "solar":44},   "HU":{"wind":328, "solar":29},
    "IE": {"wind":2400,  "solar":1},    "IT":{"wind":8750,"solar":19100},
    "LT": {"wind":290,   "solar":69},   "LU":{"wind":60,  "solar":116},
    "LV": {"wind":70,    "solar":2},    "NL":{"wind":3641,"solar":1429},
    "NO": {"wind":860,   "solar":14},   "PL":{"wind":5186,"solar":87},
    "PT": {"wind":4826,  "solar":429},  "RO":{"wind":2923,"solar":1249},
    "SI": {"wind":3,     "solar":263},  "SK":{"wind":3,   "solar":532},
    "SE": {"wind":3029,  "solar":104},  "UK":{"wind":13563,"solar":9000},
}

valid = [c for c,cap in installed_capacity_2015.items() 
         if cap['wind']>0 and cap['solar']>0]
pv   = pv[valid];   wind = wind[valid]

pv_gen   = pv.mul({c:installed_capacity_2015[c]['solar'] for c in valid}, axis=1)
wind_gen = wind.mul({c:installed_capacity_2015[c]['wind']  for c in valid}, axis=1)

mean_solar = pv_gen.mean()
mean_wind  = wind_gen.mean()

iso_to_name = {
    'AT':'Austria','BE':'Belgium','BG':'Bulgaria','CH':'Switzerland',
    'CZ':'Czech Republic','DE':'Germany','DK':'Denmark','EE':'Estonia',
    'ES':'Spain','FI':'Finland','FR':'France','EL':'Greece','HR':'Croatia',
    'HU':'Hungary','IE':'Ireland','IT':'Italy','LT':'Lithuania',
    'LU':'Luxembourg','LV':'Latvia','NL':'Netherlands','NO':'Norway',
    'PL':'Poland','PT':'Portugal','RO':'Romania','SI':'Slovenia',
    'SK':'Slovakia','SE':'Sweden','UK':'United Kingdom'
}
mean_wind  = mean_wind.rename(index=iso_to_name)
mean_solar = mean_solar.rename(index=iso_to_name)

# —————————————————————————————————————————————————————————————
# 2) Build the GeoDataFrame of Europe
shp   = natural_earth('110m','cultural','admin_0_countries')
world = gpd.read_file(shp)

europe = (
    world[world['ADMIN'].isin(mean_wind.index)]
     .set_index('ADMIN')
     .assign(mean_wind=mean_wind, mean_solar=mean_solar)
     .explode(index_parts=False)
)

def keep_main(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return geom

europe['geometry'] = europe.geometry.map(keep_main)
europe = europe.to_crs(epsg=4326)

# Mask outliers by representative_point()
rp = europe.geometry.representative_point()
europe['rp_lon'], europe['rp_lat'] = rp.x, rp.y

mask = (
    (europe.rp_lon < -10) | (europe.rp_lon > 35) |
    (europe.rp_lat < 35)   | (europe.rp_lat > 72)
)
europe = europe[~mask].drop(columns=['rp_lon','rp_lat'])

# For labeling by ISO2
name_to_iso = {v:k for k,v in iso_to_name.items()}

# —————————————————————————————————————————————————————————————
# 3) Set up scales & plot
cmap      = 'viridis'
wind_norm = LogNorm(vmin=europe['mean_wind'].min()+1,
                    vmax=europe['mean_wind'].max())
solar_norm= LogNorm(vmin=europe['mean_solar'].min()+1,
                    vmax=europe['mean_solar'].max())

fig, axes = plt.subplots(
    1, 2, figsize=(16,8),
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# Super‐title and spacing
fig.suptitle(
    "Mean Hourly Generation by Country (2015, MW, log scale)",
    fontsize=16, fontweight='bold'
)
# reduce the space *between* maps, leave room at top for suptitle
fig.subplots_adjust(wspace=-0.42, top=0.85)

for ax, (col, norm, short_title) in zip(
    axes,
    [
      ('mean_wind',  wind_norm,  'Wind Generation'),
      ('mean_solar', solar_norm, 'PV Generation')
    ]
):
    # short subplot title
    ax.set_title(short_title, fontsize=14, pad=12)
    ax.set_extent([-10, 35, 35, 72], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', color='gray')

    europe.plot(
        column=col,
        cmap=cmap,
        norm=norm,
        ax=ax,
        transform=ccrs.PlateCarree(),
        edgecolor='white',
        linewidth=0.8,
        legend=False
    )

    # horizontal colorbar exactly under this map
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    pos = ax.get_position()
    cbar_ax = fig.add_axes([
        pos.x0,           # align left
        pos.y0 - 0.03,    # just below map
        pos.width,        # same width
        0.02              # thickness
    ])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    # you can uncomment to label the bar
    # cbar.set_label('Mean hourly generation (MW, log scale)', fontsize=12)

    # label *every* country by ISO2 inside map
    for country, row in europe.iterrows():
        iso = name_to_iso[country]
        pt  = row.geometry.representative_point()
        txt = ax.text(
            pt.x, pt.y, iso,
            transform=ccrs.PlateCarree(),
            ha='center', va='center',
            fontsize=6, fontweight='bold'
        )
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=1.5, foreground="white"),
            PathEffects.Normal()
        ])
plt.savefig('heatmap_mean_log_scale.png',
            dpi=300,            # 300 DPI for print‐quality
            bbox_inches='tight' # trim extra white space
)
plt.show()

