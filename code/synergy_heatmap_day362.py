#!/usr/bin/env python3
"""
Geographic heatmap of mean synergy ratios — day_362, tol=0.05, 384 MW.

Countries with no day_362 data (HR, IT, SI) are shown in light gray.
Colormap is diverging around 1.0: below 1 = negative synergy (red),
above 1 = positive synergy (green).
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import natural_earth
from matplotlib.colors import Normalize
import pycountry

# ── 1. Load pre-computed CI results ──────────────────────────────────────────

df_ci = pd.read_csv("../results/battery_CI_day362_tol005_384.csv")
df_map = df_ci[["country", "mean_ratio"]].rename(columns={"country": "ISO2"})

# ── 2. ISO2 → ISO3 mapping ────────────────────────────────────────────────────

iso2to3 = {c.alpha_2: c.alpha_3 for c in pycountry.countries}
iso2to3.update({
    "UK": "GBR", "EL": "GRC", "GR": "GRC", "CZ": "CZE",
    "BA": "BIH", "ME": "MNE", "MK": "MKD", "RS": "SRB", "XK": "KOS",
})
df_map["ISO3"] = df_map["ISO2"].map(iso2to3)

# ── 3. Load world geometry ────────────────────────────────────────────────────

shp = natural_earth("110m", "cultural", "admin_0_countries")
world = gpd.read_file(shp)

# All 28 study countries (including those missing day_362 data)
all_iso3 = set(df_map["ISO3"].dropna()) | {"HRV", "ITA", "SVN"}

df_geo = (
    world[world.ADM0_A3.isin(all_iso3)]
    .merge(df_map, left_on="ADM0_A3", right_on="ISO3", how="left")
    .explode(index_parts=False)
)

# ── 4. Remove overseas fragments ──────────────────────────────────────────────

df_geo = df_geo.to_crs(epsg=4326)
rp = df_geo.geometry.representative_point()
df_geo["lon"], df_geo["lat"] = rp.x, rp.y

fr_mask = (
    (df_geo.ISO2 == "FR") &
    ((df_geo.lon < -5.5) | (df_geo.lon > 8.2) |
     (df_geo.lat < 41.3) | (df_geo.lat > 51.1))
)
no_mask = (df_geo.ISO2 == "NO") & (df_geo.lat > 72)
df_geo = df_geo[~(fr_mask | no_mask)].drop(columns=["lon", "lat"])
df_geo = df_geo.to_crs(epsg=4326)

# ── 5. Split into data and missing subsets ────────────────────────────────────

df_has_data = df_geo[df_geo["mean_ratio"].notna()].copy()
df_missing  = df_geo[df_geo["mean_ratio"].isna()].copy()

# ── 6. Plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(
    figsize=(12, 8),
    subplot_kw={"projection": PlateCarree()}
)

# Sequential norm from 1.0 (no synergy) to max ratio
vmin = 1.0
vmax = df_has_data["mean_ratio"].max()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = "YlGn"

df_has_data.plot(
    column="mean_ratio",
    cmap=cmap, norm=norm,
    edgecolor="black", linewidth=0.5,
    ax=ax, transform=PlateCarree(),
    legend=False,
)

# Missing countries in gray
df_missing.plot(
    color="#CCCCCC",
    edgecolor="black", linewidth=0.5,
    ax=ax, transform=PlateCarree(),
)

# ── 7. Colorbar ───────────────────────────────────────────────────────────────

ax.set_facecolor("#f0f0f0")
ax.set_xticks([])
ax.set_yticks([])

pos = ax.get_position()
cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Mean Synergy Ratio", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# ── 8. Country labels ─────────────────────────────────────────────────────────

all_labels = df_geo.dissolve(by="ISO2").representative_point()
for iso2, pt in all_labels.geometry.items():
    txt = ax.text(
        pt.x, pt.y, iso2,
        transform=PlateCarree(),
        ha="center", va="center",
        fontsize=7, fontweight="bold",
    )
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=1.5, foreground="white"),
        PathEffects.Normal(),
    ])

# ── 9. Legend for missing countries ──────────────────────────────────────────

missing_patch = mpatches.Patch(color="#CCCCCC", label="No day_362 data (HR, IT, SI)")
ax.legend(handles=[missing_patch], loc="lower left", fontsize=10)

ax.set_title(
    "Mean Synergy Ratio by Country\n(day_362, tol=0.05, 384 MW installed capacity)",
    fontsize=15, pad=14,
)

out_path = "../results/synergy_heatmap_day362_tol005_384.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
