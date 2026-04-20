# -*- coding: utf-8 -*-
"""
Beach width statistics + transect correlation structure (Samoa)

- Uses ONLY the master beach-width CSV
- Assumes transect ID column is 'id' (string)
- Computes:
    * Pairwise correlation matrix between transects
    * Correlation heatmap
    * Rolling-mean time series
    * Monthly & annual site-wide averages
    * Correlation vs distance
    * Semivariogram-like distance decay plot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# ------------ USER PATHS / SETTINGS -------------
master_csv     = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv"

out_table_dir  = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\statistics"
out_fig_dir    = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\statistics\figures"
save_figs      = True

rolling_n      = 5        # number of observations for rolling mean

# Path to transects (one shapefile with many lines)
transects_fp       = r"Y:\OPC\beachWidthTool\Sites\Samoa\transects\transects_10m.shp"
TRANSECT_ID_FIELD  = "id"       # ← updated: YOUR shapefile ID column
TARGET_EPSG        = 32610      # CRS in meters
# ------------------------------------------------

os.makedirs(out_table_dir, exist_ok=True)
os.makedirs(out_fig_dir, exist_ok=True)

# ---------- LOAD MASTER ----------
df = pd.read_csv(master_csv)

# If CSV has Python-colored 'transects' column, rename:
if "transect" in df.columns:
    df.rename(columns={"transect": "id"}, inplace=True)

required_cols = {"id", "date", "width_m"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Master CSV is missing columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "width_m", "id"])
df["id"] = df["id"].astype(str)
df = df.sort_values(["date", "id"])

# ---------- BUILD WIDE TABLE ----------
wide = (
    df.pivot_table(index="date",
                   columns="id",
                   values="width_m",
                   aggfunc="mean")
)

wide_clean = wide.dropna(how="any")
if wide_clean.empty:
    raise ValueError("No overlapping dates across all transects after cleaning.")

print("Wide table shape (dates x transects):", wide_clean.shape)

# ---------- 1) PAIRWISE CORRELATION ----------
corr_matrix = wide_clean.corr()

corr_csv_path = os.path.join(out_table_dir, "pairwise_correlation_matrix_master.csv")
corr_matrix.to_csv(corr_csv_path)
print(f"Pairwise correlation matrix saved → {corr_csv_path}")

# Heatmap
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)

transects = corr_matrix.columns.tolist()
ax.set_xticks(range(len(transects)))
ax.set_xticklabels(transects, rotation=45, ha="right")
ax.set_yticks(range(len(transects)))
ax.set_yticklabels(transects)

fig.colorbar(im, ax=ax, label="Correlation coefficient (r)")
ax.set_title("Pairwise Correlation Between Transects (id)")
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "pairwise_corr_heatmap_master.png"), dpi=200)
plt.show()

print("\nPairwise correlation matrix:\n", corr_matrix)

# ---------- 2) ROLLING AVERAGE ----------
wide_roll = wide.rolling(rolling_n, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(10, 5))
for tr in wide_roll.columns:
    ax.plot(wide_roll.index, wide_roll[tr], lw=1.2, label=tr)

ax.set_title(f"Rolling Mean Beach Width (window={rolling_n})")
ax.set_xlabel("Date")
ax.set_ylabel("Width (m)")
ax.grid(True, ls="--", alpha=0.5)
#ax.legend(ncol=3, fontsize=7)
plt.tight_layout()

if save_figs:
    fig.savefig(os.path.join(out_fig_dir, f"rolling_mean_{rolling_n}obs.png"), dpi=200)

plt.show()

# ---------- 3) MONTHLY & ANNUAL AVERAGES ----------
monthly = df.groupby(pd.Grouper(key="date", freq="M"))["width_m"].mean()
monthly.to_frame("width_mean_m").to_csv(
    os.path.join(out_table_dir, "monthly_mean_all_transects.csv")
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(monthly.index, monthly.values, marker="o", lw=1.5)
ax.set_title("Monthly Mean Beach Width (All Transects)")
ax.set_ylabel("Width (m)")
ax.grid(True)
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "monthly_mean_all.png"), dpi=200)
plt.show()

annual = df.groupby(pd.Grouper(key="date", freq="Y"))["width_m"].mean()
annual.to_frame("width_mean_m").to_csv(
    os.path.join(out_table_dir, "annual_mean_all_transects.csv")
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(annual.index, annual.values, marker="o", lw=1.5)
ax.set_title("Annual Mean Beach Width (All Transects)")
ax.set_ylabel("Width (m)")
ax.grid(True)
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "annual_mean_all.png"), dpi=200)
plt.show()

# ---------- 4) CORRELATION vs DISTANCE (SEMIVARIOGRAM-LIKE) ----------

tgdf = gpd.read_file(transects_fp)
tgdf = tgdf.to_crs(TARGET_EPSG)

if TRANSECT_ID_FIELD not in tgdf.columns:
    raise ValueError(f"ID field '{TRANSECT_ID_FIELD}' missing in shapefile.")

tgdf["id_str"] = tgdf[TRANSECT_ID_FIELD].astype(str)
centroids = tgdf.geometry.centroid
id_to_xy = {
    row["id_str"]: (centroids.iloc[i].x, centroids.iloc[i].y)
    for i, row in tgdf.iterrows()
}

trs = list(corr_matrix.columns)
pairs = []
for i in range(len(trs)):
    for j in range(i + 1, len(trs)):
        t1, t2 = trs[i], trs[j]
        if (t1 not in id_to_xy) or (t2 not in id_to_xy):
            continue
        x1, y1 = id_to_xy[t1]
        x2, y2 = id_to_xy[t2]
        dist = np.hypot(x2 - x1, y2 - y1)
        r = corr_matrix.loc[t1, t2]
        pairs.append({"id1": t1, "id2": t2, "distance_m": dist, "corr": r})

pairs_df = pd.DataFrame(pairs)
pairs_df["semivar"] = 1 - pairs_df["corr"]

pairs_df.to_csv(os.path.join(out_table_dir, "dist_vs_corr.csv"), index=False)

# Scatter correlation vs distance
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(pairs_df["distance_m"], pairs_df["corr"], alpha=0.7)
ax.set_xlabel("Transect separation (m)")
ax.set_ylabel("Correlation (r)")
ax.set_title("Correlation vs Distance")
ax.grid(True)
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "corr_vs_distance.png"), dpi=200)
plt.show()

# Semivariogram-style binning
nbins = 15
bins = np.linspace(0, pairs_df["distance_m"].max(), nbins + 1)
pairs_df["dist_bin"] = pd.cut(pairs_df["distance_m"], bins=bins)

gamma = pairs_df.groupby("dist_bin")["semivar"].mean()
h = pairs_df.groupby("dist_bin")["distance_m"].mean()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(h, gamma, marker="o")
ax.set_xlabel("Transect separation (m)")
ax.set_ylabel("Semivariance (γ ≈ 1 - r)")
ax.set_title("Semivariogram-like Plot")
ax.grid(True)
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "semivariogram.png"), dpi=200)
plt.show()

# =========================================================
# 5) ALONGSHORE CORRELATIONS & MAPS FOR MULTIPLE REFERENCES
# =========================================================

# --- alongshore coordinate for each transect (only once) ---
tgdf = tgdf.copy()
tgdf["centroid_x"] = tgdf.geometry.centroid.x
tgdf["centroid_y"] = tgdf.geometry.centroid.y

# sort alongshore (if shoreline is more N–S, swap to centroid_y)
tgdf_sorted = tgdf.sort_values("centroid_x")

xs = tgdf_sorted["centroid_x"].values
ys = tgdf_sorted["centroid_y"].values

s = np.zeros(len(xs))
for i in range(1, len(xs)):
    dx = xs[i] - xs[i - 1]
    dy = ys[i] - ys[i - 1]
    s[i] = s[i - 1] + np.hypot(dx, dy)

tgdf_sorted["s_coord_m"] = s  # alongshore distance in meters
id_to_s = dict(zip(tgdf_sorted["id_str"], tgdf_sorted["s_coord_m"]))

# ---------------------------------------------
# CHOOSE MULTIPLE REFERENCE TRANSECTS
# ---------------------------------------------
# Option A: pick 3 alongshore positions: updrift, middle, downdrift
n_tr = len(tgdf_sorted)
ref_ids = [
    tgdf_sorted["id_str"].iloc[0],              # near one end
    tgdf_sorted["id_str"].iloc[n_tr // 2],      # middle
    tgdf_sorted["id_str"].iloc[-1],             # other end
]

# If you prefer explicit IDs, you can instead do:
# ref_ids = ["10", "50", "120"]

print("Reference transects:", ref_ids)

# ---------------------------------------------
# LOOP OVER EACH REFERENCE TRANSECT
# ---------------------------------------------
for REF_ID in ref_ids:
    if REF_ID not in corr_matrix.columns:
        print(f"Skipping ref {REF_ID}: not in correlation matrix")
        continue

    print(f"\nProcessing reference transect: {REF_ID}")

    # correlation of every transect with this reference
    corr_ref = corr_matrix[REF_ID].rename("corr_with_ref")

    # build alongshore correlation table
    along_df = (
        pd.DataFrame({"id_str": corr_ref.index, "corr_with_ref": corr_ref.values})
          .dropna()
    )
    along_df["s_coord_m"] = along_df["id_str"].map(id_to_s)
    s_ref = id_to_s[REF_ID]
    along_df["dist_from_ref_m"] = along_df["s_coord_m"] - s_ref

    # -------- 5a) correlation vs distance from this reference --------
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(along_df["dist_from_ref_m"], along_df["corr_with_ref"], alpha=0.7)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Alongshore distance from reference transect (m)")
    ax.set_ylabel(f"Correlation r with transect {REF_ID}")
    ax.set_title(f"Alongshore Correlation Structure (ref = {REF_ID})")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    if save_figs:
        fig.savefig(
            os.path.join(out_fig_dir, f"alongshore_corr_ref_{REF_ID}.png"),
            dpi=200
        )
    plt.show()

    # -------- 5b) map colored by correlation with this reference -----
    tgdf_corr = tgdf.merge(
        corr_ref,
        left_on="id_str",
        right_index=True,
        how="left"
    )

    vals = tgdf_corr["corr_with_ref"].values
    vmax = np.nanmax(np.abs(vals))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(6, 6))
    mappable = tgdf_corr.plot(
        column="corr_with_ref",
        cmap="coolwarm",
        linewidth=2,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        ax=ax,
    )

    # arrow to reference transect centroid
    ref_geom = tgdf_corr.loc[tgdf_corr["id_str"] == REF_ID, "geometry"].iloc[0]
    ref_centroid = ref_geom.centroid
    rx, ry = ref_centroid.x, ref_centroid.y

    ax.annotate(
        f"Ref {REF_ID}",
        xy=(rx, ry),                # point to centroid (data coords)
        xytext=(0.05, 0.95),        # label pos in axes fraction
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            linewidth=1.5,
        ),
        fontsize=10,
        ha="left",
        va="top",
    )

    ax.set_title(f"Transects Colored by Correlation with {REF_ID}")
    ax.set_axis_off()

    # label colorbar
    cbar = mappable.get_figure().axes[-1]
    cbar.set_ylabel("Correlation coefficient (r)", fontsize=11)

    plt.tight_layout()
    if save_figs:
        fig.savefig(
            os.path.join(out_fig_dir, f"transects_corr_map_ref_{REF_ID}.png"),
            dpi=200
        )
    plt.show()



#plot average time series 
# ---------- SITE-AVERAGE WIDTH PER DATE (ONE LINE TIMESERIES) ----------

# average across all transects for each date
# (uses all available transects; skips NaNs by default)
daily_mean = wide.mean(axis=1)

# save to CSV (optional)
daily_mean.to_frame("mean_width_m").to_csv(
    os.path.join(out_table_dir, "daily_mean_width_all_transects.csv")
)

# plot single timeseries
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily_mean.index, daily_mean.values, lw=1.5)
ax.set_title("Mean Beach Width per Date (All Transects)")
ax.set_xlabel("Date")
ax.set_ylabel("Width (m)")
ax.grid(True, ls="--", alpha=0.5)
plt.tight_layout()

if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "daily_mean_width_all_transects.png"), dpi=200)

plt.show()

'''
OUTLIERS ANALYSIS AND TIMESERIES ROLLING AVERAGES
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- LOAD DATA ----------
samoa_ts = pd.read_csv(
    r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\statistics\daily_mean_width_all_transects.csv"
)
samo_ts = pd.read_csv(
    r"Y:\OPC\beachWidthTool\Sites\SantaMonica\BeachWidth\statistics\daily_mean_width_all_transects.csv"
)

# Standardize column names
samoa_ts.columns = ["date", "width_m"]
samo_ts.columns = ["date", "width_m"]

# Convert to datetime
samoa_ts["date"] = pd.to_datetime(samoa_ts["date"])
samo_ts["date"] = pd.to_datetime(samo_ts["date"])

# ---------- OUTLIER DETECTION (Tukey IQR) ----------
def find_outliers(df):
    q1, q3 = df["width_m"].quantile([0.25, 0.75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (df["width_m"] < low) | (df["width_m"] > high)
    return df[mask], low, high

samoa_outliers, low_sam, high_sam = find_outliers(samoa_ts)
samo_outliers, low_sm, high_sm = find_outliers(samo_ts)


#plotting the outliers


print("Samoa outliers:")
print(samoa_outliers)

print("\nSanta Monica outliers:")
print(samo_outliers)

# ---------- PLOT ----------
plt.figure(figsize=(12, 6))

# Main lines
plt.plot(samoa_ts["date"], samoa_ts["width_m"], label="Samoa Mean Width", lw=1.4)
plt.plot(samo_ts["date"], samo_ts["width_m"], label="Santa Monica Mean Width", lw=1.4)

# Outlier points
plt.scatter(samoa_outliers["date"], samoa_outliers["width_m"], color="red", label="Samoa outliers", zorder=5)
plt.scatter(samo_outliers["date"], samo_outliers["width_m"], color="black", label="Santa Monica outliers", zorder=5)

# Annotate outliers with dates
for _, r in samoa_outliers.iterrows():
    plt.annotate(r["date"].strftime("%Y-%m-%d"),
                 (r["date"], r["width_m"]),
                 textcoords="offset points",
                 xytext=(0, 8),
                 ha="center",
                 fontsize=7,
                 color="red")

for _, r in samo_outliers.iterrows():
    plt.annotate(r["date"].strftime("%Y-%m-%d"),
                 (r["date"], r["width_m"]),
                 textcoords="offset points",
                 xytext=(0, -10),
                 ha="center",
                 fontsize=7,
                 color="black")

plt.title("Mean Beach Width — Samoa vs Santa Monica\n(Outliers Annotated)")
plt.xlabel("Date")
plt.ylabel("Mean Width (m)")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()




'''
Clean Samo dataset and plot (removed 28 oulier dates)

'''
samo_cleaned = samo_ts.drop(samo_outliers.index).reset_index(drop = True)

#plot cleaned dataset for samo without outliers
plt.plot(samo_cleaned['date'], samo_cleaned['width_m'], label = "Santa Monica Beach Width", lw = 1.4)
plt.plot(samoa_ts["date"], samoa_ts["width_m"], label="Samoa Mean Width", lw=1.4)
plt.legend()
plt.show()



# =========================================================
# 5) EOF ANALYSIS: SPATIOTEMPORAL PATTERNS OF BEACH WIDTH
# =========================================================

from numpy.linalg import svd

# ----- build anomaly matrix (time × transects) -----
# wide_clean: rows = dates, columns = transect ids, no NaNs
X = wide_clean.values
dates_eof = wide_clean.index
tran_ids = wide_clean.columns

# remove temporal mean at each transect
X_anom = X - np.nanmean(X, axis=0)

# ----- EOF via SVD -----
# X_anom = U * S * Vt
U, s, Vt = svd(X_anom, full_matrices=False)

# PCs (time series) and EOFs (spatial patterns)
PCs = U * s          # shape: (ntime, nmodes)
EOFs = Vt            # shape: (nmodes, ntransects)

# variance explained by each mode
var = s**2
var_frac = var / var.sum()
# ===== PLOT VARIANCE FRACTIONS (SCREE PLOT) =====

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(np.arange(1, len(var_frac)+1), var_frac*100,
        marker='o', lw=1.5, color='dodgerblue')

ax.set_title("Variance Explained by EOF Modes (Scree Plot)")
ax.set_xlabel("EOF Mode Number")
ax.set_ylabel("Variance Explained (%)")
ax.grid(True, alpha=0.4)

if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "EOF_variance_fractions.png"), dpi=200)

plt.show()


# cumulative variance (optional)
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(np.arange(1, len(var_frac)+1),
        np.cumsum(var_frac)*100,
        marker='o', lw=1.5, color='darkred')

ax.set_title("Cumulative Variance Explained by EOF Modes")
ax.set_xlabel("EOF Mode Number")
ax.set_ylabel("Cumulative Variance (%)")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.4)

if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "EOF_cumulative_variance.png"), dpi=200)

plt.show()

print("EOF variance fractions (first 10 modes):")
for k in range(min(10, len(var_frac))):
    print(f" Mode {k+1}: {var_frac[k]*100:.2f}%")

# choose how many modes to keep
n_modes_plot = 3

# put EOFs into a DataFrame for easy merging
eof_df = pd.DataFrame(
    EOFs[:n_modes_plot].T,  # shape: (ntransects, n_modes_plot)
    index=tran_ids,
    columns=[f'EOF{k+1}' for k in range(n_modes_plot)]
)

# PCs DataFrame (optional)
pc_df = pd.DataFrame(
    PCs[:, :n_modes_plot],
    index=dates_eof,
    columns=[f'PC{k+1}' for k in range(n_modes_plot)]
)

# ----- plot PC time series -----
fig, axes = plt.subplots(n_modes_plot, 1, figsize=(10, 2.5*n_modes_plot), sharex=True)
if n_modes_plot == 1:
    axes = [axes]

for k in range(n_modes_plot):
    ax = axes[k]
    ax.plot(pc_df.index, pc_df[f'PC{k+1}'], lw=1.2)
    ax.set_ylabel(f'PC{k+1}')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'PC{k+1}  (Var = {var_frac[k]*100:.1f}%)')

axes[-1].set_xlabel('Date')
plt.tight_layout()
if save_figs:
    fig.savefig(os.path.join(out_fig_dir, "EOF_PCs_timeseries.png"), dpi=200)
plt.show()

# ----- map EOF modes onto transect geometries -----
# merge EOF loadings with transect GeoDataFrame
tgdf_eof = tgdf.merge(
    eof_df,
    left_on="id_str",
    right_index=True,
    how="inner"
)

# function to plot one EOF spatial mode
def plot_eof_map(gdf, colname, mode_num, outname=None):
    vals = gdf[colname].values
    vmax = np.nanmax(np.abs(vals))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10, 4))
    gdf.plot(
        column=colname,
        cmap="coolwarm",
        linewidth=2,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        ax=ax
    )
    ax.set_title(f"EOF{mode_num} Spatial Pattern (loadings)")
    ax.set_axis_off()
    plt.tight_layout()
    if save_figs and outname is not None:
        fig.savefig(os.path.join(out_fig_dir, outname), dpi=200)
    plt.show()

# plot first few EOF spatial patterns
for k in range(n_modes_plot):
    col = f'EOF{k+1}'
    plot_eof_map(tgdf_eof, col, k+1, outname=f"EOF{k+1}_spatial.png")

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:28:55 2025

@author: mcallahan
"""




"""
STORM ANALYSIS – Eureka / Arcata (USW00024213 + ACV ASOS)
Daily precip (mm), daily max wind (knots), daily min pressure (hPa)
"""

import requests
import pandas as pd
import io
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. DAILY PRECIP FROM GHCN (NOAA)
# ---------------------------------------------------------
station_id = "USW00024213"
url = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station_id}.csv"

resp = requests.get(url)
if resp.status_code != 200:
    raise RuntimeError(f"Could not download data: HTTP {resp.status_code}")

ghcn = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])

# filter years
ghcn = ghcn[(ghcn["DATE"].dt.year >= 2017) & (ghcn["DATE"].dt.year <= 2025)]

# rename + convert precip (tenths of mm -> mm)
ghcn = ghcn.rename(columns={
    "DATE": "date",
    "PRCP": "precip_mm",
})

ghcn["precip_mm"] = ghcn["precip_mm"] / 10.0   # tenths mm -> mm

# (optional) convert TMAX/TMIN from tenths °C to °C if you care
if "TMAX" in ghcn.columns:
    ghcn["TMAX_C"] = ghcn["TMAX"] / 10.0
if "TMIN" in ghcn.columns:
    ghcn["TMIN_C"] = ghcn["TMIN"] / 10.0

# keep core columns
precip_daily = ghcn[["date", "precip_mm"]].copy()

# save
out_csv = r"Y:\OPC\beachWidthTool\Sites\Samoa\eureka_weather_2017_2025.csv"
precip_daily.to_csv(out_csv, index=False)
print("Saved precip file:", out_csv)

# plot precip
plt.figure(figsize=(12, 5))
plt.plot(precip_daily["date"], precip_daily["precip_mm"],
         label="Daily Precipitation (mm)", lw=1.3)
plt.title("Daily Precipitation – Eureka (2017–2025)")
plt.ylabel("Total Precipitation per Day (mm)")
plt.xlabel("Date")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. HOURLY WIND + PRESSURE FROM ASOS (ACV)
# ---------------------------------------------------------
wind_path = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\ACV.csv"

ACV_df = pd.read_csv(wind_path)

wind_clean = ACV_df[["valid", "tmpf", "drct", "sknt", "alti", "mslp"]].copy()
wind_clean["valid"] = pd.to_datetime(wind_clean["valid"], errors="coerce")
wind_clean = wind_clean.dropna(subset=["valid"])

wind_clean = wind_clean[
    (wind_clean["valid"].dt.year >= 2017)
    & (wind_clean["valid"].dt.year <= 2025)
]

# convert wind + pressure to numeric (handle "M")
wind_clean["sknt"] = pd.to_numeric(wind_clean["sknt"], errors="coerce")
wind_clean["mslp"] = pd.to_numeric(wind_clean["mslp"], errors="coerce")

# drop rows where both are missing
wind_clean = wind_clean.dropna(subset=["sknt", "mslp"], how="all")



# date-only column for daily grouping
wind_clean["date"] = wind_clean["valid"].dt.date

# ---------------------------------------------------------
# 3. DAILY WIND + DAILY MIN PRESSURE
# ---------------------------------------------------------
daily_met = (
    wind_clean
    .groupby("date")
    .agg(
        daily_max_sust_knots=("sknt", "max"),
        daily_mean_sust_knots=("sknt", "mean"),
        daily_min_press_hpa=("mslp", "min"),
    )
    .reset_index()
)

# make date datetime for plotting/merging
daily_met["date"] = pd.to_datetime(daily_met["date"])

# ---------------------------------------------------------
# 4. PLOTS – DAILY MAX WIND & DAILY MIN PRESSURE
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(daily_met["date"], daily_met["daily_max_sust_knots"],
         c="r", lw=1.2, label="Daily Max Wind Speed (knots)")
plt.title("Daily Max Wind Speed – Eureka Airport (2017–2025)")
plt.xlabel("Date")
plt.ylabel("Wind Speed (knots)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(daily_met["date"], daily_met["daily_min_press_hpa"],
         lw=1.2, label="Daily Min Sea-Level Pressure (hPa)")
plt.title("Daily Minimum Sea-Level Pressure – Eureka (2017–2025)")
plt.xlabel("Date")
plt.ylabel("Pressure (hPa)")
   # useful range for coastal storms
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. (OPTIONAL) MERGE DAILY MET WITH DAILY PRECIP
# ---------------------------------------------------------
daily_all = pd.merge(
    daily_met,
    precip_daily,
    on="date",
    how="left"
)

print(daily_all.head())

#----------------------------------------------------------------
#6. FIND STORM EVENTS BY COMBINING PRESSURE AND WIND THRESHOLDS
#-----------------------------------------------------------------

'''
daily all columns are 

date  daily_max_sust_knots  daily_mean_sust_knots  daily_min_press_hpa   precip_mm


we are going to use thresholds found in literature to determine storm conditions and create a dataset of storm days
'''
import numpy as np
import matplotlib.pyplot as plt

def norm_xcorr(x, y, max_lag):
    # remove NaN and center
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    x = df['x'].values - np.mean(df['x'])
    y = df['y'].values - np.mean(df['y'])

    lags = np.arange(-max_lag, max_lag+1)
    R = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            R[i] = np.corrcoef(x[:lag], y[-lag:])[0,1]
        elif lag > 0:
            R[i] = np.corrcoef(x[lag:], y[:-lag])[0,1]
        else:
            R[i] = np.corrcoef(x, y)[0,1]

    return R, lags
df = daily_all[['daily_max_sust_knots', 'daily_min_press_hpa']].dropna()
x = df['daily_max_sust_knots']
y = df['daily_min_press_hpa']

R, lags = norm_xcorr(x, y, max_lag=25)

plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Wind and Pressure")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()

#this is not correlating the way it should. I am trying the hourly dataset, since technically I smoothed the timeseries

hourly_wind = wind_clean[['sknt', 'mslp']].dropna()
x_hourly = hourly_wind['sknt']
y_hourly = hourly_wind['mslp']

R, lags = norm_xcorr(y_hourly, x_hourly, max_lag=25)

plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Hourly Wind and Pressure")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()

#ok that was worse 

# but lets correlate these data with the beach width to see what we can find
#first, atmospheric pressure and beach width 

samoa_ts = pd.read_csv(
    r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\statistics\daily_mean_width_all_transects.csv"
)
pressure_ts = daily_all['daily_min_press_hpa'].dropna()
R, lags= norm_xcorr(pressure_ts, samoa_ts['mean_width_m'], 50)
plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Pressure and Daily Mean Width")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()

#Second, wind and beach width 
wind_ts = daily_all['daily_max_sust_knots'].dropna()
R, lags= norm_xcorr(wind_ts, samoa_ts['mean_width_m'], 50)
plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Max Sustained Windspeed and Daily Mean Width")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show() 


#third, temperature and beach width (expected relationship due to seasonal cycle)
temp_ts = wind_clean[['valid','tmpf']].dropna()
temp_ts['valid'] = pd.to_datetime(temp_ts['valid'])
temp_ts['date'] = temp_ts['valid'].dt.date
temp_ts['tmpf'] = pd.to_numeric(temp_ts['tmpf'], errors='coerce')

daily_temp = (
    temp_ts
    .groupby('date')['tmpf']
    .mean()
    .reset_index()
)
daily_temp['date'] = pd.to_datetime(daily_temp['date'])

#plot daily mean temperature
plt.figure(figsize=(10,5))
plt.plot(daily_temp['date'], daily_temp['tmpf'], c = 'c', label = 'Mean Temperature (F)',)
plt.ylabel('Daily Mean Air Temp (F)')
plt.xlabel('Date')
plt.title('Daily Average Air Temperature in Humbolt Bay (F)')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()

#now correlation
R, lags= norm_xcorr(temp_ts['tmpf'], samoa_ts['mean_width_m'], 50)
plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Mean Temperature and Daily Mean Width")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()



'''
WAVES 
'''

import os
import glob
import pandas as pd

# -----------------------------
# 1. Folder with .txt.gz files
# -----------------------------
waves_folder = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\waves"

# pattern for all compressed NDBC files
pattern = os.path.join(waves_folder, "*.txt.gz")
file_list = glob.glob(pattern)

print("Found", len(file_list), "files")
if not file_list:
    raise FileNotFoundError("No .txt.gz files found in folder.")

# -----------------------------
# 2. Read and combine all files
# -----------------------------
dfs = []

for fpath in file_list:
    print("Reading:", os.path.basename(fpath))
    
    df = pd.read_csv(
        fpath,
        compression="gzip",
        delim_whitespace=True,
        skiprows=[1],     # skip NDBC header lines
        na_values=["MM"]
    )
    
    # fix year column name (#YY -> YY)
    if "#YY" in df.columns:
        df = df.rename(columns={"#YY": "YY"})
    
    # build datetime column
    # NDBC stdmet columns: YY, MM, DD, hh, mm
    df["time"] = pd.to_datetime(
        dict(
            year=df["YY"],
            month=df["MM"],
            day=df["DD"],
            hour=df["hh"],
            minute=df["mm"]
        ),
        errors="coerce"
    )
    
    df = df.dropna(subset=["time"])
    
    # keep only what we care about: time and wave height
    if "WVHT" not in df.columns:
        raise KeyError(f"'WVHT' column not found in {fpath}")
    
    df_sub = df[["time", "WVHT"]].copy()
    
    dfs.append(df_sub)

# concatenate all years/files into one DataFrame
waves_all = pd.concat(dfs, ignore_index=True)

# set time as index and sort
waves_all = waves_all.set_index("time").sort_index()

print("Combined hourly rows:", len(waves_all))

# -----------------------------
# 3. Save combined hourly CSV (optional)
# -----------------------------
combined_hourly_csv = os.path.join(
    waves_folder, "waves_hourly_combined.csv"
)
waves_all.to_csv(combined_hourly_csv)
print("Saved combined hourly waves to:", combined_hourly_csv)

# -----------------------------
# 4. Create daily mean wave heights
# -----------------------------
# WVHT is significant wave height in meters
daily_waves = waves_all["WVHT"].resample("D").mean().to_frame(name="WVHT_mean")

print("Daily rows:", len(daily_waves))

# -----------------------------
# 5. Save daily mean DataFrame to CSV
# -----------------------------
daily_csv = os.path.join(
    waves_folder, "waves_daily_mean_WVHT.csv"
)
daily_waves.to_csv(daily_csv)
print("Saved daily mean wave heights to:", daily_csv)

#lets plot daily mean wave height
plt.figure(figsize=(10,5))
plt.plot(daily_waves.index, daily_waves['WVHT_mean'], c = 'g', label = 'Daily Mean Wave Height',)
plt.ylabel('Daily Mean Wave Height (m)')
plt.xlabel('Date')
plt.title('Daily Average Wave Height in Humbolt Bay (meters)')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()

#correlate wave height iwth beach width
#set indexes, align, and merge to get the same time frame. 
daily_waves = daily_waves.copy()
daily_waves.index = pd.to_datetime(daily_waves.index)

samoa = samoa_ts.copy()
samoa['date'] = pd.to_datetime(samoa['date'])
samoa = samoa.set_index('date')
#merge to keep overlapping dates
merged = daily_waves.join(samoa['mean_width_m'], how='inner')
merged = merged.dropna()
#extract arrays
x = merged['WVHT_mean'].values
y = merged['mean_width_m'].values


#now correlation, need to align first

R, lags= norm_xcorr(x,y, 50)
plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Mean Wave Height (m) and Daily Mean Width (m)")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()



# --- FINAL MERGE: beach width + met + temp + waves ---

# 1) Beach width: ensure datetime
samoa = samoa_ts.copy()
samoa['date'] = pd.to_datetime(samoa['date'])

# 2) Met (wind/pressure/precip) - already daily_all
daily_all['date'] = pd.to_datetime(daily_all['date'])

# 3) Temperature: rename tmpf -> daily_mean_tmpf_F so it's clear
daily_temp = daily_temp.copy()
daily_temp['date'] = pd.to_datetime(daily_temp['date'])
daily_temp = daily_temp.rename(columns={'tmpf': 'daily_mean_tmpf_F'})

# 4) Waves: daily_waves index -> date column
waves_df = daily_waves.reset_index()     # index currently is datetime
waves_df = waves_df.rename(columns={'index': 'date'}) if 'index' in waves_df.columns else waves_df
if 'time' in waves_df.columns:
    waves_df = waves_df.rename(columns={'time': 'date'})
waves_df['date'] = pd.to_datetime(waves_df['date'])


# Start from beach width as the base (only dates where we have beach width)
# samoa_ts should have a column like 'mean_width_m'
full = samoa.merge(
    daily_all,
    on='date',
    how='left'
)

# Add daily mean temperature
full = full.merge(
    daily_temp[['date', 'daily_mean_tmpf_F']],
    on='date',
    how='left'
)

# Add daily mean wave height


full = full.merge(
    waves_df[['date', 'WVHT_mean']],
    on='date',
    how='left'
)

print("Full merged shape:", full.shape)
print(full.head())
print(full.columns)


out_full = r"Y:\OPC\beachWidthTool\Sites\Samoa\samoa_beachwidth_full_forcing.csv"
full.to_csv(out_full, index=False)
print("Saved full forcing dataset to:", out_full)


#full is my new dataset, lets see how the correlations change
x = full['WVHT_mean']
y = full['mean_width_m']
R, lags= norm_xcorr(x,y, 50)
plt.figure(figsize=(10,5))
plt.plot(lags, R, marker='o')
plt.axhline(0, color='k', lw=0.7)
plt.title("Cross-Correlation Between Daily Mean Wave Height (m) and Daily Mean Width (m)")
plt.xlabel("Lag (days)")
plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)
plt.show()
#looks the same
plt.figure(figsize=(10,5))
plt.plot(full['date'], full['WVHT_mean'], c = 'g', label = 'Daily Mean Wave Height',)
plt.ylabel('Daily Mean Wave Height (m)')
plt.xlabel('Date')
plt.title('Daily Average Wave Height in Humbolt Bay (meters)')
plt.legend()
plt.grid(True, alpha = 0.3)
plt.show()


# =========================================================
# FORCING-ONLY DATASET: WIND + PRESSURE + PRECIP + TEMP + WAVES
# (FULL TIME RANGE, NOT LIMITED BY BEACH WIDTH DATES)
# =========================================================

# 1) Make sure all the component dataframes have proper datetime columns

# daily_met: already has 'date' as datetime and:
#   ['date', 'daily_max_sust_knots', 'daily_mean_sust_knots', 'daily_min_press_hpa']
daily_met = daily_met.copy()
daily_met['date'] = pd.to_datetime(daily_met['date'])

# precip_daily: ['date', 'precip_mm']
precip_daily = precip_daily.copy()
precip_daily['date'] = pd.to_datetime(precip_daily['date'])

# daily_temp: we already renamed tmpf -> daily_mean_tmpf_F above
daily_temp = daily_temp.copy()
daily_temp['date'] = pd.to_datetime(daily_temp['date'])

# waves_df: came from daily_waves (WVHT_mean)
waves_df = waves_df.copy()
waves_df['date'] = pd.to_datetime(waves_df['date'])

# 2) Start from daily_met and outer-merge everything to keep the full met record
forcing = daily_met.merge(
    precip_daily,
    on='date',
    how='outer'
)

forcing = forcing.merge(
    daily_temp[['date', 'daily_mean_tmpf_F']],
    on='date',
    how='outer'
)

forcing = forcing.merge(
    waves_df[['date', 'WVHT_mean']],
    on='date',
    how='outer'
)

# 3) Sort by date and reset index
forcing = forcing.sort_values('date').reset_index(drop=True)

print("Forcing-only dataframe shape:", forcing.shape)
print(forcing.head())
print(forcing.columns)

# 4) Save to CSV
forcing_out = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_forcing_fulltimeseries.csv"
forcing.to_csv(forcing_out, index=False)
print("Saved forcing-only full timeseries to:", forcing_out)



#forcing is our storm finding timeseries
# =========================================================
# STORM COMPOSITING: IDENTIFY STORM DAYS / STORM EVENTS
# USING THE 'forcing' DATAFRAME
# =========================================================



# 1) Make sure forcing is sorted and clean
forcing = forcing.copy()
forcing = forcing.sort_values('date').reset_index(drop=True)

# 2) USER-ADJUSTABLE STORM THRESHOLDS
#    (tune these as you learn more / check histograms)
P_THRESH     = 995.0   # hPa: low pressure threshold
WIND_THRESH  = 32.0    # knots: strong wind
WVHT_THRESH  = 3.0     # m: high waves (significant wave height)
PRECIP_THRESH = 10.0   # mm: heavy daily rain

# 3) BUILD BOOLEAN "STORM DAY" MASK
#    Here: a storm day is when pressure is low AND wind is strong,
#    OR waves are large, OR precip is heavy.
storm_mask = (
    ((forcing['daily_min_press_hpa'] <= P_THRESH) &
     (forcing['daily_max_sust_knots'] >= WIND_THRESH))
    |
    (forcing['WVHT_mean'] >= WVHT_THRESH)
    |
    (forcing['precip_mm'] >= PRECIP_THRESH)
)

forcing['is_storm_day'] = storm_mask

# 4) GROUP CONSECUTIVE STORM DAYS INTO STORM EVENTS
#    Each continuous run of True in is_storm_day is one event.

# start a new event when today is storm and yesterday was not
event_start_flag = storm_mask & ~storm_mask.shift(fill_value=False)
# cumulative sum of starts → event IDs
forcing['storm_event_id'] = event_start_flag.cumsum()
# non-storm days get NaN event id
forcing.loc[~storm_mask, 'storm_event_id'] = np.nan

# 5) SUMMARIZE EACH STORM EVENT: start, end, duration, peak intensity
storm_days = forcing.dropna(subset=['storm_event_id']).copy()
storm_days['storm_event_id'] = storm_days['storm_event_id'].astype(int)

def _center_date(g):
    """Define storm center as date of minimum pressure within the event."""
    idx = g['daily_min_press_hpa'].idxmin()
    return g.loc[idx, 'date']

events = (
    storm_days
    .groupby('storm_event_id')
    .agg(
        start_date   = ('date', 'min'),
        end_date     = ('date', 'max'),
        duration_d   = ('date', lambda s: (s.max() - s.min()).days + 1),
        min_press_hpa= ('daily_min_press_hpa', 'min'),
        max_wind_kt  = ('daily_max_sust_knots', 'max'),
        max_waves_m  = ('WVHT_mean', 'max'),
        total_precip_mm = ('precip_mm', 'sum')
    )
    .reset_index()
)

centers = (
    storm_days
    .groupby('storm_event_id')
    .apply(_center_date)
    .rename('center_date')
    .reset_index()
)

events = events.merge(centers, on='storm_event_id', how='left')

print("Identified storm events:")
print(events.head())

# 6) SAVE STORM EVENT TABLE
storm_events_csv = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_storm_events_from_forcing.csv"
events.to_csv(storm_events_csv, index=False)
print("Saved storm events to:", storm_events_csv)

# 7) OPTIONAL: ALSO SAVE A DAY-BY-DAY STORM FLAG FILE
forcing_out_with_flags = r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_forcing_with_stormflags.csv"
forcing.to_csv(forcing_out_with_flags, index=False)
print("Saved forcing with storm day flags to:", forcing_out_with_flags)



'''
plotting all storms on the time series
'''

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Make sure dates are datetime
full_plot = full.copy()
full_plot['date'] = pd.to_datetime(full_plot['date'])

events_plot = events.copy()
events_plot['start_date']  = pd.to_datetime(events_plot['start_date'])
events_plot['end_date']    = pd.to_datetime(events_plot['end_date'])
events_plot['center_date'] = pd.to_datetime(events_plot['center_date'])

# Optional: if you want to only keep "real" storms, you can filter here, e.g.
# events_plot = events_plot[events_plot['max_wind_kt'] >= 20]

fig, ax = plt.subplots(figsize=(12, 5))

# 1) Beach width time series
ax.plot(full_plot['date'], full_plot['mean_width_m'],
        lw=1.4, color='k', label='Samoa mean width')

# 2) Shade storm events
for _, ev in events_plot.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.60, linewidth=0)

ax.set_title("Samoa Mean Beach Width with Storm Events Shaded")
ax.set_xlabel("Date")
ax.set_ylabel("Mean width (m)")
ax.grid(True, alpha=0.3)

# Legend: line for width, patch for storms
storm_patch = mpatches.Patch(color='red', alpha=0.12, label='Storm event')
ax.legend(handles=[ax.lines[0], storm_patch], loc='best')

plt.tight_layout()
plt.show()


'''
winter 2021 investigation
'''
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ----------------------------------
# 0. TIME WINDOW
# ----------------------------------
start_zoom = pd.Timestamp("2019-06-01")
end_zoom   = pd.Timestamp("2020-06-30")

# ----------------------------------
# 1. WIDTH SERIES (only when we have data)
# ----------------------------------
width_zoom = samoa_ts.copy()
width_zoom['date'] = pd.to_datetime(width_zoom['date'])
width_zoom = width_zoom[(width_zoom['date'] >= start_zoom) &
                        (width_zoom['date'] <= end_zoom)]

# ----------------------------------
# 2. FORCING SERIES (all days, not limited by width)
#    Build a 'forcing' dataframe if you don't already have one
#    daily_all: wind + pressure + precip (date)
#    daily_temp: daily_mean_tmpf_F (date)
#    waves_df: WVHT_mean (date)
# ----------------------------------
forcing = daily_all.copy()
forcing['date'] = pd.to_datetime(forcing['date'])

# add temperature
forcing = forcing.merge(
    daily_temp[['date', 'daily_mean_tmpf_F']],
    on='date',
    how='outer'
)

# add waves
forcing = forcing.merge(
    waves_df[['date', 'WVHT_mean']],
    on='date',
    how='outer'
)

forcing_zoom = forcing[(forcing['date'] >= start_zoom) &
                       (forcing['date'] <= end_zoom)].copy()

# ----------------------------------
# 3. STORM EVENTS IN WINDOW
# ----------------------------------
events_zoom = events.copy()
events_zoom['start_date']  = pd.to_datetime(events_zoom['start_date'])
events_zoom['end_date']    = pd.to_datetime(events_zoom['end_date'])
events_zoom['center_date'] = pd.to_datetime(events_zoom['center_date'])

events_zoom = events_zoom[
    (events_zoom['end_date'] >= start_zoom) &
    (events_zoom['start_date'] <= end_zoom)
].copy()

print("Storms in zoom window:")
print(events_zoom[['storm_event_id', 'start_date', 'end_date',
                   'max_wind_kt', 'max_waves_m']])

# ----------------------------------
# 4. PLOT 5-PANEL FIGURE
# ----------------------------------
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

# 1) Beach width (only dates with width data)
ax = axes[0]
ax.plot(width_zoom['date'], width_zoom['mean_width_m'],
        lw=1.4, color='k', label='Mean width')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.20, linewidth=0)
ax.set_ylabel("Width (m)")
ax.set_title(f"Samoa Beach Width & Climate Dynamics ({start_zoom} to {end_zoom})\nStorm events shaded")
ax.grid(True, alpha=0.3)

# 2) Waves (all daily WVHT)
ax = axes[1]
ax.plot(forcing_zoom['date'], forcing_zoom['WVHT_mean'],
        lw=1.0, color='g')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.20, linewidth=0)
ax.set_ylabel("WVHT (m)")
ax.set_title("Daily Mean Wave Height")
ax.grid(True, alpha=0.3)

# 3) Wind (all daily max winds)
ax = axes[2]
ax.plot(forcing_zoom['date'], forcing_zoom['daily_max_sust_knots'],
        lw=1.0, color='orange')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.20, linewidth=0)
ax.set_ylabel("Max wind (kt)")
ax.set_title("Daily Max Sustained Wind")
ax.grid(True, alpha=0.3)

# 4) Precipitation (all days)
ax = axes[3]
ax.plot(forcing_zoom['date'], forcing_zoom['precip_mm'],
        lw=1.0, color='b')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.20, linewidth=0)
ax.set_ylabel("Precip (mm)")
ax.set_title("Daily Precipitation")
ax.grid(True, alpha=0.3)

# 5) Pressure (all days)
ax = axes[4]
ax.plot(forcing_zoom['date'], forcing_zoom['daily_min_press_hpa'],
        lw=1.0, color='purple')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.20, linewidth=0)
ax.set_ylabel("Min MSLP (hPa)")
ax.set_title("Daily Minimum Sea-level Pressure")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Date")

# Legend for storms on top panel
storm_patch = mpatches.Patch(color='red', alpha=0.20, label='Storm event')
axes[0].legend(handles=[axes[0].lines[0], storm_patch], loc='best')

plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
"""
Net change in beach width for Samoa (2020-06 → 2021-06)
Mapped to transects with red/blue color scale (erosion/accretion)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize

# ----------------------------------------------------
# 0. USER PATHS — UPDATE THESE IF NEEDED
# ----------------------------------------------------

# Beach-width master *long* table
master_csv = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv"

# Transect shapefile
transects_fp = r"Y:\OPC\beachWidthTool\Sites\Samoa\transects\transects_10m.shp"

# Field in the shapefile matching transect ID in the CSV
TRANSECT_ID_FIELD = "id"

# CRS used for distance/plotting
TARGET_EPSG = 32610

# ----------------------------------------------------
# 1. LOAD BEACH-WIDTH DATA
# ----------------------------------------------------
df = pd.read_csv(master_csv)

# Standardize columns
if "transect" in df.columns:
    df.rename(columns={"transect": "id"}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df["id"] = df["id"].astype(str)

# Window of interest
t_start = start_zoom
t_end   = end_zoom

bw_win = df[(df["date"] >= t_start) & (df["date"] <= t_end)].copy()

# Wide matrix (date × transect)
wide_win = bw_win.pivot_table(
    index="date",
    columns="id",
    values="width_m"
).dropna(axis=1, how='all')

print("Transects with data in window:", len(wide_win.columns))

# ----------------------------------------------------
# 2. CALCULATE NET CHANGE PER TRANSECT
# ----------------------------------------------------
net_change = {}

for tr in wide_win.columns:
    s = wide_win[tr].dropna()
    if len(s) < 2:
        continue
    net_change[tr] = s.iloc[-1] - s.iloc[0]

net_change = pd.Series(net_change, name="net_change")
print("\nNet change stats:")
print(net_change.describe())
print(net_change.head())

if net_change.empty:
    raise ValueError("No transects have ≥2 observations in this window!")

# ----------------------------------------------------
# 3. LOAD TRANSECTS + MERGE NET CHANGE
# ----------------------------------------------------
tgdf = gpd.read_file(transects_fp).to_crs(TARGET_EPSG)

if TRANSECT_ID_FIELD not in tgdf.columns:
    raise KeyError(f"Shapefile missing ID field '{TRANSECT_ID_FIELD}'")

tgdf["id_str"] = tgdf[TRANSECT_ID_FIELD].astype(str)

tgdf = tgdf.merge(net_change, left_on="id_str", right_index=True, how="left")

# ----------------------------------------------------
# 4. MAP NET BEACH-WIDTH CHANGE
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 12))

# symmetric color scale centered at zero
vmax = np.nanmax(np.abs(tgdf["net_change"]))
vmax = 1 if (np.isnan(vmax) or vmax == 0) else vmax

norm = Normalize(vmin=-vmax, vmax=vmax)

tgdf.plot(
    ax=ax,
    column="net_change",
    cmap="coolwarm",
    linewidth=2,
    vmin=-vmax,
    vmax=vmax,
    norm=norm
)

ax.set_title(
    f"Net Beach Width Change ({t_start} → {t_end})\n"
    "Blue = Erosion   |   Red = Accretion",
    fontsize=12
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
sm._A = []
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)

cbar.set_label("Net Change (m)")

plt.tight_layout()
plt.show()

#test arrows to find transect range of restoration site
import numpy as np
from matplotlib.colors import Normalize

# --- existing map code (unchanged) ---
fig, ax = plt.subplots(figsize=(6, 12))

# symmetric color scale centered at zero
vmax = np.nanmax(np.abs(tgdf["net_change"]))
vmax = 1 if (np.isnan(vmax) or vmax == 0) else vmax
norm = Normalize(vmin=-vmax, vmax=vmax)

tgdf.plot(
    ax=ax,
    column="net_change",
    cmap="coolwarm",
    linewidth=2,
    vmin=-vmax,
    vmax=vmax,
    norm=norm
)

ax.set_title(
    f"Net Beach Width Change ({t_start} - {t_end})\n"
    "Blue = Erosion   |   Red = Accretion",
    fontsize=12
)
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")

# colorbar
sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
sm._A = []
cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
cbar.set_label("Net Change (m)")

# -------------------------------------------------
# NEW: highlight two transects with arrows
# -------------------------------------------------

# IDs to highlight (as strings to match tgdf['id_str'])
highlight_ids = ["210", "260"]  # <- change these to iterate / explore

# compute centroids if not already done
tgdf["centroid"] = tgdf.geometry.centroid

for tid in highlight_ids:
    row = tgdf.loc[tgdf["id_str"] == tid]
    if row.empty:
        print(f"Transect {tid} not found in tgdf, skipping.")
        continue

    cx = row["centroid"].iloc[0].x
    cy = row["centroid"].iloc[0].y

    # choose an offset for where the label will sit
    # (you can tweak these to move the text around)
    x_text = cx + 40   # 40 m east of line
    y_text = cy + 40   # 40 m north of line

    ax.annotate(
        f"Transect {tid}",
        xy=(cx, cy),             # arrow points here
        xytext=(x_text, y_text), # label text location
        fontsize=9,
        color="black",
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1.2
        ),
        ha="left",
        va="bottom"
    )

plt.tight_layout()
plt.show()


#plot of transects in restoration site vs outside 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 1. Define window (same as case study)
# ------------------------------
t_start = pd.Timestamp("2021-06-01")
t_end   = pd.Timestamp("2022-06-30")

# ------------------------------
# 2. Prep beach-width dataframe
#    df should already be your master:
#    columns: ['id', 'date', 'width_m']
# ------------------------------
bw = df.copy()
bw['date'] = pd.to_datetime(bw['date'])
bw['id'] = bw['id'].astype(str)

# keep only dates in window
bw_win = bw[(bw['date'] >= t_start) & (bw['date'] <= t_end)].copy()

# pivot: date × transect
wide_win = bw_win.pivot_table(index='date', columns='id', values='width_m')

# ------------------------------
# 3. Subset transects 0–150
# ------------------------------
# build list of IDs as strings
subset_ids = [str(i) for i in range(110, 165)]

# only keep those that actually exist in the data
cols_to_plot = [tr for tr in subset_ids if tr in wide_win.columns]

wide_sub = wide_win[cols_to_plot].copy()

print("Windowed matrix shape (dates × transects 0–150 present):", wide_sub.shape)
print("Transects included:", cols_to_plot[:10], "...")  # preview

# ------------------------------
# 4. Plot time series for transects 0–150 (within window)
# ------------------------------
plt.figure(figsize=(10, 6))

for tr in cols_to_plot:
    plt.plot(wide_sub.index, wide_sub[tr],
             lw=0.8, alpha=0.5)   # light lines so overlap isn't too crazy

plt.title("Beach Width Time Series (European beach grass area)\n2020-06 to 2021-06")
plt.xlabel("Date")
plt.ylabel("Width (m)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#plotting the average
# ------------------------------
# 5. Plot mean width of this section
# ------------------------------

# compute mean across transects (row-wise)
section_mean = wide_sub.mean(axis=1)

plt.figure(figsize=(10, 4))
plt.plot(section_mean.index, section_mean.values,
         lw=1.8, color='k', label='Mean width (transects 210–259)')

plt.title("Mean Beach Width – Restoration Area\n2020-06 to 2021-06")
plt.xlabel("Date")
plt.ylabel("Width (m)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


num_window_dates = bw_win['date'].nunique()
print("Beach-width observation dates in window:", num_window_dates)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import Normalize

# --------------------------------------------------
# 1. PREP WINDOWED MATRIX FOR EOF
#    wide_win: (time × transects), index = date, cols = id
# --------------------------------------------------
X = wide_win.copy()

# make sure index is datetime and sorted for time interpolation
X.index = pd.to_datetime(X.index)
X = X.sort_index()

print("Original windowed matrix shape:", X.shape)

# Optional: drop transects with *very* poor coverage (< 50% valid)
col_valid_frac = X.notna().mean(axis=0)
good_cols = col_valid_frac[col_valid_frac >= 0.5].index
X = X[good_cols]

print("After dropping bad columns, shape:", X.shape)

# --------------------------------------------------
# 2. FILL MISSING VALUES (short gaps) & ANOMALIES
# --------------------------------------------------
# interpolate in time per transect (limit=5 days gap) then ffill/bfill
X_filled = X.interpolate(method="time", limit=5).ffill().bfill()

# remove time mean -> anomalies
X_anom = X_filled - X_filled.mean(axis=0)

print("Matrix used for EOF (after fill):", X_anom.shape)

# --------------------------------------------------
# 3. EOF VIA SVD
# --------------------------------------------------
# SVD of (Nt × Nx) matrix
U, S, Vt = np.linalg.svd(X_anom.values, full_matrices=False)

eigvals = S**2
varfrac = eigvals / eigvals.sum()

print("\nVariance fractions (first few):")
for i, vf in enumerate(varfrac[:5]):
    print(f"  EOF{i+1}: {vf*100:.2f}%")

EOFs = Vt                  # shape (modes × transects)
PCs  = U @ np.diag(S)      # shape (time × modes)

# put EOFs into a DataFrame: rows = EOF1, EOF2, ..., cols = transect IDs
eof_df = pd.DataFrame(
    EOFs,
    columns=X_anom.columns.astype(str),
    index=[f"EOF{i+1}" for i in range(EOFs.shape[0])]
)

# --------------------------------------------------
# 4. LOAD TRANSECTS FOR SPATIAL PLOTTING
# --------------------------------------------------
tgdf_eof = gpd.read_file(transects_fp).to_crs(TARGET_EPSG)

if TRANSECT_ID_FIELD not in tgdf_eof.columns:
    raise KeyError(f"Shapefile missing ID field '{TRANSECT_ID_FIELD}'")

tgdf_eof["id_str"] = tgdf_eof[TRANSECT_ID_FIELD].astype(str)

# make sure eof_df columns are strings to match id_str
eof_df.columns = eof_df.columns.astype(str)

# --------------------------------------------------
# 5. FUNCTION TO PLOT EOF MODE SPATIALLY
# --------------------------------------------------
def plot_eof_mode_spatial(mode_idx, tgdf_base, eof_table, varfrac):
    """
    mode_idx : 0-based index (0 = EOF1, 1 = EOF2, ...)
    tgdf_base: GeoDataFrame with 'id_str' and geometry
    eof_table: DataFrame, index = ['EOF1', 'EOF2', ...], columns = transect IDs
    varfrac  : array of variance fractions from EOF
    """
    mode_name = f"EOF{mode_idx+1}"
    if mode_name not in eof_table.index:
        print(f"{mode_name} not found in eof_table; skipping.")
        return

    # copy so we don't overwrite original tgdf
    g = tgdf_base.copy()

    # get loadings for this mode (Series: index = transect id, values = loading)
    load = eof_table.loc[mode_name]

    # merge into GeoDataFrame
    g = g.merge(load.rename("eof_loading"),
                left_on="id_str", right_index=True, how="left")

    # symmetric color scale about zero
    vmax = np.nanmax(np.abs(g["eof_loading"]))
    if np.isnan(vmax) or vmax == 0:
        vmax = 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 12))

    g.plot(
        ax=ax,
        column="eof_loading",
        cmap="coolwarm",
        linewidth=2,
        vmin=-vmax,
        vmax=vmax,
        norm=norm
    )

    ax.set_title(
        f"{mode_name} Spatial Pattern\n"
        f"Variance: {varfrac[mode_idx]*100:.1f}%",
        fontsize=12
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        shrink=0.6,
        fraction=0.03,
        pad=0.01
    )
    cbar.set_label(f"{mode_name} loading (m-equivalent)")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# 6. PLOT EOF1–EOF3 SPATIALLY ALONGSHORE
# --------------------------------------------------
plot_eof_mode_spatial(0, tgdf_eof, eof_df, varfrac)  # EOF1
plot_eof_mode_spatial(1, tgdf_eof, eof_df, varfrac)  # EOF2
plot_eof_mode_spatial(2, tgdf_eof, eof_df, varfrac)  # EOF3

import matplotlib.pyplot as plt
import numpy as np

# time axis from your anomaly matrix
time = X_anom.index

# make a helper to standardize PCs for nicer comparison
def standardize(pc):
    return (pc - np.mean(pc)) / np.std(pc)

pc1 = PCs[:, 0]
pc2 = PCs[:, 1]
pc3 = PCs[:, 2]

pc1_std = standardize(pc1)
pc2_std = standardize(pc2)
pc3_std = standardize(pc3)

# --- separate panels for each PC ---
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(time, pc1_std, lw=1.4, color='k')
axes[0].set_ylabel("PC1 (std units)")
axes[0].set_title(f"PC1 – Temporal Pattern (Var = {varfrac[0]*100:.1f}%)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, pc2_std, lw=1.3, color='tab:blue')
axes[1].set_ylabel("PC2 (std units)")
axes[1].set_title(f"PC2 – Temporal Pattern (Var = {varfrac[1]*100:.1f}%)")
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, pc3_std, lw=1.3, color='tab:orange')
axes[2].set_ylabel("PC3 (std units)")
axes[2].set_title(f"PC3 – Temporal Pattern (Var = {varfrac[2]*100:.1f}%)")
axes[2].set_xlabel("Date")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# --- Load beach width (your daily_mean_width_all_transects.csv) ---
bw = pd.read_csv(
    r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\statistics\daily_mean_width_all_transects.csv"
)

# Standardize column names
bw.columns = ["date", "width_m"]

# Convert date and sort
bw["date"] = pd.to_datetime(bw["date"])
bw = bw.sort_values("date")

# Optionally remove NaN
bw = bw.dropna(subset=["width_m"])

# Optional: Smooth (rolling mean)
bw["width_smoothed"] = bw["width_m"].rolling(7, center=True).mean()

# --- Plot ---
plt.figure(figsize=(14, 6))

plt.plot(bw["date"], bw["width_m"], label="Daily Mean Width", color="steelblue", lw=1.2)

# plot smoothed
plt.plot(bw["date"], bw["width_smoothed"],
         label="7-day Smoothed", color="black", lw=2, alpha=0.8)

plt.title("Samoa Beach — Daily Mean Beach Width Time Series")
plt.xlabel("Date")
plt.ylabel("Beach Width (m)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()




#regression to understand which variables drive variability
# PCs is (Nt x Nmodes)
PC1 = PCs[:, 0]
PC2 = PCs[:, 1]
PC3 = PCs[:, 2]

pc_df = pd.DataFrame({
    "date": X_anom.index,
    "PC1": PC1,
    "PC2": PC2,
    "PC3": PC3
})

pc_df['date'] = pd.to_datetime(pc_df['date'])

# Merge onto forcings
regdf = pc_df.merge(
    full[['date','WVHT_mean','daily_max_sust_knots','daily_min_press_hpa',
          'daily_mean_tmpf_F','precip_mm']],
    on='date', how='inner'
).dropna()

print(regdf.head())
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Predictors and scaling
# -----------------------------
X = regdf[['WVHT_mean','daily_max_sust_knots','daily_min_press_hpa',
           'daily_mean_tmpf_F','precip_mm']]
predictors = X.columns.tolist()

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# -----------------------------
# 2. Fit separate models for PC1, PC2, PC3
# -----------------------------
models = {}
r2_scores = {}

for pcname in ['PC1', 'PC2', 'PC3']:
    y = regdf[pcname].values
    m = LinearRegression().fit(Xs, y)
    models[pcname] = m
    r2_scores[pcname] = m.score(Xs, y)

    print(f"\n---- Regression for {pcname} ----")
    print("R²:", r2_scores[pcname])
    for name, coef in zip(predictors, m.coef_):
        print(f"{name}: {coef:.4f}")

# -----------------------------
# 3. Plotting helper
# -----------------------------
def plot_reg_horizontal(model, predictors, title):
    coefs = model.coef_

    plt.figure(figsize=(7,5))
    plt.barh(predictors, coefs, alpha=0.85)
    plt.axvline(0, color="k", lw=1)

    for i, coef in enumerate(coefs):
        plt.text(coef, i, f"{coef:.1f}",
                 va='center',
                 ha='left' if coef > 0 else 'right',
                 fontsize=9)

    plt.xlabel("Standardized coefficient")
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 4. Make the three bar plots
# -----------------------------
plot_reg_horizontal(models['PC1'], predictors,
                    f"PC1 Regression Coefficients (R²={r2_scores['PC1']:.2f})")

plot_reg_horizontal(models['PC2'], predictors,
                    f"PC2 Regression Coefficients (R²={r2_scores['PC2']:.2f})")

plot_reg_horizontal(models['PC3'], predictors,
                    f"PC3 Regression Coefficients (R²={r2_scores['PC3']:.2f})")


# ============================================================
# EOF + Regression on FULL Beach-Width Time Series (All Years)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# 1. Build full wide matrix: date × transect
# ------------------------------------------
bw_full = df.copy()
bw_full['date'] = pd.to_datetime(bw_full['date'])
bw_full['id'] = bw_full['id'].astype(str)

wide_full = bw_full.pivot_table(
    index='date',
    columns='id',
    values='width_m'
)

# sort by date just to be safe
wide_full = wide_full.sort_index()

print("Full wide matrix shape (dates × transects):", wide_full.shape)

# ------------------------------------------
# 2. Remove time mean → anomalies
#    (Do NOT window here, this is full record)
# ------------------------------------------
X_full = wide_full.copy()

# drop transects that are entirely NaN
X_full = X_full.dropna(axis=1, how='all')

# remove column-wise mean (time mean)
X_anom_full = X_full - X_full.mean(axis=0)

# for a “strict” EOF, use only dates where all transects are present
X_anom_full = X_anom_full.dropna(axis=0, how='any')

print("EOF full matrix shape (after NaN removal):", X_anom_full.shape)

# ------------------------------------------
# 3. SVD for EOF: X_anom_full = U S Vᵀ
# ------------------------------------------
U_full, S_full, Vt_full = np.linalg.svd(X_anom_full, full_matrices=False)

# eigenvalues and variance fractions
eigvals_full = S_full**2
varfrac_full = eigvals_full / eigvals_full.sum()

print("\nVariance fractions (full record):")
for i, vf in enumerate(varfrac_full[:5]):
    print(f" Mode {i+1}: {vf*100:.2f}%")

# EOF spatial patterns and PCs
EOFs_full = Vt_full              # (Nmodes × Ntransects)
PCs_full  = U_full @ np.diag(S_full)  # (Ntimes × Nmodes)

# ------------------------------------------
# 4. Build PC dataframe with dates
# ------------------------------------------
time_full = X_anom_full.index  # these are the dates kept after NaN drop

PC1_full = PCs_full[:, 0]
PC2_full = PCs_full[:, 1]
PC3_full = PCs_full[:, 2]

pc_df_full = pd.DataFrame({
    "date": time_full,
    "PC1": PC1_full,
    "PC2": PC2_full,
    "PC3": PC3_full
})
pc_df_full['date'] = pd.to_datetime(pc_df_full['date'])

# ------------------------------------------
# 5. Merge PCs with full forcing time series
# ------------------------------------------
# Assumes `full` has forcing for all/most dates and a 'date' column
forcing_cols = ['WVHT_mean',
                'daily_max_sust_knots',
                'daily_min_press_hpa',
                'daily_mean_tmpf_F',
                'precip_mm']

regdf_full = pc_df_full.merge(
    full[['date'] + forcing_cols],
    on='date',
    how='inner'
).dropna()

print("\nRegression dataframe (full) date range:")
print(regdf_full['date'].min(), "→", regdf_full['date'].max())
print("Regdf_full shape:", regdf_full.shape)

# ------------------------------------------
# 6. Standardize predictors and fit models
# ------------------------------------------
X_forc = regdf_full[forcing_cols].values
predictors = forcing_cols

scaler = StandardScaler()
Xs_forc = scaler.fit_transform(X_forc)

models_full = {}
r2_full = {}

for pcname in ['PC1', 'PC2', 'PC3']:
    y_pc = regdf_full[pcname].values
    m = LinearRegression().fit(Xs_forc, y_pc)
    models_full[pcname] = m
    r2_full[pcname] = m.score(Xs_forc, y_pc)

    print(f"\n---- FULL-RECORD Regression for {pcname} ----")
    print("R²:", r2_full[pcname])
    for name, coef in zip(predictors, m.coef_):
        print(f"{name}: {coef:.4f}")

# ------------------------------------------
# 7. Plot PC time series (full record)
# ------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(regdf_full['date'], regdf_full['PC1'], label=f"PC1 ({varfrac_full[0]*100:.1f}%)")
plt.plot(regdf_full['date'], regdf_full['PC2'], label=f"PC2 ({varfrac_full[1]*100:.1f}%)")
plt.plot(regdf_full['date'], regdf_full['PC3'], label=f"PC3 ({varfrac_full[2]*100:.1f}%)")

plt.title("Principal Components of Samoa Beach Width (Full Record)")
plt.xlabel("Date")
plt.ylabel("PC amplitude")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------
# 8. Helper: Horizontal bar plot of coefficients
# ------------------------------------------
def plot_reg_horizontal(model, predictors, title):
    coefs = model.coef_

    plt.figure(figsize=(7, 5))
    plt.barh(predictors, coefs, alpha=0.85)
    plt.axvline(0, color="k", lw=1)

    for i, coef in enumerate(coefs):
        plt.text(coef, i,
                 f"{coef:.1f}",
                 va='center',
                 ha='left' if coef > 0 else 'right',
                 fontsize=9)

    plt.xlabel("Standardized coefficient")
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------
# 9. Make regression barplots for PC1–PC3
# ------------------------------------------
plot_reg_horizontal(
    models_full['PC1'],
    predictors,
    f"PC1 Regression Coefficients (Full Record, R²={r2_full['PC1']:.2f})"
)

plot_reg_horizontal(
    models_full['PC2'],
    predictors,
    f"PC2 Regression Coefficients (Full Record, R²={r2_full['PC2']:.2f})"
)

plot_reg_horizontal(
    models_full['PC3'],
    predictors,
    f"PC3 Regression Coefficients (Full Record, R²={r2_full['PC3']:.2f})"
)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pycwt as wavelet

# =================================================
# 0. Patch pycwt's wct_significance to accept 'mother'
# =================================================
# This avoids: TypeError: wct_significance() got an unexpected keyword argument 'mother'

# Save original function
_orig_wct_significance = wavelet.wct_significance

def _wct_significance_patched(lag1, lag2, dt, J, s0, dj,
                              sig1=0.0, sig2=0.0,
                              significance_level=0.95):
    """
    Wrapper that ignores 'mother' and calls the original
    wct_significance with the arguments it actually expects.
    """
    return _orig_wct_significance(lag1, lag2, dt, J, s0, dj,
                                  sig1, sig2, significance_level)

# Replace in pycwt
wavelet.wct_significance = _wct_significance_patched

# =================================================
# 1. Load and align beach width + wave height series
# =================================================
bw = samoa_ts.copy()
bw['date'] = pd.to_datetime(bw['date'])
bw = bw.dropna()

waves = daily_waves.copy()
waves.index = pd.to_datetime(waves.index)
waves = waves.rename(columns={'WVHT_mean': 'waves'})

# Join only overlapping dates
df = bw.set_index('date').join(waves, how='inner').dropna()

print("Length of aligned series:", len(df))
print(df.head())

# Extract arrays
x = df['mean_width_m'].values
y = df['waves'].values

# =================================================
# 2. Preprocess: remove mean (you could also detrend if you want)
# =================================================
x = x - np.mean(x)
y = y - np.mean(y)

dt = 1.0  # sampling = 1 day

# =================================================
# 3. Wavelet Coherence (still returns significance,
#    but we will ignore it in plotting)

WCT, aWCT, coi, freq, signif = wavelet.wct(x, y, dt=dt)

period = 1.0 / freq      # convert freq to period (days)
T = np.arange(len(x))    # simple time index; you can swap in df.index if you prefer

# =================================================
# 4. Plot WTC magnitude
# =================================================
fig, ax = plt.subplots(figsize=(12, 6))

cf = ax.contourf(
    T,
    period,
    WCT,
    levels=100,
    cmap='viridis'
)

ax.set_yscale("log")
ax.set_ylabel("Period (days)")
ax.set_xlabel("Time index")
ax.set_title("Wavelet Coherence: Beach Width vs Wave Height")

# Cone of influence
ax.fill_between(
    T,
    coi,
    y2=period.max(),
    color="white",
    alpha=0.4,
    hatch="//"
)

cbar = plt.colorbar(cf, ax=ax)
cbar.set_label("Coherence")

plt.tight_layout()
plt.show()





#wave model cdip, variability in wave height across sites

#SHAPE0100