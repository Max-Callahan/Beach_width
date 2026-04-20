# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 12:09:53 2025

@author: mcallahan
"""

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


width_master_csv = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv"

transects = r"Y:\OPC\beachWidthTool\Sites\Samoa\transects\transects_10m.shp"
transect_id = 'id'

#coordinate system of samoa beach
epsg = 32610

min_fraction = 0.6
max_interp_gap_days = 7



#bring in csv of beach widths and transects 
df = pd.read_csv(width_master_csv)

#need to rename column in width_csv to match my transects naming convention with 'id'
df = df.rename(columns = {'transect': 'id'})

#cleaning data 
df['date'] = pd.to_datetime(df['date'], errors = 'coerce')
df = df.dropna(subset = ['date', 'width_m', 'id'])
df['id'] = df['id'].astype(str)
df = df.sort_values(['date', 'id'])


#convert to pivot table
wide = df.pivot_table(
    index = 'date',
    columns = 'id',
    values = 'width_m',
    aggfunc = 'mean'
)

wide = wide.sort_index()
#wide is now my 69 dates by 399 transect table

#interpolate along time 
wide_interp = wide.interpolate(method= 'time',
                               limit = 7,
                               limit_direction = 'both')

#fill any nans with column mean (just in case )
wide_interp = wide_interp.fillna(wide_interp.mean())

#feature matrix
X = wide_interp.T.copy()

transect_ids = X.index.tolist()


#standardize each transect time series
for i in range(X.shape[0]):
    row = X.iloc[i,:].values
    row_mean = np.mean(row)
    row_std = np.std(row)
    if row_std == 0:
        X.iloc[i,:] = 0.0
    else:
        X.iloc[i,:] = (row -row_mean)/ row_std
        
X_features = X.values
        


#trying different Ks for Kmeans

for k in range(2,9):
    km = KMeans(n_clusters = k, random_state = 0, n_init = 10)
    labels_k = km.fit_predict(X_features)
    
    try:
        score = silhouette_score(X_features, labels_k)
    except Exception as e:
        score = np.nan
        
    print(f" K = {k}: silhouette = {score:.3f}")
    
    
k_final = 2

kmeans = KMeans(n_clusters = k_final, random_state = 0, n_init = 50)
cluster_labels = kmeans.fit_predict(X_features)


cluster_df = pd.DataFrame({
    'id_str': transect_ids,
    'cluster': cluster_labels
    })


#loading transects and coloring byu 






# -*- coding: utf-8 -*-
"""
SPATIAL CLUSTERING OF SAMOA TRANSECTS (BEACH WIDTH DYNAMICS)

Standalone script that:
    1. Loads the beach-width master CSV (long format).
    2. Builds a date × transect wide matrix of width_m.
    3. Cleans + interpolates time series per transect.
    4. Standardizes each transect time series.
    5. Runs K-means clustering on transect time series.
    6. Computes alongshore distance from transect shapefile.
    7. Keeps results in memory and makes plots (no saving).

Outputs in memory:
    - wide, wide_clean, wide_interp
    - X_features (cluster feature matrix)
    - cluster_df (transect → cluster)
    - tgdf_clusters (GeoDataFrame with geometry + cluster + alongshore distance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# 0. USER SETTINGS / PATHS
# ============================================================

# Beach-width master (LONG) CSV
# Required columns: 'id' (or 'transect'), 'date', 'width_m'
MASTER_CSV = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv"

# Transect shapefile (one file with all transect lines)
TRANSECTS_SHP = r"Y:\OPC\beachWidthTool\Sites\Samoa\transects\transects_10m.shp"
TRANSECT_ID_FIELD = "id"      # field in shapefile that matches CSV transect ID

# Coordinate system for alongshore distance (meters)
TARGET_EPSG = 32610

# Data cleaning settings
MIN_VALID_FRACTION   = 0.6   # minimum fraction of non-NaN points required per transect
MAX_INTERP_GAP_DAYS  = 60    # max gap (days) to fill by interpolation

# Cluster settings
N_CLUSTERS_FIXED = 5        # your chosen K for final model
TRY_MULTIPLE_K   = True
K_RANGE          = range(2, 9)

# ============================================================
# 1. LOAD BEACH-WIDTH MASTER AND BUILD WIDE MATRIX
# ============================================================

print("Loading beach-width master CSV...")
df = pd.read_csv(MASTER_CSV)

# Standardize column names
if "transect" in df.columns and "id" not in df.columns:
    df = df.rename(columns={"transect": "id"})

required_cols = {"id", "date", "width_m"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Master CSV is missing columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "width_m", "id"])
df["id"] = df["id"].astype(str)
df = df.sort_values(["date", "id"])

print("Total rows in master:", len(df))

wide = df.pivot_table(
    index="date",
    columns="id",
    values="width_m",
    aggfunc="mean"
).sort_index()

print("Wide matrix shape (dates × transects):", wide.shape)

# ============================================================
# 2. REMOVE TRANSECTS WITH TOO MANY NaNs
# ============================================================

valid_fraction = wide.notna().mean(axis=0)
good_transects = valid_fraction[valid_fraction >= MIN_VALID_FRACTION].index

print(f"Transects kept (>= {MIN_VALID_FRACTION*100:.0f}% valid):", len(good_transects))

wide_clean = wide[good_transects].copy()

# ============================================================
# 3. FILL SHORT GAPS + STANDARDIZE PER TRANSECT
# ============================================================

# interpolate along time
wide_interp = wide_clean.interpolate(
    method="time",
    limit=MAX_INTERP_GAP_DAYS,
    limit_direction="both"
)

# fill any remaining NaNs with column mean
wide_interp = wide_interp.fillna(wide_interp.mean())

# build feature matrix: rows = transects, cols = time steps
X_raw = wide_interp.T.copy()   # (n_transects × n_dates)
transect_ids = X_raw.index.tolist()

print("Feature matrix shape before standardization:", X_raw.shape)

# standardize each transect's time series
X_std = X_raw.copy()
for i in range(X_std.shape[0]):
    row = X_std.iloc[i, :].values
    m  = np.mean(row)
    s  = np.std(row)
    if s == 0:
        X_std.iloc[i, :] = 0.0
    else:
        X_std.iloc[i, :] = (row - m) / s

X_features = X_std.values
print("Feature matrix ready for clustering:", X_features.shape)

# ============================================================
# 4. OPTIONAL: TRY MULTIPLE K VALUES (SILHOUETTE)
# ============================================================

if TRY_MULTIPLE_K:
    print("\nSilhouette scores for different K:")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels_k = km.fit_predict(X_features)
        try:
            score = silhouette_score(X_features, labels_k)
        except Exception:
            score = np.nan
        print(f"  K = {k}: silhouette = {score:.3f}")

# ============================================================
# 5. FINAL K-MEANS CLUSTERING
# ============================================================

k_final = N_CLUSTERS_FIXED
print(f"\nRunning final K-means with K = {k_final} ...")

kmeans = KMeans(n_clusters=k_final, random_state=0, n_init=50)
cluster_labels = kmeans.fit_predict(X_features)

cluster_df = pd.DataFrame({
    "id_str": transect_ids,
    "cluster": cluster_labels
})

print("\nCluster counts:")
print(cluster_df["cluster"].value_counts().sort_index())

# ============================================================
# 6. LOAD TRANSECTS + COMPUTE ALONGSHORE DISTANCE
# ============================================================

print("\nLoading transects and computing alongshore distance...")

tgdf = gpd.read_file(TRANSECTS_SHP)
tgdf = tgdf.to_crs(TARGET_EPSG)

if TRANSECT_ID_FIELD not in tgdf.columns:
    raise ValueError(f"Shapefile missing field '{TRANSECT_ID_FIELD}'")

tgdf["id_str"] = tgdf[TRANSECT_ID_FIELD].astype(str)

tgdf["centroid"]   = tgdf.geometry.centroid
tgdf["centroid_x"] = tgdf["centroid"].x
tgdf["centroid_y"] = tgdf["centroid"].y

# sort alongshore by x (swap to y if needed)
tgdf_sorted = tgdf.sort_values("centroid_x").reset_index(drop=True)

xs = tgdf_sorted["centroid_x"].values
ys = tgdf_sorted["centroid_y"].values

s_along = np.zeros(len(xs))
for i in range(1, len(xs)):
    dx = xs[i] - xs[i - 1]
    dy = ys[i] - ys[i - 1]
    s_along[i] = s_along[i - 1] + np.hypot(dx, dy)

tgdf_sorted["s_along_m"] = s_along

# map id → alongshore
id_to_s = dict(zip(tgdf_sorted["id_str"], tgdf_sorted["s_along_m"]))

# ============================================================
# 7. MERGE CLUSTERS INTO TRANSECT GEODATAFRAME
# ============================================================

tgdf_clusters = tgdf_sorted.merge(
    cluster_df,
    on="id_str",
    how="left"
)

# Keep just the subset with clusters for plotting numeric stuff
plot_df = tgdf_clusters.dropna(subset=["cluster"]).copy()
plot_df["cluster"] = plot_df["cluster"].astype(int)

# ============================================================
# 8. PLOT: CLUSTER VS ALONGSHORE DISTANCE
# ============================================================

plt.figure(figsize=(10, 4))
plt.scatter(plot_df["s_along_m"], plot_df["cluster"],
            c=plot_df["cluster"], cmap="tab10", s=25)

plt.xlabel("Alongshore distance (m)")
plt.ylabel("Cluster ID")
plt.title(f"Samoa transect clusters (K = {k_final}) — alongshore structure")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 9. PLOT: MAP OF TRANSECTS COLORED BY CLUSTER
# ============================================================

fig, ax = plt.subplots(figsize=(6, 10))

tgdf_clusters.plot(
    column="cluster",
    ax=ax,
    cmap="tab10",
    linewidth=2,
    legend=True,
    missing_kwds={
        "color": "lightgrey",
        "label": "No cluster"
    }
)

ax.set_title(f"Samoa transect clusters (K = {k_final})")
ax.set_axis_off()
plt.tight_layout()
plt.show()

# At this point, you have in memory:
#   wide, wide_clean, wide_interp
#   X_features, cluster_df, tgdf_clusters, plot_df
# You can poke them in the console / notebook as needed.
