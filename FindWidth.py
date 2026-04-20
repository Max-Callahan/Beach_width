# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:43:42 2025

@author: mcallahan
"""
# -*- coding: utf-8 -*-
"""
Samoa width-extraction script
Per-date waterlines + per-date backlines + multi-feature transect shapefile
"""

import os, re, glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.geometry import shape
from shapely.ops import unary_union
import rasterio
from rasterio import features

# -------- USER PATHS --------
# Multi-feature transect shapefile (each feature = one transect)
transects_fp = r"Y:\OPC\beachWidthTool\Sites\Samoa\Transects\transects_10m.shp"
TRANSECT_ID_FIELD = "FID"   # field in transects_10m.shp that identifies each transect

# Rasters: one WATER and one BACK (veg) per date, both value==1 where line exists
water_dir    = r"Y:\OPC\beachWidthTool\processing\Samoa\WaterLines"
back_dir     = r"Y:\OPC\beachWidthTool\processing\Samoa\VegetationLines"

out_dir      = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth"
master_csv   = r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv"
# ----------------------------

TARGET_EPSG = 32610     # projected CRS in meters
SNAP_BUF_M  = 2.0       # catch pixel gaps at intersections


# ---------- helpers ----------
def extract_date_key(fname: str) -> str:
    """
    Extract 'YYYY-MM-DD' from filename like:
    YYYYMMDD or YYYY_MM_DD or YYYY-MM-DD.
    """
    base = os.path.basename(fname)
    m = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', base)
    if not m:
        raise ValueError(f"Could not parse date from filename: {base}")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def raster_to_lines(raster_path):
    """
    Vectorize pixels == 1 into a single union geometry.
    Returns (union_geom, crs, pixel_size).
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        mask = (band == 1)
        pxsz = max(abs(src.transform.a), abs(src.transform.e))
        geoms = [shape(gm) for gm, val in features.shapes(
                    band, mask=mask, transform=src.transform)]
        if not geoms:
            return None, src.crs, pxsz
        return unary_union(geoms), src.crs, pxsz


def intersect_point_on_transect(tr: LineString, line_union, snap_buf_m: float):
    """
    Return a Point where transect meets line set; use small buffer if fragmented.
    """
    if line_union is None:
        return None

    inter = tr.intersection(line_union)
    # simple point intersection
    if isinstance(inter, Point):
        return inter

    # any intersection geometry: take representative point & snap to transect
    if not inter.is_empty:
        rp = inter.representative_point()
        return tr.interpolate(tr.project(rp))

    # fallback: buffer transect to catch near-misses
    inter2 = tr.buffer(snap_buf_m).intersection(line_union)
    if inter2.is_empty:
        return None
    rp2 = inter2.representative_point()
    return tr.interpolate(tr.project(rp2))


# ---------- 1) Pair water/back rasters by date ----------
water_map = {}
for f in glob.glob(os.path.join(water_dir, "*.tif")):
    try:
        water_map[extract_date_key(f)] = f
    except Exception:
        pass

back_map = {}
for f in glob.glob(os.path.join(back_dir, "*.tif")):
    try:
        back_map[extract_date_key(f)] = f
    except Exception:
        pass

dates = sorted(set(water_map) & set(back_map))
if not dates:
    raise ValueError("No matching dates between waterlines and backlines.")

print(f"Found {len(dates)} dates with both waterline and backline rasters.")

# ---------- 2) Precompute WATER/BACK unions per date (projected) ----------
date_unions = {}  # date -> (water_union_proj, back_union_proj)

for d in dates:
    w_union, w_crs, _ = raster_to_lines(water_map[d])
    b_union, b_crs, _ = raster_to_lines(back_map[d])

    if (w_union is None) or (b_union is None):
        date_unions[d] = (None, None)
        continue

    w_proj = gpd.GeoDataFrame(geometry=[w_union], crs=w_crs).to_crs(TARGET_EPSG).geometry.iloc[0]
    b_proj = gpd.GeoDataFrame(geometry=[b_union], crs=b_crs).to_crs(TARGET_EPSG).geometry.iloc[0]
    date_unions[d] = (w_proj, b_proj)

# ---------- 3) Load transects (single multi-feature shapefile) ----------
gdf = gpd.read_file(transects_fp)
if gdf.crs is None:
    raise ValueError("Transect shapefile has no CRS.")
gdf = gdf.to_crs(TARGET_EPSG)

# ensure ID field exists
if TRANSECT_ID_FIELD not in gdf.columns:
    gdf[TRANSECT_ID_FIELD] = gdf.index.astype(str)

os.makedirs(out_dir, exist_ok=True)
all_rows = []

# ---------- 4) Loop over transects + dates and compute widths ----------
for idx, row in gdf.iterrows():
    tr_geom = row.geometry
    tr_name = str(row[TRANSECT_ID_FIELD])

    rows = []
    for d in dates:
        w_union_proj, b_union_proj = date_unions[d]

        if (w_union_proj is None) or (b_union_proj is None):
            rows.append({
                "transect": tr_name,
                "date": d,
                "width_m": np.nan,
                "note": "missing waterline/backline geometry"
            })
            continue

        w_pt = intersect_point_on_transect(tr_geom, w_union_proj, SNAP_BUF_M)
        b_pt = intersect_point_on_transect(tr_geom, b_union_proj, SNAP_BUF_M)

        if (w_pt is None) or (b_pt is None):
            rows.append({
                "transect": tr_name,
                "date": d,
                "width_m": np.nan,
                "note": "no intersection"
            })
            continue

        width = float(w_pt.distance(b_pt))
        rows.append({
            "transect": tr_name,
            "date": d,
            "width_m": width,
            "note": ""
        })

    # per-transect CSV (optional but nice to have)
   # df_tr = pd.DataFrame(rows).sort_values(["transect", "date"])
    #out_fp = os.path.join(out_dir, f"{tr_name}.csv")
    #df_tr.to_csv(out_fp, index=False)
    #print(f"Saved {len(df_tr)} rows → {out_fp}")

    all_rows.extend(rows)

# ---------- 5) Master CSV ----------
pd.DataFrame(all_rows).sort_values(["transect", "date"]).to_csv(master_csv, index=False)
print(f"Saved master CSV → {master_csv}")

























# -*- coding: utf-8 -*-
"""
Compute beach width along transects for Santa Monica
using:
  - a static backline (single shapefile or raster)
  - multiple dated waterline rasters (value == 1 along wet/dry line)
  - a single multi-feature transect shapefile

Output: one master CSV of width per transect per date.
"""

import os, re, glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union
import rasterio
from rasterio import features

# -------- USER PATHS --------
transects_fp = r"Y:\OPC\beachWidthTool\Sites\SantaMonica\transects\transects_10m.shp"
TRANSECT_ID_FIELD = "id"   # <-- change if your transect shapefile uses a different ID field

water_dir   = r"Y:\OPC\beachWidthTool\processing\SantaMonica\WetDryLines"  # waterline rasters (tif, value=1)
back_path   = r"Y:\OPC\beachWidthTool\Sites\SantaMonica\StaticBackLine\staticBikePath.shp"  # static backline (shp or tif)

out_dir     = r"Y:\OPC\beachWidthTool\Sites\SantaMonica\BeachWidth"
master_csv  = os.path.join(out_dir, "beach_widths_master.csv")
# ----------------------------

TARGET_EPSG = 32610    # projected CRS in meters (use what you used elsewhere)
SNAP_BUF_M  = 2.0      # buffer to catch small gaps / fragmented lines


# ---------- HELPERS ----------
def extract_date_key(fname: str) -> str:
    """
    Extract 'YYYY-MM-DD' from filename like:
    YYYYMMDD, YYYY_MM_DD, or YYYY-MM-DD.
    """
    base = os.path.basename(fname)
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", base)
    if not m:
        raise ValueError(f"Could not parse date from filename: {base}")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"


def raster_to_union(raster_path):
    """
    Vectorize pixels == 1 into a single union geometry.
    Returns (union_geom, crs, pixel_size).
    """
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        mask = (band == 1)
        pxsz = max(abs(src.transform.a), abs(src.transform.e))
        geoms = [shape(gm) for gm, val in features.shapes(
                    band, mask=mask, transform=src.transform)]
        if not geoms:
            return None, src.crs, pxsz
        return unary_union(geoms), src.crs, pxsz


def vector_to_union(vector_path):
    """
    Dissolve all features in a vector file into a single geometry.
    Returns (union_geom, crs).
    """
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        return None, None
    return gdf.unary_union, gdf.crs


def path_to_union(path):
    """
    Open raster OR vector and return (union_geom, crs).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tif":
        u, crs, _ = raster_to_union(path)
        return u, crs
    else:
        return vector_to_union(path)


def intersect_point_on_transect(tr: LineString, line_union, snap_buf_m: float):
    """
    Return a Point where transect meets the line geometry.
    If intersection is fragmented or narrow, use a small buffer around the transect.
    """
    if line_union is None:
        return None

    inter = tr.intersection(line_union)

    # direct point intersection
    if isinstance(inter, Point):
        return inter

    # some geometry -> snap representative point to transect
    if not inter.is_empty:
        rp = inter.representative_point()
        return tr.interpolate(tr.project(rp))

    # fallback: buffer the transect to catch near-misses
    inter2 = tr.buffer(snap_buf_m).intersection(line_union)
    if inter2.is_empty:
        return None

    rp2 = inter2.representative_point()
    return tr.interpolate(tr.project(rp2))


# ---------- 1) STATIC BACKLINE (once) ----------
back_union_raw, back_crs = path_to_union(back_path)
if back_union_raw is None or back_crs is None:
    raise ValueError("Static backline is empty or has no CRS.")

back_union_proj = (
    gpd.GeoDataFrame(geometry=[back_union_raw], crs=back_crs)
       .to_crs(TARGET_EPSG)
       .geometry.iloc[0]
)

# ---------- 2) WATERLINE RASTERS (by date) ----------
water_map = {}
for f in glob.glob(os.path.join(water_dir, "*.tif")):
    try:
        water_map[extract_date_key(f)] = f
    except Exception:
        # skip filenames without a parseable date
        pass

dates = sorted(water_map)
if not dates:
    raise ValueError("No waterline rasters found or filenames do not contain parseable dates.")

print(f"Found {len(dates)} waterline dates.")

# Precompute projected union for each waterline date
date_water_unions = {}  # date -> projected union geom or None
for d in dates:
    w_union_raw, w_crs, _ = raster_to_union(water_map[d])
    if w_union_raw is None:
        date_water_unions[d] = None
        continue
    w_proj = (
        gpd.GeoDataFrame(geometry=[w_union_raw], crs=w_crs)
          .to_crs(TARGET_EPSG)
          .geometry.iloc[0]
    )
    date_water_unions[d] = w_proj

# ---------- 3) LOAD TRANSECTS (multi-feature shapefile) ----------
os.makedirs(out_dir, exist_ok=True)

tgdf = gpd.read_file(transects_fp)
if tgdf.crs is None:
    raise ValueError("Transect shapefile has no CRS.")
tgdf = tgdf.to_crs(TARGET_EPSG)

if TRANSECT_ID_FIELD not in tgdf.columns:
    # fallback to using index if no explicit ID field
    tgdf[TRANSECT_ID_FIELD] = tgdf.index

tgdf[TRANSECT_ID_FIELD] = tgdf[TRANSECT_ID_FIELD].astype(str)

# ---------- 4) LOOP TRANSECTS × DATES, COMPUTE WIDTHS ----------
all_rows = []

for idx, row in tgdf.iterrows():
    tr_geom = row.geometry
    tr_id   = row[TRANSECT_ID_FIELD]  # as string

    for d in dates:
        w_union_proj = date_water_unions[d]

        if w_union_proj is None:
            all_rows.append({
                "id": tr_id,
                "date": d,
                "width_m": np.nan,
                "note": "missing waterline geometry"
            })
            continue

        w_pt = intersect_point_on_transect(tr_geom, w_union_proj, SNAP_BUF_M)
        b_pt = intersect_point_on_transect(tr_geom, back_union_proj, SNAP_BUF_M)

        if (w_pt is None) or (b_pt is None):
            all_rows.append({
                "id": tr_id,
                "date": d,
                "width_m": np.nan,
                "note": "no intersection"
            })
            continue

        width = float(w_pt.distance(b_pt))
        all_rows.append({
            "id": tr_id,
            "date": d,
            "width_m": width,
            "note": ""
        })

# ---------- 5) MASTER CSV ONLY ----------
df_out = pd.DataFrame(all_rows)
df_out = df_out.sort_values(["id", "date"])
df_out.to_csv(master_csv, index=False)

print(f"Saved {len(df_out)} rows → {master_csv}")

