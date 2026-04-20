# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:07:00 2025

@author: mcallahan
"""

import math
from shapely.geometry import LineString

def make_perpendicular_transects(ref_line: LineString,
                                 spacing_m: float = 20.0,
                                 half_length_m: float = 200.0):
    """
    Create perpendicular transects every `spacing_m` meters
    along a reference LineString.

    Each transect is centered on the reference line and extends
    `half_length_m` in both directions (so full length = 2 * half_length_m).

    Returns a list of LineString transects.
    """
    transects = []

    # total length of the reference line (units = CRS units, so meters if CRS is projected)
    L = ref_line.length

    # how far to move along the line for finite-difference tangent approximation
    # keep it small but non-zero
    delta = spacing_m / 2.0

    # distances along the line where we create transects
    # include endpoint with +1
    n_steps = int(L // spacing_m)
    distances = [i * spacing_m for i in range(n_steps + 1)]

    for d in distances:
        # point on the line at distance d
        pt = ref_line.interpolate(d)

        # estimate tangent direction using a small window around d
        d1 = max(d - delta, 0.0)
        d2 = min(d + delta, L)

        # if we have no window (degenerate at very start/end), skip
        if d2 == d1:
            continue

        p1 = ref_line.interpolate(d1)
        p2 = ref_line.interpolate(d2)

        # tangent vector (p1 -> p2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y

        # if tangent is zero length (weird geometry), skip
        if dx == 0 and dy == 0:
            continue

        # normal (perpendicular) vector: rotate (dx,dy) by +90°: (-dy, dx)
        nx = -dy
        ny = dx

        # normalize to unit vector
        norm = math.hypot(nx, ny)
        nx /= norm
        ny /= norm

        # endpoints of transect: move half_length_m along ± normal
        x0 = pt.x - nx * half_length_m
        y0 = pt.y - ny * half_length_m
        x1 = pt.x + nx * half_length_m
        y1 = pt.y + ny * half_length_m

        tr = LineString([(x0, y0), (x1, y1)])
        transects.append(tr)

    return transects
import geopandas as gpd

# 1) Read reference line (must be in projected CRS, meters!)
ref_fp = r"Y:\OPC\beachWidthTool\Sites\SantaMonica\transects\Samo_reference_line.shp"
ref_gdf = gpd.read_file(ref_fp)

if ref_gdf.crs is None:
    raise ValueError("Reference line has no CRS; define one before using.")

# Reproject to a metric CRS (example: UTM zone 10N)
ref_gdf = ref_gdf.to_crs(32610)

# Assume a single reference line; if multiple, you can loop
ref_line = ref_gdf.geometry.iloc[0]

# 2) Generate transects every 10 m, 100 m long (±50 m from the line)
transects = make_perpendicular_transects(ref_line,
                                         spacing_m=20.0,
                                         half_length_m=500.0)

# 3) Build a GeoDataFrame and save
tr_gdf = gpd.GeoDataFrame({"id": range(len(transects))},
                          geometry=transects,
                          crs=ref_gdf.crs)

out_transects_fp = r"Y:\OPC\beachWidthTool\Sites\SantaMonica\transects\transects_10m.shp"
tr_gdf.to_file(out_transects_fp)
print(f"Saved {len(transects)} transects → {out_transects_fp}")
