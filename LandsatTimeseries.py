# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:26:56 2025

@author: mcallahan
"""

# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# USER SETTINGS
# -------------------------------------------------
folder = r"Y:\OPC\beachWidthTool\CoastSat\usa_CA_0288_timeseries"  # <-- change this
date_col = "dates UTC"
width_col = "chainage (m)"

beach_widths = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\BeachWidth\beach_widths_master.csv")
beach_widths['date'] = pd.to_datetime(beach_widths['date'])
average_bw_perdate = (beach_widths.groupby('date')['width_m'].mean().reset_index())
# force Samoa series to UTC (assumes dates are already UTC-like)
average_bw_perdate["date"] = pd.to_datetime(
    average_bw_perdate["date"], utc=True
)


#test_csv = pd.read_csv(r"Y:\OPC\beachWidthTool\CoastSat\usa_CA_0288_timeseries\usa_CA_0288-0000.csv")

# -------------------------------------------------
# 1) Collect CSV files
# -------------------------------------------------
file_list = sorted(glob.glob(os.path.join(folder, "*.csv")))

if len(file_list) == 0:
    raise RuntimeError("No CSV files found")

# -------------------------------------------------
# 2) Read and merge (outer join on date)
# -------------------------------------------------
merged = None

for fpath in file_list:
    df = pd.read_csv(fpath)
    # normalize column names (removes hidden unicode junk)
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(u"\xa0", " ", regex=False)
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # rename width column to keep transects separate
    transect_name = os.path.splitext(os.path.basename(fpath))[0]
    df = df[[date_col, width_col]].rename(
        columns={width_col: transect_name}
    )

    if merged is None:
        merged = df
    else:
        merged = pd.merge(
            merged,
            df,
            on=date_col,
            how="outer"
        )

# sort by date
merged = merged.sort_values(date_col).reset_index(drop=True)

# -------------------------------------------------
# 3) Compute average width per date (ignore NaNs)
# -------------------------------------------------
transect_cols = merged.columns.drop(date_col)

merged["ls_width_m"] = merged[transect_cols].mean(axis=1)
merged['date'] = pd.to_datetime(merged[date_col])
# -------------------------------------------------
# 4) Plot average beach width
# -------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(
    merged[date_col],
    merged["ls_width_m"],
    lw=1.8,
    color="k"
)
#planet timeseries
plt.plot(
    average_bw_perdate['date'],
    average_bw_perdate['width_m'],
    lw = 1.8,
    color = 'c'
)
plt.xlabel("Date")
plt.ylabel("Average Beach Width (m)")
plt.title("Average Beach Width Time Series")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()



# --- Make sure both date columns are datetime (no merging needed) ---
average_bw_perdate["date"] = pd.to_datetime(average_bw_perdate["date"], errors="coerce")
merged["date"] = pd.to_datetime(merged[date_col], errors="coerce")

# --- Drop bad dates (keep NaNs in widths if you want, but dates must exist) ---
average_bw_perdate = average_bw_perdate.dropna(subset=["date"]).sort_values("date")
merged = merged.dropna(subset=["date"]).sort_values("date")

# --- Normalize (z-score) separately ---
my_mean = average_bw_perdate["width_m"].mean()
my_std  = average_bw_perdate["width_m"].std()
average_bw_perdate["my_z"] = (average_bw_perdate["width_m"] - my_mean) / my_std

ls_mean = merged["ls_width_m"].mean()
ls_std  = merged["ls_width_m"].std()
merged["ls_z"] = (merged["ls_width_m"] - ls_mean) / ls_std

# --- Plot on the same axes ---
plt.figure(figsize=(12,4))
plt.plot(average_bw_perdate["date"], average_bw_perdate["my_z"], lw=1.6, label="My series (z)")
plt.plot(merged["date"], merged["ls_z"], lw=1.6, label="Landsat/CoastSat (z)")
plt.axhline(0, lw=1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Relative Change Comparison (Normalized, No Merge)")
plt.xlabel("Date")
plt.ylabel("Normalized units (z-score)")
plt.tight_layout()
plt.show()
