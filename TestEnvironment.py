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
               color='red', alpha=0.12, linewidth=0)

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

# ----------- TIME WINDOW -----------
start_zoom = pd.Timestamp("2020-06-01")
end_zoom   = pd.Timestamp("2021-06-30")

# ----------- PREP DATA -------------
full_zoom = full.copy()
full_zoom['date'] = pd.to_datetime(full_zoom['date'])
full_zoom = full_zoom[(full_zoom['date'] >= start_zoom) &
                      (full_zoom['date'] <= end_zoom)].copy()

events_zoom = events.copy()
events_zoom['start_date']  = pd.to_datetime(events_zoom['start_date'])
events_zoom['end_date']    = pd.to_datetime(events_zoom['end_date'])
events_zoom['center_date'] = pd.to_datetime(events_zoom['center_date'])

# keep only storms that intersect the zoom window
events_zoom = events_zoom[
    (events_zoom['end_date'] >= start_zoom) &
    (events_zoom['start_date'] <= end_zoom)
].copy()

print("Storms in zoom window:")
print(events_zoom[['storm_event_id', 'start_date', 'end_date',
                   'max_wind_kt', 'max_waves_m']])

# ----------- PLOT 5-PANEL FIGURE -----------
fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

# 1) Beach width
ax = axes[0]
ax.plot(full_zoom['date'], full_zoom['mean_width_m'], lw=1.4, color='k')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.12, linewidth=0)
ax.set_ylabel("Width (m)")
ax.set_title("Samoa Beach Width & Forcing (2020-06 to 2021-06)\nStorm events shaded")
ax.grid(True, alpha=0.3)

# 2) Waves (WVHT)
ax = axes[1]
ax.plot(full_zoom['date'], full_zoom['WVHT_mean'], lw=1.0, color='g')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.12, linewidth=0)
ax.set_ylabel("WVHT (m)")
ax.set_title("Daily Mean Wave Height")
ax.grid(True, alpha=0.3)

# 3) Wind
ax = axes[2]
ax.plot(full_zoom['date'], full_zoom['daily_max_sust_knots'],
        lw=1.0, color='orange')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.12, linewidth=0)
ax.set_ylabel("Max wind (kt)")
ax.set_title("Daily Max Sustained Wind")
ax.grid(True, alpha=0.3)

# 4) Precipitation
ax = axes[3]
ax.plot(full_zoom['date'], full_zoom['precip_mm'], lw=1.0, color='b')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.12, linewidth=0)
ax.set_ylabel("Precip (mm)")
ax.set_title("Daily Precipitation")
ax.grid(True, alpha=0.3)

# 5) Pressure
ax = axes[4]
ax.plot(full_zoom['date'], full_zoom['daily_min_press_hpa'],
        lw=1.0, color='purple')
for _, ev in events_zoom.iterrows():
    ax.axvspan(ev['start_date'], ev['end_date'],
               color='red', alpha=0.12, linewidth=0)
ax.set_ylabel("Min MSLP (hPa)")
ax.set_title("Daily Minimum Sea-level Pressure")
ax.grid(True, alpha=0.3)
ax.set_xlabel("Date")

# legend for storms
storm_patch = mpatches.Patch(color='red', alpha=0.12, label='Storm event')
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
t_start = pd.Timestamp("2020-06-01")
t_end   = pd.Timestamp("2021-06-30")

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
    "Net Beach Width Change (2020-06 → 2021-06)\n"
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


num_window_dates = bw_win['date'].nunique()
print("Beach-width observation dates in window:", num_window_dates)
