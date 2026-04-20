# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:14:19 2025

@author: mcallahan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- file paths ----
samoa_fp = r"Y:\OPC\beachWidthTool\processing\Samoa\TransectWidth\beach_widths_master.csv"
samoa_df = pd.read_csv(samoa_fp)

samoa_df['date'] = pd.to_datetime(samoa_df['date'], errors='coerce')
samoa_df = samoa_df.dropna(subset=['date','width_m'])
samoa_df = samoa_df.sort_values('date')

# average across transects for each timestamp
samoa_daily = (
    samoa_df.groupby('date')['width_m']
    .mean()
    .rename("samoa_width")
)
print(samoa_daily.head())

# ---- santa monica master file ----
sm_fp = r"Y:\OPC\beachWidthTool\processing\SantaMonica\TransectWidth\beach_widths_master.csv"
sm_df = pd.read_csv(sm_fp)

sm_df['date'] = pd.to_datetime(sm_df['date'], errors='coerce')
sm_df = sm_df.dropna(subset=['date','width_m'])
sm_df = sm_df.sort_values('date')

sm_daily = (
    sm_df.groupby('date')['width_m']
    .mean()
    .rename("sm_width")
)
print(sm_daily.head())


# inner join -> only dates existing in BOTH datasets
combo = pd.concat([samoa_daily, sm_daily], axis=1, join='inner').dropna()
print(combo.head())

pearson_corr = combo.corr().iloc[0,1]
print("Pearson correlation:", pearson_corr)

print(combo.corr())


def norm_xcorr(x1, x2, lags, dt=1):
    x1 = x1 - np.nanmean(x1)
    x2 = x2 - np.nanmean(x2)

    R = np.full((2*lags+1,), np.nan)
    k = np.arange(-lags, lags+1) * dt

    for i, n in enumerate(k):
        if n < 0:
            x2_shift = x2[-n:]
            x1_shift = x1[:len(x2_shift)]
        elif n > 0:
            x2_shift = x2[:-n]
            x1_shift = x1[n:]
        else:
            x2_shift = x2
            x1_shift = x1

        R[i] = np.corrcoef(x1_shift, x2_shift)[0,1]

    return R, k

lags = 60  # example: +/- 60 time steps
x1 = combo['samoa_width'].values
x2 = combo['sm_width'].values

R, k = norm_xcorr(x1, x2, lags, dt=1)


plt.figure(figsize=(8,4))
plt.plot(k, R, lw=2)
plt.axhline(0, color='k', lw=1)
plt.xlabel("Lag (days)")
plt.ylabel("Cross-correlation")
plt.title("Samoa vs Santa Monica Cross-Correlation")
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pearson_corr = combo.corr().iloc[0,1]

plt.figure(figsize=(6,5))
sns.regplot(x=combo['samoa_width'], y=combo['sm_width'], ci=None, scatter_kws={'s':25, 'alpha':0.7})
plt.xlabel("Samoa Avg Width (m)")
plt.ylabel("Santa Monica Avg Width (m)")
plt.title(f"Scatter: Samoa vs Santa Monica\nPearson r = {pearson_corr:.3f}")
plt.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,4))
plt.plot(combo.index, combo['samoa_width'], label="Samoa", lw=1.4)
plt.plot(combo.index, combo['sm_width'], label="Santa Monica", lw=1.4)
plt.xlabel("Date")
plt.ylabel("Avg Beach Width (m)")
plt.title("Samoa vs Santa Monica — Time Series")
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()


window = 60  # about 2 months if daily-ish data

rolling_corr = (
    combo['samoa_width']
    .rolling(window)
    .corr(combo['sm_width'])
)

plt.figure(figsize=(10,4))
plt.plot(rolling_corr.index, rolling_corr.values, lw=2)
plt.axhline(0, color='k', lw=1)
plt.ylabel(f"Rolling Corr ({window}-pt)")
plt.title("Rolling Correlation: Samoa vs Santa Monica")
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.show()



