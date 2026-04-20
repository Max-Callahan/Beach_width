# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 11:07:27 2026

@author: mcallahan
"""

import pandas as pd
import numpy as np


df = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\Tides\2024_rawtide.csv", parse_dates = ['Time (GMT)'])

# ---- interpret datetime as GMT/UTC, convert to Pacific ----
df["datetime_local"] = (
    df["Time (GMT)"]
    .dt.tz_localize("UTC")                 # GMT == UTC for tz purposes
    .dt.tz_convert("America/Los_Angeles")  # Pacific time (PST/PDT handled)
)

# ---- filter to 10:00 Pacific time ----
df_10am = df[df["datetime_local"].dt.hour == 10].copy()

# ---- average tide at 10am ----
avg_10am = df_10am["Verified (ft)"].mean()        # change "tide_m" to your tide column

# ---- keep days within ±0.4 m of that average ----
df_10am["date_local"] = df_10am["datetime_local"].dt.date
filtered = df_10am[(df_10am["Verified (ft)"] - avg_10am).abs() <= 0.4]

# ---- result: list of days + 10am tide ----
result = filtered[["date_local", "datetime_local", "Verified (ft)"]].sort_values("datetime_local")
print("Average tide at 10am Pacific:", avg_10am)
print(result)

# optional save
#result.to_csv("tide_days_within_0p4m_of_10am_avg.csv", index=False)
