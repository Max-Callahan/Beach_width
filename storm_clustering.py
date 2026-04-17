import os
import glob
import io
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#Cleaning and preprocessing weather csv, already merged from weather script
#weather data gathering 
weather_df = pd.read_csv(r"Y:\OPC\beachWidthTool\Sites\Samoa\Weather\samoa_forcing_fulltimeseries.csv")
start_year = 1984
end_year = 2025
weather_df.columns
weather_df.head()

#make sure dat is datetime and the dataframe is in order
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df = weather_df.sort_values('date')
weather_df = weather_df.dropna()

features = ['daily_max_sust_knots', 'daily_mean_sust_knots', 'daily_min_press_hpa',  'WVHT_mean']

#feature array
X = weather_df[features]

#standardize variables, I have disproportionately high pressure values vs wave height for example
X_std = StandardScaler().fit_transform(X)



##########################################################################
#Isolation Forest analysis 
#########################################################################


#isolation Forest anomaly detection (alternative to Kmeans clustering)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

#isolation forest
#contamination = expected proportion of storm days in dataset
#0.1 means 10% of days flagged (need to tune)
iso_forest = IsolationForest(
    n_estimators = 200,
    contamination = 0.1,
    random_state = 42
)

weather_df['anomaly'] = iso_forest.fit_predict(X_std)
weather_df['anomaly_score'] = iso_forest.decision_function(X_std)

# After fitting the iso forest, apply physical storm conditions
storm_threshold = {
    'daily_max_sust_knots': weather_df['daily_max_sust_knots'].quantile(0.90),  # upper outliers only
    'daily_mean_sust_knots': weather_df['daily_mean_sust_knots'].quantile(0.90),
    'WVHT_mean': weather_df['WVHT_mean'].quantile(0.90),
    'daily_min_press_hpa': weather_df['daily_min_press_hpa'].quantile(0.10)     # lower outliers only
}

print("Storm thresholds:")
for k, v in storm_threshold.items():
    print(f"  {k}: {v:.2f}")

# A flagged day is only a storm if it exceeds physical thresholds
storm_mask = (
    (weather_df['anomaly'] == -1) &  # flagged by iso forest
    (
        (weather_df['daily_max_sust_knots'] >= storm_threshold['daily_max_sust_knots']) &
        (weather_df['daily_mean_sust_knots'] >= storm_threshold['daily_mean_sust_knots'])&
        (weather_df['daily_min_press_hpa'] <= storm_threshold['daily_min_press_hpa']) |
        (weather_df['WVHT_mean'] >= storm_threshold['WVHT_mean']) 
    )
)

weather_df['storm_flag'] = storm_mask.astype(int)  # 1 = storm, 0 = normal

storm_days = weather_df[weather_df['storm_flag'] == 1]
normal_days = weather_df[weather_df['storm_flag'] == 0]

print(f"\nTotal days:  {len(weather_df)}")
print(f"Storm days:  {len(storm_days)} ({len(storm_days)/len(weather_df)*100:.1f}%)")
print(f"Normal days: {len(normal_days)} ({len(normal_days)/len(weather_df)*100:.1f}%)")



#plotting ISO forest output

# --- Plot 1: Time series with flagged storms ---
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

plot_vars = {
    'daily_max_sust_knots': 'Max Sustained Wind (knots)',
    'daily_mean_sust_knots': 'Mean Sustained Wind (knots)',
    'daily_min_press_hpa': 'Min Pressure (hPa)',
    'WVHT_mean': 'Mean Wave Height (m)'
}

for ax, (col, label) in zip(axes, plot_vars.items()):
    ax.plot(weather_df['date'], weather_df[col], color='steelblue', linewidth=0.8)
    ax.scatter(storm_days['date'], storm_days[col], color='red', s=10, zorder=5, label='Flagged storm')
    ax.set_ylabel(label, fontsize=9)

axes[0].set_title('Isolation Forest — Storm Day Detection')
axes[0].legend(loc='upper right')
axes[-1].set_xlabel('Date')

plt.tight_layout()
plt.show()

# --- Plot 2: Anomaly score over time ---
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(weather_df['date'], weather_df['anomaly_score'], color='gray', linewidth=0.8)
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Decision boundary')
ax.set_ylabel('Anomaly Score')
ax.set_xlabel('Date')
ax.set_title('Anomaly Score Over Time')
ax.legend()
plt.tight_layout()
plt.show()


#####################################################
#Plotting the storm days using clustered data
#####################################################



weather_df['date'] = pd.to_datetime(weather_df['date'])
storm_days['date'] = pd.to_datetime(storm_days['date'])

# Create a binary storm column on the full dataframe
weather_df['is_storm'] = weather_df['date'].isin(storm_days['date']).astype(int)
#weather_df['rough_weather'] = weather_df['date'].isin(rough_days_df['date']).astype(int)


fig, ax = plt.subplots(figsize=(14, 3))

ax.bar(weather_df['date'], weather_df['storm_flag'], color='red', width=1.5)
#ax.bar(weather_df['date'], weather_df['rough_weather'], color = 'orange', width = 1.0)
ax.set_title('Storm Days')
ax.set_xlabel('Date')
ax.set_yticks([0, 1])
ax.set_yticklabels(['No Storm', 'Storm'])
ax.set_ylim(0, 1.2)

plt.tight_layout()
#plt.savefig('/mnt/user-data/outputs/storm_indicator.png', dpi=150)
plt.show()