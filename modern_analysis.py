import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import numpy as np
import math
from scipy.stats.mstats import winsorize
from statsmodels.stats.weightstats import ztest

#exploratory data analysis:
#Find statistically significant shift in burn times using a z-test

print("Imports loaded")
gdb_path = "Bushfire Extents - Historical (2025).gdb"
 
layer_name = "National_Historical_Bushfire_Extents_v4"

fire_data = gpd.read_file(gdb_path, layer = layer_name)

fire_data['ignition_date'] = pd.to_datetime(fire_data['ignition_date'], errors='coerce', utc=True)
fire_data['capture_date'] = pd.to_datetime(fire_data['capture_date'], errors='coerce', utc=True)
fire_data['extinguish_date'] = pd.to_datetime(fire_data['extinguish_date'], errors='coerce', utc=True)

fire_data['ignition_date'] = fire_data['ignition_date'].dt.tz_localize(None)
fire_data['extinguish_date'] = fire_data['extinguish_date'].dt.tz_localize(None)

fire_data['year'] = fire_data['ignition_date'].dt.year

threshold_area = 2000

filtered_data = fire_data[fire_data['area_ha'] >= threshold_area]

#Find burn times of these datas
filtered_data['burn_time'] = (filtered_data['extinguish_date'] - filtered_data['ignition_date']).dt.total_seconds()/3600
filtered_data = filtered_data[filtered_data['burn_time'].notna() & (filtered_data['burn_time'] >= 0)]

yearly_mean_burn_time = filtered_data.groupby('year')['burn_time'].mean().reset_index()

#removing burn-time outliers for visual purposes. Want to find a year to use in my z-test hypothesis
visual_burn_data = yearly_mean_burn_time[yearly_mean_burn_time['burn_time'] <= 20000]

plt.figure(figsize=(12,6))
plt.scatter(visual_burn_data['year'], visual_burn_data['burn_time'])
plt.title("Yearly Average Burn Time (hr)")
plt.xlabel("Year")
plt.ylabel("Burn Time (hr)")
plt.show()
#Visually, 2004 looks like a change-point (Aligns with change-point detection method in analysis.py)

#Winsorize raw data to deal with extreme outliers (Black Friday event)

burn_times_wins = winsorize(filtered_data['burn_time'], limits=[0, 0.01])

#run z-test
split_year = 2004

historical = filtered_data[filtered_data['year'] < split_year]['burn_time']
modern = filtered_data[filtered_data['year'] >= split_year]['burn_time']

z_stat, p_value = ztest(historical, modern)
print(z_stat, p_value)

