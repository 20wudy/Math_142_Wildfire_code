import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import numpy as np
import math

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

modern_fire_data = fire_data[fire_data['year'] > 2004]

threshold_area = 2000

filtered_data = modern_fire_data[modern_fire_data['area_ha'] >= threshold_area]

large_fires = filtered_data[filtered_data['area_ha'] >= 10000].copy()

small_fires = filtered_data[filtered_data['area_ha'] <10000].copy()

#Find burn times of these datas
large_fires['burn_time'] = (large_fires['extinguish_date'] - large_fires['ignition_date']).dt.total_seconds()/3600
large_fires = large_fires[large_fires['burn_time'].notna() & (large_fires['burn_time'] >= 0)]

small_fires['burn_time'] = (small_fires['extinguish_date'] - small_fires['ignition_date']).dt.total_seconds()/3600
small_fires = small_fires[small_fires['burn_time'].notna() & (small_fires['burn_time'] >= 0)]


mean_burn_time_large = large_fires['burn_time'].mean()
mean_burn_time_small = small_fires['burn_time'].mean()

print("Large: ", mean_burn_time_large)
print("Small: ", mean_burn_time_small)


