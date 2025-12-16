import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import numpy as np

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

# ---- FIRE FREQUENCY BY YEAR ----

fires_per_year = fire_data.groupby('year').size().sort_index()

#Graph frequency of extreme fire events
q75 = fire_data['area_ha'].quantile(0.75)
extreme_fire = fire_data[fire_data['area_ha'] >= q75]

frequency_of_extreme = extreme_fire.groupby('year').size().sort_index()

print(frequency_of_extreme.get(1995))

'''
plt.figure(figsize=(12,6))
plt.scatter(frequency_of_extreme.index, frequency_of_extreme.values)
plt.title("Yearly Frequency of Extreme Fires")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.show()
'''

#frequency_of_extreme.to_csv('extreme_fire_frequency.csv', header=True)

'''
print(frequency_of_extreme)
print(frequency_of_extreme.index)
print(frequency_of_extreme.values)
'''