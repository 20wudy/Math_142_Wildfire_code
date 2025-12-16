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

cut_off = 2004
#2004
modern_fire_data = fire_data[fire_data['year'] > cut_off]

print("\nModern dataset (2005-2025), Australia:")

# ---- FIRE AREA STATISTICS ----

total_area = modern_fire_data['area_ha'].sum()
mean_area = modern_fire_data['area_ha'].mean()
max_area = modern_fire_data['area_ha'].max()
min_area = modern_fire_data['area_ha'].min()

print("\n--- Fire Area Statistics (hectares)---")
print(f"Total burned area: {total_area:,.0f} ha")
print(f"Mean fire area: {mean_area:,.1f} ha")
print(f"Max fire area: {max_area:,.0f} ha")
print(f"Min fire area: {min_area:,.0f} ha")


# ---- FIRE FREQUENCY BY YEAR ----

fires_per_year = modern_fire_data.groupby('year').size().sort_index()

print("\n--- Fire Frequency by Year ---")
print(fires_per_year)


# ---- FIRE FREQUENCY BY STATE ----
fires_per_state_year = modern_fire_data.groupby(['state','year']).size().unstack(fill_value=0)
print("\n--- Fire Frequency by State and Year ---")
print(fires_per_state_year)


modern_vic = modern_fire_data[modern_fire_data['state'] == 'VIC (Victoria)']

print("\nModern dataset 2005-2025, Victoria:")
print("\nNumber of Victoria fires: ", len(modern_vic))

# ---- VICTORIA FIRE AREA STATISTICS ----
total_area_vic = modern_vic['area_ha'].sum()
mean_area_vic = modern_vic['area_ha'].mean()
max_area_vic = modern_vic['area_ha'].max()
min_area_vic = modern_vic['area_ha'].min()

print("\n--- Fire Area Statistics (hectares)---")
print(f"Total burned area: {total_area_vic:,.0f} ha")
print(f"Mean fire area: {mean_area_vic:,.1f} ha")
print(f"Max fire area: {max_area_vic:,.0f} ha")
print(f"Min fire area: {min_area_vic:,.0f} ha")

# ---- VICTORIA FIRE FREQUENCY BY YEAR ----

fires_per_year_vic = modern_vic.groupby('year').size().sort_index()

print("\n--- Fire Frequency by Year in Victoria ---")
print(fires_per_year_vic)

#Plot yearly fire frequency

plt.figure(figsize=(12,6))
plt.scatter(fires_per_year.index, fires_per_year.values)
plt.title("Yearly Fire Frequency in Australia")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.show()

#Graph frequency of extreme fire events
q75 = modern_fire_data['area_ha'].quantile(0.75)
extreme_fire = modern_fire_data[modern_fire_data['area_ha'] >= q75]

frequency_of_extreme = extreme_fire.groupby('year').size().sort_index()

plt.figure(figsize=(12,6))
plt.scatter(frequency_of_extreme.index, frequency_of_extreme.values)
plt.title("Yearly Frequency of Extreme Fires")
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.show()

print("\n--- Frequency of Extreme Fires by Year, Australia")
print(frequency_of_extreme)