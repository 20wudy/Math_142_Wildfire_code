
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ruptures as rpt
import numpy as np

#Included in this code: 
# Statistics for entire dataset
# Plot average burn-time per year for entire dataset
# Plot average burn-time per year excluding outlier 
# Algorithm for finding a "modern cut-off" date
# Plot average burn-time per year in modern context (2005-2025)
# Burn-time statistics in modern context
# Plot average burn area per year in modern context

print("Imports loaded")
gdb_path = "Bushfire Extents - Historical (2025).gdb"
 
layer_name = "National_Historical_Bushfire_Extents_v4"

fire_data = gpd.read_file(gdb_path, layer = layer_name)

#fire_data.to_csv("fire_data_raw.csv", index=False)

print("Dataset loaded successfully. ")
print("Number of fire polygons: ", len(fire_data))
#print("\nColumn names: ")
#print(fire_data.columns)

''' 
layers found in geo database: 
NT_Historical_Bushfire_Extents_v1
National_Historical_Bushfire_Extents_v4
'''



# ---- CONVERT DATES ----
fire_data['ignition_date'] = pd.to_datetime(fire_data['ignition_date'], errors='coerce', utc=True)
fire_data['capture_date'] = pd.to_datetime(fire_data['capture_date'], errors='coerce', utc=True)
fire_data['extinguish_date'] = pd.to_datetime(fire_data['extinguish_date'], errors='coerce', utc=True)


#stats for entire dataset, Australia, 1898-2025, before filtering:

#COMMENTED OUT FOR RUNTIME EFFICIENCY
print("Statistics for entire dataset, Australia, 1899-2025, before filtering")
# ---- FIRE AREA STATISTICS ----
total_area = fire_data['area_ha'].sum()
mean_area = fire_data['area_ha'].mean()
max_area = fire_data['area_ha'].max()
min_area = fire_data['area_ha'].min()

print("\n--- Fire Area Statistics (hectares), entire dataset---")
print(f"Total burned area: {total_area:,.0f} ha")
print(f"Mean fire area: {mean_area:,.1f} ha")
print(f"Max fire area: {max_area:,.0f} ha")
print(f"Min fire area: {min_area:,.0f} ha")

# ---- FIRE FREQUENCY BY YEAR ----
fire_data['year'] = fire_data['ignition_date'].dt.year
fires_per_year = fire_data.groupby('year').size().sort_index()

print("\n--- Fire Frequency by Year ---")
print(fires_per_year)

# ---- FIRE FREQUENCY BY STATE ----
fires_per_state_year = fire_data.groupby(['state','year']).size().unstack(fill_value=0)
print("\n--- Fire Frequency by State and Year ---")
print(fires_per_state_year)

#Calculate burn-time stats, that can help us predict the behavior of modern wildfires

#STEP 1 Clean data: fix time zone conflict, Remove timezone info so both are tz-naive
#Documentation summary: Although both timestamps are documented as UTC, the raw data encodes them inconsistently (tz-aware vs tz-naive), requiring normalization before temporal analysis
fire_data['ignition_date'] = fire_data['ignition_date'].dt.tz_localize(None)
fire_data['extinguish_date'] = fire_data['extinguish_date'].dt.tz_localize(None)


#STEP 2 Create column with burn time. Remove bad data (negative values)
#Time is in hours
fire_data['burn_time'] = (fire_data['extinguish_date'] - fire_data['ignition_date']).dt.total_seconds()/3600
fire_data = fire_data[fire_data['burn_time'].notna() & (fire_data['burn_time'] >= 0)]
print(fire_data.columns)

#STEP 3 Find modern cut-off date 
#Method: Plot average burn time per year. Look for sudden shifts that may be indicative of modernization

#Group data by year they started and compute their average burn time
fire_data['year'] = fire_data['ignition_date'].dt.year
yearly_mean_burn = fire_data.groupby('year')['burn_time'].mean().reset_index()

print("Year range:", fire_data['year'].min(), "to", fire_data['year'].max())
print("Number of rows:", len(fire_data))
#print(fire_data[['ignition_date','extinguish_date','year']].head(10))


#Plot it and look for visual markers
#Exclude the Black Friday bushfire outlier and make scatter plot easier to see
filtered_burn_data = yearly_mean_burn[yearly_mean_burn['burn_time'] <= 10000]

#BLOCK
#COMMENTED OUT FOR RUNTIME EFFICIENCY
print("Generating plot for Yearly Average Wildfire Burn Time 1899-2025, excluding year with Black Friday outlier, for visualization purposes")
plt.figure(figsize=(12,6))
plt.scatter(filtered_burn_data['year'], filtered_burn_data['burn_time'])
plt.title("Yearly Average Wildfire Burn Time")
plt.xlabel("Year")
plt.ylabel("Burn Time (hours)")
plt.show()


#Run algorithm to identify shifts in burn times using least squared error

#BLOCK
#COMMENTED OUT FOR RUNTIME EFFICIENCY
print("\nRunning algorithm for identifying change points in burn time data")
signal = filtered_burn_data['burn_time'].values
model = "l2"
algor = rpt.Pelt(model=model).fit(signal)
#indices in the signal array
change_points = algor.predict(pen = 900000) 
#The years corresponsing to the change points, excluding last in list because that is not a data point
potential_cutoffs = yearly_mean_burn['year'].iloc[change_points[:-1]].values
print("Potential cut-off years: ", potential_cutoffs)
#modern cut-off is 2004, so exclude all data 2004 and prior. Test for statistical significance is in modern_analysis


#Truncating data to exclude year of cut-off and prior (For scatter plot visualization purposes)
filtered_burn_data = yearly_mean_burn[yearly_mean_burn['year'] > 2004]

#BLOCK
#COMMENTED OUT FOR RUNTIME EFFICIENCY
print("\nGenerating plot for Yearly Average Wildfire Burn Time, 2005-2025")
plt.figure(figsize=(12,6))
plt.scatter(filtered_burn_data['year'], filtered_burn_data['burn_time'])
plt.title("Yearly Average Wildfire Burn Time")
plt.xlabel("Year")
plt.ylabel("Burn Time (hours)")
plt.show()


#STEP 4 calculate modern burn-time stats, Austalia, 2005-2025
modern_fire_data = fire_data[fire_data['year'] > 2004]

yearly_mean_area = modern_fire_data.groupby('year')['area_ha'].mean().reset_index()

#plotting yearly average fire area to look for visual outlier and see trends in the years 2005-modern
#BLOCK
print("\nGenerating plot for Yearly Average Wildfire Area, 2005-2025")
plt.figure(figsize=(12,6))
plt.scatter(yearly_mean_area['year'], yearly_mean_area['area_ha'])
plt.title("Yearly Average Wildfire Area")
plt.xlabel("Year")
plt.ylabel("Area (hectares)")
plt.show()
#No visual outliers found

# ---- BURN TIME STATISTICS ---
print("\nModern dataset (2005-2025), Australia")

mean_burn_time = modern_fire_data['burn_time'].mean()
max_burn_time = modern_fire_data['burn_time'].max()
min_burn_time = modern_fire_data['burn_time'].min()

print("\n--- Burn Time Statistics (hours) ---")
print(f"Mean burn time: {mean_burn_time:,.1f} hr")
print(f"Max burn time: {max_burn_time:,.0f} hr")#
print(f"Min burn time: {min_burn_time:,.0f} hr")

print("\nModern dataset (2005-2025), Victoria")

#Victoria fires only:
modern_vic = modern_fire_data[modern_fire_data['state'] == 'VIC (Victoria)']

# ---- VICTORIA BURN TIME STATISTICS ---
mean_burn_time_vic = modern_vic['burn_time'].mean()
max_burn_time_vic = modern_vic['burn_time'].max()
min_burn_time_vic = modern_vic['burn_time'].min()

print("\n--- Burn Time Statistics (hours) ---")
print(f"Mean burn time: {mean_burn_time_vic:,.1f} hr")
print(f"Max burn time: {max_burn_time_vic:,.0f} hr")
print(f"Min burn time: {min_burn_time_vic:,.0f} hr")

#Debugging:
print(modern_fire_data['state'].unique())
print("Victoria rows after 2005:", len(modern_vic))
print("Non-null burn_time values in Victoria:", modern_vic['burn_time'].notna().sum())
print(modern_vic[['ignition_date', 'extinguish_date', 'burn_time']].head(10))
print(fire_data[fire_data['state'] == 'VIC (Victoria)']['year'].describe())
#print(fire_data[['state','ignition_date','year']].tail(20))
#Conclusion: Vicoria does not have enough data points from 2005-2025 to calculate statistics
