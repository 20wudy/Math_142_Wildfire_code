import xarray as xr
import pandas as pd

ds = pd.read_excel("NRM_fire_proj_summary.xlsx")
print(ds.columns)

FFDI_Col = "Avg Ann CFFDI"

ds_1995 = ds[ds["Year"] == 1995].copy()

baseline_FFDI = ds_1995[FFDI_Col].mean()

print("baseline FFDI (1995): ", baseline_FFDI)

ds_2030 = ds[ds["Year"] == 2030].copy()

future_FFDI_2030 = ds_2030[FFDI_Col].mean()

print("Average FFDI for 2030: ", future_FFDI_2030)

extreme_fire_events_1995 = 1325
#Calculated in frequency_predictions

predicted_2030_fire_events = extreme_fire_events_1995 * future_FFDI_2030/baseline_FFDI

print("Estimated frequency of extreme fire events in 2030: ", predicted_2030_fire_events)

ds_2050 = ds[ds["Year"] == 2050].copy()

future_FFDI_2050 = ds_2050[FFDI_Col].mean()

print("Average FFDI for 2050: ", future_FFDI_2050)

predicted_2050_fire_events = extreme_fire_events_1995 * future_FFDI_2050/baseline_FFDI

print("Estimated frequency of extreme fire events in 2050: ", predicted_2050_fire_events)

predicted_2035_fire_events = ((predicted_2050_fire_events - predicted_2030_fire_events)/20) + predicted_2030_fire_events

print("Estimated frequency of extreme fire events in 2035: ", predicted_2035_fire_events)


