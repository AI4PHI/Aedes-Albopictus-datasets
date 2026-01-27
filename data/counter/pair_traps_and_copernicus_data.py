#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


df = pd.read_pickle("albopictus.pkl")
df = df[df["end_date"] < pd.Timestamp("2021-01-01")]
df = df[df["end_date"] >= pd.Timestamp("2020-01-01")]


# In[3]:


# Convert date fields to datetime format
df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"] = pd.to_datetime(df["end_date"])

# Display min/max dates to understand data timeframe
print("Minimum start_date:", df["start_date"].min())
print("Maximum start_date:", df["start_date"].max())
print("Minimum end_date:", df["end_date"].min())
print("Maximum end_date:", df["end_date"].max())

# Show date range span in days
print(f"Date range spans {(df['end_date'].max() - df['start_date'].min()).days} days")


# In[4]:


import seaborn as sns

import matplotlib.pyplot as plt

# Convert date columns to datetime if not already
if not pd.api.types.is_datetime64_any_dtype(df["start_date"]):
    df["start_date"] = pd.to_datetime(df["start_date"])
if not pd.api.types.is_datetime64_any_dtype(df["end_date"]):
    df["end_date"] = pd.to_datetime(df["end_date"])

# Get basic date statistics
print(f"Date range in dataset:")
print(f"Start dates: {df['start_date'].min()} to {df['start_date'].max()}")
print(f"End dates: {df['end_date'].min()} to {df['end_date'].max()}")

# Check entries with end date before 2020
before_2020 = df[df["end_date"] < "2020-01-01"]
print(f"\nEntries with end date before 2020: {len(before_2020)} ({len(before_2020)/len(df)*100:.2f}%)")

# Plot the distribution of end dates by month
plt.figure(figsize=(12, 6))
df['end_date_month'] = df['end_date'].dt.to_period('M')
date_counts = df['end_date_month'].value_counts().sort_index()
date_counts.plot(kind='bar')
plt.title('Number of Records by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

# Check data by country
country_counts = df['country'].value_counts()
print("\nRecord count by country:")
print(country_counts)

# Check presence/absence ratio
presence_counts = df['occurrenceStatus'].value_counts()
print("\nOccurrence status distribution:")
print(presence_counts)


# In[5]:


import importlib as imp
import AIedes_data.counter.process_copernicus_data_claude as process_copernicus_data_claude
imp.reload(process_copernicus_data_claude)
from AIedes_data.counter.process_copernicus_data_claude import extract_climate_data_to_df, aggregate_to_monthly


# In[6]:


df.iloc[0]["end_date"]


# In[7]:


import os
import pandas as pd
import xarray as xr
# Import from existing classifier CDS infrastructure
import sys
sys.path.append('../classifier/src')
from era5_land_downloader import ERA5LandDownloader

# Input variables (base names to search for in NetCDF files)
input_vars = [
    #"total_evaporation",
    "total_precipitation", 
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature", "2m_temperature",
    #"surface_net_thermal_radiation", "surface_net_solar_radiation", "surface_pressure", "skin_temperature",
    #"surface_sensible_heat_flux", "surface_latent_heat_flux", "surface_thermal_radiation_downwards",
    "volumetric_soil_water_layer_1", #"potential_evaporation", "evaporation_from_vegetation_transpiration",
]

year = 2020  # Base year
freq = "daily"

# Check if the original path exists, otherwise create a local directory
original_path_dir = f"/eos/jeodpp/data/projects/ETOHA/DATA/ClimateData/Copernicus_data/europe/data/"
local_path_dir = "./copernicus_climate_data/europe/data/"

if os.path.exists(os.path.dirname(original_path_dir.rstrip('/'))):
    path_dir = original_path_dir
    print(f"✅ Using original data directory: {path_dir}")
else:
    path_dir = local_path_dir
    # Create the local directory structure
    os.makedirs(path_dir, exist_ok=True)
    print(f"📁 Original path not accessible, created local directory: {path_dir}")
    print(f"   Climate data will be downloaded to: {os.path.abspath(path_dir)}")

# Define columns for latitude, longitude, and time
lat_col = "decimalLatitude"
lon_col = "decimalLongitude"
time_col = "end_date"
time_window = "89D"  # Time window for extracting data
months_to_average = 3  # Number of months to average data over
days_per_month = 30  # Number of days in each month
time_window_avg = f"{days_per_month * months_to_average - 1}D"  # Time window for averaging data
# Ensure 'end_date' is in datetime format
df[time_col] = pd.to_datetime(df[time_col])

# Determine the years needed based on the dataframe's end_date column
first_date_needed = df[time_col].min() - pd.Timedelta(time_window_avg)
last_date_needed = df[time_col].max()

first_year_needed = first_date_needed.year
last_year_needed = last_date_needed.year

print(f"Data extraction required from {first_year_needed} to {last_year_needed}.")

# Initialize ERA5-Land downloader (using existing CDS API setup)
base_dir = path_dir.replace("/europe/data/", "")
downloader = ERA5LandDownloader(base_output_dir=base_dir)
print(f"🔧 Initialized downloader with base directory: {base_dir}")

# Iterate over each base climate variable name
for var in input_vars:
    print(f"\nProcessing base variable: {var}")

    # List to store datasets for merging
    datasets = []

    # Load each required year's data
    for yr in range(first_year_needed, last_year_needed + 1):
        if var in [
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature", "2m_temperature",
        "surface_net_thermal_radiation", "surface_net_solar_radiation", "surface_pressure", "skin_temperature",
        "surface_sensible_heat_flux", "surface_latent_heat_flux", "surface_thermal_radiation_downwards",
        "volumetric_soil_water_layer_1"]:
            input_file = f"{path_dir}{yr}/{var}_{freq}_stats_{yr}.nc"
        else:
            input_file = f"{path_dir}{yr}/{var}_{freq}_cum_{yr}.nc"


        if os.path.exists(input_file):
            print(f"Loading dataset: {input_file}")
            try:
                ds = xr.open_dataset(input_file, engine='netcdf4')  # Explicit engine specification
                datasets.append(ds)
            except Exception as e:
                print(f"❌ Failed to load existing file {input_file}: {e}")
                print("Attempting to redownload...")
        else:
            print(f"Warning: File {input_file} does not exist. Attempting to download...")
            try:
                # Download and process the missing data
                processed_file = downloader.ensure_data_available(var, yr, freq)
                if processed_file.exists():
                    print(f"✅ Successfully downloaded and processed: {processed_file}")
                    try:
                        ds = xr.open_dataset(processed_file, engine='netcdf4')
                        datasets.append(ds)
                    except Exception as e:
                        print(f"❌ Failed to load processed file {processed_file}: {e}")
                else:
                    print(f"❌ Failed to download data for {var} {yr}")
            except Exception as e:
                print(f"❌ Error downloading {var} for {yr}: {e}")
                print("Continuing with available data...")

    # Merge all loaded datasets for this variable
    if datasets:
        ds_merged = xr.concat(datasets, dim="time")
        print(f"Merged {len(datasets)} datasets for {var}.")

        # Get all available variable names inside the NetCDF file
        available_vars = list(ds_merged.data_vars.keys())
        print(f"Variables available in dataset: {available_vars}")

        # Iterate over all available variables inside this dataset
        for climate_var in available_vars:
            print(f"Extracted data for: {climate_var} -- monthly average")            
            df = extract_climate_data_to_df(
                df, ds_merged, climate_var, lat_col=lat_col, lon_col=lon_col, time_col=time_col, time_window=time_window_avg
            )
            df[climate_var+"_monthly"] = df[climate_var].apply(lambda x: aggregate_to_monthly(x, num_months=months_to_average))


            print(f"Extracted data for: {climate_var} daily")
            # Call the function to extract climate data for the DataFrame
            df = extract_climate_data_to_df(
                df, ds_merged, climate_var, lat_col=lat_col, lon_col=lon_col, time_col=time_col, time_window=time_window
            )
    else:
        print(f"No datasets available for {var}. Skipping extraction.")



# In[8]:


df.to_csv("albopictus_with_climate_3m.csv")
df.to_pickle("albopictus_with_climate_3m.pkl")


# In[9]:


df.columns


# In[ ]:




