import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_climate_data_to_df(df, ds, variable, 
                               lat_col="decimalLatitude", 
                               lon_col="decimalLongitude", 
                               time_col="end_date", 
                               time_window="30D"):
    """
    Extract climate data for multiple locations and store results in a new column as arrays.
    Also store/update a flag (in column 'climate_nan') that indicates if the extracted array 
    contains any NaN values ("yes" if any NaN are present, "no" otherwise).
    
    Displays a progress bar and prints NaN and valid row counts dynamically.

    Parameters:
        df: Pandas DataFrame containing location and time information.
        nc_file: Path to the NetCDF file.
        variable: Name of the climate variable to extract.
        lat_col: Column name for latitude in the DataFrame.
        lon_col: Column name for longitude in the DataFrame.
        time_col: Column name for timestamps in the DataFrame.
        time_window: Time window as a string (e.g., "30D").

    Returns:
        A copy of the input DataFrame with:
            - A new column f"{variable}_values" that stores extracted climate values as arrays.
            - A column "climate_nan" which is updated (or created) to "yes" for rows where the 
              extraction contains any NaN values, and "no" otherwise.
    """
    # Create a copy so we don't modify the original DataFrame
    df_copy = df.copy()

    # Convert time column to datetime (if not already)
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])

    climate_values = []
    # This will store a flag ("yes"/"no") for each row
    computed_nan_flags = []  

    total_nan_rows = 0  # count of rows that contain any NaN in extracted array
    total_valid_rows = 0  # count of rows that are fully valid

    # Use tqdm to display progress (iterating over DataFrame rows)
    progress_bar = tqdm(df_copy.iterrows(), total=len(df_copy), 
                        desc="Extracting Climate Data", leave=True)
    
    for i, row in progress_bar:
        lat = row[lat_col]
        lon = row[lon_col]
        time = row[time_col]

        # Define the time window: here we use time_window as the period before the given date
        end_date = time
        start_date = end_date - pd.Timedelta(time_window)

        # Extract the climate data for this location and time window.
        # (Assumes extract_climate_data() is defined elsewhere and accepts an open dataset)
        climate_data = extract_climate_data(
            ds, variable, lat, lon, start_date, end_date
        )

        # Convert the returned dictionary values into a numpy array
        values_array = np.array(list(climate_data.values()))
        climate_values.append(values_array)

        # Determine if there is any NaN in the extracted values for this row
        if np.isnan(values_array).any():
            computed_nan_flags.append("yes")
            total_nan_rows += 1
            # Optionally print additional info (e.g., country and municipality if available)
            #if "country" in row and "municipality" in row:
            #    tqdm.write(f"Row: {row['country']} -- {row['municipality']} has NaN values.")
        else:
            computed_nan_flags.append("no")
            total_valid_rows += 1

        # Update the progress bar with current totals
        progress_bar.set_postfix(nan_rows=total_nan_rows, valid_rows=total_valid_rows)

    # Add the extracted climate values as a new column
    df_copy[f"{variable}"] = climate_values

    # Check if a "climate_nan" column already exists:
    if "climate_nan" in df_copy.columns:
        # If present, update rows that need to be marked as "yes"
        # (i.e., if the new computed flag is "yes", update the column value to "yes")
        new_flags = np.array(computed_nan_flags)
        # For rows where new_flags is "yes", update the existing column; otherwise keep its current value.
        df_copy["climate_nan"] = np.where(new_flags == "yes", "yes", df_copy["climate_nan"])
    else:
        # If not present, simply add the new column with the computed flags.
        df_copy["climate_nan"] = computed_nan_flags

    tqdm.write("\nFinal Extraction Summary:")
    tqdm.write(f"Total rows with NaN values in extracted data: {total_nan_rows}")
    tqdm.write(f"Total fully valid rows extracted: {total_valid_rows}")

    return df_copy


def bilinear_interpolation_nan(x, y, x1, x2, y1, y2, Q11, Q12, Q21, Q22):
    """
    Perform bilinear interpolation, ignoring NaN values if at least one of the neighs is not NaN.

    Parameters:
        x, y: Target coordinates.
        x1, x2: Bounding longitudes.
        y1, y2: Bounding latitudes.
        Q11, Q12, Q21, Q22: Climate values at the four surrounding grid points.

    Returns:
        Interpolated value at (x, y) or NaN if not enough data.
    """
    # Store valid (non-NaN) values
    valid_points = []
    valid_weights = []

    # Compute weights and store only valid points
    if not np.isnan(Q11):
        w11 = (x2 - x) * (y2 - y)
        valid_points.append(Q11)
        valid_weights.append(w11)
    if not np.isnan(Q21):
        w21 = (x - x1) * (y2 - y)
        valid_points.append(Q21)
        valid_weights.append(w21)
    if not np.isnan(Q12):
        w12 = (x2 - x) * (y - y1)
        valid_points.append(Q12)
        valid_weights.append(w12)
    if not np.isnan(Q22):
        w22 = (x - x1) * (y - y1)
        valid_points.append(Q22)
        valid_weights.append(w22)

    # If all values are NaN, return NaN
    if len(valid_points) == 0:
        return np.nan

    # Compute weighted sum (normalize by sum of weights)
    return np.sum(np.array(valid_points) * np.array(valid_weights)) / np.sum(valid_weights)

def extract_climate_data(ds, variable, lat, lon, start_date, end_date):
    """
    Extract climate data for a given location and time window using bilinear interpolation,
    ignoring NaN values in the surrounding points.
    
    Parameters:
        nc_file: NetCDF file.
        variable: Name of the climate variable to extract.
        lat, lon: Target latitude and longitude.
        start_date, end_date: Time window for extraction.
    
    Returns:
        Time series of interpolated values.
    """
    # Convert timestamps to strings for xarray compatibility
    start_date_str = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

    # Determine correct time dimension name
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

    # Extract time range using the correct dimension
    ds = ds.sel({time_dim: slice(start_date, end_date)})

    # Get latitude and longitude arrays
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Find the four nearest grid points
    lat1 = lats[lats <= lat].max()
    lat2 = lats[lats >= lat].min()
    lon1 = lons[lons <= lon].max()
    lon2 = lons[lons >= lon].min()

    # Extract values at four surrounding points for all time steps
    Q11 = ds[variable].sel(latitude=lat1, longitude=lon1).values
    Q12 = ds[variable].sel(latitude=lat2, longitude=lon1).values
    Q21 = ds[variable].sel(latitude=lat1, longitude=lon2).values
    Q22 = ds[variable].sel(latitude=lat2, longitude=lon2).values

    # Perform bilinear interpolation for each time step, handling NaNs
    interpolated_values = np.array([
        bilinear_interpolation_nan(lon, lat, lon1, lon2, lat1, lat2, q11, q12, q21, q22)
        for q11, q12, q21, q22 in zip(Q11, Q12, Q21, Q22)
    ])

    # Return results as a dictionary (time series) using the same time dimension
    times = ds[time_dim].values
    return dict(zip(times, interpolated_values))

def aggregate_to_monthly(avg_values, num_months=12):
    if not isinstance(avg_values, np.ndarray):
        return np.nan  # Handle missing or incorrect data format

    num_days = len(avg_values)
    days_per_month = num_days // num_months  # Typically ~30 days

    if num_days % num_months != 0:
        print(f"Warning: Data length {num_days} is not perfectly divisible by {num_months}, truncating extra days.")

    # Reshape to (12, ~30) and compute the mean along axis 1 (monthly mean)
    return avg_values[:days_per_month * num_months].reshape(num_months, -1).mean(axis=1)
