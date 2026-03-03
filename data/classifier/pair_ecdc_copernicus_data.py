#!/usr/bin/env python3
"""
Script to pair ECDC and Copernicus data for Aedes albopictus analysis.
"""

import argparse
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import fiona
import importlib
import warnings
warnings.filterwarnings('ignore')

# Custom modules
# Custom modules - change these lines
from src import copernicus
from src import aedes_suitability
from src import unified_climate_downloader

def setup_directories():
    """Create necessary directories"""
    Path("data/img").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

def load_ecdc_data(name_file='20230828_VectorFlatFileGDB.gdb', parent_dir="./data/input/"):
    """Load and process ECDC vector data"""
    zip_file_path = parent_dir + name_file
    zip_gdb_path = f"zip://{zip_file_path}"
    
    # List all layers in the Geodatabase
    layers = fiona.listlayers(zip_gdb_path)
    print(f"Available layers: {layers}")
    
    # Read the GDB file
    gdf = gpd.read_file(zip_gdb_path)
    
    # Create status mapping
    code_to_status = {
        'INV001A': 'Established',
        'INV002A': 'Introduced',
        'INV003A': 'Absent',
        'INV004A': 'No data',
        'INV999A': 'Unknown',
        'NAT001A': 'Present',
        'NAT002A': 'Absent',
        'NAT003A': 'Absent',
        'NAT004A': 'No data',
        'NAT005A': 'Introduced',
        'NAT999A': 'Unknown'
    }
    
    gdf['status'] = gdf['AssessedDistributionStatus'].map(code_to_status)
    
    return gdf

def analyze_mosquito_data(gdf, year):
    """Analyze mosquito data and create visualizations"""
    # Filter for mosquitoes
    mosquitoes = gdf[gdf['VectorCategoryCode'] == 'Mosq']
    
    # Filter for Aedes mosquitoes
    aedes_mosquitoes = mosquitoes[mosquitoes['VectorSpeciesName'].str.contains('aedes', case=False, na=False)]
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.countplot(y='VectorSpeciesName', hue='status', data=aedes_mosquitoes)
    plt.xlabel('Count')
    plt.ylabel('Vector Species Name')
    plt.title('Distribution of Mosquito Species by Assessed Distribution Status')
    plt.legend(title='Assessed Distribution Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"data/img/aedes_species_distribution_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Filter for Aedes albopictus
    gdf_albo = gdf[gdf['VectorSpeciesName'].str.contains('albo', case=False, na=False)]
    
    # Create presence numeric mapping
    gdf_albo['presence_numeric'] = gdf_albo['status'].map({
        'Absent': 0, 'Present': 1, "Established": 1, 
        'No data': 3, "Introduced": 2, "Unknown": 3
    })
    
    # Status frequency plot
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='status', data=gdf_albo, palette='viridis')
    plt.title('Frequency of Status in Aedes albopictus')
    plt.xlabel('Status')
    plt.ylabel('Frequency')
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.savefig(f"data/img/albopictus_status_frequency_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Geographic distribution map
    plt.figure(figsize=(12, 8))
    gdf_albo.plot(column='status', legend=True)
    plt.title('Geographic Distribution of Aedes albopictus Status')
    plt.savefig(f"data/img/albopictus_geographic_distribution_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return gdf_albo

def load_climate_data(year, climate_source='cordex'):
    """
    Load and process climate data from either CORDEX or ERA5-Land.
    
    Parameters:
        year: Year for analysis
        climate_source: 'cordex' (default) or 'era5_land'
    """
    print(f"📡 Loading {climate_source.upper()} climate data for {year}...")
    
    # Use unified downloader
    p_month, t_month = unified_climate_downloader.load_climate_data_unified(
        year=int(year),
        climate_source=climate_source,
        out_dir="./data/inputs/"
    )
    
    print("Dataset shapes:")
    print("Precipitation:", p_month.shape)
    print("Temperature:", t_month.shape)
    
    # Build climate dataframe
    climate_df = unified_climate_downloader.create_climate_dataframe(
        p_month=p_month,
        t_month=t_month,
        year=int(year),
        add_monthly_columns=True
    )
    
    print(f"\n📊 DataFrame: {climate_df.shape} columns: {list(climate_df.columns)}")
    
    # Create climate maps with climate_source
    create_climate_maps(p_month, t_month, year, climate_source)
    
    return climate_df

def create_climate_maps(p_month, t_month, year, climate_source='cordex'):
    """Create climate visualization maps"""
    # January maps
    january_date = 1
    p_january = p_month.sel(month=january_date, method='nearest')
    t_january = t_month.sel(month=january_date, method='nearest')
    
    # Precipitation map
    plt.figure(figsize=(12, 6))
    p_january.plot(cmap='Blues', robust=True)
    plt.title(f'Total Precipitation in January ({year}) - {climate_source.upper()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f"data/img/precipitation_january_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Temperature map
    plt.figure(figsize=(12, 6))
    t_january.plot(cmap='coolwarm', robust=True)
    plt.title(f'2m Temperature in January ({year}) - {climate_source.upper()}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f"data/img/temperature_january_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()

def merge_datasets(gdf_albo, climate_df):
    """Merge ECDC and Copernicus datasets"""
    # Convert to GeoDataFrames
    gdf_albo = gpd.GeoDataFrame(gdf_albo, geometry='geometry')
    gdf_copernicus = gpd.GeoDataFrame(
        climate_df,
        geometry=gpd.points_from_xy(climate_df['longitude'], climate_df['latitude']),
        crs="EPSG:4326"
    )
    
    # Ensure same CRS
    if gdf_albo.crs != gdf_copernicus.crs:
        gdf_albo = gdf_albo.to_crs(gdf_copernicus.crs)
    
    # Spatial join
    gdf_merged = gpd.sjoin(
        gdf_copernicus,
        gdf_albo,
        how='left',
        predicate='within'
    )
    
    gdf_merged = gdf_merged.drop(columns=['index_right'], errors='ignore')
    
    return gdf_merged

def calculate_suitability(gdf_merged, year, climate_source='cordex'):
    """Calculate climate suitability for Aedes albopictus"""
    # Filter training data
    gdf_training = gdf_merged[gdf_merged["presence_numeric"] < 4].copy()
    
    df = gdf_training.copy()
    
    from src.aedes_suitability import aedes_precipitation_suitability, aedes_temperature_suitability
    
    # Calculate suitability
    tavg = np.stack(df['temperature_2m_monthly'].values)
    p_annual = np.stack(df['precipitation_monthly'].values).sum(axis=1)
    
    df['Precipitation Suitable'] = aedes_precipitation_suitability(p_annual)
    df['Temperature Suitable'] = aedes_temperature_suitability(tavg)
    df['Suitable'] = np.logical_and(df['Precipitation Suitable'], df['Temperature Suitable'])
    
    # Create suitability plots with climate_source
    create_suitability_plots(df, year, climate_source)
    
    return df

def create_suitability_plots(df, year, climate_source='cordex'):
    """Create suitability analysis plots"""
    # Count plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    sns.countplot(x='Suitable', data=df, ax=axes[0], palette='Set2')
    axes[0].set_title(f'Overall Suitability - {climate_source.upper()}')
    
    sns.countplot(x='Temperature Suitable', data=df, ax=axes[1], palette='RdYlBu_r')
    axes[1].set_title(f'Temperature Suitability - {climate_source.upper()}')
    
    sns.countplot(x='Precipitation Suitable', data=df, ax=axes[2], palette='Blues')
    axes[2].set_title(f'Precipitation Suitability - {climate_source.upper()}')
    
    # New: Presence but Not Suitable count
    df_present = df[df['presence_numeric'] == 1]
    df_present_not_suitable = df_present[df_present['Suitable'] == False]
    counts = pd.DataFrame({
        'Category': ['Present & Suitable', 'Present & Not Suitable'],
        'Count': [len(df_present) - len(df_present_not_suitable), len(df_present_not_suitable)]
    })
    sns.barplot(x='Category', y='Count', data=counts, ax=axes[3], palette='OrRd')
    axes[3].set_title(f'Present: Suitable vs Not - {climate_source.upper()}')
    axes[3].tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    plt.savefig(f"data/img/suitability_counts_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Geographic suitability maps
    df_web = df.to_crs(epsg=3857)
    
    # Temperature suitability map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df_web.plot(
        column='Temperature Suitable',
        categorical=True,
        legend=True,
        markersize=1,
        ax=ax,
        cmap='RdYlBu_r',
        alpha=0.8,
        zorder=2
    )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df_web.crs, zoom=5)
    ax.set_title(f"Temperature Suitability Map - {climate_source.upper()} ({year})")
    ax.set_axis_off()
    plt.savefig(f"data/img/temperature_suitability_map_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Precipitation suitability map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df_web.plot(
        column='Precipitation Suitable',
        categorical=True,
        legend=True,
        markersize=1,
        ax=ax,
        cmap='Blues',
        alpha=0.8,
        zorder=2
    )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df_web.crs, zoom=5)
    ax.set_title(f"Precipitation Suitability Map - {climate_source.upper()} ({year})")
    ax.set_axis_off()
    plt.savefig(f"data/img/precipitation_suitability_map_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Overall suitability map
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    df_web.plot(
        column='Suitable',
        categorical=True,
        legend=True,
        markersize=1,
        ax=ax,
        cmap='RdYlGn',
        alpha=0.8,
        zorder=2
    )
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df_web.crs, zoom=5)
    ax.set_title(f"Overall Suitability Map - {climate_source.upper()} ({year})")
    ax.set_axis_off()
    plt.savefig(f"data/img/overall_suitability_map_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()

def filter_european_data(df, year, climate_source='cordex'):
    """Filter data for continental European countries"""
    continental_european_country_codes = [
        'AL', 'AD', 'AT', 'BY', 'BE', 'BA', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 
        'EE', 'FI', 'FR', 'GR', 'EL', 'HR', 'HU', 'IS', 'IE', 'IT', 'XK', 'LI', 
        'LT', 'LU', 'LV', 'MC', 'MD', 'ME', 'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 
        'RO', 'SM', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'UA', 'GB', 'UK', 'VA'
    ]
    
    # Filter for European NUTS3 codes
    df_european_nuts3 = df[df["LocationCode"].str[:2].isin(continental_european_country_codes)]
    df_european_nuts3 = df_european_nuts3[df_european_nuts3["latitude"] > 34]
    df_european_nuts3 = df_european_nuts3[df_european_nuts3["latitude"] < 75]
    
    # Create high-resolution aggregation plot
    df_aggregated = df_european_nuts3.groupby([
        df_european_nuts3['latitude'].round(3),
        df_european_nuts3['longitude'].round(3)
    ]).Suitable.max().reset_index()
    
    plt.figure(figsize=(10, 8), dpi=150)
    plt.scatter(
        df_aggregated['longitude'], 
        df_aggregated['latitude'], 
        c=df_aggregated['Suitable'], 
        cmap='RdYlGn',
        s=0.01,
        alpha=0.9
    )
    plt.colorbar(label='Suitable (1) / Not suitable (0)')
    plt.title(f'High-Resolution Aggregation - {climate_source.upper()} ({year})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(f"data/img/european_suitability_aggregation_{climate_source}_{year}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return df_european_nuts3

def save_results_to_database(df, year, climate_source='cordex', output_dir='./data/outputs/'):
    """
    Save results to a clean CSV database with standardized naming.
    Keeps original column names but adds metadata.
    
    Parameters:
        df: DataFrame with results
        year: Year of analysis
        climate_source: 'cordex' or 'era5_land'
        output_dir: Output directory path
    """
    import os
    from datetime import datetime
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a clean copy of the dataframe
    df_clean = df.copy()
    
    # Remove geometry column if present (can't be saved in CSV)
    if 'geometry' in df_clean.columns:
        # Extract lat/lon if not already present
        if 'latitude' not in df_clean.columns and hasattr(df_clean['geometry'].iloc[0], 'y'):
            df_clean['latitude'] = df_clean['geometry'].y
        if 'longitude' not in df_clean.columns and hasattr(df_clean['geometry'].iloc[0], 'x'):
            df_clean['longitude'] = df_clean['geometry'].x
        df_clean = df_clean.drop(columns=['geometry'])
    
    # Check for any remaining object columns and handle them
    object_cols = df_clean.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Try to convert to string, handle lists/arrays
        try:
            first_val = df_clean[col].iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                # Convert arrays to string representation
                df_clean[col] = df_clean[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, (list, np.ndarray)) else str(x))
            else:
                df_clean[col] = df_clean[col].astype(str)
        except Exception as e:
            print(f"⚠️  Warning: Could not convert column {col}: {e}")
            df_clean = df_clean.drop(columns=[col])
    
    # Add metadata columns
    df_clean['analysis_year'] = year
    df_clean['climate_data_source'] = climate_source
    df_clean['processing_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Reorder columns: metadata first, then rest
    metadata_cols = ['analysis_year', 'climate_data_source', 'processing_date']
    other_cols = [col for col in df_clean.columns if col not in metadata_cols]
    df_clean = df_clean[metadata_cols + other_cols]
    
    # Generate filename with climate source
    filename = f"ecdc_albopictus_{climate_source}_{year}.csv"
    output_path = os.path.join(output_dir, filename)
    
    # Save to CSV
    df_clean.to_csv(output_path, index=False)
    
    # Always save as zip
    compressed_path = output_path.replace('.csv', '.zip')
    df_clean.to_csv(compressed_path, index=False, compression='zip')
    
    # Remove uncompressed CSV
    os.remove(output_path)
    
    print(f"✅ Database saved: {compressed_path}")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Columns: {len(df_clean.columns)}")
    print(f"   File size: {os.path.getsize(compressed_path) / 1024:.1f} KB")
    
    # Print data summary
    print("\n📊 Data Summary:")
    print(f"   Total records: {len(df_clean)}")
    if 'presence_numeric' in df_clean.columns:
        print(f"   Presence distribution:")
        print(df_clean['presence_numeric'].value_counts().sort_index().to_string())
    if 'Suitable' in df_clean.columns:
        suitable_count = df_clean['Suitable'].sum()
        suitable_pct = suitable_count / len(df_clean) * 100
        print(f"   Climate suitable: {suitable_count} ({suitable_pct:.1f}%)")
    
    return compressed_path

def analyze_presence_not_suitable(df, year, climate_source='cordex'):
    """
    Analyze and map grid points labelled as present/established by ECDC
    but flagged as climatically unsuitable by the QC filters.
    
    These points highlight potential labelling artefacts caused by 
    spatial heterogeneity within NUTS-3 surveillance polygons (e.g., 
    mountain tops within a polygon that is labelled as 'present').
    
    Parameters:
        df: GeoDataFrame with suitability flags and presence_numeric
        year: Analysis year
        climate_source: 'cordex' or 'era5_land'
        
    Returns:
        df_conflict: GeoDataFrame of conflicting points
    """
    # Select present/established points that are not suitable
    mask_present = df['presence_numeric'] == 1
    mask_not_suitable = df['Suitable'] == False
    df_conflict = df[mask_present & mask_not_suitable].copy()
    
    n_present = mask_present.sum()
    n_conflict = len(df_conflict)
    pct = (n_conflict / n_present * 100) if n_present > 0 else 0.0
    
    print(f"\n🔍 Presence vs Suitability Conflict Analysis ({climate_source.upper()}):")
    print(f"   Total present/established points: {n_present}")
    print(f"   Present but NOT suitable:         {n_conflict} ({pct:.1f}%)")
    
    if n_conflict > 0:
        # Break down by which filter fails
        mask_temp_fail = df[mask_present]['Temperature Suitable'] == False
        mask_precip_fail = df[mask_present]['Precipitation Suitable'] == False
        mask_both_fail = mask_temp_fail & mask_precip_fail
        
        print(f"   - Temperature filter fails:       {mask_temp_fail.sum()}")
        print(f"   - Precipitation filter fails:      {mask_precip_fail.sum()}")
        print(f"   - Both filters fail:               {mask_both_fail.sum()}")
        
        # Create conflict category column for mapping
        df_present = df[mask_present].copy()
        df_present['suitability_status'] = 'Present & Suitable'
        df_present.loc[mask_not_suitable[mask_present].values, 'suitability_status'] = 'Present & Not Suitable'
        
        # --- Map 1: Presence points coloured by suitability status ---
        df_present_web = df_present.to_crs(epsg=3857)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        
        # Plot suitable present points first (green, small)
        mask_ok = df_present_web['suitability_status'] == 'Present & Suitable'
        df_present_web[mask_ok].plot(
            ax=ax, color='#2ca02c', markersize=1, alpha=0.5, 
            label='Present & Suitable', zorder=2
        )
        # Plot conflicting points on top (red, larger)
        df_present_web[~mask_ok].plot(
            ax=ax, color='#d62728', markersize=3, alpha=0.8,
            label='Present & Not Suitable', zorder=3
        )
        
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df_present_web.crs, zoom=5)
        ax.legend(loc='lower left', fontsize=10, markerscale=5)
        ax.set_title(
            f"ECDC Present Points: Suitable vs Not Suitable\n"
            f"{climate_source.upper()} ({year}) — {n_conflict} conflicts out of {n_present} ({pct:.1f}%)",
            fontsize=12
        )
        ax.set_axis_off()
        plt.savefig(
            f"data/img/presence_not_suitable_map_{climate_source}_{year}.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        # --- Map 2: Conflict points coloured by failing filter ---
        df_conflict_detail = df_conflict.copy()
        conditions = [
            (~df_conflict_detail['Temperature Suitable']) & (~df_conflict_detail['Precipitation Suitable']),
            ~df_conflict_detail['Temperature Suitable'],
            ~df_conflict_detail['Precipitation Suitable'],
        ]
        choices = ['Both fail', 'Temperature fails', 'Precipitation fails']
        df_conflict_detail['failure_reason'] = np.select(conditions, choices, default='Unknown')
        
        df_conflict_web = df_conflict_detail.to_crs(epsg=3857)
        
        color_map = {
            'Temperature fails': '#d62728',
            'Precipitation fails': '#1f77b4',
            'Both fail': '#7f007f',
        }
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        for reason, color in color_map.items():
            subset = df_conflict_web[df_conflict_web['failure_reason'] == reason]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, markersize=3, alpha=0.8, label=reason, zorder=3)
        
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=df_conflict_web.crs, zoom=5)
        ax.legend(loc='lower left', fontsize=10, markerscale=5)
        ax.set_title(
            f"Present but Not Suitable — Failure Reasons\n"
            f"{climate_source.upper()} ({year}) — {n_conflict} points",
            fontsize=12
        )
        ax.set_axis_off()
        plt.savefig(
            f"data/img/presence_not_suitable_reasons_{climate_source}_{year}.png",
            dpi=150, bbox_inches='tight'
        )
        plt.close()
        
        print(f"   📊 Maps saved to data/img/presence_not_suitable_*_{climate_source}_{year}.png")
    else:
        print("   ✅ No conflicts found — all present points are climatically suitable.")
    
    return df_conflict

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Pair ECDC and Copernicus data for Aedes albopictus analysis')
    parser.add_argument('--year', type=str, default='2020', help='Year for analysis (default: 2020)')
    parser.add_argument('--parent-dir', type=str, default='./data/inputs/', 
                       help='Parent directory containing ECDC data (default: ./data/inputs)')
    parser.add_argument('--ecdc-file', type=str, default='20230828_VectorFlatFileGDB.gdb.zip',
                       help='ECDC file name (default: 20230828_VectorFlatFileGDB.gdb.zip)')
    parser.add_argument('--climate-source', type=str, default='cordex', 
                       choices=['cordex', 'era5_land'],
                       help='Climate data source: cordex (default) or era5_land')
    parser.add_argument('--output-dir', type=str, default='./data/outputs/',
                       help='Output directory for results (default: ./data/outputs/)')
    args = parser.parse_args()
    
    year = args.year
    parent_dir = args.parent_dir
    ecdc_file = args.ecdc_file
    climate_source = args.climate_source
    output_dir = args.output_dir
    
    print(f"Processing data for year: {year}")
    print(f"Using climate source: {climate_source}")
    print(f"Using parent directory: {parent_dir}")
    print(f"Using ECDC file: {ecdc_file}")
    
    # Setup
    setup_directories()
    
    # Load and process ECDC data
    print("\n" + "="*50)
    print("Loading ECDC data...")
    gdf = load_ecdc_data(name_file=ecdc_file, parent_dir=parent_dir)
    gdf_albo = analyze_mosquito_data(gdf, year)
    
    # Load climate data with specified source
    print("\n" + "="*50)
    print("Loading climate data...")
    climate_df = load_climate_data(year, climate_source=climate_source)
    
    # Merge datasets
    print("\n" + "="*50)
    print("Merging datasets...")
    gdf_merged = merge_datasets(gdf_albo, climate_df)
    
    # Calculate suitability
    print("\n" + "="*50)
    print("Calculating suitability...")
    df_with_suitability = calculate_suitability(gdf_merged, year, climate_source=climate_source)
    
    # Analyze presence-but-not-suitable conflicts
    print("\n" + "="*50)
    print("Analyzing presence vs suitability conflicts...")
    df_conflicts = analyze_presence_not_suitable(df_with_suitability, year, climate_source=climate_source)
    
    # Filter for European data
    print("\n" + "="*50)
    print("Processing European data...")
    df_european = filter_european_data(df_with_suitability, year, climate_source=climate_source)
    
    # Save to clean database format
    print("\n" + "="*50)
    print("Saving results to database...")
    output_path = save_results_to_database(
        df_european, 
        year=year, 
        climate_source=climate_source,
        output_dir=output_dir
    )
    
    print("\n" + "="*50)
    print("✅ Analysis complete!")
    print(f"📁 Results saved to: {output_path}")
    print(f"🖼️  Visualizations in: ./data/img/")

if __name__ == "__main__":
    main()