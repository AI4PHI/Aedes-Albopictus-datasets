#!/usr/bin/env python
"""
Aedes albopictus Surveillance Data Analysis

This module processes mosquito surveillance data focusing on Aedes albopictus
(Asian tiger mosquito). It includes data loading, cleaning, coordinate
standardization, trap identification, temporal analysis, and weekly occurrence
rate calculations.

Author: Biazzin
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlbopictusDataProcessor:
    """
    A class to process Aedes albopictus surveillance data.

    This class handles the complete pipeline from raw data loading to
    processed data export, including coordinate cleaning, temporal analysis,
    and rate calculations.
    """

    def __init__(self, data_dir: str = "../input_data/dwca-aimsurv-v2.3"):
        """
        Initialize the processor with data directory.

        Args:
            data_dir (str): Directory containing the raw data files
        """
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        self.data_dir = (script_dir / data_dir).resolve()
        self.event_data = None
        self.occurrence_data = None
        self.albopictus_data = None
        self.filtered_data = None
        self.summary = {}  # collected run statistics for README/reporting

    def load_data(self) -> None:
        """Load event and occurrence data from CSV files."""
        logger.info("Loading event and occurrence data...")

        event_file = self.data_dir / "event.txt"
        occurrence_file = self.data_dir / "occurrence.txt"

        if not event_file.exists() or not occurrence_file.exists():
            raise FileNotFoundError(f"Data files not found in {self.data_dir}")

        self.event_data = pd.read_csv(event_file, delimiter='\t', low_memory=False)
        self.occurrence_data = pd.read_csv(occurrence_file, delimiter='\t', low_memory=False)

        logger.info(f"Loaded {len(self.occurrence_data)} occurrence records")

        self.summary["raw"] = {
            "occurrence_records_total": int(len(self.occurrence_data)),
            "event_records_total": int(len(self.event_data)) if self.event_data is not None else None,
            "unique_scientific_names": int(self.occurrence_data["scientificName"].nunique(dropna=True)),
            "top_scientific_names": (
                self.occurrence_data["scientificName"]
                .value_counts(dropna=False)
                .head(15)
                .to_dict()
            ),
            "life_stage_counts_raw": (
                self.occurrence_data.get("lifeStage", pd.Series(dtype="object"))
                .value_counts(dropna=False)
                .to_dict()
            ),
            "zeros_total_raw": int((self.occurrence_data.get("individualCount") == 0).sum())
            if "individualCount" in self.occurrence_data.columns else None,
        }

        # Cross-tab: counts per species per lifeStage (top N species to keep file small)
        if "scientificName" in self.occurrence_data.columns and "lifeStage" in self.occurrence_data.columns:
            top_species = (
                self.occurrence_data["scientificName"]
                .value_counts(dropna=False)
                .head(15)
                .index
            )
            ct = pd.crosstab(
                self.occurrence_data.loc[self.occurrence_data["scientificName"].isin(top_species), "scientificName"],
                self.occurrence_data.loc[self.occurrence_data["scientificName"].isin(top_species), "lifeStage"],
                dropna=False
            )
            self.summary["raw"]["life_stage_by_species_top15"] = ct.to_dict()

    def extract_albopictus_data(self) -> None:
        """Extract Aedes albopictus data including zero counts."""
        logger.info("Extracting Aedes albopictus data...")

        if self.occurrence_data is None:
            raise ValueError("Occurrence data not loaded. Call load_data() first.")

        # Get albopictus records
        albopictus = self.occurrence_data[
            self.occurrence_data["scientificName"] == "Aedes albopictus (Skuse, 1894)"
        ].copy()

        # Include zero count records (negative observations are important)
        zeros = self.occurrence_data[self.occurrence_data["individualCount"] == 0]

        self.summary["extraction"] = {
            "albopictus_records": int(len(albopictus)),
            "zeros_records_all_species": int(len(zeros)),
            "zeros_records_albopictus": int((albopictus["individualCount"] == 0).sum()) if "individualCount" in albopictus.columns else None,
            "zeros_records_non_albopictus": int(len(zeros) - int((self.occurrence_data[
                self.occurrence_data["scientificName"] == "Aedes albopictus (Skuse, 1894)"
            ]["individualCount"] == 0).sum())) if "individualCount" in self.occurrence_data.columns else None,
            "life_stage_counts_albopictus": albopictus.get("lifeStage", pd.Series(dtype="object")).value_counts(dropna=False).to_dict(),
        }

        self.albopictus_data = pd.concat([albopictus, zeros], ignore_index=True).copy()
        self.summary["extraction"]["post_concat_total_records"] = int(len(self.albopictus_data))
        logger.info(f"Extracted {len(self.albopictus_data)} albopictus records (including zeros)")

    def clean_coordinates(self) -> None:
        """Clean and standardize coordinate data - replicating notebook's exact logic."""
        logger.info("Cleaning coordinate data...")

        if self.albopictus_data is None:
            raise ValueError("Albopictus data not extracted. Call extract_albopictus_data() first.")

        coord_before = self.albopictus_data[["decimalLatitude", "decimalLongitude"]].copy()

        # Remove degree symbols and whitespace
        self.albopictus_data['decimalLatitude'] = (
            self.albopictus_data['decimalLatitude']
            .astype(str)
            .str.replace('°', '', regex=False)
            .str.strip()
        )
        self.albopictus_data['decimalLongitude'] = (
            self.albopictus_data['decimalLongitude']
            .astype(str)
            .str.replace('°', '', regex=False)
            .str.strip()
        )

        # Count quick “fix indicators”
        deg_sym_lat = coord_before["decimalLatitude"].astype(str).str.contains("°", na=False).sum()
        deg_sym_lon = coord_before["decimalLongitude"].astype(str).str.contains("°", na=False).sum()
        comma_lat = coord_before["decimalLatitude"].astype(str).str.contains(",", na=False).sum()
        comma_lon = coord_before["decimalLongitude"].astype(str).str.contains(",", na=False).sum()

        # Create a temporary copy (replicating notebook's albopictus_ variable)
        albopictus_temp = self.albopictus_data.copy()

        # Replace commas with dots before converting
        albopictus_temp['decimalLatitude'] = (
            albopictus_temp['decimalLatitude'].str.replace(',', '.', regex=False)
        )
        albopictus_temp['decimalLongitude'] = (
            albopictus_temp['decimalLongitude'].str.replace(',', '.', regex=False)
        )

        # Convert to numeric
        albopictus_temp['decimalLatitude'] = pd.to_numeric(
            albopictus_temp['decimalLatitude'], errors='coerce'
        )
        albopictus_temp['decimalLongitude'] = pd.to_numeric(
            albopictus_temp['decimalLongitude'], errors='coerce'
        )

        # NOTE: remove the notebook’s redundant overwrite that undoes comma fixing.
        albopictus_temp["decimalLatitude"] = pd.to_numeric(albopictus_temp["decimalLatitude"], errors="coerce")
        albopictus_temp["decimalLongitude"] = pd.to_numeric(albopictus_temp["decimalLongitude"], errors="coerce")

        # Remove records with missing coordinates
        initial_count = len(albopictus_temp)
        nan_count = int(albopictus_temp[["decimalLatitude", "decimalLongitude"]].isna().any(axis=1).sum())

        self.albopictus_data = albopictus_temp.dropna(subset=['decimalLatitude', 'decimalLongitude'])
        kept_count = int(len(self.albopictus_data))
        removed_count = initial_count - kept_count

        self.summary["coordinates"] = {
            "records_before": int(initial_count),
            "degree_symbol_lat_rows": int(deg_sym_lat),
            "degree_symbol_lon_rows": int(deg_sym_lon),
            "comma_decimal_lat_rows": int(comma_lat),
            "comma_decimal_lon_rows": int(comma_lon),
            "rows_with_nan_coords_after_parse": int(nan_count),
            "records_dropped_missing_coords": int(removed_count),
            "records_after": int(kept_count),
        }

        logger.info(f"Removed {removed_count} records due to missing coordinates")

    def create_trap_ids(self) -> None:
        """Create unique trap IDs based on coordinate pairs."""
        logger.info("Creating unique trap IDs...")

        self.albopictus_data['id_trap'] = self.albopictus_data.groupby(
            ['decimalLatitude', 'decimalLongitude']
        ).ngroup()

        n_traps = int(self.albopictus_data['id_trap'].nunique())
        n_measures = int(len(self.albopictus_data))

        # Records-per-trap quick stats
        per_trap = self.albopictus_data.groupby("id_trap").size()
        self.summary["traps"] = {
            "trap_id_definition": "ngroup() over exact (decimalLatitude, decimalLongitude) pairs",
            "unique_traps": n_traps,
            "total_measures": n_measures,
            "measures_per_trap_min": int(per_trap.min()) if len(per_trap) else 0,
            "measures_per_trap_median": float(per_trap.median()) if len(per_trap) else 0.0,
            "measures_per_trap_max": int(per_trap.max()) if len(per_trap) else 0,
        }

        logger.info(f"Created {n_traps} unique traps with {n_measures} total measures")

    def process_temporal_data(self) -> None:
        """Process temporal data including date splitting and duration calculation."""
        logger.info("Processing temporal data...")

        # Split event dates
        self.albopictus_data[['temp_start', 'temp_end']] = (
            self.albopictus_data['eventDate'].str.split('/', expand=True)
        )

        # Convert to datetime
        self.albopictus_data['temp_start'] = pd.to_datetime(
            self.albopictus_data['temp_start'], format='%Y-%m-%d', errors='coerce'
        )
        self.albopictus_data['temp_end'] = pd.to_datetime(
            self.albopictus_data['temp_end'], format='%Y-%m-%d', errors='coerce'
        )

        # Assign start and end dates correctly
        self.albopictus_data['start_date'] = self.albopictus_data[['temp_start', 'temp_end']].min(axis=1)
        self.albopictus_data['end_date'] = self.albopictus_data[['temp_start', 'temp_end']].max(axis=1)

        # Clean up temporary columns
        self.albopictus_data.drop(columns=['temp_start', 'temp_end'], inplace=True)

        # Calculate time differences in days
        self.albopictus_data['time_diff'] = (
            self.albopictus_data['end_date'] - self.albopictus_data['start_date']
        ).dt.days

        # Fix zero time differences SAFELY: only time_diff, not entire rows
        zero_time_diff_mask = self.albopictus_data['time_diff'] == 0
        zero_time_diff_count = int(zero_time_diff_mask.sum())
        if zero_time_diff_count > 0:
            self.albopictus_data.loc[zero_time_diff_mask, 'time_diff'] = 1
        self.summary["temporal"] = {
            "zero_time_diff_fixed_to_1_day": int(zero_time_diff_count),
            "time_diff_min": int(self.albopictus_data["time_diff"].min()) if "time_diff" in self.albopictus_data.columns else None,
            "time_diff_max": int(self.albopictus_data["time_diff"].max()) if "time_diff" in self.albopictus_data.columns else None,
        }

        logger.info(f"Fixed {zero_time_diff_count} records with zero time difference")

    def validate_sampling_effort(self) -> None:
        """Validate sampling effort against calculated time differences."""
        logger.info("Validating sampling effort...")

        # Extract numeric values from sampling effort
        self.albopictus_data['samplingEffort Days'] = (
            self.albopictus_data['samplingEffort']
            .str.extract(r'(\d+)')
            .astype(float)
        )

        # Calculate difference between reported and calculated effort
        self.albopictus_data['samplingEffort Diff'] = (
            self.albopictus_data['samplingEffort Days'] - self.albopictus_data['time_diff']
        )

        # Mark records to keep (difference <= 2 days)
        # Create the keep column with the exact same logic as notebook
        keep_condition = (np.abs(self.albopictus_data['samplingEffort Diff']) <= 2)

        # Convert to object dtype to match notebook format
        # The notebook's keep column becomes object dtype due to the row corruption
        self.albopictus_data['keep'] = keep_condition.astype('object')

        keep_stats = self.albopictus_data['keep'].value_counts(dropna=False)
        self.summary["sampling_effort_validation"] = {
            "kept_true": int(keep_stats.get(True, 0)),
            "kept_false": int(keep_stats.get(False, 0)),
            "kept_nan": int(keep_stats.get(np.nan, 0)) if np.nan in keep_stats.index else 0,
        }

        logger.info(f"Validation results - Keep: {keep_stats.get(True, 0)}, "
                   f"Discard: {keep_stats.get(False, 0)}")

    def calculate_weekly_rates(self) -> None:
        """Calculate weekly occurrence rates."""
        logger.info("Calculating weekly occurrence rates...")

        self.albopictus_data['weeklyRate'] = (
            7 * self.albopictus_data['individualCount'] / self.albopictus_data['time_diff']
        )

    def compute_previous_weekly_rates_by_effort(self, delta_days: int = 1) -> None:
        """
        For each row in albopictus_data, find the 1-period-ago and 2-periods-ago
        weeklyRate measurements (within +/- delta_days of the implied start date),
        and store them in new columns 'prev_weeklyRate' and 'prev2_weeklyRate'.
        If multiple candidates are found, takes their mean.
        """
        logger.info("Computing previous weekly rates (prev_weeklyRate, prev2_weeklyRate)...")

        df = self.albopictus_data

        # Ensure correct dtypes
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df["samplingEffort Days"] = pd.to_numeric(df["samplingEffort Days"], errors="coerce")

        df["prev_weeklyRate"] = np.nan
        df["prev2_weeklyRate"] = np.nan

        multiple_prev = 0
        multiple_prev2 = 0

        for i, row in df.iterrows():
            trap = row["id_trap"]
            end = row["end_date"]
            eff = row["samplingEffort Days"]

            # Skip rows with missing values
            if pd.isna(trap) or pd.isna(end) or pd.isna(eff):
                continue

            start = end - pd.Timedelta(days=eff)

            # 1-period-ago window
            w0 = start - pd.Timedelta(days=delta_days)
            w1 = start + pd.Timedelta(days=delta_days)
            prev_rows = df[
                (df["id_trap"] == trap) &
                (df.index != i) &
                (df["end_date"] > w0) &
                (df["end_date"] < w1)
            ]["weeklyRate"]

            if len(prev_rows) == 1:
                df.at[i, "prev_weeklyRate"] = prev_rows.iloc[0]
            elif len(prev_rows) > 1:
                df.at[i, "prev_weeklyRate"] = prev_rows.mean()
                multiple_prev += 1

            # 2-periods-ago window (end_date - 2*eff, ±delta_days)
            prev2_center = end - pd.Timedelta(days=2 * eff)
            w0 = prev2_center - pd.Timedelta(days=delta_days)
            w1 = prev2_center + pd.Timedelta(days=delta_days)
            prev2_rows = df[
                (df["id_trap"] == trap) &
                (df.index != i) &
                (df["end_date"] > w0) &
                (df["end_date"] < w1)
            ]["weeklyRate"]

            if len(prev2_rows) == 1:
                df.at[i, "prev2_weeklyRate"] = prev2_rows.iloc[0]
            elif len(prev2_rows) > 1:
                df.at[i, "prev2_weeklyRate"] = prev2_rows.mean()
                multiple_prev2 += 1

        total = len(df)
        found1 = df["prev_weeklyRate"].notna().sum()
        miss1 = df["prev_weeklyRate"].isna().sum()
        found2 = df["prev2_weeklyRate"].notna().sum()
        miss2 = df["prev2_weeklyRate"].isna().sum()

        logger.info(f"Total rows:                                  {total}")
        logger.info(f"Entries with prev_weeklyRate found:          {found1}")
        logger.info(f"Entries with prev_weeklyRate missing:        {miss1}")
        logger.info(f"Used mean for >1 prev  candidates:           {multiple_prev}")
        logger.info(f"Entries with prev2_weeklyRate found:         {found2}")
        logger.info(f"Entries with prev2_weeklyRate missing:       {miss2}")
        logger.info(f"Used mean for >1 prev2 candidates:           {multiple_prev2}")

        self.albopictus_data = df

    def filter_data(self) -> None:
        """Apply final filters to the data."""
        logger.info("Applying final filters...")

        # Filter by life stages
        valid_life_stages = ["Egg", "Adult", "Larva"]
        initial_count = int(len(self.albopictus_data))

        life_stage_counts_before = (
            self.albopictus_data.get("lifeStage", pd.Series(dtype="object"))
            .value_counts(dropna=False)
            .to_dict()
        )

        self.filtered_data = self.albopictus_data[
            self.albopictus_data["lifeStage"].isin(valid_life_stages)
        ].copy()

        life_stage_counts_after = (
            self.filtered_data.get("lifeStage", pd.Series(dtype="object"))
            .value_counts(dropna=False)
            .to_dict()
        )

        life_stage_removed = initial_count - len(self.filtered_data)
        logger.info(f"Removed {life_stage_removed} records due to invalid life stages")

        # Filter by validation flag
        validation_initial = int(len(self.filtered_data))
        self.filtered_data = self.filtered_data[self.filtered_data["keep"] == True]
        validation_removed = validation_initial - len(self.filtered_data)

        logger.info(f"Removed {validation_removed} records due to validation failures")
        logger.info(f"Final dataset: {len(self.filtered_data)} records")

        final_count = int(len(self.filtered_data))

        self.summary["final_filtering"] = {
            "records_before_filtering": int(initial_count),
            "life_stage_counts_before": life_stage_counts_before,
            "records_after_life_stage_filter": int(validation_initial),
            "life_stage_counts_after": life_stage_counts_after,
            "records_removed_validation_fail": int(validation_removed),
            "final_records": int(final_count),
            "final_unique_traps": int(self.filtered_data["id_trap"].nunique()) if self.filtered_data is not None else None,
            "final_zeros": int((self.filtered_data["individualCount"] == 0).sum()) if self.filtered_data is not None and "individualCount" in self.filtered_data.columns else None,
        }

    def analyze_duplicates(self) -> None:
        """Analyze duplicate entries in the dataset."""
        logger.info("Analyzing duplicate entries...")

        # Group by trap and date to find duplicates
        trap_date_counts = (
            self.filtered_data
            .groupby(['id_trap', 'end_date'])
            .size()
            .reset_index(name='count')
        )

        multiple_entries = trap_date_counts[trap_date_counts['count'] > 1]

        if len(multiple_entries) > 0:
            total_cases = len(multiple_entries)
            max_entries = multiple_entries['count'].max()
            affected_traps = multiple_entries['id_trap'].nunique()

            logger.info(f"Found {total_cases} cases with multiple entries")
            logger.info(f"Maximum entries per trap/date: {max_entries}")
            logger.info(f"Affected traps: {affected_traps}")
        else:
            logger.info("No duplicate entries found")

    def plot_time_diff_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot the distribution of time differences."""
        time_diff_distribution = self.albopictus_data['time_diff'].dropna()
        frequency_counts = time_diff_distribution.value_counts().sort_index()

        plt.figure(figsize=(12, 6))
        bars = plt.bar(frequency_counts.index, frequency_counts.values,
                      color='skyblue', edgecolor='black')

        # Add annotations
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                        ha='center', va='bottom', fontsize=9)

        plt.title('Distribution of Sampling Time Differences')
        plt.xlabel('Time Difference (days)')
        plt.ylabel('Frequency')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_weekly_rate_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot the distribution of weekly rates."""
        plt.figure(figsize=(10, 6))
        self.albopictus_data["weeklyRate"].hist(bins=3000)
        plt.xlim(0, 20)
        plt.yscale("log")
        plt.title('Distribution of Weekly Occurrence Rates')
        plt.xlabel('Weekly Rate')
        plt.ylabel('Frequency (log scale)')

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_data(self, csv_path: str = "albopictus.csv.zip",
                  pickle_path: str = "albopictus.pkl", save_dir: str = "../output_data",
                  stats_dir: str = "../output_stats") -> None:
        """
        Save the processed data to compressed CSV (ZIP) and pickle formats.
        
        Args:
            csv_path: Filename for compressed CSV output (default: albopictus.csv.zip)
            pickle_path: Filename for pickle output (default: albopictus.pkl)
            save_dir: Directory for saving outputs (default: ../output_data relative to src/)
            stats_dir: Directory for saving stats JSON (default: ../output_stats relative to src/)
        """
        # Resolve path relative to this script's location
        script_dir = Path(__file__).parent
        save_dir_path = (script_dir / save_dir).resolve()
        save_dir_path.mkdir(parents=True, exist_ok=True)
        
        stats_dir_path = (script_dir / stats_dir).resolve()
        stats_dir_path.mkdir(parents=True, exist_ok=True)
        
        csv_full_path = save_dir_path / csv_path
        pickle_full_path = save_dir_path / pickle_path
        
        logger.info(f"Saving data to {csv_full_path} and {pickle_full_path}...")

        if self.filtered_data is None:
            raise ValueError("No filtered data to save. Run the complete pipeline first.")

        self.filtered_data.to_csv(csv_full_path, index=False)
        self.filtered_data.to_pickle(pickle_full_path)

        # Save run summary to the stats directory (for README / provenance)
        summary_path = stats_dir_path / "albopictus_summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.summary, f, indent=2, default=str)
            logger.info(f"Saved processing summary to {summary_path}")
        except Exception as e:
            logger.warning(f"Failed to save summary JSON: {e}")

        logger.info("Data saved successfully")

    def get_summary_stats(self) -> dict:
        """Get summary statistics of the processed data."""
        if self.filtered_data is None:
            return {}

        stats = {
            'total_records': len(self.filtered_data),
            'unique_traps': self.filtered_data['id_trap'].nunique(),
            'date_range': (
                self.filtered_data['end_date'].min(),
                self.filtered_data['end_date'].max()
            ),
            'life_stages': self.filtered_data['lifeStage'].value_counts().to_dict(),
            'countries': self.filtered_data.get('country', pd.Series()).value_counts().to_dict(),
            'individual_count_stats': {
                'mean': self.filtered_data['individualCount'].mean(),
                'median': self.filtered_data['individualCount'].median(),
                'max': self.filtered_data['individualCount'].max(),
                'zero_counts': (self.filtered_data['individualCount'] == 0).sum(),
                'positive_counts': (self.filtered_data['individualCount'] > 0).sum()
            }
        }

        return stats

    def run_complete_pipeline(self) -> None:
        """Run the complete data processing pipeline."""
        logger.info("Starting complete Aedes albopictus data processing pipeline")

        try:
            self.load_data()
            self.extract_albopictus_data()
            self.clean_coordinates()
            self.create_trap_ids()
            self.process_temporal_data()
            self.validate_sampling_effort()
            self.calculate_weekly_rates()
            self.compute_previous_weekly_rates_by_effort(delta_days=3)
            self.filter_data()
            self.analyze_duplicates()

            # Print summary statistics
            stats = self.get_summary_stats()
            logger.info("Pipeline completed successfully!")
            logger.info(f"Final dataset summary: {stats}")

        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main function to run the data processing pipeline."""
    processor = AlbopictusDataProcessor()

    try:
        processor.run_complete_pipeline()

        # Save the processed data
        processor.save_data()

        # Generate plots (optional)
        # processor.plot_time_diff_distribution("time_diff_distribution.png")
        # processor.plot_weekly_rate_distribution("weekly_rate_distribution.png")

        print("Data processing completed successfully!")
        print(f"Processed data saved to albopictus.csv and albopictus.pkl")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())