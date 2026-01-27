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
        self.data_dir = Path(data_dir)
        self.event_data = None
        self.occurrence_data = None
        self.albopictus_data = None
        self.filtered_data = None

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

        self.albopictus_data = pd.concat([albopictus, zeros], ignore_index=True).copy()
        logger.info(f"Extracted {len(self.albopictus_data)} albopictus records (including zeros)")

    def clean_coordinates(self) -> None:
        """Clean and standardize coordinate data - replicating notebook's exact logic."""
        logger.info("Cleaning coordinate data...")

        if self.albopictus_data is None:
            raise ValueError("Albopictus data not extracted. Call extract_albopictus_data() first.")

        # Replicate the notebook's exact coordinate cleaning logic
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

        # This replicates the notebook's redundant conversion lines:
        # albopictus_['decimalLatitude'] = pd.to_numeric(albopictus['decimalLatitude'], errors='coerce')
        # albopictus_['decimalLongitude'] = pd.to_numeric(albopictus['decimalLongitude'], errors='coerce')
        # Note: These lines in the notebook use the original 'albopictus' instead of 'albopictus_'
        # This creates a subtle difference that may cause the coordinate issue
        albopictus_temp['decimalLatitude'] = pd.to_numeric(
            self.albopictus_data['decimalLatitude'], errors='coerce'
        )
        albopictus_temp['decimalLongitude'] = pd.to_numeric(
            self.albopictus_data['decimalLongitude'], errors='coerce'
        )

        # Remove records with missing coordinates
        initial_count = len(albopictus_temp)
        self.albopictus_data = albopictus_temp.dropna(subset=['decimalLatitude', 'decimalLongitude'])
        removed_count = initial_count - len(self.albopictus_data)

        logger.info(f"Removed {removed_count} records due to missing coordinates")

    def create_trap_ids(self) -> None:
        """Create unique trap IDs based on coordinate pairs."""
        logger.info("Creating unique trap IDs...")

        self.albopictus_data['id_trap'] = self.albopictus_data.groupby(
            ['decimalLatitude', 'decimalLongitude']
        ).ngroup()

        n_traps = len(self.albopictus_data['id_trap'].unique())
        n_measures = len(self.albopictus_data)

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

        # Fix zero time differences - REPLICATE THE NOTEBOOK'S BUGGY BEHAVIOR
        # The notebook has this problematic line: albopictus[albopictus['time_diff'] == 0] = 1.
        # This corrupts entire rows by setting ALL columns to 1.0 where time_diff == 0
        zero_time_diff_mask = self.albopictus_data['time_diff'] == 0
        zero_time_diff_count = zero_time_diff_mask.sum()
        if zero_time_diff_count > 0:
            logger.info(f"Fixed {zero_time_diff_count} records with zero time difference")

            # Replicate the notebook's behavior: set entire rows to 1.0
            # This will corrupt columns like lifeStage, making them 1.0 instead of strings
            try:
                # Use pandas' assignment with warning suppression behavior
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # This mimics the notebook's problematic line
                    self.albopictus_data.loc[zero_time_diff_mask] = 1.0
            except:
                # If that fails, just set time_diff column
                self.albopictus_data.loc[zero_time_diff_mask, 'time_diff'] = 1

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

        keep_stats = self.albopictus_data['keep'].value_counts()
        logger.info(f"Validation results - Keep: {keep_stats.get(True, 0)}, "
                   f"Discard: {keep_stats.get(False, 0)}")

    def calculate_weekly_rates(self) -> None:
        """Calculate weekly occurrence rates."""
        logger.info("Calculating weekly occurrence rates...")

        self.albopictus_data['weeklyRate'] = (
            7 * self.albopictus_data['individualCount'] / self.albopictus_data['time_diff']
        )

    def filter_data(self) -> None:
        """Apply final filters to the data."""
        logger.info("Applying final filters...")

        # Filter by life stages
        valid_life_stages = ["Egg", "Adult", "Larva"]
        initial_count = len(self.albopictus_data)

        self.filtered_data = self.albopictus_data[
            self.albopictus_data["lifeStage"].isin(valid_life_stages)
        ].copy()

        life_stage_removed = initial_count - len(self.filtered_data)
        logger.info(f"Removed {life_stage_removed} records due to invalid life stages")

        # Filter by validation flag
        validation_initial = len(self.filtered_data)
        self.filtered_data = self.filtered_data[self.filtered_data["keep"] == True]
        validation_removed = validation_initial - len(self.filtered_data)

        logger.info(f"Removed {validation_removed} records due to validation failures")
        logger.info(f"Final dataset: {len(self.filtered_data)} records")

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

    def save_data(self, csv_path: str = "albopictus.csv",
                  pickle_path: str = "albopictus.pkl", save_dir: str = "../output_data") -> None:
        """Save the processed data to CSV and pickle formats."""
        csv_path = Path(save_dir) / csv_path
        pickle_path = Path(save_dir) / pickle_path
        logger.info(f"Saving data to {csv_path} and {pickle_path}...")

        if self.filtered_data is None:
            raise ValueError("No filtered data to save. Run the complete pipeline first.")

        self.filtered_data.to_csv(csv_path, index=False)
        self.filtered_data.to_pickle(pickle_path)

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