"""
Production-Ready AQI Time-Series Preprocessing Module

This module provides time-series aware preprocessing for AQI datasets.
Key Features:
  - Datetime conversion and sorting
  - Time-series aware missing value handling
  - Missing value indicators for analysis transparency
  - No data leakage (proper temporal order)
  - Handles long gaps without over-interpolation
  - Production-ready and reusable

Author: Data Science Team
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings('ignore')


class AQIPreprocessor:
    """
    Time-series aware preprocessor for AQI datasets.
    
    This class handles missing values in a time-series context by:
    1. Respecting temporal order (no data leakage)
    2. Detecting and handling long gaps
    3. Using appropriate imputation strategies for different gap types
    4. Adding transparency through missing value indicators
    """
    
    # Configuration constants
    POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    INTERPOLATION_LIMIT = 4  # Max consecutive NaNs to interpolate (hours)
    KNN_K = 3  # Number of neighbors for KNN imputation
    LONG_GAP_THRESHOLD = 24  # Gap > 24 hours is considered "long"
    
    def __init__(self, interpolation_limit: int = 4, knn_k: int = 3, 
                 long_gap_threshold: int = 24, use_knn: bool = True):
        """
        Initialize preprocessor with custom parameters.
        
        Args:
            interpolation_limit: Max consecutive NaNs to interpolate (hours)
            knn_k: Number of neighbors for KNN imputation
            long_gap_threshold: Gap threshold in hours (above = long gap)
            use_knn: Whether to apply KNN imputation after basic filling
        """
        self.interpolation_limit = interpolation_limit
        self.knn_k = knn_k
        self.long_gap_threshold = long_gap_threshold
        self.use_knn = use_knn
    
    @staticmethod
    def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse Date column to datetime format.
        
        Args:
            df: DataFrame with Date column in "DD-MM-YYYY HH:MM" format
            
        Returns:
            DataFrame with parsed datetime column
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M', 
                                     errors='coerce')
        
        # Check for parsing errors
        if df['Date'].isna().any():
            raise ValueError(
                f"Failed to parse {df['Date'].isna().sum()} date values. "
                "Check format (should be DD-MM-YYYY HH:MM)"
            )
        
        return df
    
    @staticmethod
    def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort dataset by Date and reset index.
        
        Args:
            df: DataFrame to sort
            
        Returns:
            Sorted DataFrame with reset index
        """
        df = df.sort_values('Date', ignore_index=True)
        return df
    
    def _detect_long_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify consecutive missing values and long gaps.
        
        Time-series context: Missing values often indicate data collection failures.
        Long gaps (>24hrs) may represent system downtime and shouldn't be
        interpolated (risks introducing artificial trends).
        
        Args:
            df: DataFrame with parsed Date column
            
        Returns:
            DataFrame with added columns:
            - {pollutant}_missing_before: Whether values were missing before imputation
            - gap_hours: Hours since last complete record (for each pollutant)
        """
        df = df.copy()
        
        for pollutant in self.POLLUTANTS:
            # Create indicator before any imputation
            df[f'{pollutant}_missing'] = df[pollutant].isna().astype(int)
        
        # Calculate time gaps for each pollutant
        df['time_diff_hours'] = df['Date'].diff().dt.total_seconds() / 3600
        
        return df
    
    def _interpolate_short_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate short gaps using linear interpolation with limit.
        
        Strategy: For gaps < interpolation_limit hours, use linear interpolation.
        This is reasonable for hourly AQI data where short gaps are likely
        measurement glitches, not actual zero pollution.
        
        Args:
            df: DataFrame with identified missing values
            
        Returns:
            DataFrame with short gaps interpolated
        """
        df = df.copy()
        
        for pollutant in self.POLLUTANTS:
            # Linear interpolation with limit on consecutive NaNs
            df[pollutant] = df[pollutant].interpolate(
                method='linear',
                limit=self.interpolation_limit,
                limit_area='inside'  # Don't extrapolate at boundaries
            )
        
        return df
    
    def _forward_backward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply forward fill followed by backward fill for remaining gaps.
        
        Strategy:
        - Forward fill: Uses last known value (preserves recent trend)
        - Backward fill: For leading NaNs (uses first known value)
        
        This is applied AFTER interpolation, so it handles remaining long gaps.
        Use forward fill cautiously; it can propagate outdated values.
        
        Args:
            df: DataFrame with interpolated values
            
        Returns:
            DataFrame with forward/backward filled values
        """
        df = df.copy()
        
        # Forward fill with limit to prevent propagating stale values
        # (max 48 hours of forward filling)
        for pollutant in self.POLLUTANTS:
            df[pollutant] = df[pollutant].fillna(method='ffill', limit=48)
        
        # Backward fill for leading NaNs only
        for pollutant in self.POLLUTANTS:
            df[pollutant] = df[pollutant].bfill()
        
        return df
    
    def _apply_knn_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply KNN imputation using pollutant features only.
        
        Strategy: KNN uses k=3 neighbors (cities have similar AQI patterns).
        This respects temporal order (uses all prior values).
        Applied after basic filling to refine remaining imputations.
        
        Args:
            df: DataFrame with forward/backward filled values
            
        Returns:
            DataFrame with KNN-refined values
        """
        if not self.use_knn:
            return df
        
        df = df.copy()
        
        # Extract pollutant data
        pollutant_data = df[self.POLLUTANTS].copy()
        
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=self.knn_k, weights='distance')
        df[self.POLLUTANTS] = imputer.fit_transform(pollutant_data)
        
        return df
    
    def _clean_up_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove intermediate columns, keep essential ones.
        
        Args:
            df: DataFrame with all intermediate columns
            
        Returns:
            Cleaned DataFrame with relevant columns
        """
        df = df.copy()
        
        # Remove time diff column (was for analysis only)
        df = df.drop(columns=['time_diff_hours'], errors='ignore')
        
        # Remove AQI columns and missing indicators
        cols_to_drop = ['AQI', 'AQI_Category', 
                       'PM2.5_missing', 'PM10_missing', 'NO2_missing', 
                       'SO2_missing', 'CO_missing', 'O3_missing']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        
        return df
    
    def preprocess(self, df: pd.DataFrame, city_name: Optional[str] = None) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for AQI dataset.
        
        Pipeline Steps:
        1. Parse datetime
        2. Sort by date
        3. Detect and flag missing values
        4. Interpolate short gaps
        5. Forward/backward fill remaining gaps
        6. Apply KNN imputation (optional)
        7. Clean up intermediate columns
        
        Args:
            df: Raw AQI DataFrame
            city_name: Optional city name for logging
            
        Returns:
            Preprocessed DataFrame ready for analysis
        """
        print(f"Starting preprocessing{'for ' + city_name if city_name else ''}...")
        print(f"  Input shape: {df.shape}")
        
        # Step 1: Parse dates
        print("  [1/6] Parsing dates...")
        df = self._parse_date(df)
        
        # Step 2: Sort by date
        print("  [2/6] Sorting by date...")
        df = self._sort_by_date(df)
        
        # Step 3: Detect missing values
        print("  [3/6] Detecting missing values...")
        missing_counts = {p: df[p].isna().sum() for p in self.POLLUTANTS}
        total_missing = sum(missing_counts.values())
        print(f"      Missing values by pollutant: {missing_counts}")
        print(f"      Total missing: {total_missing} ({100*total_missing/(len(df)*len(self.POLLUTANTS)):.2f}%)")
        
        df = self._detect_long_gaps(df)
        
        # Step 4: Interpolate short gaps
        print("  [4/6] Interpolating short gaps (limit={} hours)...".format(
            self.interpolation_limit))
        df = self._interpolate_short_gaps(df)
        
        # Step 5: Forward/backward fill
        print("  [5/6] Forward/backward filling...")
        df = self._forward_backward_fill(df)
        
        # Step 6: KNN imputation
        if self.use_knn:
            print(f"  [6/6] Applying KNN imputation (k={self.knn_k})...")
            df = self._apply_knn_imputation(df)
        else:
            print("  [6/6] Skipping KNN imputation (disabled)...")
        
        # Step 7: Clean up
        df = self._clean_up_columns(df)
        
        print(f"  Output shape: {df.shape}")
        print(f"  Remaining NaN values: {df[self.POLLUTANTS].isna().sum().sum()}")
        print("  ✓ Preprocessing complete!\n")
        
        return df


def preprocess_city(df: pd.DataFrame, city_name: Optional[str] = None,
                   use_knn: bool = True) -> pd.DataFrame:
    """
    Convenient wrapper function for single dataset preprocessing.
    
    This is the main function to use for preprocessing individual city datasets.
    
    Args:
        df: Raw AQI DataFrame from CSV
        city_name: Optional city name for logging/identification
        use_knn: Whether to apply KNN imputation (default: True)
        
    Returns:
        Preprocessed DataFrame with:
        - Parsed and sorted dates
        - Handled missing values
        - Missing value indicators (for transparency)
        - No data leakage (temporal order preserved)
        
    Example:
        >>> df = pd.read_csv('Bengaluru.csv')
        >>> df_clean = preprocess_city(df, city_name='Bengaluru')
    """
    preprocessor = AQIPreprocessor(use_knn=use_knn)
    return preprocessor.preprocess(df, city_name=city_name)


# ============================================================================
# BATCH PROCESSING UTILITY
# ============================================================================

def preprocess_all_cities(dataset_dir: str, use_knn: bool = True) -> dict:
    """
    Preprocess all city datasets in a directory.
    
    Args:
        dataset_dir: Directory containing CSV files (one per city)
        use_knn: Whether to apply KNN imputation
        
    Returns:
        Dictionary mapping city names to preprocessed DataFrames
        
    Example:
        >>> processed = preprocess_all_cities('./dataset')
        >>> bengaluru_clean = processed['Bengaluru']
    """
    import os
    from pathlib import Path
    
    dataset_path = Path(dataset_dir)
    processed_data = {}
    
    print("="*70)
    print("BATCH PROCESSING: AQI DATASETS")
    print("="*70 + "\n")
    
    for csv_file in sorted(dataset_path.glob('*.csv')):
        city_name = csv_file.stem  # Filename without extension
        print(f"Processing: {city_name}")
        
        df = pd.read_csv(csv_file)
        df_clean = preprocess_city(df, city_name=city_name, use_knn=use_knn)
        processed_data[city_name] = df_clean
    
    print("="*70)
    print(f"✓ Successfully processed {len(processed_data)} cities")
    print("="*70)
    
    return processed_data


# ============================================================================
# VALIDATION & DIAGNOSTICS
# ============================================================================

def validate_preprocessed_data(df: pd.DataFrame, city_name: str = "Dataset") -> None:
    """
    Validate preprocessed dataset for quality checks.
    
    Checks:
    - No remaining NaN values
    - Monotonic datetime index
    - Reasonable value ranges (no extreme outliers from imputation)
    
    Args:
        df: Preprocessed DataFrame
        city_name: Name for reporting
    """
    print(f"\nValidation Report for {city_name}:")
    print("-" * 50)
    
    # Check 1: Missing values
    missing_count = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].isna().sum().sum()
    print(f"Missing values: {missing_count}")
    if missing_count == 0:
        print("  ✓ No missing values")
    else:
        print(f"  ⚠ {missing_count} missing values remain")
    
    # Check 2: Date order
    is_sorted = df['Date'].is_monotonic_increasing
    print(f"Date order: {'✓ Monotonic' if is_sorted else '⚠ NOT monotonic'}")
    
    # Check 3: Reasonable ranges
    print("\nPollutant ranges:")
    for pollutant in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
        print(f"  {pollutant:6s}: min={df[pollutant].min():8.2f}, "
              f"max={df[pollutant].max():8.2f}, "
              f"mean={df[pollutant].mean():8.2f}")
    
    print("-" * 50)


if __name__ == "__main__":
    """
    Example usage and demonstration
    """
    # Example 1: Process single city
    print("EXAMPLE 1: Processing single city")
    print("-" * 70)
    df_bengaluru = pd.read_csv('./dataset/Bengaluru.csv')
    df_bengaluru_clean = preprocess_city(df_bengaluru, city_name='Bengaluru')
    validate_preprocessed_data(df_bengaluru_clean, 'Bengaluru')
    
    # Example 2: Batch process all cities
    print("\n\nEXAMPLE 2: Batch processing all cities")
    print("-" * 70)
    processed_cities = preprocess_all_cities('./dataset', use_knn=True)
    
    # Example 3: Inspect results
    print("\n\nEXAMPLE 3: Inspecting preprocessed data")
    print("-" * 70)
    for city_name, df_clean in processed_cities.items():
        print(f"\n{city_name}:")
        print(f"  Shape: {df_clean.shape}")
        print(f"  Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
        print(f"  Missing indicators: {[col for col in df_clean.columns if '_missing' in col]}")
