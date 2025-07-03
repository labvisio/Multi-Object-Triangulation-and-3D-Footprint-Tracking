# -*- coding: utf-8 -*-
"""
Odometry Data Pre-processing Utility.

This script reads a raw odometry data file (in CSV format) and enriches it
by adding several new columns with formatted and relative time information.
This pre-processing makes the data more human-readable and easier to use for
time-based analysis and plotting.

The script performs the following main tasks:
1.  Reads an input CSV file containing at least a 'timestamp' column in Unix
    seconds, along with odometry data (e.g., 'x_odom', 'y_odom').
2.  Adds new columns:
    -   Absolute timestamps formatted as HH:MM:SS.ffffff.
    -   Relative time in seconds since the first timestamp.
    -   Relative time formatted as a human-readable string.
3.  Saves the processed data to a new output CSV file.
4.  Prints a summary report including the time range, total duration, and
    approximate sampling rate.
"""

import pandas as pd
from datetime import datetime

def process_odometry_file(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Processes a raw odometry CSV file to add formatted and relative time columns.

    This function reads a CSV file, converts the Unix timestamp column into various
    human-readable and relative time formats with microsecond precision, and saves
    the resulting DataFrame to a new CSV file.

    Args:
        input_file (str): The path to the input odometry CSV file.
        output_file (str): The path where the processed CSV file will be saved.

    Returns:
        pd.DataFrame: The processed pandas DataFrame with the added time columns.

    New Columns Created:
    -   `time_hms`: Absolute time formatted as HH:MM:SS.
    -   `time_hms_precise`: Absolute time with microsecond precision (HH:MM:SS.ffffff).
    -   `relative_time`: Time elapsed since the first record (in seconds, as float).
    -   `relative_seconds`: Alias for `relative_time`.
    -   `relative_time_precise`: Timedelta object converted to a string.
    -   `relative_time_formatted`: Cleaned, human-readable relative time string.
    -   `timestamp_original`: A copy of the original timestamp for reference.
    """
    # Read the source CSV file into a pandas DataFrame.
    df = pd.read_csv(input_file)
    
    # --- Time Column Generation ---

    # Convert Unix timestamp (in seconds) to a standard datetime object, then format it.
    # %f provides microsecond precision.
    df['time_hms_precise'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S.%f')
    df['time_hms'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')

    # Calculate relative time by subtracting the initial timestamp from all others.
    start_time = df['timestamp'].iloc[0]
    df['relative_time'] = df['timestamp'] - start_time
    df['relative_seconds'] = df['relative_time'] # Create a clear alias for relative time in seconds.
    
    # Format the relative time (which is in seconds) into a timedelta string for readability.
    df['relative_time_precise'] = pd.to_timedelta(df['relative_time'], unit='s').astype(str)
    
    # Clean up the timedelta string format by removing the "0 days " prefix.
    df['relative_time_formatted'] = df['relative_time_precise'].str.replace('0 days ', '', regex=False)
    
    # Keep a copy of the original timestamp for easy reference and validation.
    df['timestamp_original'] = df['timestamp']
    
    # Save the processed DataFrame to a new CSV file without the pandas index.
    df.to_csv(output_file, index=False)
    
    # --- Summary Report ---
    print(f"Processed file has been saved to: {output_file}")
    print(f"Total number of records: {len(df)}")
    print(f"Time period (UTC): from {df['time_hms_precise'].iloc[0]} to {df['time_hms_precise'].iloc[-1]}")
    print(f"Total duration: {df['relative_seconds'].iloc[-1]:.6f} seconds")
    
    return df

# --- Script Execution ---

if __name__ == "__main__":
    # Define input and output filenames
    input_filename = 'odometry_data.csv'
    output_filename = 'odometry_data_processed.csv'
    
    # Call the processing function
    processed_df = process_odometry_file(input_filename, output_filename)

    # Display the first few rows of the processed data as an example
    print("\n--- First 5 Rows of Processed Data (High Precision) ---")
    print(processed_df[[
        'x_odom', 'y_odom', 'timestamp_original', 
        'time_hms_precise', 'relative_seconds', 'relative_time_formatted'
    ]].head())

    # Display a comparison to show the precision of the conversion
    print("\n--- Precision Comparison ---")
    print("Original Timestamp vs. Formatted Time:")
    for i in range(min(3, len(processed_df))): # Show up to 3 rows
        print(f"  {processed_df['timestamp_original'].iloc[i]:.6f} -> {processed_df['time_hms_precise'].iloc[i]}")

    # --- Basic Analysis ---
    # Calculate the average time difference between consecutive samples
    mean_interval = processed_df['relative_seconds'].diff().mean()
    # Calculate the approximate sampling rate in Hertz (Hz)
    sampling_rate = 1 / mean_interval if mean_interval > 0 else 0
    
    print(f"\nAverage interval between samples: {mean_interval:.6f} seconds")
    print(f"Approximate sampling rate: {sampling_rate:.1f} Hz")