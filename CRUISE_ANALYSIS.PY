import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
# Constants matched with plotv14.py
GEAR_RATIO = 6.09        
WHEEL_RADIUS_M = 0.261

def select_file():
    """Open GUI to select a CSV file."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CSV File for Cruise Analysis",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

def process_file(file_path):
    """Load, filter, and extract cruise data from the file."""
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"Loading file: {file_path}")
    
    try:
        # Load data
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
            
        print(f"Data Loaded. Shape: {df.shape}")

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check for Cruising Status Column
        if 'MCU_Cruising_Status' not in df.columns:
            print("ERROR: Column 'MCU_Cruising_Status' not found in the CSV.")
            print("Available columns:", list(df.columns))
            return

        # Parse Timestamps
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        else:
            print("WARNING: 'timestamps' column not found. Time difference cannot be calculated accurately.")
            df['timestamps'] = np.nan

        # ---------------------------------------------------------
        # 1. CALCULATE DERIVED COLUMNS (Speed, Time Diff)
        # ---------------------------------------------------------
        
        # Calculate Speed if missing (Logic from plotv14.py)
        if 'Speed_kmh' not in df.columns:
            if 'Current_Rotational_Speed' in df.columns:
                print("Calculating Speed_kmh from Current_Rotational_Speed...")
                wheel_rpm = df['Current_Rotational_Speed'] / GEAR_RATIO
                df['Speed_kmh'] = (wheel_rpm * 2 * np.pi / 60) * WHEEL_RADIUS_M * 3.6
            else:
                df['Speed_kmh'] = np.nan
                print("WARNING: Could not determine Speed (missing 'Speed_kmh' and 'Current_Rotational_Speed').")

        # Calculate Time Difference (Difference between consecutive rows)
        if not df['timestamps'].isna().all():
            df['Time_Diff_Seconds'] = df['timestamps'].diff().dt.total_seconds()
        else:
            df['Time_Diff_Seconds'] = np.nan

        # ---------------------------------------------------------
        # 2. IDENTIFY TARGET COLUMNS
        # ---------------------------------------------------------
        # We need: SPEED , BATTERY CURRENT , BATTERY VOLTAGE, AND TIME DIFFERENCE
        
        # Speed
        speed_col = 'Speed_kmh' if 'Speed_kmh' in df.columns else None
        
        # Battery Current
        # Try 'Battery_current1' (standard), then 'pack_current', then 'Current'
        current_col = None
        for col in ['Battery_current1', 'Battery_current', 'pack_current', 'Current']:
            if col in df.columns:
                current_col = col
                break
        
        # Battery Voltage
        # Try 'Battery_voltage' (standard), then 'pack_voltage', then 'Voltage'
        voltage_col = None
        for col in ['Battery_voltage', 'pack_voltage', 'Voltage']:
            if col in df.columns:
                voltage_col = col
                break

        print(f"Using columns -> Speed: {speed_col}, Current: {current_col}, Voltage: {voltage_col}")

        # ---------------------------------------------------------
        # 3. FILTER FOR CRUISE MODE
        # ---------------------------------------------------------
        print("Filtering for Cruise Mode (MCU_Cruising_Status == 1)...")
        # Ensure it's numeric 1
        cruise_df = df[df['MCU_Cruising_Status'] == 1].copy()
        
        if cruise_df.empty:
            print("No data found where MCU_Cruising_Status is 1 (Cruise Mode).")
            return

        # ---------------------------------------------------------
        # 4. EXTRACT AND SAVE
        # ---------------------------------------------------------
        output_df = pd.DataFrame()
        output_df['Timestamp'] = cruise_df['timestamps']
        
        if speed_col:
            output_df['Speed_kmh'] = cruise_df[speed_col]
        else:
            output_df['Speed_kmh'] = "N/A"
            
        if current_col:
            output_df['Battery_Current_A'] = cruise_df[current_col]
        else:
            output_df['Battery_Current_A'] = "N/A"
            
        if voltage_col:
            output_df['Battery_Voltage_V'] = cruise_df[voltage_col]
        else:
            output_df['Battery_Voltage_V'] = "N/A"
            
        output_df['Time_Diff_Seconds'] = cruise_df['Time_Diff_Seconds']

        # Determine Output Filename
        input_dir = os.path.dirname(file_path)
        input_name = os.path.basename(file_path)
        output_name = os.path.splitext(input_name)[0] + "_CRUISE_DATA.csv"
        output_path = os.path.join(input_dir, output_name)

        try:
            output_df.to_csv(output_path, index=False)
            print("="*60)
            print(f"SUCCESS! Extracted {len(output_df)} rows.")
            print(f"File saved to: {output_path}")
            print("="*60)
        except Exception as e:
            print(f"Error saving output file: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Select a file from the dialog window...")
    selected_path = select_file()
    if selected_path:
        process_file(selected_path)
    else:
        print("Operation cancelled.")
