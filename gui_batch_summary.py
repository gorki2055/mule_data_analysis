import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import glob
import re
import threading
import sys
from scipy.signal import medfilt
# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\2877D237"
# ROOT_DIR_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
OUTPUT_ROOT_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS"
OUTPUT_FILENAME = "mule1_test_summary.csv"

GEAR_RATIO = 6.09    #6.36 and 6.09
WHEEL_RADIUS_M = 0.261
RESAMPLE_RATE = '100ms'

# Ride mode mapping
MODE_MAP = {
    0: "Suste",
    1: "Thikka", 
    2: "Babbal"
}

# ============================================================================
# LOGGING
# ============================================================================
class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        if self.text_widget:
            self.text_widget.configure(state=tk.NORMAL)
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state=tk.DISABLED)

    def flush(self):
        self.terminal.flush()

# ============================================================================
# PROCESSING LOGIC
# ============================================================================
def extract_metadata(filename):
    """
    Extracts Date and Shift from filename like 'MULE1_1-2-2026_MORNING.csv'
    Returns: (Date, Shift) or (None, None)
    """
    # Pattern: MULE2_{Date}_{Shift}.csv (case insensitive)
    # Handles: MULE2_1-2-2026_MORNING.csv or MULE2_1-27-2026_DAY_RIDE.csv
    # Matches date like 1-2-2026 or -1-2-2026 (handling possible dash prefix in user examples)
    # Group 2 (Shift) now accepts letters and underscores to match "DAY_RIDE"
    # UPDATED: Now matches MULE followed by any digit (MULE1, MULE2, etc.)
    pattern = r"MULE\d+_[-]*(\d+-\d+-\d+)_([A-Za-z_]+)\.csv"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).upper()
    return None, None

def process_single_file(file_path):
    """
    Calculates summary metrics for a single file.
    Returns a dict of metrics.
    """
    try:
        # 1. Load Data
        df = pd.read_csv(file_path, engine='c', low_memory=False)
        df.columns = df.columns.str.strip()
        
        # 2. Fix Voltage Units (mV -> V)
        if 'Battery_voltage' in df.columns and df['Battery_voltage'].mean() > 1000:
             df['Battery_voltage'] = df['Battery_voltage'] / 1000.0
             
        # 3. Parse Timestamps
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        # 4. Process Current
        if 'Battery_current1' in df.columns:
            df['Processed_Current'] = df['Battery_current1'].values
            if 'Charge_discharge_status' in df.columns:
                charging_mask = df['Charge_discharge_status'] == 1
                df.loc[charging_mask, 'Processed_Current'] = -np.abs(df.loc[charging_mask, 'Battery_current1'].values)
        else:
            return None # Vital data missing

        # Filter out current values above 185A
        initial_len = len(df)
        df = df[df['Processed_Current'] <= 185]
        if initial_len - len(df) > 0:
            print(f"Filtered {initial_len - len(df)} rows with Current > 185A")

        # Filter out current values less than -50A
        initial_len = len(df)
        df = df[df['Processed_Current'] >= -50]
        if initial_len - len(df) > 0:
            print(f"Filtered {initial_len - len(df)} rows with Current < -50A")

        # 5. Filter Temperatures
        for sensor in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']:
            if sensor in df.columns:
                # Apply median filter to remove spikes ensuring odd kernel size
                df[sensor] = medfilt(df[sensor], kernel_size=69)

        # Ride Mode Mapping
        if 'MCU_Speed_Gear' in df.columns:
            df["Ride_Mode"] = df["MCU_Speed_Gear"].map(MODE_MAP)
        else:
            df["Ride_Mode"] = None

        # 6. Resample
        df_ts = df.set_index('timestamps').select_dtypes(include=[np.number])
        df_res = df_ts.resample(RESAMPLE_RATE).mean().interpolate(method='linear')

        # Resample Ride Mode
        if 'Ride_Mode' in df.columns and not df['Ride_Mode'].isnull().all():
            mode_series = df.set_index('timestamps')['Ride_Mode']
            mode_resampled = mode_series.resample(RESAMPLE_RATE).first()
            df_res['Ride_Mode'] = mode_resampled.ffill()
        else:
            df_res['Ride_Mode'] = None
        
        # 7. Filter Spikes (Power > 20kW)
        temp_power = df_res['Battery_voltage'] * df_res['Processed_Current']
        df_res = df_res[temp_power <= 20000] # 20kW Limit
        
        # 8. Metrics Calculation
        metrics = {}
        fixed_dt_hours = 0.1 / 3600.0
        
        # Power & Energy
        power_w = df_res['Battery_voltage'] * df_res['Processed_Current']
        energy_slice_wh = (power_w * fixed_dt_hours)
        metrics['Discharge_Wh'] = energy_slice_wh.clip(lower=0).sum()
        metrics['Regen_Wh'] = (-energy_slice_wh).clip(lower=0).sum()
        metrics['Net_Wh'] = metrics['Discharge_Wh'] - metrics['Regen_Wh']
        metrics['Net_Wh'] = metrics['Discharge_Wh'] - metrics['Regen_Wh']
        metrics['Max_Current_A'] = df_res['Processed_Current'].max()

        # Ride Mode Energy
        metrics['Suste_Wh'] = 0
        metrics['Thikka_Wh'] = 0
        metrics['Babbal_Wh'] = 0

        if 'Ride_Mode' in df_res.columns and not df_res['Ride_Mode'].isnull().all():
             # Calculate energy per mode (only positive Discharge energy)
             # Using the energy_slice_wh we already calculated
             df_res['Energy_Slice_Wh'] = energy_slice_wh
             
             # Group by logic
             # We want consumption (Discharge), so we filter where Energy > 0
             df_discharge = df_res[df_res['Energy_Slice_Wh'] > 0]
             
             if not df_discharge.empty:
                 mode_energy = df_discharge.groupby('Ride_Mode')['Energy_Slice_Wh'].sum()
                 metrics['Suste_Wh'] = mode_energy.get('Suste', 0)
                 metrics['Thikka_Wh'] = mode_energy.get('Thikka', 0)
                 metrics['Babbal_Wh'] = mode_energy.get('Babbal', 0)

        # Calculate Percentages
        total_discharge = metrics['Discharge_Wh']
        if total_discharge > 0:
            metrics['Suste_Wh_%'] = (metrics['Suste_Wh'] / total_discharge) * 100
            metrics['Thikka_Wh_%'] = (metrics['Thikka_Wh'] / total_discharge) * 100
            metrics['Babbal_Wh_%'] = (metrics['Babbal_Wh'] / total_discharge) * 100
        else:
            metrics['Suste_Wh_%'] = 0
            metrics['Thikka_Wh_%'] = 0
            metrics['Babbal_Wh_%'] = 0
        
        # Peak Power
        metrics['Peak_Power_kW'] = (df_res['Battery_voltage'] * df_res['Processed_Current']).max() / 1000.0
        
        # Distance
        metrics['Distance_km'] = 0
        if 'Current_Rotational_Speed' in df_res.columns:
            wheel_rpm = df_res['Current_Rotational_Speed'] / GEAR_RATIO
            speed_kmh = (wheel_rpm * 2 * np.pi / 60) * WHEEL_RADIUS_M * 3.6
            speed_kmh = np.where(speed_kmh > 100, 0, speed_kmh)
            
            # Calculate distance slice
            distance_slice_km = speed_kmh * fixed_dt_hours
            metrics['Distance_km'] = distance_slice_km.sum()
            metrics['Max_Speed_kmh'] = speed_kmh.max()
            metrics['Avg_Speed_kmh'] = speed_kmh.mean()

            # Ride Mode Distance
            if 'Ride_Mode' in df_res.columns and not df_res['Ride_Mode'].isnull().all():
                df_res['Distance_Slice_km'] = distance_slice_km
                mode_distance = df_res.groupby('Ride_Mode')['Distance_Slice_km'].sum()
                metrics['Suste_km'] = mode_distance.get('Suste', 0)
                metrics['Thikka_km'] = mode_distance.get('Thikka', 0)
                metrics['Babbal_km'] = mode_distance.get('Babbal', 0)
            else:
                metrics['Suste_km'] = 0
                metrics['Thikka_km'] = 0
                metrics['Babbal_km'] = 0
            
        # Efficiency
        if metrics['Distance_km'] > 0:
            metrics['Efficiency_Wh_km'] = metrics['Net_Wh'] / metrics['Distance_km']
        else:
            metrics['Efficiency_Wh_km'] = 0
            
        # Temperatures
        for sensor in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']:
            if sensor in df_res.columns:
                metrics[f'Max_{sensor}_C'] = df_res[sensor].max()
            else:
                metrics[f'Max_{sensor}_C'] = None

        # SOC Metrics
        if 'SOC' in df.columns:
             metrics['Start_SOC'] = df['SOC'].iloc[0]
             metrics['End_SOC'] = df['SOC'].iloc[-1]
             metrics['SOC_Diff'] = metrics['Start_SOC'] - metrics['End_SOC']
        else:
             metrics['Start_SOC'] = None
             metrics['End_SOC'] = None
             metrics['SOC_Diff'] = None
                
        metrics['Duration_Samples'] = len(df)
        
        return metrics
        
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

# ============================================================================
# GUI CLASS
# ============================================================================
class BatchSummaryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MULE Batch Summary Generator")
        self.root.geometry("700x500")
        
        # Frames
        self.frame_top = tk.Frame(root, pady=10)
        self.frame_top.pack()
        
        # Config UI
        tk.Label(self.frame_top, text="Root Directory:").grid(row=0, column=0, sticky="e")
        self.entry_root = tk.Entry(self.frame_top, width=60)
        self.entry_root.insert(0, ROOT_DIR_DEFAULT)
        self.entry_root.grid(row=0, column=1, padx=5)
        tk.Button(self.frame_top, text="Browse", command=self.browse_root).grid(row=0, column=2)
        
        tk.Button(self.frame_top, text="RUN BATCH SUMMARY", command=self.start_batch, 
                  bg="green", fg="white", font=("Arial", 12, "bold")).grid(row=2, column=0, columnspan=3, pady=15)
        
        # Log Area
        self.log_area = scrolledtext.ScrolledText(root, width=80, height=20, state=tk.DISABLED)
        self.log_area.pack(padx=10, pady=5)
        
        # Redirect Print
        self.logger = PrintLogger(self.log_area)
        sys.stdout = self.logger

    def browse_root(self):
        d = filedialog.askdirectory()
        if d:
            self.entry_root.delete(0, tk.END)
            self.entry_root.insert(0, d)

    def start_batch(self):
        threading.Thread(target=self.run_process).start()

    def run_process(self):
        root_dir = self.entry_root.get()
        output_dir = OUTPUT_ROOT_DEFAULT
        
        print("\n" + "="*60)
        print("STARTING BATCH SUMMARY GENERATION")
        print("="*60)
        print(f"Scanning: {root_dir}")
        
        # 1. SCAN PHASE
        files_to_process = []
        
        for root, _, files in os.walk(root_dir):
            for file in files:
                # Updated to check for MULE1 or MULE2 to be more flexible
                if (file.lower().startswith("mule1_") or file.lower().startswith("mule2_")) and file.lower().endswith(".csv") and "charging" not in file.lower():
                    full_path = os.path.join(root, file)
                    date_str, shift_str = extract_metadata(file)
                    
                    if date_str:
                        # Attempt to parse date for sorting
                        try:
                            date_obj = pd.to_datetime(date_str, format="%m-%d-%Y", errors='coerce')
                            if pd.isna(date_obj):
                                date_obj = pd.to_datetime(date_str, format="%d-%m-%Y", errors='coerce')
                        except:
                            date_obj = pd.Timestamp.max # Push to end if invalid
                            
                        files_to_process.append({
                            'path': full_path,
                            'filename': file,
                            'date_str': date_str,
                            'shift_str': shift_str,
                            'date_obj': date_obj
                        })

        if not files_to_process:
            print("\n❌ No valid files found.")
            return

        # 2. SORT PHASE
        # Sort by Date then Shift
        files_to_process.sort(key=lambda x: (x['date_obj'], x['shift_str']))
        
        print(f"Found {len(files_to_process)} files. Processing in chronological order...\n")

        # 3. PROCESS PHASE
        processed_data = []
        
        for i, item in enumerate(files_to_process, 1):
            print(f"[{i}/{len(files_to_process)}] Processing: {item['filename']} ... ", end="")
            
            metrics = process_single_file(item['path'])
            
            if metrics:
                metrics['Filename'] = item['filename']
                metrics['Date'] = item['date_str']
                metrics['Shift'] = item['shift_str']
                processed_data.append(metrics)
                print("DONE")
            else:
                print("FAILED")
                        
        if not processed_data:
            print("\n❌ No files were successfully processed.")
            return
            
        # Create DataFrame
        df_summary = pd.DataFrame(processed_data)
            
        # Reorder columns
        cols = ['Date', 'Shift', 'Filename', 'Distance_km', 'Efficiency_Wh_km', 
                'Net_Wh', 
                'Suste_Wh', 'Suste_Wh_%', 'Suste_km',
                'Thikka_Wh', 'Thikka_Wh_%', 'Thikka_km',
                'Babbal_Wh', 'Babbal_Wh_%', 'Babbal_km',
                'Discharge_Wh', 'Regen_Wh', 'Max_Current_A', 'Peak_Power_kW',
                'Start_SOC', 'End_SOC', 'SOC_Diff',
                'Max_Ntc_Mos_C', 'Max_Ntc_Com_C', 'Max_Ntc_3_C']
        
        # Add remaining cols
        remaining = [c for c in df_summary.columns if c not in cols]
        df_summary = df_summary[cols + remaining]
        
        # Round all numeric columns to 2 decimals
        df_summary = df_summary.round(2)
        
        # Save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_path = os.path.join(output_dir, OUTPUT_FILENAME)
        df_summary.to_csv(out_path, index=False)
        
        print("\n" + "="*60)
        print(f"✅ BATCH SUMMARY COMPLETE")
        print(f"Files Processed: {len(df_summary)}")
        print(f"Saved to: {out_path}")
        print("="*60)

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchSummaryGUI(root)
    root.mainloop()
