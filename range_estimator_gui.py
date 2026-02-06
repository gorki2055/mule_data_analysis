# Range Estimator: Calculation Logic
# This document details how 
# range_estimator_gui.py
#  processes raw data to calculate range, energy, and efficiency.

# 1. Session Identification
# The program first separates the data into valid "riding sessions" to ensure we don't include charging time in the analysis.

# Input: charger_connected column (values like b'connected' or b'not connected').
# Logic:
# Scans the file for all rows where the charger is NOT connected.
# Groups these consecutive rows into "segments".
# SOC Check: For each segment, it calculates Start_SOC - End_SOC.
# Filtering: Only segments with an SOC Drop ≥ 90% (or user-defined value) are kept for analysis.
# 2. metrics Calculation (Per Segment)
# Once a valid segment is found, the data is resampled to a fixed 100ms interval (0.1s) to ensure accurate integration over time.

# A. Energy (Wh)
# Energy is calculated by integrating instantaneous power over time.

# Instantaneous Power (W): $$ P_t = \text{Voltage}_t \times \text{Current}_t $$ (Note: Current spikes > 185A and Power spikes > 20kW are filtered out before this step)

# Energy Increments (Wh): $$ E_{\text{step}} = P_t \times \frac{0.1 \text{ sec}}{3600 \text{ sec/hr}} $$

# Net Energy:

# Discharge Wh: Sum of all positive $E_{\text{step}}$.
# Regen Wh: Sum of all negative $E_{\text{step}}$ (absolute value).
# Net Wh: $\text{Discharge Wh} - \text{Regen Wh}$
# B. Distance (km)
# Distance is calculated from wheel speed, not GPS, for higher consistency.

# Wheel Speed (km/h): $$ \text{Speed} = \left( \frac{\text{Motor RPM}}{\text{Gear Ratio}} \right) \times \text{Wheel Circumference} \times \text{Const} $$

# Gear Ratio: 6.09
# Wheel Radius: 0.261 m
# Const: Conversion factors for RPM to km/h.
# Distance Increment: $$ D_{\text{step}} = \text{Speed}_t \times \frac{0.1 \text{ sec}}{3600} $$

# Total Distance: Sum of all $D_{\text{step}}$.
#
# Range_km:
# This metric explicitly reports the distance covered strictly within the period
# defined by the Start_SOC and End_SOC of the session.
# Formula: Sum of distance steps between the timestamp of Start_SOC and timestamp of End_SOC.

# C. Efficiency (Wh/km)
# $$ \text{Efficiency} = \frac{\text{Net Energy (Wh)}}{\text{Total Distance (km)}} $$

# 3. Ride Mode Analysis
# The program splits the energy and distance totals based on the active Ride Mode (Suste, Thikka, Babbal).

# Logic: For every 100ms time step, the energy consumed and distance traveled are attributed to the Ride_Mode active at that specific moment.
# Output: % of Total Energy consumed in each mode. $$ \text{Mode %} = \frac{\text{Energy consumed in Mode}}{\text{Total Discharge Energy}} \times 100 $$

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import re
import threading
import sys
from scipy.signal import medfilt

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3877D237"
OUTPUT_ROOT_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS"
OUTPUT_FILENAME = "range_test_summary_mule2_90%_.csv"

GEAR_RATIO = 6.09
WHEEL_RADIUS_M = 0.261
RESAMPLE_RATE = '100ms'
MIN_SOC_DROP = 90.0

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
    """
    pattern = r"MULE\d+_[-]*(\d+-\d+-\d+)_([A-Za-z_]+)\.csv"
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).upper()
    return None, None

def find_valid_segments(df):
    """
    Identifies segments where charger is NOT connected.
    Returns list of (start_idx, end_idx) tuples.
    """
    # Create mask: True if connected, False if not
    # Handles byte strings and regular strings just in case
    # User said: "charger_connected" -> b'connected' or b'not connected'
    
    # fuzzy match for charger_connected
    charger_col = None
    if 'charger_connected' in df.columns:
        charger_col = 'charger_connected'
    else:
        # Case insensitive search
        for c in df.columns:
            if 'charger' in c.lower() and 'connected' in c.lower():
                charger_col = c
                break
    
    if not charger_col:
        return []

    # Clean column content
    # Handle b'...' string representation if it was loaded as string "b'connected'"
    # or actual bytes if loaded differently.
    
    def is_connected(val):
        s = str(val).lower()
        return "not connected" not in s and "connected" in s

    # Vectorized check might be complex with mixed types, let's try mapping
    is_charging = df[charger_col].apply(is_connected)
    
    # We want segments where is_charging is FALSE
    valid_mask = ~is_charging
    
    # Find contiguous groups of True
    # diff() gives True where value changes
    # cumsum() gives group IDs
    
    # We only care about valid_mask being True
    # If the whole file is valid, return one segment
    if valid_mask.all():
        return [(df.index[0], df.index[-1])]
    
    if not valid_mask.any():
        return []

    # Identify changes
    # 0 = invalid (connected), 1 = valid (not connected)
    # diff != 0 means transition
    
    # Get indices where mask is True
    valid_indices = df.index[valid_mask]
    
    if len(valid_indices) == 0:
        return []

    # Group consecutive indices
    # We can use a trick: index - counter. If consecutive, difference is constant.
    from itertools import groupby
    from operator import itemgetter
    
    segments = []
    for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        segments.append((group[0], group[-1]))
        
    return segments

def process_segment(df, start_idx, end_idx, file_metrics_base):
    """
    Process a specific segment of the dataframe.
    Returns metrics dict if SOC drop >= MIN_SOC_DROP, else None.
    """
    # Slice the dataframe with .loc (inclusive) or iloc
    # The indices from find_valid_segments are label-based if logical, but let's assume default integer index
    segment = df.loc[start_idx:end_idx].copy()
    
    if len(segment) < 100: # Ignore tiny segments
        return None
        
    # Check SOC Drop
    if 'SOC' not in segment.columns:
        return None
        
    start_soc = segment['SOC'].iloc[0]
    end_soc = segment['SOC'].iloc[-1]
    soc_drop = start_soc - end_soc
    
    if soc_drop < MIN_SOC_DROP:
        return None

    print(f"  -> Found Candidate: SOC Drop {soc_drop:.1f}%")

    # Proceed with full calculation for this segment
    metrics = file_metrics_base.copy()
    metrics['Start_SOC'] = start_soc
    metrics['End_SOC'] = end_soc
    metrics['SOC_Diff'] = soc_drop
    metrics['Segment_Start'] = segment['timestamps'].iloc[0]
    metrics['Segment_End'] = segment['timestamps'].iloc[-1]
    
    # 1. Resample Segment (for accurate integration)
    df_ts = segment.set_index('timestamps').select_dtypes(include=[np.number])
    # Ensure no duplicate index
    if df_ts.index.duplicated().any():
        df_ts = df_ts[~df_ts.index.duplicated()]
        
    df_res = df_ts.resample(RESAMPLE_RATE).mean().interpolate(method='linear')

    # Ride Mode Handling
    if 'Ride_Mode' in segment.columns:
        mode_series = segment.set_index('timestamps')['Ride_Mode']
        if mode_series.index.duplicated().any():
            mode_series = mode_series[~mode_series.index.duplicated()]
        mode_resampled = mode_series.resample(RESAMPLE_RATE).first().ffill()
        df_res['Ride_Mode'] = mode_resampled
    else:
        df_res['Ride_Mode'] = None

    # Filter Spikes (Power > 20kW)
    if 'Battery_voltage' in df_res.columns and 'Processed_Current' in df_res.columns:
         temp_power = df_res['Battery_voltage'] * df_res['Processed_Current']
         df_res = df_res[temp_power <= 20000]

    # Metrics Calculation
    fixed_dt_hours = 0.1 / 3600.0
    
    # Power & Energy
    if 'Battery_voltage' in df_res.columns and 'Processed_Current' in df_res.columns:
        power_w = df_res['Battery_voltage'] * df_res['Processed_Current']
        energy_slice_wh = (power_w * fixed_dt_hours)
        
        metrics['Discharge_Wh'] = energy_slice_wh.clip(lower=0).sum()
        metrics['Regen_Wh'] = (-energy_slice_wh).clip(lower=0).sum()
        metrics['Net_Wh'] = metrics['Discharge_Wh'] - metrics['Regen_Wh']
        metrics['Max_Current_A'] = df_res['Processed_Current'].max()
        metrics['Peak_Power_kW'] = power_w.max() / 1000.0

        # Ride Mode Energy
        metrics['Suste_Wh'] = 0
        metrics['Thikka_Wh'] = 0
        metrics['Babbal_Wh'] = 0
        
        if 'Ride_Mode' in df_res.columns and not df_res['Ride_Mode'].isnull().all():
             df_res['Energy_Slice_Wh'] = energy_slice_wh
             df_discharge = df_res[df_res['Energy_Slice_Wh'] > 0]
             
             if not df_discharge.empty:
                 mode_energy = df_discharge.groupby('Ride_Mode')['Energy_Slice_Wh'].sum()
                 metrics['Suste_Wh'] = mode_energy.get('Suste', 0)
                 metrics['Thikka_Wh'] = mode_energy.get('Thikka', 0)
                 metrics['Babbal_Wh'] = mode_energy.get('Babbal', 0)
        
        total_discharge = metrics['Discharge_Wh']
        if total_discharge > 0:
            metrics['Suste_Wh_%'] = (metrics['Suste_Wh'] / total_discharge) * 100
            metrics['Thikka_Wh_%'] = (metrics['Thikka_Wh'] / total_discharge) * 100
            metrics['Babbal_Wh_%'] = (metrics['Babbal_Wh'] / total_discharge) * 100
        else:
            metrics['Suste_Wh_%'] = 0; metrics['Thikka_Wh_%'] = 0; metrics['Babbal_Wh_%'] = 0

    # Distance
    metrics['Distance_km'] = 0
    if 'Current_Rotational_Speed' in df_res.columns:
        wheel_rpm = df_res['Current_Rotational_Speed'] / GEAR_RATIO
        speed_kmh = (wheel_rpm * 2 * np.pi / 60) * WHEEL_RADIUS_M * 3.6
        speed_kmh = np.where(speed_kmh > 100, 0, speed_kmh)
        
        distance_slice_km = speed_kmh * fixed_dt_hours
        metrics['Distance_km'] = distance_slice_km.sum()
        
        # Range_km: Distance covered strictly within this session's Start SOC and End SOC period.
        # This matches the user requirement: "distance covered between this period soc_drop = start_soc - end_soc"
        metrics['Range_km'] = metrics['Distance_km']
        
        metrics['Avg_Speed_kmh'] = speed_kmh.mean()

        if 'Ride_Mode' in df_res.columns:
            df_res['Distance_Slice_km'] = distance_slice_km
            mode_distance = df_res.groupby('Ride_Mode')['Distance_Slice_km'].sum()
            metrics['Suste_km'] = mode_distance.get('Suste', 0)
            metrics['Thikka_km'] = mode_distance.get('Thikka', 0)
            metrics['Babbal_km'] = mode_distance.get('Babbal', 0)
        else:
            metrics['Suste_km'] = 0; metrics['Thikka_km'] = 0; metrics['Babbal_km'] = 0

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
            
    metrics['Duration_Samples'] = len(segment)
    return metrics


def process_file_ranges(file_path):
    """
    Process a file to find all valid range test segments.
    """
    try:
        # Load heavy data (only columns we need initially to speed up?) 
        # No, we need most columns for metrics.
        try:
            df = pd.read_csv(file_path, engine='c', low_memory=False)
        except:
             df = pd.read_csv(file_path, engine='python', low_memory=False)
             
        df.columns = df.columns.str.strip()
        
        # Pre-process vital columns
        if 'Battery_voltage' in df.columns and df['Battery_voltage'].mean() > 1000:
             df['Battery_voltage'] = df['Battery_voltage'] / 1000.0
             
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps').reset_index(drop=True)
        
        if 'Battery_current1' in df.columns:
            df['Processed_Current'] = df['Battery_current1'].values
            if 'Charge_discharge_status' in df.columns:
                charging_mask = df['Charge_discharge_status'] == 1
                df.loc[charging_mask, 'Processed_Current'] = -np.abs(df.loc[charging_mask, 'Battery_current1'].values)
        else:
            return []

        # Filter Current Limits so they don't spoil average
        df = df[df['Processed_Current'] <= 185]
        df = df[df['Processed_Current'] >= -50]

        # Filter Temp Spikes
        for sensor in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']:
            if sensor in df.columns:
                df[sensor] = medfilt(df[sensor], kernel_size=69)

        if 'MCU_Speed_Gear' in df.columns:
            df["Ride_Mode"] = df["MCU_Speed_Gear"].map(MODE_MAP)

        # FIND SEGMENTS
        segments = find_valid_segments(df)
        
        base_metrics = {}
        # Pre-extract file metadata
        date, shift = extract_metadata(os.path.basename(file_path))
        base_metrics['Filename'] = os.path.basename(file_path)
        base_metrics['Date'] = date
        base_metrics['Shift'] = shift

        results = []
        for start, end in segments:
            res = process_segment(df, start, end, base_metrics)
            if res:
                results.append(res)
                
        return results

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return []

# ============================================================================
# GUI CLASS
# ============================================================================
class RangeEstimatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Range Test Analyzer (SOC >= 90%)")
        self.root.geometry("800x600")
        
        # Frames
        self.frame_top = tk.Frame(root, pady=10)
        self.frame_top.pack()
        
        # Config UI
        tk.Label(self.frame_top, text="Root Directory:").grid(row=0, column=0, sticky="e")
        self.entry_root = tk.Entry(self.frame_top, width=60)
        self.entry_root.insert(0, ROOT_DIR_DEFAULT)
        self.entry_root.grid(row=0, column=1, padx=5)
        tk.Button(self.frame_top, text="Browse", command=self.browse_root).grid(row=0, column=2)
        
        tk.Label(self.frame_top, text="Min SOC Drop (%):").grid(row=1, column=0, sticky="e")
        self.entry_soc = tk.Entry(self.frame_top, width=10)
        self.entry_soc.insert(0, str(MIN_SOC_DROP))
        self.entry_soc.grid(row=1, column=1, sticky="w", padx=5)

        tk.Button(self.frame_top, text="FIND RANGE SESSIONS", command=self.start_batch, 
                  bg="blue", fg="white", font=("Arial", 12, "bold")).grid(row=2, column=0, columnspan=3, pady=15)
        
        # Log Area
        self.log_area = scrolledtext.ScrolledText(root, width=90, height=25, state=tk.DISABLED)
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
        # Update global constant from UI
        global MIN_SOC_DROP
        try:
            MIN_SOC_DROP = float(self.entry_soc.get())
        except:
            messagebox.showerror("Error", "Invalid SOC Drop value")
            return
            
        threading.Thread(target=self.run_process).start()

    def run_process(self):
        root_dir = self.entry_root.get()
        output_dir = OUTPUT_ROOT_DEFAULT
        
        print("\n" + "="*60)
        print("STARTING RANGE SESSION ANALYSIS")
        print(f"Criteria: SOC Drop >= {MIN_SOC_DROP}% | No Charger Connected")
        print("="*60)
        print(f"Scanning: {root_dir}")
        
        # 1. SCAN PHASE
        files_to_process = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if (file.lower().startswith("mule1_") or file.lower().startswith("mule2_")) and file.lower().endswith(".csv") and "charging" not in file.lower():
                    full_path = os.path.join(root, file)
                    date_str, shift_str = extract_metadata(file)
                    # Simplified date sorting logic
                    try:
                        date_obj = pd.to_datetime(date_str, format="%m-%d-%Y", errors='coerce')
                    except:
                        date_obj = pd.Timestamp.max

                    files_to_process.append({
                        'path': full_path,
                        'filename': file,
                        'date_obj': date_obj,
                        'shift': shift_str
                    })

        files_to_process.sort(key=lambda x: (x['date_obj'], x['shift']))
        print(f"Found {len(files_to_process)} potentially relevant files.")

        # 2. PROCESS PHASE
        all_range_sessions = []
        
        for i, item in enumerate(files_to_process, 1):
            print(f"[{i}/{len(files_to_process)}] Checking: {item['filename']} ... ", end="")
            
            sessions = process_file_ranges(item['path'])
            
            if sessions:
                print(f"FOUND {len(sessions)} Valid Session(s)!")
                all_range_sessions.extend(sessions)
            else:
                print("No match")
                        
        if not all_range_sessions:
            print("\n❌ No range sessions found meeting the criteria.")
            return
            
        # 3. OUTPUT PHASE
        df_summary = pd.DataFrame(all_range_sessions)
            
        # Reorder columns
        # Reorder columns
        cols = ['Date', 'Shift', 'Filename', 'Segment_Start', 'Segment_End', 
                'Distance_km', 'Range_km', 'Efficiency_Wh_km', 'Net_Wh', 
                'Start_SOC', 'End_SOC', 'SOC_Diff',
                'Avg_Speed_kmh',
                'Suste_Wh_%', 'Thikka_Wh_%', 'Babbal_Wh_%',
                'Suste_km', 'Thikka_km', 'Babbal_km',
                'Max_Ntc_Mos_C', 'Max_Ntc_Com_C', 'Max_Ntc_3_C']
        
        final_cols = [c for c in cols if c in df_summary.columns]
        remaining = [c for c in df_summary.columns if c not in final_cols]
        df_summary = df_summary[final_cols + remaining]
        
        df_summary = df_summary.round(2)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        out_path = os.path.join(output_dir, OUTPUT_FILENAME)
        df_summary.to_csv(out_path, index=False)
        
        print("\n" + "="*60)
        print(f"✅ RANGE ANALYSIS COMPLETE")
        print(f"Sessions Found: {len(df_summary)}")
        print(f"Saved to: {out_path}")
        print("="*60)

if __name__ == "__main__":
    root = tk.Tk()
    app = RangeEstimatorGUI(root)
    root.mainloop()
