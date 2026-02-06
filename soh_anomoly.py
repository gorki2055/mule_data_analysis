"""
================================================================================
SOH ANOMALY DETECTION & DIAGNOSTIC TOOL
================================================================================

PURPOSE:
    This tool automates the scanning of battery log files (CSV) to identify 
    unnatural shifts in State of Health (SOH). It distinguishes between 
    expected battery degradation and sudden anomalies (drops or jumps) that 
    may indicate BMS errors, sensor noise, or cell issues.

LOGIC FLOW:
    1. Input: Selects data via GUI or Command Line Interface (CLI).
    2. Alignment: Dynamically maps column aliases for Voltage and Current.
    3. Detection: Uses a sliding window to compare SOH values over a 
       user-defined 'Lag' period.
    4. Validation: Implements a 'Stability Check'—if the SOH returns to 
       baseline within the 'Window' period, it is flagged as noise and ignored.
    5. Visualization: Generates 3-panel PNG reports (SOH, Voltage, Current) 
       for every detected anomaly to assist in root-cause analysis.

USAGE:
    Run the script directly to open the File/Folder selector:
        python <script_name>.py

    Or use CLI arguments for batch processing:
        python <script_name>.py --root "./logs" --threshold 1.5 --window 60

OUTPUTS:
    - CSV Report: 'mule_soh_anomalies.csv' (List of all detected events)
    - Graphs: 'soh_anomaly_graph/' (Visual plots for each event)

DEPENDENCIES:
    pandas, numpy, matplotlib, tkinter
================================================================================
"""
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import filedialog, messagebox

# Configuration
DEFAULT_THRESHOLD = 1.0     # 1% Drop/Jump
DEFAULT_WINDOW = 30         # seconds
LAG_TIME = 1.0              # seconds
GRAPH_OUTPUT_DIR = "soh_anomaly_graph"

# Possible column names to look for (Case insensitive)
VOLTAGE_ALIASES = ['Battery_voltage', 'voltage', 'v_bat', 'pack_v']
CURRENT_ALIASES = ['Battery_current1', 'current', 'i_bat', 'pack_i']

def find_column(df, aliases):
    """Helper to find the first matching column name from a list of aliases."""
    cols_lower = [c.lower() for c in df.columns]
    for alias in aliases:
        if alias.lower() in cols_lower:
            return df.columns[cols_lower.index(alias.lower())]
            return df.columns[cols_lower.index(alias)]
    return None

def find_csv_files(root_dir):
    csv_files = []
    if os.path.isfile(root_dir):
        return [root_dir]
        
    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv") and "mule" in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files

def select_input_gui():
    """GUI to select folder or file."""
    root = tk.Tk()
    root.title("SOH Anomaly Detection")
    # Center the window
    window_width = 300
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    selection = {"path": None, "type": None}
    
    def select_folder():
        path = filedialog.askdirectory(title="Select Folder Containing Logs")
        if path:
            selection["path"] = path
            selection["type"] = "folder"
            root.destroy()

    def select_file():
        path = filedialog.askopenfilename(title="Select Single CSV Log", filetypes=[("CSV Files", "*.csv")])
        if path:
            selection["path"] = path
            selection["type"] = "file"
            root.destroy()

    tk.Label(root, text="Select Input Data Source", font=("Arial", 12)).pack(pady=10)
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    
    tk.Button(btn_frame, text="Select Folder", command=select_folder, width=15, height=2).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Select File", command=select_file, width=15, height=2).pack(side=tk.LEFT, padx=5)
    
    root.mainloop()
    return selection["path"]

def plot_anomalies(df, anomalies, filename, output_folder, volt_col, curr_col):
    """Generates a 3-panel plot (SOH, Voltage, Current)."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Create 3 subplots sharing the X-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # --- Plot 1: SOH ---
    ax1.plot(df['timestamps'], df['SOH'], label='SOH', color='#1f77b4', linewidth=1.5)
    
    # Highlight Anomalies
    anomaly_times = anomalies['timestamps']
    anomaly_values = anomalies['After_SOH']
    ax1.scatter(anomaly_times, anomaly_values, color='red', s=60, zorder=5, label='Anomaly')
    
    ax1.set_ylabel("SOH (%)")
    ax1.set_title(f"Anomaly Analysis: {filename}", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')

    # --- Plot 2: Voltage ---
    if volt_col:
        ax2.plot(df['timestamps'], df[volt_col], label='Voltage', color='#2ca02c', linewidth=1)
        ax2.set_ylabel(f"Voltage ({volt_col})")
        
        # Mark the anomaly time on the voltage graph too
        for at in anomaly_times:
            ax2.axvline(x=at, color='red', linestyle=':', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, "Voltage Data Not Found", ha='center', transform=ax2.transAxes)

    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 3: Current ---
    if curr_col:
        ax3.plot(df['timestamps'], df[curr_col], label='Current', color='#d62728', linewidth=1)
        ax3.set_ylabel(f"Current ({curr_col})")
        
        # Mark the anomaly time on the current graph too
        for at in anomaly_times:
            ax3.axvline(x=at, color='red', linestyle=':', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "Current Data Not Found", ha='center', transform=ax3.transAxes)

    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xlabel("Time (UTC)")

    # Formatting Dates
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, f"{filename}_Analysis.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"    -> Graph saved: {save_path}")

def process_file_sliding(file_path, threshold, valid_window, lag_seconds):
    try:
        try:
            df = pd.read_csv(file_path, engine='python', low_memory=False)
        except:
            df = pd.read_csv(file_path, low_memory=False)
        
        df.columns = df.columns.str.strip()
        
        if 'timestamps' not in df.columns or 'SOH' not in df.columns:
            return None, None, None, None

        # Identify Voltage/Current columns dynamically
        volt_col = find_column(df, VOLTAGE_ALIASES)
        curr_col = find_column(df, CURRENT_ALIASES)

        # --- Voltage Processing (Reference: plotv14.py) ---
        if volt_col:
            # Convert mV to V if mean > 1000
            if df[volt_col].mean() > 1000:
                df[volt_col] = df[volt_col] / 1000.0
            
            # Filter Voltage Range [20, 120]
            df = df[(df[volt_col] >= 20) & (df[volt_col] <= 120)]
        
        # --- Current Processing (Reference: plotv14.py) ---
        if curr_col:
            # Handle Charge/Discharge Status if present
            if 'Charge_discharge_status' in df.columns:
                # Force negative sign for charging (1 = Charging)
                # Using vectorized approach for speed instead of apply
                mask_charging = df['Charge_discharge_status'] == 1
                df.loc[mask_charging, curr_col] = -df.loc[mask_charging, curr_col].abs()
            
            # Filter Current Range [-50, 185]
            df = df[(df[curr_col] >= -50) & (df[curr_col] <= 185)]
            
        # --- Power Filtering (Reference: plotv14.py) ---
        if volt_col and curr_col:
            # Calculate tentative power for filtering spikes
            temp_power = df[volt_col] * df[curr_col]
            # Filter Rows > 20kW (20000W) - Spikes/Noise
            df = df[temp_power <= 20000]

        # --- Filter Invalid SOH Data ---
        df = df[(df['SOH'] >= 0) & (df['SOH'] <= 100)]
        
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        if df.empty: return None, None, None, None

        times = df['timestamps'].values
        sohs = df['SOH'].values
        n = len(times)
        
        lag_ns = np.timedelta64(int(lag_seconds * 1_000_000_000), 'ns')
        window_ns = np.timedelta64(int(valid_window * 1_000_000_000), 'ns')
        
        anomalies = []
        last_event_time = None
        
        left = 0
        for right in range(n):
            current_time = times[right]
            
            while left < right and (current_time - times[left]) > lag_ns:
                left += 1
            
            # 4a. BACKWARD STABILITY CHECK
            if left > 0:
                prev_soh = sohs[left-1]
                time_gap_prev = times[left] - times[left-1]
                if time_gap_prev < (lag_ns * 2):
                    if abs(sohs[left] - prev_soh) > threshold:
                        continue 

            start_soh = sohs[left]
            current_soh = sohs[right]
            diff = current_soh - start_soh
            
            if abs(diff) > threshold:
                if last_event_time is not None:
                    if (current_time - last_event_time) <= window_ns:
                        continue
                
                valid_end_time = current_time + window_ns
                end_idx = np.searchsorted(times, valid_end_time, side='right')
                future_sohs = sohs[right+1 : end_idx]
                
                is_noise = False
                if len(future_sohs) > 0:
                    dist_to_start = np.abs(future_sohs - start_soh)
                    if np.any(dist_to_start < threshold):
                        is_noise = True
                
                if not is_noise:
                    anomalies.append({
                        'timestamps': current_time,
                        'Filename': os.path.basename(file_path),
                        'Anomaly_Type': "Drop" if diff < 0 else "Increase",
                        'Before_SOH': start_soh,
                        'After_SOH': current_soh,
                        'Magnitude': abs(diff),
                        'Full_Path': file_path
                    })
                    last_event_time = current_time

        if anomalies:
            return pd.DataFrame(anomalies), df, volt_col, curr_col
        
    except Exception as e:
        print(f"Error {os.path.basename(file_path)}: {e}")
        return None, None, None, None
    return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="High-Frequency SOH Anomaly Detection & Plotting")
    # Removed default root to trigger GUI if not provided
    parser.add_argument("--root", "-r", type=str, default=None, help="Root directory or single file path")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Drop %")
    parser.add_argument("--lag", "-l", type=float, default=LAG_TIME, help="Detection lag (seconds)")
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Validation window (seconds)")
    parser.add_argument("--output", "-o", type=str, default="mule_soh_anomalies.csv")
    
    args = parser.parse_args()
    
    target_path = args.root
    
    # If no root argument provided, Launch GUI
    if target_path is None:
        target_path = select_input_gui()
    
    if not target_path:
        print("No input selected. Exiting.")
        return

    all_anomalies = []
    
    # Logic to handle if target_path is a file vs directory
    files = find_csv_files(target_path)
    
    if not files:
        print("No Valid CSV Files Found.")
        return

    count = 0 
    for f in files:
        print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
        
        res_anomalies, res_df, v_col, i_col = process_file_sliding(f, args.threshold, args.window, args.lag)
        
        if res_anomalies is not None and not res_anomalies.empty:
             all_anomalies.append(res_anomalies)
             print(f"\n[!] FOUND SOH ANOMALY in {os.path.basename(f)}")
             
             # --- GENERATE PLOT (SOH + Voltage + Current) ---
             plot_anomalies(res_df, res_anomalies, os.path.basename(f), GRAPH_OUTPUT_DIR, v_col, i_col)
             
        count += 1
        
    print("\nProcessing Complete.")
    
    if all_anomalies:
        final_df = pd.concat(all_anomalies, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print(f"Report saved: {args.output}")
        print(f"Graphs saved in folder: {GRAPH_OUTPUT_DIR}")
    else:
        print("No SOH anomalies found.")

if __name__ == "__main__":
    main()