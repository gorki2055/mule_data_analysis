# This code is designed to detect sudden, sharp changes (anomalies) in State of Charge (SOC) data within log files. Instead of looking for slow battery drain, it looks for "step changes"—where the battery percentage jumps or drops significantly within a very short time (e.g., 1 second).

# Here is the step-by-step logic:

# 1. File Discovery (find_csv_files)
# What it does: Scans a specific folder recursively.

# Filter: It looks only for files that end in .csv AND contain the word "mule" in the name (filtering for specific device logs).

# 2. Data Cleaning (process_file_sliding)
# Loading: Reads the CSV file into memory.

# Sanity Check:

# Invalid Range: It removes any rows where SOC is > 100 or < 0 (the specific request you made).

# Timestamps: It converts the time column to datetime objects and sorts the data chronologically. If timestamps are missing, those rows are dropped.

# 3. The "Two-Pointer" Sliding Window
# This is the core engine of the script. It uses a highly efficient method to check for drops without iterating through the data thousands of times.

# The Setup: Imagine two pointers, Left and Right, moving through your timeline.

# Right Pointer: Represents "Now" (the current data point being checked).

# Left Pointer: Represents "1 Second Ago" (the comparison point).

# The Movement:

# As Right moves forward, Left chases it. The code ensures the time gap between Right and Left is exactly the lag_seconds (default 1.0s).

# The Check:

# It calculates Difference = SOC[Right] - SOC[Left].

# If the difference is bigger than the threshold (default 3%), it flags a potential event.

# 4. Noise Filtering & Validation
# Real-world sensor data is noisy. A battery might read "90% -> 50% -> 90%" in one second due to a glitch. The code prevents flagging this as a real drop using two techniques:

# Look-Ahead Validation:

# When a drop is detected, the code looks at the next 30 seconds of data.

# If the SOC bounces back to the original value (start_soc) within that window, the code marks it as Noise and ignores it.

# If the SOC stays dropped, it is considered a Valid Event.

# Deduplication (Debouncing):

# If a valid drop is found, the code stops looking for new drops for the next 30 seconds (valid_window). This prevents one big drop from generating 50 duplicate error alerts.

# 5. Output
# It collects all confirmed anomalies (timestamp, magnitude, filename).

# It saves a single summary CSV report (mule1_soc_anomaly_highfreq.csv) containing all events found across all files.

# Why this logic is efficient:
# Numpy Arrays: It converts data to Numpy arrays (lines 44-45), which is roughly 100x faster than standard Python loops.

# One Pass: It only iterates through the data once (O(N) complexity) rather than re-scanning for every point.


import pandas as pd
import os
import argparse
import numpy as np

# Configuration
DEFAULT_ROOT = r"D:\kushal\LOG\LOG\2877D237"
DEFAULT_THRESHOLD = 3.0
DEFAULT_WINDOW = 60 # seconds (Validation window)
LAG_TIME = 1.0      # seconds (Look-back time for drop detection)

def find_csv_files(root_dir):
    csv_files = []
    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv") and "mule" in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_file_sliding(file_path, threshold, valid_window, lag_seconds):
    try:
        # 1. Fast Load
        try:
            df = pd.read_csv(file_path, engine='python', low_memory=False)
        except:
            df = pd.read_csv(file_path, low_memory=False)
        
        df.columns = df.columns.str.strip()
        if 'timestamps' not in df.columns or 'SOC' not in df.columns:
            return None

        # --- [MODIFIED START] Filter Invalid SOC Data ---
        # Keep only rows where SOC is between 0 and 100 (inclusive)
        df = df[(df['SOC'] >= 0) & (df['SOC'] <= 100)]
        
        if df.empty: return None
        # --- [MODIFIED END] ---

        # 2. Parse & Sort
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        if df.empty: return None

        # 3. Prepare Arrays (Numpy is 100x faster than Pandas iterrows)
        times = df['timestamps'].values
        socs = df['SOC'].values
        n = len(times)
        
        # Convert lag/window to nanoseconds (int64) for direct comparison
        lag_ns = np.timedelta64(int(lag_seconds * 1_000_000_000), 'ns')
        window_ns = np.timedelta64(int(valid_window * 1_000_000_000), 'ns')
        
        anomalies = []
        last_event_time = None
        
        # --- SLIDING WINDOW LOGIC (Two Pointers) ---
        # 'left' pointer tracks the time ~1 second ago
        left = 0
        
        for right in range(n):
            current_time = times[right]
            
            # 4. Advance 'left' pointer until it is approximately 'lag_seconds' behind 'right'
            # We want: times[right] - times[left] <= lag_ns
            while left < right and (current_time - times[left]) > lag_ns:
                left += 1
            
            # Calculate Drop over this window
            start_soc = socs[left]
            current_soc = socs[right]
            diff = current_soc - start_soc
            
            # 5. Threshold Check
            if abs(diff) > threshold:
                
                # --- DEDUPLICATION ---
                if last_event_time is not None:
                    if (current_time - last_event_time) <= window_ns:
                        continue
                
                # --- VALIDATION (Look Ahead) ---
                # Check if it's noise: Does SOC return to 'start_soc' in the next 30s?
                
                valid_end_time = current_time + window_ns
                end_idx = np.searchsorted(times, valid_end_time, side='right')
                
                # Slice future data
                future_socs = socs[right+1 : end_idx]
                
                is_noise = False
                if len(future_socs) > 0:
                    # Check distance to the ORIGINAL starting value
                    dist_to_start = np.abs(future_socs - start_soc)
                    if np.any(dist_to_start < threshold):
                        is_noise = True
                
                if not is_noise:
                    # VALID EVENT
                    anomaly_type = "Drop" if diff < 0 else "Increase"
                    
                    anomalies.append({
                        'timestamps': current_time,
                        'Filename': os.path.basename(file_path),
                        'Anomaly_Type': anomaly_type,
                        'Before_SOC': start_soc,
                        'After_SOC': current_soc,
                        'Magnitude': abs(diff),
                        'Full_Path': file_path
                    })
                    
                    last_event_time = current_time

        if anomalies:
            return pd.DataFrame(anomalies)
        
    except Exception as e:
        print(f"Error {os.path.basename(file_path)}: {e}")
        return None
    return None

def main():
    parser = argparse.ArgumentParser(description="High-Frequency SOC Anomaly Detection")
    parser.add_argument("--root", "-r", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="Drop %")
    parser.add_argument("--lag", "-l", type=float, default=LAG_TIME, help="Detection lag (seconds)")
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Validation window (seconds)")
    parser.add_argument("--output", "-o", type=str, default="mule2_soc_anomaly_highfreq.csv")
    
    args = parser.parse_args()
    
    all_anomalies = []
    files = find_csv_files(args.root)
    
    if not files and args.root == DEFAULT_ROOT:
         files = find_csv_files(".")

    count = 0 
    for f in files:
        print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
        res = process_file_sliding(f, args.threshold, args.window, args.lag)
        if res is not None and not res.empty:
             all_anomalies.append(res)
             print(f"\n[!] FOUND EVENTS in {os.path.basename(f)}")
        count += 1
        
    print("\nProcessing Complete.")
    
    if all_anomalies:
        final_df = pd.concat(all_anomalies, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print(f"Report saved: {args.output}")
        print(final_df.head())
    else:
        print("No events found.")

if __name__ == "__main__":
    main()