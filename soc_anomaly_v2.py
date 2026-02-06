import pandas as pd
import os
import argparse
import numpy as np

# Configuration
DEFAULT_ROOT = r"D:\kushal\LOG\LOG\3ACF5792"
DEFAULT_THRESHOLD = 3.0
DEFAULT_WINDOW = 30 # seconds (Validation window)
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

        # --- Filter Invalid SOC Data ---
        # 1. Remove impossible values
        df = df[(df['SOC'] >= 0) & (df['SOC'] <= 100)]
        
        # 2. Parse Timestamps (Force UTC to avoid timezone mixing issues)
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce', utc=True)
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        if df.empty: return None

        # 3. Prepare Arrays
        times = df['timestamps'].values
        socs = df['SOC'].values
        n = len(times)
        
        # Convert constants to nanoseconds
        lag_ns = np.timedelta64(int(lag_seconds * 1_000_000_000), 'ns')
        window_ns = np.timedelta64(int(valid_window * 1_000_000_000), 'ns')
        
        anomalies = []
        last_event_time = None
        
        # --- SLIDING WINDOW LOGIC ---
        left = 0
        
        for right in range(n):
            current_time = times[right]
            
            # 4. Advance 'left' pointer to maintain lag gap
            while left < right and (current_time - times[left]) > lag_ns:
                left += 1
            
            # --- [NEW] 4a. BACKWARD STABILITY CHECK ---
            # Before we calculate a drop/increase, we must ensure the 'start' point (left)
            # wasn't just a glitch itself. If socs[left] came from nowhere, it's not a valid baseline.
            if left > 0:
                prev_soc = socs[left-1]
                time_gap_prev = times[left] - times[left-1]
                
                # If the previous point was very recent (within 2 seconds) 
                # AND the value jumped wildly to get here, then 'left' is unstable.
                if time_gap_prev < (lag_ns * 2):
                    if abs(socs[left] - prev_soc) > threshold:
                        # The baseline is unstable (part of a spike). Skip comparison.
                        continue
            # ------------------------------------------

            start_soc = socs[left]
            current_soc = socs[right]
            diff = current_soc - start_soc
            
            # 5. Threshold Check
            if abs(diff) > threshold:
                
                # Deduplication
                if last_event_time is not None:
                    if (current_time - last_event_time) <= window_ns:
                        continue
                
                # --- [EXISTING] VALIDATION (Look Ahead) ---
                # Check if it's noise: Does SOC return to 'start_soc' in the next 30s?
                valid_end_time = current_time + window_ns
                end_idx = np.searchsorted(times, valid_end_time, side='right')
                future_socs = socs[right+1 : end_idx]
                
                is_noise = False
                if len(future_socs) > 0:
                    # If any future point returns close to the start value, it's noise
                    dist_to_start = np.abs(future_socs - start_soc)
                    if np.any(dist_to_start < threshold):
                        is_noise = True
                
                if not is_noise:
                    anomalies.append({
                        'timestamps': current_time,
                        'Filename': os.path.basename(file_path),
                        'Anomaly_Type': "Drop" if diff < 0 else "Increase",
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
    parser.add_argument("--output", "-o", type=str, default="mule1_soc_anomaly_highfreq.csv")
    
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