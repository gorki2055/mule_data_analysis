
# import pandas as pd
# import os
# import argparse
# import sys
# from datetime import timedelta

# # Default Configuration
# DEFAULT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
# DEFAULT_THRESHOLD = 3.0
# DEFAULT_WINDOW = 30 # seconds

# def find_csv_files(root_dir):
#     csv_files = []
#     print(f"Scanning {root_dir}...")
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.lower().endswith(".csv") and "mule" in file.lower():
#                  # Matches logic in other scripts to filter mostly relevant files
#                  # But sticking to extension is safer, user said "ride session folder"
#                 csv_files.append(os.path.join(root, file))
#     print(f"Found {len(csv_files)} csv files.")
#     return csv_files

# def process_file(file_path, threshold, window_sec):
#     try:
#         # Load Data
#         try:
#             df = pd.read_csv(file_path, engine='python', low_memory=False)
#         except:
#              df = pd.read_csv(file_path, low_memory=False)
        
#         df.columns = df.columns.str.strip()

#         # Check required columns
#         if 'timestamps' not in df.columns or 'SOC' not in df.columns:
#             return None # Skip

#         # Parse Timestamps
#         df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
#         df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
#         if df.empty:
#             return None

#         # Logic: Find max SOC in the past 'window_sec' seconds
#         # We index by time
#         df_idx = df.set_index('timestamps').sort_index()
        
#         # Determine rolling window size string
#         window_str = f"{window_sec}s"
        
#         # Calculate Rolling Max
#         # 'closed'='both' or default includes the endpoints. 
#         # We want to compare current point to the MAX of the last 30s.
#         df_idx['soc_rolling_max'] = df_idx['SOC'].rolling(window_str).max()
        
#         # Calculate Drop
#         df_idx['soc_drop'] = df_idx['soc_rolling_max'] - df_idx['SOC']
        
#         # Filter Anomalies
#         anomalies = df_idx[df_idx['soc_drop'] > threshold].copy()
        
#         if not anomalies.empty:
#             anomalies['Filename'] = os.path.basename(file_path)
#             anomalies['Full_Path'] = file_path
#             # Return relevant columns
#             cols = ['Filename', 'SOC', 'soc_rolling_max', 'soc_drop', 'Full_Path']
#             return anomalies[cols].reset_index() # timestamps becomes a column
            
#     except Exception as e:
#         print(f"Error processing {os.path.basename(file_path)}: {e}")
#         return None
        
#     return None

# def main():
#     parser = argparse.ArgumentParser(description="Recursively find sudden SOC drops in CSV logs.")
#     parser.add_argument("--root", "-r", type=str, default=DEFAULT_ROOT, help="Root directory to scan.")
#     parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="SOC drop threshold (%%).")
#     parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Time window in seconds.")
#     parser.add_argument("--output", "-o", type=str, default="soc_drop_report.csv", help="Output CSV filename.")
    
#     args = parser.parse_args()
    
#     all_anomalies = []
    
#     files = find_csv_files(args.root)
    
#     if not files:
#         print("No CSV files found.")
#         # Try current directory as fallback if default path doesn't exist
#         if args.root == DEFAULT_ROOT and not os.path.exists(DEFAULT_ROOT):
#              print(f"Default path {DEFAULT_ROOT} does not exist. Scanning current directory.")
#              files = find_csv_files(".")

#     count = 0 
#     for f in files:
#         print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
#         res = process_file(f, args.threshold, args.window)
#         if res is not None:
#              all_anomalies.append(res)
#              print(f"\n[!] ALERT: Found {len(res)} drops in {os.path.basename(f)}")
#         count += 1
        
#     print("\nProcessing Complete.")
    
#     if all_anomalies:
#         final_df = pd.concat(all_anomalies, ignore_index=True)
#         final_df.to_csv(args.output, index=False)
#         print("="*60)
#         print(f"FOUND {len(final_df)} ANOMALIES")
#         print(f"Report saved to: {os.path.abspath(args.output)}")
#         print("Top 5 drops:")
#         print(final_df.sort_values('soc_drop', ascending=False).head(5)[['timestamps', 'Filename', 'soc_drop']])
#         print("="*60)
#     else:
#         print("No drops found exceeding the threshold.")

# if __name__ == "__main__":
#     main()


# import pandas as pd
# import os
# import argparse
# import sys
# from datetime import timedelta

# # Default Configuration
# # DEFAULT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
# DEFAULT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\2877D237"
# DEFAULT_THRESHOLD = 3.0
# DEFAULT_WINDOW = 30 # seconds

# def find_csv_files(root_dir):
#     csv_files = []
#     print(f"Scanning {root_dir}...")
#     for root, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.lower().endswith(".csv") and "mule" in file.lower():
#                 csv_files.append(os.path.join(root, file))
#     print(f"Found {len(csv_files)} csv files.")
#     return csv_files

# def process_file(file_path, threshold, window_sec):
#     try:
#         # Load Data
#         try:
#             df = pd.read_csv(file_path, engine='python', low_memory=False)
#         except:
#             df = pd.read_csv(file_path, low_memory=False)
        
#         df.columns = df.columns.str.strip()

#         # Check required columns
#         if 'timestamps' not in df.columns or 'SOC' not in df.columns:
#             return None # Skip

#         # Parse Timestamps
#         df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
#         df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
#         if df.empty:
#             return None

#         # Index by time for rolling calculations
#         df_idx = df.set_index('timestamps').sort_index()
        
#         # Determine rolling window size string
#         window_str = f"{window_sec}s"
        
#         # --- LOGIC UPDATE ---
#         # 1. Calculate Rolling Max (for Drops) and Rolling Min (for Increases)
#         df_idx['soc_rolling_max'] = df_idx['SOC'].rolling(window_str).max()
#         df_idx['soc_rolling_min'] = df_idx['SOC'].rolling(window_str).min()
        
#         # 2. Calculate deviations
#         df_idx['soc_drop'] = df_idx['soc_rolling_max'] - df_idx['SOC']
#         df_idx['soc_increase'] = df_idx['SOC'] - df_idx['soc_rolling_min']
        
#         # 3. Filter Anomalies (Drop > Threshold OR Increase > Threshold)
#         anomalies = df_idx[
#             (df_idx['soc_drop'] > threshold) | 
#             (df_idx['soc_increase'] > threshold)
#         ].copy()
        
#         if not anomalies.empty:
#             # Add Metadata
#             anomalies['Filename'] = os.path.basename(file_path)
#             anomalies['Full_Path'] = file_path
            
#             # Label the anomaly type
#             # We use a function to determine if it was a drop or increase
#             def get_type(row):
#                 if row['soc_drop'] > threshold:
#                     return 'Drop'
#                 else:
#                     return 'Increase'
            
#             anomalies['Anomaly_Type'] = anomalies.apply(get_type, axis=1)
            
#             # Define 'Magnitude' for easier sorting later (the size of the jump/drop)
#             anomalies['Magnitude'] = anomalies.apply(
#                 lambda row: row['soc_drop'] if row['Anomaly_Type'] == 'Drop' else row['soc_increase'], axis=1
#             )

#             # Return relevant columns
#             cols = ['Filename', 'Anomaly_Type', 'SOC', 'Magnitude', 'soc_drop', 'soc_increase', 'Full_Path']
#             return anomalies[cols].reset_index() # timestamps becomes a column
            
#     except Exception as e:
#         print(f"Error processing {os.path.basename(file_path)}: {e}")
#         return None
        
#     return None

# def main():
#     parser = argparse.ArgumentParser(description="Recursively find sudden SOC drops and increases in CSV logs.")
#     parser.add_argument("--root", "-r", type=str, default=DEFAULT_ROOT, help="Root directory to scan.")
#     parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="SOC deviation threshold (%%).")
#     parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Time window in seconds.")
#     parser.add_argument("--output", "-o", type=str, default="soc_anomaly_report.csv", help="Output CSV filename.")
    
#     args = parser.parse_args()
    
#     all_anomalies = []
    
#     files = find_csv_files(args.root)
    
#     if not files:
#         print("No CSV files found.")
#         if args.root == DEFAULT_ROOT and not os.path.exists(DEFAULT_ROOT):
#              print(f"Default path {DEFAULT_ROOT} does not exist. Scanning current directory.")
#              files = find_csv_files(".")

#     count = 0 
#     for f in files:
#         print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
#         res = process_file(f, args.threshold, args.window)
#         if res is not None:
#              all_anomalies.append(res)
#              # Count types for the console log
#              drops = len(res[res['Anomaly_Type'] == 'Drop'])
#              incs = len(res[res['Anomaly_Type'] == 'Increase'])
#              print(f"\n[!] ALERT in {os.path.basename(f)}: {drops} Drops, {incs} Increases")
#         count += 1
        
#     print("\nProcessing Complete.")
    
#     if all_anomalies:
#         final_df = pd.concat(all_anomalies, ignore_index=True)
#         final_df.to_csv(args.output, index=False)
        
#         print("="*60)
#         print(f"FOUND {len(final_df)} TOTAL ANOMALIES")
#         print(f"Report saved to: {os.path.abspath(args.output)}")
#         print("-" * 60)
        
#         # Sort by Magnitude (absolute change size)
#         top_anomalies = final_df.sort_values('Magnitude', ascending=False).head(5)
#         print("Top 5 Biggest Events:")
#         print(top_anomalies[['timestamps', 'Filename', 'Anomaly_Type', 'Magnitude']])
#         print("="*60)
#     else:
#         print("No anomalies found exceeding the threshold.")

# if __name__ == "__main__":
#     main()
import pandas as pd
import os
import argparse
import sys
import numpy as np

# Default Configuration
DEFAULT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\2877D237"
DEFAULT_THRESHOLD = 3.0
DEFAULT_WINDOW = 30 # seconds

def find_csv_files(root_dir):
    csv_files = []
    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv") and "mule" in file.lower():
                csv_files.append(os.path.join(root, file))
    print(f"Found {len(csv_files)} csv files.")
    return csv_files

def process_file_consecutive(file_path, threshold, window_sec):
    try:
        # Load Data
        try:
            df = pd.read_csv(file_path, engine='python', low_memory=False)
        except:
            df = pd.read_csv(file_path, low_memory=False)
        
        df.columns = df.columns.str.strip()

        # Check required columns
        if 'timestamps' not in df.columns or 'SOC' not in df.columns:
            return None

        # Parse Timestamps
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        if df.empty:
            return None

        # Prepare numpy arrays for fast processing
        times = df['timestamps'].values
        socs = df['SOC'].values
        
        # Calculate differences between consecutive elements
        # prepend the first element to keep shape aligned (first row has 0 diff)
        soc_diffs = np.diff(socs, prepend=socs[0])
        
        # Find indices where the change is greater than threshold
        # We care about the magnitude of change
        candidate_indices = np.where(np.abs(soc_diffs) > threshold)[0]
        
        anomalies = []
        
        # Validation Window in Nanoseconds (numpy datetime64 uses ns)
        window_ns = np.timedelta64(window_sec, 's')
        
        for i in candidate_indices:
            # Skip the very first row if it was flagged (unlikely with diff prepend logic, but safe to check)
            if i == 0: continue
                
            current_time = times[i]
            current_soc = socs[i]
            prev_soc = socs[i-1] # The value BEFORE the jump
            diff = soc_diffs[i]
            
            # --- VALIDATION STEP ---
            # Look ahead in the window to see if SOC returns to 'prev_soc' level.
            # If it returns, it's noise.
            
            # Find the end of the window in the array
            # searchsorted is binary search, very fast
            window_end_time = current_time + window_ns
            end_idx = np.searchsorted(times, window_end_time, side='right')
            
            # Slice the future data within window
            # We look from i+1 up to end_idx
            future_socs = socs[i+1 : end_idx]
            
            is_noise = False
            
            if len(future_socs) > 0:
                # Check if any future value is "close" to the prev_soc
                # "Close" defined as being within the threshold distance
                # Logic: If I dropped from 80 to 50, and I go back to 78, that's a return.
                dist_to_prev = np.abs(future_socs - prev_soc)
                
                # If any point returns to the previous level (distance < threshold), flag as noise
                if np.any(dist_to_prev < threshold):
                    is_noise = True
            
            if not is_noise:
                # Store the anomaly
                anomaly_type = "Drop" if diff < 0 else "Increase"
                anomalies.append({
                    'timestamps': current_time,
                    'Filename': os.path.basename(file_path),
                    'Anomaly_Type': anomaly_type,
                    'Before_SOC': prev_soc,
                    'After_SOC': current_soc,
                    'Magnitude': abs(diff),
                    'Full_Path': file_path
                })

        if anomalies:
            return pd.DataFrame(anomalies)
        else:
            return None

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Find SUDDEN SOC changes using Consecutive Row Comparison.")
    parser.add_argument("--root", "-r", type=str, default=DEFAULT_ROOT, help="Root directory to scan.")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="SOC deviation threshold (%%).")
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Validation window in seconds (to filter noise).")
    parser.add_argument("--output", "-o", type=str, default="soc_anomaly_consecutive.csv", help="Output CSV filename.")
    
    args = parser.parse_args()
    
    all_anomalies = []
    
    files = find_csv_files(args.root)
    
    if not files:
        print("No CSV files found.")
        if args.root == DEFAULT_ROOT and not os.path.exists(DEFAULT_ROOT):
             print(f"Default path {DEFAULT_ROOT} does not exist. Scanning current directory.")
             files = find_csv_files(".")

    count = 0 
    for f in files:
        print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
        res = process_file_consecutive(f, args.threshold, args.window)
        if res is not None and not res.empty:
             all_anomalies.append(res)
             drops = len(res[res['Anomaly_Type'] == 'Drop'])
             incs = len(res[res['Anomaly_Type'] == 'Increase'])
             print(f"\n[!] ALERT in {os.path.basename(f)}: {drops} Drops, {incs} Increases")
        count += 1
        
    print("\nProcessing Complete.")
    
    if all_anomalies:
        final_df = pd.concat(all_anomalies, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        
        print("="*60)
        print(f"FOUND {len(final_df)} VALIDATED EVENTS")
        print(f"Report saved to: {os.path.abspath(args.output)}")
        print("-" * 60)
        
        # Sort by Magnitude
        top_anomalies = final_df.sort_values('Magnitude', ascending=False).head(5)
        print("Top 5 Biggest Events:")
        print(top_anomalies[['timestamps', 'Filename', 'Anomaly_Type', 'Before_SOC', 'After_SOC', 'Magnitude']])
        print("="*60)
    else:
        print("No anomalies found exceeding the threshold.")

if __name__ == "__main__":
    main()
