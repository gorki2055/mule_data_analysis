
import pandas as pd
import os
import argparse
import sys
from datetime import timedelta

# Default Configuration
DEFAULT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
DEFAULT_THRESHOLD = 3.0
DEFAULT_WINDOW = 30 # seconds

def find_csv_files(root_dir):
    csv_files = []
    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv") and "mule" in file.lower():
                 # Matches logic in other scripts to filter mostly relevant files
                 # But sticking to extension is safer, user said "ride session folder"
                csv_files.append(os.path.join(root, file))
    print(f"Found {len(csv_files)} csv files.")
    return csv_files

def process_file(file_path, threshold, window_sec):
    try:
        # Load Data
        try:
            df = pd.read_csv(file_path, engine='python', low_memory=False)
        except:
             df = pd.read_csv(file_path, low_memory=False)
        
        df.columns = df.columns.str.strip()

        # Check required columns
        if 'timestamps' not in df.columns or 'SOC' not in df.columns:
            return None # Skip

        # Parse Timestamps
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')
        
        if df.empty:
            return None

        # Logic: Find max SOC in the past 'window_sec' seconds
        # We index by time
        df_idx = df.set_index('timestamps').sort_index()
        
        # Determine rolling window size string
        window_str = f"{window_sec}s"
        
        # Calculate Rolling Max
        # 'closed'='both' or default includes the endpoints. 
        # We want to compare current point to the MAX of the last 30s.
        df_idx['soc_rolling_max'] = df_idx['SOC'].rolling(window_str).max()
        
        # Calculate Drop
        df_idx['soc_drop'] = df_idx['soc_rolling_max'] - df_idx['SOC']
        
        # Filter Anomalies
        anomalies = df_idx[df_idx['soc_drop'] > threshold].copy()
        
        if not anomalies.empty:
            anomalies['Filename'] = os.path.basename(file_path)
            anomalies['Full_Path'] = file_path
            # Return relevant columns
            cols = ['Filename', 'SOC', 'soc_rolling_max', 'soc_drop', 'Full_Path']
            return anomalies[cols].reset_index() # timestamps becomes a column
            
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Recursively find sudden SOC drops in CSV logs.")
    parser.add_argument("--root", "-r", type=str, default=DEFAULT_ROOT, help="Root directory to scan.")
    parser.add_argument("--threshold", "-t", type=float, default=DEFAULT_THRESHOLD, help="SOC drop threshold (%%).")
    parser.add_argument("--window", "-w", type=int, default=DEFAULT_WINDOW, help="Time window in seconds.")
    parser.add_argument("--output", "-o", type=str, default="soc_drop_report.csv", help="Output CSV filename.")
    
    args = parser.parse_args()
    
    all_anomalies = []
    
    files = find_csv_files(args.root)
    
    if not files:
        print("No CSV files found.")
        # Try current directory as fallback if default path doesn't exist
        if args.root == DEFAULT_ROOT and not os.path.exists(DEFAULT_ROOT):
             print(f"Default path {DEFAULT_ROOT} does not exist. Scanning current directory.")
             files = find_csv_files(".")

    count = 0 
    for f in files:
        print(f"Processing ({count+1}/{len(files)}): {os.path.basename(f)}...", end="\r")
        res = process_file(f, args.threshold, args.window)
        if res is not None:
             all_anomalies.append(res)
             print(f"\n[!] ALERT: Found {len(res)} drops in {os.path.basename(f)}")
        count += 1
        
    print("\nProcessing Complete.")
    
    if all_anomalies:
        final_df = pd.concat(all_anomalies, ignore_index=True)
        final_df.to_csv(args.output, index=False)
        print("="*60)
        print(f"FOUND {len(final_df)} ANOMALIES")
        print(f"Report saved to: {os.path.abspath(args.output)}")
        print("Top 5 drops:")
        print(final_df.sort_values('soc_drop', ascending=False).head(5)[['timestamps', 'Filename', 'soc_drop']])
        print("="*60)
    else:
        print("No drops found exceeding the threshold.")

if __name__ == "__main__":
    main()
