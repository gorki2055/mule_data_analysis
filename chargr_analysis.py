import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
OUTPUT_ROOT_DEFAULT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS"
OUTPUT_FILENAME = "charging_session_analysis.csv"

def save_csv_safe(df, path):
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(path)
        write_header = not file_exists
        
        # If file exists, check columns to ensure they match (optional, but good practice)
        if file_exists:
            try:
                existing_df = pd.read_csv(path, nrows=0)
                if list(existing_df.columns) != list(df.columns):
                    print(f"Warning: Columns in {path} do not match. Writing with header might duplicate it or cause issues.")
            except:
                pass

        df.to_csv(path, mode='a', index=False, header=write_header)
        print(f"Data appended to: {path}")
        
    except PermissionError:
        print(f"Warning: Permission denied for {path}. File might be open.")
        base, ext = os.path.splitext(path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_path = f"{base}_recovery_{timestamp}{ext}"
        print(f"Attempting to save as: {new_path}")
        try:
            df.to_csv(new_path, index=False)
            print(f"Report saved to: {new_path}")
        except Exception as e:
            print(f"Error saving to alternative path: {e}")
    except Exception as e:
        print(f"Error saving report: {e}")

def apply_median_filter(df, column_name, window_size=25):
    """
    Applies a rolling median filter to remove sudden spikes (salt-and-pepper noise).
    Preserves edges better than a mean filter.
    """
    # Calculate rolling median
    filtered_col = df[column_name].rolling(window=window_size, center=True).median()
    
    # Fill NaN values at the start/end of the series with the original raw values
    filtered_col.fillna(df[column_name], inplace=True)
    
    return filtered_col

def process_file(file_path):
    """
    Analyzes a single charging session file and returns a list of session dictionaries.
    """
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    # print(f"Processing: {os.path.basename(file_path)}...")
    try:
        # Load only necessary columns
        use_cols = [
            'timestamps', 
            'Current_Rotational_Speed', 
            'SOC', 
            'Battery_current1', 
            'Battery_voltage', 
            'Flt_Phase_Short_circuit'
        ]

        # Determine actual case of columns in file
        try:
            df_iter = pd.read_csv(file_path, nrows=1)
        except pd.errors.EmptyDataError:
            print("  Skipping: Empty file.")
            return []
            
        actual_cols = df_iter.columns.tolist()
        
        col_map = {}
        for req_col in use_cols:
            match = next((c for c in actual_cols if c.lower() == req_col.lower()), None)
            if match:
                col_map[req_col] = match
        
        # We strictly need timestamps, SOC, and Current for valid analysis
        if 'timestamps' not in col_map or 'SOC' not in col_map or 'Battery_current1' not in col_map:
            print("  Skipping: Missing critical columns.")
            return []

        df = pd.read_csv(file_path, usecols=col_map.values())
        
        # Rename back to standard names
        rev_map = {v: k for k, v in col_map.items()}
        df.rename(columns=rev_map, inplace=True)
        
        # Convert timestamps
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df.dropna(subset=['timestamps'], inplace=True)
        df.sort_values('timestamps', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # --- APPLY FILTERING HERE ---
        # Window of 5 is usually enough to kill single-point outliers without distorting the curve
        df['Battery_current_filt'] = apply_median_filter(df, 'Battery_current1', window_size=5)

    except Exception as e:
        print(f"  Error loading file: {e}")
        return []

    # --- IDENTIFY SESSIONS ---
    # Setup for session ID
    if 'Current_Rotational_Speed' in df.columns:
        df['is_stationary'] = (df['Current_Rotational_Speed'] == 0)
    else:
        # If speed is missing, assume it's all stationary (it's a CHARGING file)
        df['is_stationary'] = True
        
    df['segment_id'] = (df['is_stationary'] != df['is_stationary'].shift()).cumsum()
    
    stationary_groups = df[df['is_stationary']].groupby('segment_id')
    
    sessions_in_file = []
    
    for segment_id, group in stationary_groups:
        if len(group) < 20: # Minimum samples to consider a valid session
            continue
            
        start_soc = group['SOC'].iloc[0]
        end_soc = group['SOC'].iloc[-1]
        start_time = group['timestamps'].iloc[0]
        
        # Check if SOC increased
        soc_delta = end_soc - start_soc
        
        if soc_delta > 0:
            session_data = {}
            base_filename = os.path.basename(file_path)
            session_data['Filename'] = base_filename
            
            # Extract Shift from Filename (MULE1_1-13-2026_DAY_CHARGING -> DAY)
            try:
                # Remove extension and split by underscore
                name_parts = os.path.splitext(base_filename)[0].split('_')
                # Expecting at least 3 parts: ID, DATE, SHIFT, ...
                if len(name_parts) >= 3:
                    session_data['Shift'] = name_parts[2]
                else:
                    session_data['Shift'] = "Unknown"
            except Exception:
                session_data['Shift'] = "Error"
            
            # --- 1. Charging Phase Analysis ---
            soc_100_idx = group[group['SOC'] >= 100].index
            
            # CASE A: Battery reached 100%
            if not soc_100_idx.empty:
                first_100_idx = soc_100_idx[0]
                charge_end_time = df.loc[first_100_idx, 'timestamps']
                
                if charge_end_time < start_time: charge_end_time = group['timestamps'].iloc[-1]
                
                charge_mask = (group.index <= first_100_idx)
                charge_part = group.loc[charge_mask]
                actual_end_soc = 100
                
                # Static Phase Analysis (After 100%)
                static_mask = (group.index >= first_100_idx) & (group['SOC'] >= 100)
                static_part = group.loc[static_mask]
                
                if len(static_part) > 1:
                    static_currents = static_part['Battery_current_filt']
                    static_current_smooth = medfilt(static_currents, kernel_size=15)
                    session_data['Average Static Current (A)'] = round(static_currents.mean(), 3)
                    session_data['Static Min Current (A)'] = round(static_currents.min(), 3)
                    session_data['Static Max Current (A)'] = round(static_current_smooth.max(), 3)
                    session_data['Static Std Dev'] = round(static_currents.std(), 4)
                else:
                    session_data['Average Static Current (A)'] = None
                    session_data['Static Min Current (A)'] = None
                    session_data['Static Max Current (A)'] = None
                    session_data['Static Std Dev'] = None

            # CASE B: Battery stopped before 100% (Updated Logic)
            else:
                # 1. Find the highest SOC reached in this specific session segment
                actual_end_soc = group['SOC'].max()
                
                # 2. Find the FIRST index where this Max SOC occurred
                #    This effectively cuts off the "tail" where the car was sitting idle
                first_max_idx = group[group['SOC'] >= actual_end_soc].index[0]
                
                # 3. Set End Time to that specific moment
                charge_end_time = df.loc[first_max_idx, 'timestamps']
                
                # 4. Slice the dataframe so averages only use the active part
                charge_mask = (group.index <= first_max_idx)
                charge_part = group.loc[charge_mask]
                
                # No static analysis for partial charges
                session_data['Average Static Current (A)'] = None
                session_data['Static Min Current (A)'] = None
                session_data['Static Max Current (A)'] = None
                session_data['Static Std Dev'] = None

            # Calculate Charge Stats using FILTERED column and Truncated Time
            charge_currents = charge_part['Battery_current_filt']
            if len(charge_currents) > 0:
                charge_currents_smooth = medfilt(charge_currents, kernel_size=25)  
                duration_mins = (charge_end_time - start_time).total_seconds() / 60.0
                
                start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                session_data['Start Time'] = start_str
                session_data['End Time'] = charge_end_time.strftime('%Y-%m-%d %H:%M:%S')
                session_data['Start SOC'] = start_soc
                session_data['End SOC'] = actual_end_soc
                session_data['Duration (mins)'] = round(duration_mins, 2)
                session_data['Duration (hr)'] = round(duration_mins / 60.0, 4)
                session_data['Avg Current (A)'] = round(charge_currents_smooth.mean(), 2)
                session_data['Charge Min Current (A)'] = round(charge_currents_smooth.min(), 2)
                session_data['Charge Max Current (A)'] = round(charge_currents_smooth.max(), 2)
                session_data['SOC Increase'] = actual_end_soc - start_soc
                
                sessions_in_file.append(session_data)
                
                # --- Optional: Generate Plots per session ---
                # generate_session_plot(df, session_data, os.path.dirname(file_path)) 

    return sessions_in_file

def generate_session_plot(df, session_data, output_dir):
    """
    Helper to generate plot for a single session.
    """
    try:
        start = session_data['Start Time']
        end = session_data['End Time']
        view_end = end + pd.Timedelta(minutes=30)
        
        mask = (df['timestamps'] >= start) & (df['timestamps'] <= view_end)
        plot_data = df.loc[mask]
        
        if plot_data.empty: return

        fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()
        
        # SOC
        color = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('SOC (%)', color=color)
        ax1.plot(plot_data['timestamps'], plot_data['SOC'], color=color, linewidth=2, label='SOC')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        
        # Current
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Current (A)', color=color)
        ax2.plot(plot_data['timestamps'], plot_data['Battery_current1'], color='tab:pink', alpha=0.3, linewidth=1, label='Raw Noise')
        ax2.plot(plot_data['timestamps'], plot_data['Battery_current_filt'], color='tab:red', linewidth=1.5, label='Filtered Current')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.axvspan(start, end, color='green', alpha=0.1, label='Charging')
        
        plt.title(f"Charging Session: {session_data['Filename']}")
        plt.tight_layout()
        
        base_name = os.path.splitext(session_data['Filename'])[0]
        plot_path = os.path.join(output_dir, f"{base_name}_plot.png")
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Error producing plot: {e}")

def main():
    print("\n" + "="*60)
    print(f"BATCH CHARGING ANALYSIS @ {datetime.now().strftime('%H:%M:%S')}")
    print(f"Scanning: {ROOT_DIR_DEFAULT}")
    print("="*60)
    
    files_to_process = []
    
    # 1. Find Files
    for root, _, files in os.walk(ROOT_DIR_DEFAULT):
        for file in files:
            # Filter Logic: Must contain "CHARGING" and end with .csv. Must NOT contain "RIDE".
            if "CHARGING" in file.upper() and "RIDE" not in file.upper() and file.lower().endswith(".csv"):
                full_path = os.path.join(root, file)
                files_to_process.append(full_path)
                
    if not files_to_process:
        print("❌ No matching 'CHARGING' files found.")
        return

    print(f"Found {len(files_to_process)} valid charging files.")
    print("-" * 60)

    # 2. Process Files
    all_sessions = []
    
    for i, fpath in enumerate(files_to_process, 1):
        fname = os.path.basename(fpath)
        print(f"[{i}/{len(files_to_process)}] Processing: {fname} ... ", end="")
        
        try:
            sessions = process_file(fpath)
            if sessions:
                all_sessions.extend(sessions)
                print(f"✅ Found {len(sessions)} session(s)")
            else:
                print("⚠️  No valid sessions")
        except Exception as e:
            print(f"❌ Failed: {e}")

    # 3. Save Summary
    if all_sessions:
        if not os.path.exists(OUTPUT_ROOT_DEFAULT):
            os.makedirs(OUTPUT_ROOT_DEFAULT)
            
        out_path = os.path.join(OUTPUT_ROOT_DEFAULT, OUTPUT_FILENAME)
        
        df_summary = pd.DataFrame(all_sessions)
        
        # Desired Column Order
        cols = [
            'Filename', 'Shift', 'Start Time', 'End Time', 'Start SOC', 'End SOC', 
            'Duration (mins)', 'Duration (hr)', 
            'Avg Current (A)', 'Charge Min Current (A)', 'Charge Max Current (A)', 
            'SOC Increase', 
            'Average Static Current (A)', 'Static Min Current (A)', 
            'Static Max Current (A)', 'Static Std Dev'
        ]
        
        # Reorder and fill missing
        for c in cols:
            if c not in df_summary.columns: df_summary[c] = None
        df_summary = df_summary[cols]
        
        save_csv_safe(df_summary, out_path)
        
        print("\n" + "="*60)
        print(f"SUMMARY COMPLETE.")
        print(f"Total Sessions Found: {len(all_sessions)}")
        print(f"Saved to: {out_path}")
        print("="*60)
    else:
        print("\n❌ No charging sessions extracted from any files.")

if __name__ == "__main__":
    main()