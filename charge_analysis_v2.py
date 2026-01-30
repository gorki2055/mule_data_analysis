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
OUTPUT_FILENAME = "charging_session_analysis_MULE1_temp.csv"

def save_csv_safe(df, path):
    """
    Saves the dataframe to CSV safely. 
    If the file is open/locked by Excel, it saves to a timestamped backup file.
    """
    try:
        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(path)
        write_header = not file_exists
        
        # If file exists, check columns to ensure they match
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

    try:
        # Load only necessary columns (Including charge_discharge_status)
        use_cols = [
            'timestamps', 
            'Current_Rotational_Speed', 
            'SOC', 
            'Battery_current1', 
            'Battery_voltage', 
            'Flt_Phase_Short_circuit',
            'charge_discharge_status',
            'Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4'
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
        # Window of 5 is usually enough to kill single-point outliers
        df['Battery_current_filt'] = apply_median_filter(df, 'Battery_current1', window_size=5)

        # --- TEMPERATURE FILTERING (MATCHING PLOTV14) ---
        ntc_sensors = ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
        valid_ntc_cols = [c for c in ntc_sensors if c in df.columns]

        for sensor in valid_ntc_cols:
            # 1. HARD MASK
            df.loc[(df[sensor] < -40) | (df[sensor] > 75), sensor] = np.nan
            
            # 2. 10-MINUTE CONSENSUS BASELINE
            # 10 mins @ 1Hz = 600 points. If 10Hz, larger. Assuming similar rate to plotv14 logic or adjusting.
            # plotv14 uses window=60001 (likely high freq). We'll use a safer large window or just skip if expensive.
            # For reported files, sampling might be 1s or 0.1s. Let's start with a simpler check or same generic logic.
            # Using a generic reasonable window for 1Hz data (600) or just relying on medfilt if dense.
            # Let's stick closer to plotv14 logic but be mindful of data size. 
            # If we assume ~1-10Hz, 60001 is huge. Let's use a simpler effective cleaner or copy plotv14 logic blindly?
            # User wants "from plotv14", so let's try to match logic but maybe scale window if needed. 
            # safe bet: use the same logic lines.
            
            baseline = df[sensor].rolling(window=60001, center=True, min_periods=1).median()
            
            # 3. SURGICAL REJECTION
            invalid_mask = (df[sensor] - baseline).abs() > 2.0
            df.loc[invalid_mask, sensor] = np.nan
            
            # 4. INTERPOLATE
            df[sensor] = df[sensor].interpolate(method='linear', limit_direction='both')
            
            # 5. FINAL SMOOTHING
            if not df[sensor].isna().all():
                df[sensor] = medfilt(df[sensor], kernel_size=31)

        # Calculate Max Temp Row-wise
        if valid_ntc_cols:
            df['Max_Temp'] = df[valid_ntc_cols].max(axis=1)
        else:
            df['Max_Temp'] = None

    except Exception as e:
        print(f"  Error loading file: {e}")
        return []

    sessions_in_file = []
    base_filename = os.path.basename(file_path)

    # Helper to extract shift name
    try:
        name_parts = os.path.splitext(base_filename)[0].split('_')
        shift_name = name_parts[2] if len(name_parts) >= 3 else "Unknown"
    except:
        shift_name = "Error"

    # ========================================================================
    # METHOD 1: PRECISE BMS STATUS (If 'charge_discharge_status' exists)
    # ========================================================================
    if 'charge_discharge_status' in df.columns:
        # Create groups where the status changes
        df['group_id'] = (df['charge_discharge_status'] != df['charge_discharge_status'].shift()).cumsum()
        grouped = df.groupby('group_id')

        for g_id, group in grouped:
            # We strictly look for STATUS = 1 (Charging)
            if group['charge_discharge_status'].iloc[0] == 1:
                
                # --- NOISE FILTER START ---
                # Check Duration and SOC Gain BEFORE processing
                temp_start_time = group['timestamps'].iloc[0]
                temp_end_time = group['timestamps'].iloc[-1]
                temp_duration_mins = (temp_end_time - temp_start_time).total_seconds() / 60.0
                temp_soc_increase = group['SOC'].iloc[-1] - group['SOC'].iloc[0]

                # Filter Rule: Ignore if shorter than 2 mins AND no significant SOC gain
                if temp_duration_mins < 2.0 and temp_soc_increase <= 0:
                    continue
                # --- NOISE FILTER END ---

                # A. Accurate Timing
                start_time = temp_start_time
                end_time = temp_end_time
                start_soc = group['SOC'].iloc[0]
                end_soc = group['SOC'].iloc[-1]

                # B. Charge Stats
                charge_currents = group['Battery_current_filt']
                charge_currents_smooth = medfilt(charge_currents, kernel_size=25)

                session_data = {
                    'Filename': base_filename,
                    'Shift': shift_name,
                    'Start Time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'End Time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'Duration (mins)': round(temp_duration_mins, 2),
                    'Duration (hr)': round(temp_duration_mins / 60.0, 4),
                    'Start SOC': start_soc,
                    'End SOC': end_soc,
                    'SOC Increase': end_soc - start_soc,
                    'Avg Current (A)': round(charge_currents_smooth.mean(), 2),
                    'Charge Min Current (A)': round(charge_currents_smooth.min(), 2),
                    'Charge Max Current (A)': round(charge_currents_smooth.max(), 2),
                    'Max Temp (C)': round(group['Max_Temp'].max(), 2) if 'Max_Temp' in df.columns and not group['Max_Temp'].isnull().all() else None,
                }

                # C. Static Phase Look-Ahead
                # Look at the rows immediately AFTER this group ends to find static draw
                last_idx = group.index[-1]
                # Look ahead up to 200 rows
                next_rows = df.loc[last_idx+1 : last_idx+200]

                if not next_rows.empty:
                    # Filter for typical static consumption (negative current, small magnitude)
                    # e.g., -5A < Current < 1A. Exclude large discharges (driving).
                    static_part = next_rows[
                        (next_rows['Battery_current_filt'] > -5) & 
                        (next_rows['Battery_current_filt'] < 1)
                    ]
                    
                    if not static_part.empty and len(static_part) > 10:
                        static_vals = static_part['Battery_current_filt']
                        session_data['Average Static Current (A)'] = round(static_vals.mean(), 3)
                        session_data['Static Min Current (A)'] = round(static_vals.min(), 3)
                        session_data['Static Max Current (A)'] = round(static_vals.max(), 3)
                        session_data['Static Std Dev'] = round(static_vals.std(), 4)
                    else:
                        # If current is high immediately after, they probably drove off
                        session_data['Average Static Current (A)'] = None
                        session_data['Static Min Current (A)'] = None
                        session_data['Static Max Current (A)'] = None
                        session_data['Static Std Dev'] = None
                else:
                    session_data['Average Static Current (A)'] = None
                    session_data['Static Min Current (A)'] = None
                    session_data['Static Max Current (A)'] = None
                    session_data['Static Std Dev'] = None

                sessions_in_file.append(session_data)

    # ========================================================================
    # METHOD 2: FALLBACK (Original logic if status column is missing)
    # ========================================================================
    else:
        # Setup for session ID based on speed or assumption
        if 'Current_Rotational_Speed' in df.columns:
            df['is_stationary'] = (df['Current_Rotational_Speed'] == 0)
        else:
            df['is_stationary'] = True
            
        df['segment_id'] = (df['is_stationary'] != df['is_stationary'].shift()).cumsum()
        stationary_groups = df[df['is_stationary']].groupby('segment_id')
        
        for segment_id, group in stationary_groups:
            if len(group) < 20: continue
                
            start_soc = group['SOC'].iloc[0]
            end_soc = group['SOC'].iloc[-1]
            start_time = group['timestamps'].iloc[0]
            
            soc_delta = end_soc - start_soc
            
            if soc_delta > 0:
                session_data = {
                    'Filename': base_filename,
                    'Shift': shift_name
                }
                
                # --- Charging Phase Analysis ---
                soc_100_idx = group[group['SOC'] >= 100].index
                
                # CASE A: Battery reached 100%
                if not soc_100_idx.empty:
                    first_100_idx = soc_100_idx[0]
                    charge_end_time = df.loc[first_100_idx, 'timestamps']
                    if charge_end_time < start_time: charge_end_time = group['timestamps'].iloc[-1]
                    
                    charge_mask = (group.index <= first_100_idx)
                    charge_part = group.loc[charge_mask]
                    actual_end_soc = 100
                    
                    # Static Stats (Using original logic for 100% hold)
                    static_mask = (group.index >= first_100_idx) & (group['SOC'] >= 100)
                    static_part = group.loc[static_mask]
                    
                    if len(static_part) > 1:
                        static_currents = static_part['Battery_current_filt']
                        session_data['Average Static Current (A)'] = round(static_currents.mean(), 3)
                        session_data['Static Min Current (A)'] = round(static_currents.min(), 3)
                        session_data['Static Max Current (A)'] = round(static_currents.max(), 3)
                        session_data['Static Std Dev'] = round(static_currents.std(), 4)
                    else:
                        session_data['Average Static Current (A)'] = None
                        session_data['Static Min Current (A)'] = None
                        session_data['Static Max Current (A)'] = None
                        session_data['Static Std Dev'] = None

                # CASE B: Battery stopped before 100% (Updated Logic)
                else:
                    actual_end_soc = group['SOC'].max()
                    # Find the FIRST index where this Max SOC occurred to cut off idle tail
                    first_max_idx = group[group['SOC'] >= actual_end_soc].index[0]
                    charge_end_time = df.loc[first_max_idx, 'timestamps']
                    
                    charge_mask = (group.index <= first_max_idx)
                    charge_part = group.loc[charge_mask]
                    
                    session_data['Average Static Current (A)'] = None
                    session_data['Static Min Current (A)'] = None
                    session_data['Static Max Current (A)'] = None
                    session_data['Static Std Dev'] = None

                # Calculate Charge Stats
                charge_currents = charge_part['Battery_current_filt']
                if len(charge_currents) > 0:
                    charge_currents_smooth = medfilt(charge_currents, kernel_size=25)  
                    duration_mins = (charge_end_time - start_time).total_seconds() / 60.0
                    
                    session_data['Start Time'] = start_time.strftime('%Y-%m-%d %H:%M:%S')
                    session_data['End Time'] = charge_end_time.strftime('%Y-%m-%d %H:%M:%S')
                    session_data['Start SOC'] = start_soc
                    session_data['End SOC'] = actual_end_soc
                    session_data['Duration (mins)'] = round(duration_mins, 2)
                    session_data['Duration (hr)'] = round(duration_mins / 60.0, 4)
                    session_data['Avg Current (A)'] = round(charge_currents_smooth.mean(), 2)
                    session_data['Charge Min Current (A)'] = round(charge_currents_smooth.min(), 2)
                    session_data['Charge Max Current (A)'] = round(charge_currents_smooth.max(), 2)
                    session_data['SOC Increase'] = actual_end_soc - start_soc
                    session_data['Max Temp (C)'] = round(charge_part['Max_Temp'].max(), 2) if 'Max_Temp' in df.columns and not charge_part['Max_Temp'].isnull().all() else None
                    
                    sessions_in_file.append(session_data)

    return sessions_in_file

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
            'Max Temp (C)',
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