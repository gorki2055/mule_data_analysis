import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ============================================================================
# CONFIGURATION
# ============================================================================
GEAR_RATIO = 6.09
WHEEL_RADIUS_M = 0.261
RESAMPLE_RATE = '100ms'

def calculate_summary(file_path):
    try:
        # 1. High Speed Load using C engine
        df = pd.read_csv(file_path, engine='c', low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Parse timestamps
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df = df.dropna(subset=['timestamps']).sort_values('timestamps')

        # 2. Vectorized Current Processing
        if 'Battery_current1' in df.columns:
            df['Processed_Current'] = df['Battery_current1'].values
            if 'Charge_discharge_status' in df.columns:
                charging_mask = df['Charge_discharge_status'] == 1
                df.loc[charging_mask, 'Processed_Current'] = -np.abs(df.loc[charging_mask, 'Battery_current1'].values)

        # 3. Resampling
        df_ts = df.set_index('timestamps').select_dtypes(include=[np.number])
        df_res = df_ts.resample(RESAMPLE_RATE).mean().interpolate(method='linear')

        # --- PEAK CURRENT INSTANCE CALCULATION ---
        # Find the absolute maximum current value and its location
        idx_peak = df_res['Processed_Current'].idxmax()
        peak_current_val = df_res.loc[idx_peak, 'Processed_Current']
        voltage_at_peak = df_res.loc[idx_peak, 'Battery_voltage']
        # -----------------------------------------

        # 4. Energy Calculations (Wh)
        fixed_dt_hours = 0.1 / 3600.0
        power_w = df_res['Battery_voltage'] * df_res['Processed_Current']
        
        energy_slice_wh = (power_w * fixed_dt_hours)
        discharge_wh = energy_slice_wh.clip(lower=0).sum()
        regen_wh = (-energy_slice_wh).clip(lower=0).sum()
        net_wh = discharge_wh - regen_wh

        # 5. Distance Calculation
        distance_km = 0
        if 'Current_Rotational_Speed' in df_res.columns:
            wheel_rpm = df_res['Current_Rotational_Speed'] / GEAR_RATIO
            speed_kmh = (wheel_rpm * 2 * np.pi / 60) * WHEEL_RADIUS_M * 3.6
            speed_kmh = np.where(speed_kmh > 100, 0, speed_kmh)
            distance_km = (speed_kmh * fixed_dt_hours).sum()

        # 6. Display Summary
        print("\n" + "="*55)
        print(f"BATTERY ANALYSIS SUMMARY: {os.path.basename(file_path)}")
        print("="*55)
        print(f"Duration:          {df['timestamps'].iloc[-1] - df['timestamps'].iloc[0]}")
        print(f"Samples:           {len(df):,}")
        print("-" * 35)
        
        # Display Peak Instance
        print(f"PEAK CURRENT INSTANCE:")
        print(f"  Max Current:     {peak_current_val:.2f} A")
        print(f"  Voltage at Peak: {voltage_at_peak:.2f} V")
        print(f"  Peak Power:      {(peak_current_val * voltage_at_peak)/1000:.2f} kW")
        print(f"  Timestamp:       {idx_peak}")
        
        print("-" * 35)
        print(f"Voltage Range:     {df_res['Battery_voltage'].min():.2f}V - {df_res['Battery_voltage'].max():.2f}V")
        print(f"Total Discharge:   {discharge_wh:.1f} Wh")
        print(f"Total Regen:       {regen_wh:.1f} Wh")
        print(f"Net Energy:        {net_wh:.1f} Wh")
        print("-" * 35)
        print(f"Distance Covered:  {distance_km:.3f} km")
        if distance_km > 0:
            print(f"Efficiency:        {net_wh/distance_km:.1f} Wh/km")
        
        # Temperature Peaks
        temp_cols = ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
        available_temps = [c for c in df_res.columns if c in temp_cols]
        if available_temps:
            print("-" * 35)
            print("Peak Temperatures:")
            for t in available_temps:
                print(f"  {t}: {df_res[t].max():.1f} °C")
        print("="*55 + "\n")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process file:\n{str(e)}")

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Battery CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if file_path:
        calculate_summary(file_path)
    root.destroy()

if __name__ == "__main__":
    select_file()