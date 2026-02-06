import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
from scipy.signal import medfilt

def select_file():
    """Opens a file dialog to select a CSV file."""
    root = tk.Tk()
    root.withdraw()
    initial_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        initialdir=initial_dir,
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

def plot_soc(file_path):
    """Loads CSV and plots SOC, Voltage (Actual vs Request), and Current (Actual vs Request)."""
    if not file_path:
        print("No file selected.")
        return

    try:
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        # --- TIME PROCESSING ---
        time_keywords = ['timestamp', 'time', 'date']
        time_col = next((col for col in df.columns if any(k in col.lower() for k in time_keywords)), None)
        
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            x_data, x_label = df[time_col], 'Time'
        else:
            x_data, x_label = df.index, 'Sample Index'

        # --- SOC PROCESSING ---
        soc_col = next((col for col in df.columns if col.lower() == 'soc'), None)

        # --- VOLTAGE PROCESSING ---
        volt_col = 'Battery_voltage'
        req_volt_col = 'Charging_Request_voltage'
        
        # Clean Actual Voltage
        if volt_col in df.columns:
            if df[volt_col].mean() > 1000:
                df[volt_col] = df[volt_col] / 1000.0
            df = df[(df[volt_col] >= 20) & (df[volt_col] <= 120)]
            df[volt_col] = medfilt(df[volt_col], kernel_size=111)
        
        # Scale Request Voltage (if in mV)
        if req_volt_col in df.columns and df[req_volt_col].mean() > 1000:
            df[req_volt_col] = df[req_volt_col] / 1000.0

        # --- CURRENT PROCESSING ---
        curr_col = 'Battery_current1'
        req_curr_col = 'Charging_Request_current'
        status_col = 'Charge_discharge_status'
        proc_curr_col = 'Processed_Current'

        if curr_col in df.columns:
            if status_col in df.columns:
                # Charging (1) should be negative current for standard convention
                df[proc_curr_col] = df.apply(
                    lambda r: -abs(r[curr_col]) if r[status_col] == 1 else r[curr_col], axis=1
                )
            else:
                df[proc_curr_col] = df[curr_col]
            df[proc_curr_col] = medfilt(df[proc_curr_col], kernel_size=111)

        # --- PLOTTING ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

        # Plot 1: SOC
        if soc_col:
            ax1.plot(x_data, df[soc_col], label='SOC %', color='tab:blue', linewidth=2)
            ax1.set_ylabel('SOC (%)')
        ax1.set_title(f'Battery Analysis: {os.path.basename(file_path)}', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # Plot 2: Voltages
        if volt_col in df.columns:
            ax2.plot(x_data, df[volt_col], label='Actual Voltage', color='tab:green', linewidth=1.5)
        if req_volt_col in df.columns:
            ax2.plot(x_data, df[req_volt_col], label='Request Voltage', color='darkgreen', linestyle='--', alpha=0.8)
        
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper right')

        # Plot 3: Currents
        if proc_curr_col in df.columns:
            ax3.plot(x_data, df[proc_curr_col], label='Actual Current', color='tab:red', linewidth=1.5)
        if req_curr_col in df.columns:
            # Note: We keep request current sign as provided in the CSV
            ax3.plot(x_data, df[req_curr_col], label='Request Current', color='darkred', linestyle='--', alpha=0.8)
            
        ax3.set_ylabel('Current (A)')
        ax3.axhline(0, color='black', linewidth=1)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right')

        ax3.set_xlabel(x_label)
        plt.tight_layout()
        
        print("Displaying plot...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    path = select_file()
    if path:
        plot_soc(path)