import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import sys
from scipy.signal import medfilt
import numpy as np

def select_file():
    """Opens a file dialog to select a CSV file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Set initial directory to the script's directory if possible
    initial_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_path = filedialog.askopenfilename(
        title="Select CSV File",
        initialdir=initial_dir,
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path

def plot_soc(file_path):
    """Loads the CSV file and plots SOC, Voltage, and Current."""
    if not file_path:
        print("No file selected.")
        return

    try:
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)

        # Identify SOC column (case-insensitive)
        soc_col = next((col for col in df.columns if col.lower() == 'soc'), None)
        
        if not soc_col:
            print("Error: 'SOC' column not found in the CSV file.")
            print("Available columns:", df.columns.tolist())
            return

        # Identify Time/Timestamp column (looks for variations)
        time_keywords = ['timestamp', 'time', 'date']
        time_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in time_keywords):
                time_col = col
                break
        
        # --- VOLTAGE PROCESSING ---
        volt_col = 'Battery_voltage'
        if volt_col in df.columns:
            print(f"Using '{volt_col}' for Voltage.")
            # Convert mV to V if necessary (mean > 1000 suggests mV)
            if df[volt_col].mean() > 1000:
                print("Detected mV, converting to V...")
                df[volt_col] = df[volt_col] / 1000.0
            # Filter outliers (optional but good based on other scripts)
            df = df[(df[volt_col] >= 20) & (df[volt_col] <= 120)]
            # Apply Median Filter
            print("Applying median filter to Voltage (kernel=111)...")
            df[volt_col] = medfilt(df[volt_col], kernel_size=111)
        
        # --- CURRENT PROCESSING ---
        target_curr_col = 'Battery_current1'
        status_col = 'Charge_discharge_status'
        processed_curr_col = 'Processed_Current'

        if target_curr_col in df.columns:
            print(f"Using '{target_curr_col}' for Current.")
            if status_col in df.columns:
                print(f"Applying charge/discharge logic using '{status_col}'...")
                # Logic: If status is Charging (1), ensure current is negative
                df[processed_curr_col] = df.apply(
                    lambda row: -abs(row[target_curr_col]) if row[status_col] == 1 
                    else row[target_curr_col], 
                    axis=1
                )
            else:
                print("Status column not found, using raw current.")
                df[processed_curr_col] = df[target_curr_col]
            
            # Apply Median Filter to Current
            print("Applying median filter to Current (kernel=111)...")
            df[processed_curr_col] = medfilt(df[processed_curr_col], kernel_size=111)
            curr_plot_col = processed_curr_col
        else:
             curr_plot_col = None

        # Setup Plotting
        if time_col:
            print(f"Using '{time_col}' as time axis.")
            # Convert to datetime for better plotting
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Drop rows where timestamp conversion failed only if important
            df = df.dropna(subset=[time_col])
            
            x_data = df[time_col]
            x_label = 'Time'
        else:
            print("No timestamp column found. Plotting by Index.")
            x_data = df.index
            x_label = 'Sample Index'
            
        # Create 3 Subplots sharing X axis
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # Plot 1: SOC
        ax1.plot(x_data, df[soc_col], label='SOC', color='tab:blue', linewidth=1.5)
        ax1.set_ylabel('SOC (%)')
        ax1.set_title(f'Analysis: {os.path.basename(file_path)}', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')

        # Plot 2: Voltage
        if volt_col in df.columns:
            ax2.plot(x_data, df[volt_col], label='Voltage', color='tab:green', linewidth=1.5)
            ax2.set_ylabel('Voltage (V)')
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'Voltage Data Not Found', ha='center', va='center')

        # Plot 3: Current
        if curr_plot_col and curr_plot_col in df.columns:
            ax3.plot(x_data, df[curr_plot_col], label='Current', color='tab:red', linewidth=1.5)
            ax3.set_ylabel('Current (A)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend(loc='upper right')
            # Add zero line for current
            ax3.axhline(0, color='black', linewidth=0.8, linestyle='-')
        else:
            ax3.text(0.5, 0.5, 'Current Data Not Found', ha='center', va='center')

        ax3.set_xlabel(x_label)
        
        # Format layout
        plt.tight_layout()
        
        # Show plot
        print("Displaying plot...")
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting SOC Plotter...")
    try:
        path = select_file()
        if path:
            plot_soc(path)
        else:
            print("Operation cancelled by user.")
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        input("Press Enter to close...")
