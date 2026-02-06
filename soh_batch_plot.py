"""
================================================================================
BATCH SOH PLOTTER
================================================================================
Purpose: Scan a folder for CSVs, combine SOH data, and plot SOH vs Time.
Usage: python soh_batch_plot.py
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import filedialog
import glob

def select_folder():
    """GUI to select folder."""
    root = tk.Tk()
    root.withdraw() # Hide main window
    path = filedialog.askdirectory(title="Select Folder Containing Logs")
    root.destroy()
    return path

def find_csv_files(root_dir):
    """Recursively find CSV files."""
    csv_files = []
    if os.path.isfile(root_dir):
        return [root_dir]
        
    print(f"Scanning {root_dir}...")
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_batch(folder_path):
    files = find_csv_files(folder_path)
    if not files:
        print("No CSV files found.")
        return

    all_data = []

    print(f"Found {len(files)} files. Processing...")
    
    for f in files:
        try:
            # Read only relevant columns to save memory
            # We need to peek at columns first or just read all and filter
            try:
                # Read valid timestamps and SOH
                df = pd.read_csv(f, usecols=lambda c: c.lower() in ['timestamps', 'soh'], engine='python')
            except ValueError:
                # Fallback if columns don't exist under those names or other read error
                continue
            except Exception:
                 continue

            df.columns = df.columns.str.strip().str.lower()
            
            if 'timestamps' in df.columns and 'soh' in df.columns:
                df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
                df = df.dropna(subset=['timestamps', 'soh'])
                
                # Filter invalid SOH
                df = df[(df['soh'] >= 0) & (df['soh'] <= 100)]
                
                if not df.empty:
                    all_data.append(df)
                    
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}")

    if not all_data:
        print("No valid SOH data found in files.")
        return

    print("Concatenating data...")
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values('timestamps')
    
    print(f"Total data points: {len(full_df)}")
    
    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(15, 8))
    plt.plot(full_df['timestamps'], full_df['soh'], color='#1f77b4', linewidth=1.5, label='SOH')
    
    plt.title("Batch SOH vs Time", fontsize=16)
    plt.ylabel("SOH (%)", fontsize=12)
    plt.xlabel("Time (UTC)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    output_path = os.path.join(folder_path, "batch_soh_plot.png")
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    target_folder = select_folder()
    if target_folder:
        process_batch(target_folder)
    else:
        print("No folder selected.")
