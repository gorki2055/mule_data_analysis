import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

try:
    from plotv14 import load_and_process_data
except ImportError:
    # Handle case where it might be run from a different directory or file missing
    load_and_process_data = None

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk")

class BatteryPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Battery Data Analysis Tool")
        self.root.geometry("1400x900")
        
        self.df = None
        self.filename = None

        # --- Top Control Frame ---
        control_frame = tk.Frame(root, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = tk.Button(control_frame, text="📂 Load CSV Data", command=self.load_file, 
                                  font=("Arial", 12, "bold"), bg="#2196F3", fg="white", padx=20, pady=5)
        self.btn_load.pack()
        
        self.lbl_status = tk.Label(control_frame, text="No file loaded", font=("Arial", 10, "italic"), fg="gray")
        self.lbl_status.pack(pady=5)

        # --- Main Plotting Area ---
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Initialize empty figure
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        
        # Toolbar
        self.toolbar_frame = tk.Frame(root)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if not file_path:
            return
            
        try:
            self.lbl_status.config(text="Loading data...", fg="orange")
            self.root.update()
            
            # Load Data
            self.current_file_path = file_path # Store for re-processing
            self.df = pd.read_csv(file_path)
            self.filename = file_path.split("/")[-1]
            
            # Process Data
            self.process_data()
            
            # Plot Data
            self.plot_data()
            
            self.lbl_status.config(text=f"Loaded: {self.filename}", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            self.lbl_status.config(text="Error loading file", fg="red")

    def process_data(self):
        # Use plotv14 logic if available
        if load_and_process_data:
            try:
                # Reload data using plotv14 logic
                # We need the full path to pass to load_and_process_data
                 file_path = getattr(self, 'current_file_path', None)
                 if file_path:
                    self.df = load_and_process_data(file_path)
                    # plotv14 handles timestamp parsing and column creation (Processed_Current)
                    
                    # Ensure time_obj exists for our plotting logic if not present
                    if 'timestamps' in self.df.columns and 'time_obj' not in self.df.columns:
                         self.df['time_obj'] = pd.to_datetime(self.df['timestamps'])
                    return
            except Exception as e:
                print(f"Error using plotv14 logic: {e}")
                # Fallback to manual processing if plotv14 fails
        
        # ... Fallback Manual Processing (Cleaned up) ...
        # 0. Clean Column Names
        self.df.columns = self.df.columns.str.strip()

        # 1. Standardize Timestamps (Reference: plotv14.py)
        os.environ['TZ'] = 'UTC0' # Ensure consistent timezone handling
        
        if 'timestamps' in self.df.columns:
            try:
                self.df['time_obj'] = pd.to_datetime(self.df['timestamps'], errors='coerce', utc=False)
            except:
                self.df['time_obj'] = pd.to_datetime(self.df['timestamps'], errors='coerce')
        elif 'timestamp' in self.df.columns:
             self.df['time_obj'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        else:
            # Fallback: Create index-based time if no timestamp column
            self.df['time_obj'] = self.df.index
            messagebox.showwarning("Warning", "No 'timestamps' column found. Using index as time.")

        # Drop invalid timestamps and sort
        if 'time_obj' in self.df.columns:
            self.df = self.df.dropna(subset=['time_obj']).copy()
            self.df = self.df.sort_values('time_obj').reset_index(drop=True)

        # 2. Check for required columns
        required_cols = ['Battery_voltage', 'Processed_Current', 'SOC']
        
        if 'Battery_current1' in self.df.columns:
            if 'Charge_discharge_status' in self.df.columns:
                # Force negative sign for charging state (1 = Charging)
                self.df['Processed_Current'] = self.df.apply(
                    lambda row: -abs(row['Battery_current1']) if row['Charge_discharge_status'] == 1 
                    else row['Battery_current1'], 
                    axis=1
                )
            else:
                self.df['Processed_Current'] = self.df['Battery_current1']
            
            # Filter out current values above 185A
            initial_len = len(self.df)
            self.df = self.df[self.df['Processed_Current'] <= 185]
            print(f"Filtered {initial_len - len(self.df)} rows with Current > 185A")

            # Filter out current values less than -50A
            initial_len = len(self.df)
            self.df = self.df[self.df['Processed_Current'] >= -50]
            print(f"Filtered {initial_len - len(self.df)} rows with Current < -50A")
        if 'Battery_current1' in self.df.columns:
            if 'Charge_discharge_status' in self.df.columns:
                # Force negative sign for charging state (1 = Charging)
                self.df['Processed_Current'] = self.df.apply(
                    lambda row: -abs(row['Battery_current1']) if row['Charge_discharge_status'] == 1 
                    else row['Battery_current1'], 
                    axis=1
                )
            else:
                self.df['Processed_Current'] = self.df['Battery_current1']
            
            # Filter out current values above 185A
            initial_len = len(self.df)
            self.df = self.df[self.df['Processed_Current'] <= 185]
            print(f"Filtered {initial_len - len(self.df)} rows with Current > 185A")

            # Filter out current values less than -50A
            initial_len = len(self.df)
            self.df = self.df[self.df['Processed_Current'] >= -50]
            print(f"Filtered {initial_len - len(self.df)} rows with Current < -50A")

        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
             messagebox.showwarning("Missing Data", f"The following columns are missing: {', '.join(missing)}\nPlots may be incomplete.")

    def plot_data(self):
        self.axs[0].clear()
        self.axs[1].clear()
        
        timestamps = self.df['time_obj']
        
        # --- TOP PLOT: Voltage & Current ---
        ax1 = self.axs[0]
        
        # Voltage (Left Axis)
        if 'Battery_voltage' in self.df.columns:
            color = 'tab:blue'
            ax1.set_ylabel('Voltage (V)', color=color, fontsize=12, fontweight='bold')
            ax1.plot(timestamps, self.df['Battery_voltage'], color=color, label='Voltage', linewidth=1.5)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
        
        # Current (Right Axis)
        if 'Processed_Current' in self.df.columns:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Current (A)', color=color, fontsize=12, fontweight='bold')
            ax2.plot(timestamps, self.df['Processed_Current'], color=color, label='Current', linewidth=1.0, alpha=0.8)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.grid(False) # Avoid clutter
            
            # Add zero line for current
            ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

        ax1.set_title(f"Battery Voltage & Current - {self.filename}", fontsize=14, fontweight='bold')
        
        # --- BOTTOM PLOT: SOC ---
        ax3 = self.axs[1]
        if 'SOC' in self.df.columns:
            color = 'tab:green'
            ax3.set_ylabel('SOC (%)', color=color, fontsize=12, fontweight='bold')
            ax3.plot(timestamps, self.df['SOC'], color=color, label='SOC', linewidth=2)
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_ylim(0, 105) # SOC is usually 0-100
            ax3.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = BatteryPlotterApp(root)
    root.mainloop()
