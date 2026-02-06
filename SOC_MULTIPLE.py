
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import medfilt

# --- 1. THE LIST OF FILES TO PROCESS ---
# I have cleaned up the formatting from your request and added 'r' for raw strings
files_to_process = [
    r"D:\kushal\LOG\LOG\3ACF5792\1-11-2026\DAY_CHARGING\MULE1_1-11-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-11-2026\MORNING_RIDE\MULE1_1-11-2026_MORNING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-11-2026\NIGHT_CHARGING\MULE1_1-11-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-13-2026\DAY_CHARGING\MULE1_1-13-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-13-2026\DAY_RIDE\MULE1_1-13-2026_DAY.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-13-2026\MORNING_RIDE\MULE1_1-13-2026_MORNING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-14-2026\NIGHT_CHARGING\MULE1_1-14-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-15-2026\DAY_CHARGING\MULE1_1-15-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-15-2026\NIGHT_CHARGING\MULE1_1-15-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-2-2026\CHARGING(10_38-13_54)\MULE1_1-2-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-2-2026\EVENING_RIDE\MULE1_1-2-2026_EVENING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-3-1026\NIGHT CHARGING(20_07-6_21)\MULE1_1-3-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-4-2026\DAY_CHARGING(12_53-14_4)\MULE1_1-4-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-4-2026\MORNING_CHARGING(10_31-12_52)\MULE1_1-4-2026_MORNING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-4-2026\MORNING_RIDE\MULE1_1-4-2026_MORNING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-5-2026\DAY_CHARGING\MULE1_1-5-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-6-2026\DAY_RIDE\MULE1_1-6-2026_DAY.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-6-2026\MORNING_CHARGING\MULE1_1-6-2026_MORNING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-6-2026\NIGHT_CHARGING\MULE1_1-6-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-7-2026\DAY_CHARGING\MULE1_1-7-2026_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-7-2026\NIGHT_CHARGING\MULE1_1-7-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\1-9-2026\NIGHT_CHARGING\MULE1_1-9-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-28-2025\EVENING_RIDE\MULE1_28-12-2025_EVENING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-28-2025\NIGHT_CHARGING\MULE1_12-28-2025_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-29-2025\11-14_14-38\MULE1_12-29-22025_DAY_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-29-2025\MULE1_29-12-2025_EVENING\MULE1_12-29-2025_EVENING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-29-2025\MULE1_29-12-2025_MORNING\MULE1_12-29-2025_MORNING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-30-2025\MULE1_30-12-2025_EVENING\MULE1_12-30-2025_EVENING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-31-2025\EVENING_RIDE\MULE1_12-31-2025_EVENING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\12-31-2025\NIGHT_CHARGING\MULE1_12-31-2025_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-1-2026\DAY_RIDE\MULE1_2-1-2026_DAY_RIDE.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-1-2026\EVENING_CHARGING\MULE1_2-1-2026_EVENING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-1-2026\NIGHT_CHARGING\MULE1_2-1-2026_NIGHT_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-2-2026\EVENING_RIDE\MULE1_2-2-2026_EVENING_RIDE.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-2-2026\MORNING_CHARGING\MULE1_2-2-2026_MORNING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-3-2026\EVENING_CHARGING\MULE1_2-3-2026_EVENING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-3-2026\MORNING_CHARGING\MULE1_2-3-2026_MORNING_CHARGING.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-3-2026\MORNING_RIDE\MULE1_2-3-2026_MORNING_RIDE.csv",
    r"D:\kushal\LOG\LOG\3ACF5792\2-3-2026\NIGHT_CHARGING\MULE1_2-3-2026_NIGHT_CHARGING.csv"
]

def process_and_plot(file_list):
    """Iterates through file list, plots SOC, and saves image."""
    
    # Create an output directory for the graphs
    output_dir = "soc_anomoly"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {os.path.abspath(output_dir)}")
    else:
        print(f"Saving graphs to: {os.path.abspath(output_dir)}")

    # Remove duplicates from list to save processing time
    unique_files = list(set(file_list))
    total_files = len(unique_files)
    
    print(f"Found {total_files} unique files to process.")
    print("-" * 50)

    for i, file_path in enumerate(unique_files):
        # Clean up path (strip whitespace)
        file_path = file_path.strip()
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"[{i+1}/{total_files}] SKIPPED (File not found): {file_path}")
            continue

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # --- DATA PROCESSING (SAME AS BEFORE) ---
            # --- DATA PROCESSING ---
            soc_col = next((col for col in df.columns if col.lower() == 'soc'), None)
            
            if not soc_col:
                print(f"[{i+1}/{total_files}] SKIPPED (No SOC column): {os.path.basename(file_path)}")
                continue

            # Standardize column names (strip spaces)
            df.columns = df.columns.str.strip()

            time_keywords = ['timestamp', 'time', 'date']
            time_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in time_keywords):
                    time_col = col
                    break
            

            # --- VOLTAGE PROCESSING ---
            volt_col = 'Battery_voltage'
            if volt_col in df.columns:
                # Convert mV to V if necessary (mean > 1000 suggests mV)
                if df[volt_col].mean() > 1000:
                    df[volt_col] = df[volt_col] / 1000.0
                # Filter outliers
                df = df[(df[volt_col] >= 20) & (df[volt_col] <= 120)]
                # Apply Median Filter
                df[volt_col] = medfilt(df[volt_col], kernel_size=111)
            
            # --- CURRENT PROCESSING ---
            curr_col = 'Battery_current1'
            status_col = 'Charge_discharge_status'
            processed_curr_col = 'Processed_Current'

            if curr_col in df.columns:
                if status_col in df.columns:
                    # Logic: If status is Charging (1), ensure current is negative
                    df[processed_curr_col] = df.apply(
                        lambda row: -abs(row[curr_col]) if row[status_col] == 1 
                        else row[curr_col], 
                        axis=1
                    )
                else:
                    df[processed_curr_col] = df[curr_col]
                
                # Apply Median Filter to Current
                df[processed_curr_col] = medfilt(df[processed_curr_col], kernel_size=111)
            
            # --- PLOTTING ---
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Handle Time Axis
            x_data = None
            x_label = 'Sample Index'
            
            if time_col:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                # Drop rows where critical data or time is missing
                df = df.dropna(subset=[time_col, soc_col])
                x_data = df[time_col]
                x_label = 'Time'
            else:
                x_data = df.index
            
            # Plot 1: SOC
            ax1.plot(x_data, df[soc_col], label='SOC', color='tab:blue', linewidth=1.5)
            ax1.set_ylabel('SOC (%)')
            ax1.set_title(f'Analysis: {os.path.basename(file_path)}')
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
            if processed_curr_col in df.columns:
                ax3.plot(x_data, df[processed_curr_col], label='Current', color='tab:red', linewidth=1.5)
                ax3.set_ylabel('Current (A)')
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.legend(loc='upper right')
                
                # Add zero line for current
                ax3.axhline(0, color='black', linewidth=0.8, linestyle='-')
            else:
                ax3.text(0.5, 0.5, 'Current Data Not Found', ha='center', va='center')

            plt.xlabel(x_label)
            plt.tight_layout()
            
            file_name = os.path.basename(file_path)

            # --- SAVING (INSTEAD OF SHOWING) ---
            # Create a safe filename for the image (replace .csv with .png)
            save_name = file_name.replace('.csv', '').replace('.CSV', '') + "_plot.png"
            save_path = os.path.join(output_dir, save_name)
            
            plt.savefig(save_path)
            plt.close() # Close plot to free memory
            
            print(f"[{i+1}/{total_files}] SUCCESS: Saved {save_name}")

        except Exception as e:
            print(f"[{i+1}/{total_files}] ERROR on {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    print("Starting Batch SOC Plotter...")
    try:
        process_and_plot(files_to_process)
        print("-" * 50)
        print("Batch processing complete.")
    except KeyboardInterrupt:
        print("\nProgram interrupted.")
    except Exception as e:
        print(f"Unexpected global error: {e}")
    
    input("Press Enter to close...")