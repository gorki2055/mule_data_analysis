import pandas as pd
import numpy as np
import os
import sys

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from plotv14 import load_and_process_data

def run_test():
    # Create a dummy CSV
    filename = "MULE1_01-01-2024_MORNING.csv" # Needs to match pattern roughly for plotv14 default?
    # plotv14 doesn't seem to enforce filename pattern for loading, just path.
    # But let's use a nice name.
    
    timestamps = pd.date_range("2024-01-01 10:00:00", periods=200, freq="100ms")
    data = {
        "timestamps": timestamps,
        "Battery_voltage": [48000] * 200, 
        "Battery_current1": [10] * 200,
        "Charge_discharge_status": [0] * 200,
        "Current_Rotational_Speed": [0] * 200,
        "Ntc_Mos": [25.0] * 200
    }

    # Add a spike
    data["Ntc_Mos"][100] = 150.0 # Spike to 150C

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print("Created dummy file with spike at index 100 (150C). Background 25C.")

    # Run processing
    try:
        # load_and_process_data should filter the spike now
        df_processed = load_and_process_data(filename)
        
        if df_processed is not None and not df_processed.empty:
            max_temp = df_processed["Ntc_Mos"].max()
            print(f"Max Ntc_Mos detected: {max_temp}")
            
            # Since kernel size is 35, a single point spike should be completely removed
            if max_temp < 50:
                print("PASS: Spike was filtered.")
            else:
                print("FAIL: Spike was NOT filtered.")
        else:
            print("FAIL: Processing returned None or empty.")
            
    except Exception as e:
        print(f"FAIL: Exception occurred: {e}")
    
    # Clean up
    if os.path.exists(filename):
        os.remove(filename)

if __name__ == "__main__":
    run_test()
