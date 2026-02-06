import pandas as pd
import sys
import os

# Add current dir to path to import the module
sys.path.append(os.getcwd())
import range_estimator_gui

file_path = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\2877D237\1-30-2026\MORNING_RIDE\MULE2_1-30-2026_MORNING_RIDE.csv"

print(f"Checking file: {file_path}")
try:
    # Read only header
    df_header = pd.read_csv(file_path, nrows=1)
    # Print columns nicely
    print("--- COLUMNS ---")
    for col in df_header.columns:
        print(f"'{col}'")
    print("--- END COLUMNS ---")
    
    # Check for charger connected
    matches = [c for c in df_header.columns if 'charger' in c.lower() or 'connected' in c.lower()]
    print(f"Potential Matches: {matches}")

    if 'charger_connected' in df_header.columns or matches:
        target_col = 'charger_connected' if 'charger_connected' in df_header.columns else matches[0] if matches else None
        
        if target_col:
            print(f"Using column: {target_col}")
            # Now verify the logic with a small chunk
            print("Reading small chunk to test logic...")
            df_chunk = pd.read_csv(file_path, nrows=5000)
            df_chunk.columns = df_chunk.columns.str.strip()
            
            # Print unique values in this column
            print(f"Unique values in {target_col}: {df_chunk[target_col].unique()}")
        else:
            print("No suitable column found.")

except Exception as e:
    print(f"Error: {e}")
