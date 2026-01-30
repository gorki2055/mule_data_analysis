import os
import glob
import pandas as pd
import plotv14
import traceback

def generate_summary_csv(input_dir='.', output_file='summary.csv', files=None, save_csv=True):
    """
    Scans for CSV files, processes them using plotv14, and aggregates key metrics into a summary CSV.
    """
    # Find all CSV files in the directory
    if files:
        csv_files = files
    else:
        csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    summary_data = []
    
    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        # Skip the output file itself if it exists
        if os.path.basename(file_path) == output_file:
            continue
            
        print(f"\nProcessing {os.path.basename(file_path)}...")
        try:
            # 1. Load Data
            df_raw = plotv14.load_and_process_data(file_path)
            if len(df_raw) == 0:
                print(f"Skipping {file_path}: No valid data found.")
                continue
                
            # 2. Resample Data
            df_resampled = plotv14.resample_data(df_raw)
            if len(df_resampled) == 0:
                print(f"Skipping {file_path}: No valid data after resampling.")
                continue
                
            # 3. Calculate Metrics
            # Note: calculate_metrics returns (df, metrics_dict)
            _, metrics = plotv14.calculate_metrics(df_resampled)
            
            # 4. Extract Key Metrics for Summary
            row = {
                'Filename': os.path.basename(file_path),
                'Max Current (A)': metrics.get('max_current'),
                'Start SOC (%)': metrics.get('start_soc'),
                'End SOC (%)': metrics.get('end_soc'),
                'SOC Drop (%)': metrics.get('soc_drop'),
                'Discharge Energy (Wh)': metrics.get('cumulative_wh_discharge'),
                'Regen Energy (Wh)': metrics.get('cumulative_wh_charge'),
                'Net Energy (Wh)': metrics.get('net_energy_wh'),
                'Distance (km)': metrics.get('distance_km'),
                'Efficiency (Wh/km)': metrics.get('wh_per_km'),
                'Current Limit Violations': metrics.get('current_limit_violations'),
            }
            
            # Add Thermal Metrics
            for sensor in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']:
                row[f'{sensor} Max (C)'] = metrics.get(f'{sensor}_max')
                row[f'{sensor} Avg (C)'] = metrics.get(f'{sensor}_avg')

            # Add Pack Thermal Stats
            if 'pack_thermal' in metrics:
                pack = metrics['pack_thermal']
                row['Pack Avg Max Temp (C)'] = pack.get('avg_max_temp')
                row['Pack Peak Max Temp (C)'] = pack.get('peak_max_temp')
                row['Pack Max Delta Temp (C)'] = pack.get('max_delta_temp')

            summary_data.append(row)
            print("✓ Added to summary.")
            
        except Exception as e:
            print(f"⚠ Error processing {file_path}: {e}")
            traceback.print_exc()
            
    # 5. Write to CSV or Print
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        if save_csv:
            output_path = os.path.join(input_dir, output_file)
            df_summary.to_csv(output_path, index=False)
            print(f"\n✅ Summary successfully written to: {output_path}")
        else:
            print("\n" + "="*80)
            print(" SUMMARY DATA ")
            print("="*80)
            # Configure pandas display options to ensure all columns/rows are visible in text
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            pd.set_option('display.max_rows', None)
            print(df_summary.to_string(index=False))
            print("="*80 + "\n")
            
    else:
        print("\n❌ No data was processed successfully. Summary file not created.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_summary_csv(current_dir)