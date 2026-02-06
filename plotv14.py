"""
================================================================================
BATTERY SYSTEM ANALYSIS TOOL (Synced with Batch Logic)
================================================================================
Purpose: Comprehensive battery analysis with accurate distance calculation
Features: Voltage/current analysis, SOC tracking, temperature monitoring, 
          distance/energy, ride mode efficiency comparison, current status visualization
Input: CSV file with timestamps, voltage, current, SOC, temperature data
Output: 13 essential plots + comprehensive summary with ride mode analysis
================================================================================
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns
import os
from scipy.signal import medfilt
import tkinter as tk
from tkinter import filedialog

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = "MULE1_1-8-2026_MORNING.csv"
GEAR_RATIO = 6.09        # GEAR RATIO AFTER 4 JAN IS 6.09(QS) before 4jan 6.36
WHEEL_RADIUS_M = 0.261
RESAMPLE_RATE = '100ms'  # Matches Script 2 exactly
RPM_THRESHOLD = 5.0      # Threshold to distinguish Regen (Moving) vs Charger (Stopped)

# Ride mode mapping
MODE_MAP = {
    0: "Suste",
    1: "Thikka", 
    2: "Babbal"
}

# Status color mapping
STATUS_COLORS = {
    0: '#FF0000',  # Discharging - Red
    1: '#00AA00',  # Charging - Green
    2: '#0000FF'   # Idle - Blue
}

STATUS_NAMES = {
    0: 'Discharging',
    1: 'Charging',
    2: 'Idle'
}

# Plot styling
PLOT_STYLES = {
    'fig_size_wide': (14, 6),
    'fig_size_standard': (10, 7),
    'line_width_thin': 0.8,
    'line_width_standard': 1.0,
    'line_width_thick': 2.0,
    'grid_alpha': 0.3,
    'marker_size': 6,
    'dpi': 150,
    'colors': {
        'voltage': '#1f77b4',
        'current': '#d62728',
        'soc': '#ff7f0e',
        'power': '#9467bd',
        'permissible': '#2ca02c',
        'energy': '#27AE60',
        'suste': '#2ca02c',
        'thikka': '#ff7f0e',
        'babbal': '#d62728'
    }
}

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

def load_and_process_data(file_path):
    """Load and prepare battery data with ride mode mapping."""
    import os
    os.environ['TZ'] = 'UTC0'
    
    try:
        df = pd.read_csv(file_path, engine='python', low_memory=False)
    except:
        print("⚠ Retrying with 'c' engine...")
        df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded: {df.shape}")
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse timestamps
    if 'timestamps' in df.columns:
        try:
            df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce', utc=False)
        except:
            df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
    
    # Convert mV to V if needed (Matches Script 2)
    if 'Battery_voltage' in df.columns:
        if df['Battery_voltage'].mean() > 1000:
            df['Battery_voltage'] = df['Battery_voltage'] / 1000.0
        df = df[df['Battery_voltage'] <= 120]
        df = df[df['Battery_voltage'] >= 20]
    
    # Map ride modes
    if 'MCU_Speed_Gear' in df.columns:
        df["Ride_Mode"] = df["MCU_Speed_Gear"].map(MODE_MAP)
        print(f"Ride modes detected: {df['Ride_Mode'].dropna().unique().tolist()}")
    else:
        df["Ride_Mode"] = None
    
    # Process battery current based on status (Matches Script 2 Logic)
    if 'Battery_current1' in df.columns:
        if 'Charge_discharge_status' in df.columns:
            # Force negative sign for charging state (1 = Charging)
            df['Processed_Current'] = df.apply(
                lambda row: -abs(row['Battery_current1']) if row['Charge_discharge_status'] == 1 
                else row['Battery_current1'], 
                axis=1
            )
        else:
            df['Processed_Current'] = df['Battery_current1']
        
        # Filter out current values above 185A
        initial_len = len(df)
        df = df[df['Processed_Current'] <= 185]
        print(f"Filtered {initial_len - len(df)} rows with Current > 185A")

        # Filter out current values less than -50A
        initial_len = len(df)
        df = df[df['Processed_Current'] >= -50]
        print(f"Filtered {initial_len - len(df)} rows with Current < -50A")
    
    # Drop invalid timestamps
    df = df.dropna(subset=['timestamps']).copy()
    print(f"Rows preserved for analysis: {len(df)}")
    
    # Sort by timestamps
    df = df.sort_values('timestamps').reset_index(drop=True)

    # Filter Temperatures (Matches Script 2 Logic)
    ntc_sensors = ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']

    for sensor in ntc_sensors:
        if sensor in df.columns:
            # 1. HARD MASK
            df.loc[(df[sensor] < -40) | (df[sensor] > 75), sensor] = np.nan
            
            # 2. 10-MINUTE CONSENSUS BASELINE
            baseline = df[sensor].rolling(window=60001, center=True, min_periods=1).median()
            
            # 3. SURGICAL REJECTION
            invalid_mask = (df[sensor] - baseline).abs() > 2.0
            df.loc[invalid_mask, sensor] = np.nan
            
            # 4. INTERPOLATE
            df[sensor] = df[sensor].interpolate(method='linear', limit_direction='both')
            
            # 5. FINAL SMOOTHING
            if not df[sensor].isna().all():
                df[sensor] = medfilt(df[sensor], kernel_size=31)
    
    # Filter Permissible Current Spikes
    if 'Permissible_discharge_current' in df.columns:
        # Use a kernel size of 11 for effective spike removal
        df['Permissible_discharge_current'] = medfilt(df['Permissible_discharge_current'], kernel_size=11)
        print("✓ Applied median filter to Permissible_discharge_current")

    return df

def resample_data(df):
    """Resample data and Apply STRICT Power Filtering (Matches Script 2)."""
    print("⏳ Resampling data & Filtering Spikes...")
    
    # Create a copy for resampling
    df_ts = df.set_index('timestamps')
    
    # 1. Resample Numeric Data (Mean)
    numeric_cols = df_ts.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df_ts[numeric_cols]
    df_resampled = df_numeric.resample(RESAMPLE_RATE).mean()
    
    # 2. INTERPOLATE Numeric Data
    df_resampled = df_resampled.interpolate(method='linear')
    
    # 3. Handle Categorical Data
    if 'Ride_Mode' in df.columns:
        mode_series = df.set_index('timestamps')['Ride_Mode']
        mode_resampled = mode_series.resample(RESAMPLE_RATE).first()
        df_resampled['Ride_Mode'] = mode_resampled.ffill()
    
    if 'Charge_discharge_status' in df.columns:
        status_series = df.set_index('timestamps')['Charge_discharge_status']
        status_resampled = status_series.resample(RESAMPLE_RATE).first()
        df_resampled['Charge_discharge_status'] = status_resampled.ffill()

    # 4. --- STRICT SPIKE FILTERING (Script 2 Logic) ---
    # We calculate power immediately and drop the rows.
    if 'Battery_voltage' in df_resampled.columns and 'Processed_Current' in df_resampled.columns:
        # Calculate tentative power
        df_resampled['Power_W'] = df_resampled['Battery_voltage'] * df_resampled['Processed_Current']
        
        # Filter Rows > 20kW (20000W)
        initial_count = len(df_resampled)
        mask = df_resampled['Power_W'] <= 20000 
        df_resampled = df_resampled[mask].copy()
        
        dropped_count = initial_count - len(df_resampled)
        
        # Also calculate kW for plotting
        df_resampled['Power_kW'] = df_resampled['Power_W'] / 1000.0
        
        if dropped_count > 0:
            print(f"⚠ Filtered {dropped_count} rows with Power > 20kW (Spikes/Noise)")

    print(f"✓ Resampling & Filtering complete: {len(df_resampled)} rows")
    return df_resampled.reset_index()

# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_metrics(df):
    """Calculate metrics using Script 2 Logic (Regen vs Charger)."""
    print("⏳ Calculating metrics...")
    metrics = {}
    
    # Basic electrical metrics (From Filtered Data)
    metrics['max_current'] = df['Processed_Current'].max()
    metrics['start_soc'] = df['SOC'].iloc[0] if 'SOC' in df.columns else None
    metrics['end_soc'] = df['SOC'].iloc[-1] if 'SOC' in df.columns else None
    if metrics['start_soc'] and metrics['end_soc']:
        metrics['soc_drop'] = metrics['start_soc'] - metrics['end_soc']
    
    # Absolute current for mode analysis
    df["Battery_current_abs"] = df["Processed_Current"].abs()
    
    # --- FIXED TIME STEP & ENERGY (Script 2 Logic) ---
    fixed_dt_seconds = 0.1 
    fixed_dt_hours = fixed_dt_seconds / 3600.0
    df['time_diff_hours'] = fixed_dt_hours
    
    # Calculate Instant Energy Increment
    # Note: Power_W is already calculated and filtered in resample_data
    df['Energy_increment_Wh'] = df['Power_W'] * df['time_diff_hours']
    
    # --- REGEN VS CHARGER DISTINCTION (Script 2 Logic) ---
    is_charging = df['Processed_Current'] < 0
    
    # Check if vehicle is moving (Regen) vs Stopped (Charger)
    if 'Current_Rotational_Speed' in df.columns:
        is_moving = df['Current_Rotational_Speed'] > RPM_THRESHOLD
    else:
        # Fallback if no speed data: assume all charging is plug-in if we can't tell
        is_moving = False 
    
    # 1. Discharge: Energy > 0
    df['Energy_discharge_Wh'] = df['Energy_increment_Wh'].clip(lower=0)
    
    # 2. Regen: Energy < 0 AND Moving
    # We use -Energy to make the Wh value positive for reporting
    df['Regen_Energy_Wh'] = 0.0
    df.loc[(is_charging) & (is_moving), 'Regen_Energy_Wh'] = -df.loc[(is_charging) & (is_moving), 'Energy_increment_Wh']
    
    # 3. Charger: Energy < 0 AND Stopped
    df['Charger_Energy_Wh'] = 0.0
    df.loc[(is_charging) & (~is_moving), 'Charger_Energy_Wh'] = -df.loc[(is_charging) & (~is_moving), 'Energy_increment_Wh']

    # Cumulative calculations
    df['Cumulative_Energy_Wh'] = df['Energy_discharge_Wh'].cumsum()
    df['Cumulative_Regen_Wh'] = df['Regen_Energy_Wh'].cumsum()
    df['Cumulative_Charger_Wh'] = df['Charger_Energy_Wh'].cumsum()
    
    # Net Energy = Discharge - Regen - Charger
    df['Net_Cumulative_Energy_Wh'] = df['Cumulative_Energy_Wh'] - df['Cumulative_Regen_Wh'] - df['Cumulative_Charger_Wh']
    
    # Metrics Summary
    metrics['cumulative_wh_discharge'] = df['Energy_discharge_Wh'].sum()
    metrics['cumulative_wh_regen'] = df['Regen_Energy_Wh'].sum()
    metrics['cumulative_wh_charger'] = df['Charger_Energy_Wh'].sum()
    metrics['net_energy_wh'] = metrics['cumulative_wh_discharge'] - metrics['cumulative_wh_regen'] - metrics['cumulative_wh_charger']
    
    # --- DISTANCE CALCULATION (Script 2 Physics) ---
    metrics['distance_km'] = None
    
    if 'Current_Rotational_Speed' in df.columns:
        # 1. Calculate Speed
        wheel_rpm = df['Current_Rotational_Speed'] / GEAR_RATIO
        df['Speed_kmh'] = (wheel_rpm * 2 * np.pi / 60) * WHEEL_RADIUS_M * 3.6
        
        # 2. APPLY SPEED FILTER
        df.loc[df['Speed_kmh'] > 110, 'Speed_kmh'] = 0 # Matches Script 2 (110 cutoff)
        
        # 3. Calculate Distance
        df['distance_km'] = df['Speed_kmh'] * df['time_diff_hours']
        
        # 4. Total Sum
        metrics['distance_km'] = df['distance_km'].sum()
    
    # Energy efficiency
    if metrics['distance_km'] and metrics['distance_km'] > 0.1:
        metrics['wh_per_km'] = metrics['net_energy_wh'] / metrics['distance_km']
        metrics['wh_per_km_discharge'] = metrics['cumulative_wh_discharge'] / metrics['distance_km']
    else:
        metrics['wh_per_km'] = 0
        metrics['wh_per_km_discharge'] = 0
    
    # Temperature metrics
    ntc_sensors = ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
    for sensor in ntc_sensors:
        if sensor in df.columns:
            metrics[f'{sensor}_max'] = df[sensor].max()
            metrics[f'{sensor}_avg'] = df[sensor].mean()
    
    # Current limit analysis
    if 'Permissible_discharge_current' in df.columns:
        over_limit = df[df['Battery_current1'] > df['Permissible_discharge_current']]
        metrics['current_limit_violations'] = len(over_limit)
        metrics['current_limit_percent'] = len(over_limit) / len(df) * 100 if len(df) > 0 else 0
    
    # Status distribution analysis
    if 'Charge_discharge_status' in df.columns:
        metrics['status_distribution'] = {}
        for status in [0, 1, 2]:
            count = (df['Charge_discharge_status'] == status).sum()
            percentage = (count / len(df)) * 100
            metrics['status_distribution'][status] = {
                'count': count,
                'percentage': percentage,
                'name': STATUS_NAMES[status]
            }
    
    # Processed current statistics
    if 'Processed_Current' in df.columns:
        metrics['processed_current_stats'] = {
            'min': df['Processed_Current'].min(),
            'max': df['Processed_Current'].max(),
            'mean': df['Processed_Current'].mean(),
            'std': df['Processed_Current'].std()
        }
    
    # Ride mode statistics
    if 'Ride_Mode' in df.columns:
        metrics['mode_stats'] = {}
        df_modes = df.dropna(subset=['Ride_Mode'])

        if not df_modes.empty:
            # Current stats by mode
            current_stats = df_modes.groupby("Ride_Mode")["Battery_current_abs"].agg(
                Avg_Current_A="mean",
                Max_Current_A="max"
            )
            metrics['mode_stats']['current'] = current_stats
            
            # Power stats by mode
            power_stats = df_modes.groupby("Ride_Mode")["Power_W"].agg(
                Avg_Power_W="mean",
                Max_Power_W="max"
            )
            metrics['mode_stats']['power'] = power_stats
            
            # Energy efficiency by mode
            if 'distance_km' in df_modes.columns:
                df_modes = df_modes.copy()
                energy_stats = df_modes.groupby("Ride_Mode")[["Energy_increment_Wh", "distance_km"]].sum()
                energy_stats["Wh_per_km"] = energy_stats["Energy_increment_Wh"] / energy_stats["distance_km"]
                metrics['mode_stats']['energy'] = energy_stats
    
    # Battery thermal stress analysis
    temp_cols = [c for c in ntc_sensors if c in df.columns]
    if temp_cols:
        df_temp = df[temp_cols].dropna(how="all").copy()
        if len(df_temp) > 0:
            df_temp["Temp_Max"] = df_temp[temp_cols].max(axis=1)
            df_temp["Temp_Min"] = df_temp[temp_cols].min(axis=1)
            df_temp["Temp_Delta"] = df_temp["Temp_Max"] - df_temp["Temp_Min"]
            
            thermal_stats = pd.DataFrame({
                "Avg_Temp_C": df_temp[temp_cols].mean(),
                "Max_Temp_C": df_temp[temp_cols].max(),
                "P95_Temp_C": df_temp[temp_cols].quantile(0.95)
            })
            metrics['thermal_stats'] = thermal_stats
            
            metrics['pack_thermal'] = {
                'avg_max_temp': df_temp["Temp_Max"].mean(),
                'peak_max_temp': df_temp["Temp_Max"].max(),
                'p95_temp': df_temp["Temp_Max"].quantile(0.95),
                'avg_delta_temp': df_temp["Temp_Delta"].mean(),
                'max_delta_temp': df_temp["Temp_Delta"].max()
            }
    
    print("✓ Metrics calculated successfully")
    return df, metrics

# ============================================================================
# VISUALIZATION FUNCTIONS (UNCHANGED)
# ============================================================================

def generate_battery_current_heatmap(df, output_dir):
    """Generate battery current heatmap."""
    try:
        plt.figure(figsize=(12, 4))
        sns.heatmap(
            df[['Processed_Current']].T,
            cmap="viridis",
            cbar_kws={'label': 'Battery Current (A)'}
        )
        plt.title(f"Battery Current Heatmap\nSampling: {RESAMPLE_RATE}")
        plt.xlabel(f"Time Index ({RESAMPLE_RATE} intervals)")
        plt.ylabel("Current")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "battery_current_heatmap.png"), dpi=PLOT_STYLES['dpi'])
        plt.close()
        print("✓ Generated: battery_current_heatmap.png")
    except Exception as e:
        print(f"⚠ Error generating heatmap: {e}")

def generate_power_profile_analysis(df, output_dir):
    """Generate power profile analysis plot."""
    try:
        fig = plt.figure(figsize=(PLOT_STYLES['fig_size_wide'][0], 8))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        
        # Subplot 1: Power (kW)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['timestamps'], df['Power_kW'], 
                 color=PLOT_STYLES['colors']['power'], 
                 linewidth=PLOT_STYLES['line_width_thin'],
                 label='Power (kW)')
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax1.set_ylabel('Power (kW)', fontsize=11, color=PLOT_STYLES['colors']['power'])
        ax1.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['power'])
        ax1.grid(True, alpha=PLOT_STYLES['grid_alpha'])
        ax1.legend(loc='upper left')
        ax1.set_title(f"Power Profile Analysis\nStart: {df['timestamps'].iloc[0]}", fontsize=12)
        
        # Subplot 2: Voltage and Current (dual axis)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_ylabel('Battery Voltage (V)', color=PLOT_STYLES['colors']['voltage'], fontsize=11)
        ax2.plot(df['timestamps'], df['Battery_voltage'], 
                 color=PLOT_STYLES['colors']['voltage'], 
                 linewidth=PLOT_STYLES['line_width_thin'],
                 label='Voltage (V)')
        ax2.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['voltage'])
        ax2.grid(True, alpha=PLOT_STYLES['grid_alpha'])
        
        ax2_right = ax2.twinx()
        ax2_right.set_ylabel('Battery Current (A)', color=PLOT_STYLES['colors']['current'], fontsize=11)
        ax2_right.plot(df['timestamps'], df['Processed_Current'], 
                        color=PLOT_STYLES['colors']['current'], 
                        linewidth=PLOT_STYLES['line_width_thin'],
                        label='Current (A)')
        ax2_right.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['current'])
        
        lines2 = ax2.get_lines() + ax2_right.get_lines()
        labels2 = [l.get_label() for l in lines2]
        ax2.legend(lines2, labels2, loc='upper left')
        
        # Subplot 3: Cumulative Energy (Wh)
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(df['timestamps'], df['Cumulative_Energy_Wh'], 
                 color=PLOT_STYLES['colors']['energy'], 
                 linewidth=PLOT_STYLES['line_width_standard'],
                 label='Cumulative Discharge (Wh)')
        
        if df['Cumulative_Regen_Wh'].max() > 0:
            ax3.plot(df['timestamps'], df['Cumulative_Regen_Wh'], 
                     color='#E67E22', 
                     linewidth=PLOT_STYLES['line_width_standard'],
                     linestyle='-',
                     label='Cumulative Regen (Wh)')
                     
        ax3.plot(df['timestamps'], df['Net_Cumulative_Energy_Wh'], 
                 color='#2C3E50', 
                 linewidth=PLOT_STYLES['line_width_thick'],
                 label='Net Energy (Wh)')
        
        ax3.set_ylabel('Energy (Wh)', fontsize=11)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.grid(True, alpha=PLOT_STYLES['grid_alpha'])
        ax3.legend(loc='upper left')
        
        for ax in [ax1, ax2, ax3]:
            ax.tick_params(axis='x', rotation=15)
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "power_profile_analysis.png"), dpi=PLOT_STYLES['dpi'])
        plt.close()
        print("✓ Generated: power_profile_analysis.png")
    except Exception as e:
        print(f"⚠ Error generating power profile plot: {e}")

def generate_battery_current_status_plot(df, output_dir):
    """Generate battery current with charge/discharge status plot."""
    if 'Charge_discharge_status' not in df.columns or 'Processed_Current' not in df.columns:
        print("⚠ Skipping battery current status plot - required columns not found")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot baseline
        ax.plot(df['timestamps'], df['Processed_Current'], 
                color='#CCCCCC', linewidth=2, alpha=0.5, zorder=1, label='_nolegend_')
        
        # Create colored segments
        segments_by_status = {0: [], 1: [], 2: []}
        
        i = 0
        while i < len(df):
            current_status = df.iloc[i]['Charge_discharge_status']
            segment_start = i
            while i < len(df) and df.iloc[i]['Charge_discharge_status'] == current_status:
                i += 1
            segment_end = i
            
            segment_times = df['timestamps'].iloc[segment_start:segment_end].values
            segment_currents = df['Processed_Current'].iloc[segment_start:segment_end].values
            
            if len(segment_times) > 1:
                segment_times_num = mdates.date2num(segment_times)
                points = np.array([segment_times_num, segment_currents]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                if current_status in segments_by_status:
                    segments_by_status[current_status].extend(segments)
        
        legend_handles = []
        for status in [0, 1, 2]:
            if segments_by_status[status]:
                lc = LineCollection(segments_by_status[status], 
                                  colors=STATUS_COLORS[status], 
                                  linewidths=2.5, 
                                  zorder=2)
                ax.add_collection(lc)
                legend_handles.append(plt.Line2D([0], [0], color=STATUS_COLORS[status], 
                                               linewidth=2.5, label=STATUS_NAMES[status]))
        
        ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.7, zorder=1)
        ax.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Battery Current (A)', fontsize=12, fontweight='bold')
        ax.set_title('Battery Current with Charge/Discharge Status', fontsize=14, fontweight='bold', pad=20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(handles=legend_handles, loc='best', framealpha=0.9, edgecolor='gray', fancybox=True)
        
        ax.set_xlim(df['timestamps'].min(), df['timestamps'].max())
        y_margin = (df['Processed_Current'].max() - df['Processed_Current'].min()) * 0.1
        ax.set_ylim(df['Processed_Current'].min() - y_margin, df['Processed_Current'].max() + y_margin)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "battery_current_status.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated: battery_current_status.png")
    except Exception as e:
        print(f"⚠ Error generating battery current status plot: {e}")

def generate_voltage_current_plot(df, output_dir):
    """Generate voltage and current vs time plot."""
    try:
        fig, ax1 = plt.subplots(figsize=PLOT_STYLES['fig_size_wide'])
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Battery Voltage (V)', color=PLOT_STYLES['colors']['voltage'])
        ax1.plot(df['timestamps'], df['Battery_voltage'], 
                 label="Battery Voltage (V)", 
                 color=PLOT_STYLES['colors']['voltage'], 
                 linewidth=PLOT_STYLES['line_width_thin'])
        ax1.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['voltage'])
        ax1.grid(True, alpha=PLOT_STYLES['grid_alpha'])
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Battery Current (A)', color=PLOT_STYLES['colors']['current'])
        ax2.plot(df['timestamps'], df['Processed_Current'], 
                 label="Battery Current (A)", 
                 color=PLOT_STYLES['colors']['current'], 
                 linewidth=PLOT_STYLES['line_width_thin'])
        ax2.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['current'])
        
        plt.title(f"Battery Voltage and Current vs Time\nStart: {df['timestamps'].iloc[0]}")
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "voltage_current_vs_time.png"), dpi=PLOT_STYLES['dpi'])
        plt.close()
        print("✓ Generated: voltage_current_vs_time.png")
    except Exception as e:
        print(f"⚠ Error generating voltage/current plot: {e}")

# def generate_permissible_current_plot(df, output_dir):
#     """Generate permissible vs battery current plot."""
#     try:
#         if 'Permissible_discharge_current' in df.columns and 'Processed_Current' in df.columns:
#             plt.figure(figsize=PLOT_STYLES['fig_size_wide'])

#             plt.plot(df['timestamps'], df['Processed_Current'],
#                      label="Battery Current (A)",
#                      color=PLOT_STYLES['colors']['current'],
#                      linewidth=PLOT_STYLES['line_width_thin'])

#             plt.plot(df['timestamps'], df['Permissible_discharge_current'],
#                      label="Permissible Discharge Current (A)",
#                      color=PLOT_STYLES['colors']['permissible'],
#                      linewidth=PLOT_STYLES['line_width_thin'],
#                      linestyle='-')

#             ntc_cols = [col for col in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
#                         if col in df.columns]
#             ntc_colors = ["#4a0404", '#1f77b4', '#2ca02c', "#ea954b", "#7c05ec", '#8c564b']

#             for i, col in enumerate(ntc_cols):
#                 plt.plot(df['timestamps'], df[col],
#                          label=f"{col} (°C)",
#                          color=ntc_colors[i % len(ntc_colors)],
#                          linewidth=PLOT_STYLES['line_width_thin'],
#                          linestyle='--')

#             plt.title(f"Battery Current, Permissible Current & NTC Temps\nStart: {df['timestamps'].iloc[0]}")
#             plt.xlabel("Time")
#             plt.ylabel("Current (A) / Temperature (°C)")
#             plt.legend(loc='best')
#             plt.grid(True, alpha=PLOT_STYLES['grid_alpha'])
#             plt.gcf().autofmt_xdate()
#             plt.tight_layout()

#             plt.savefig(os.path.join(output_dir, "permissible_vs_battery_current.png"),
#                         dpi=PLOT_STYLES['dpi'])
#             plt.close()
#             print("✓ Generated: permissible_vs_battery_current.png")
#     except Exception as e:
#         print(f"⚠ Error generating permissible current plot: {e}")
# from scipy.signal import medfilt
# import os
# import matplotlib.pyplot as plt

def generate_permissible_current_plot(df, output_dir, kernel_size=5):
    """Generate permissible vs filtered battery current plot with dual axes."""
    try:
        if 'Permissible_discharge_current' in df.columns and 'Processed_Current' in df.columns:
            # Create figure and primary axis (Current)
            fig, ax1 = plt.subplots(figsize=PLOT_STYLES['fig_size_wide'])
            
            # Apply Median Filter to Battery Current
            # Ensure kernel_size is odd
            filtered_current = medfilt(df['Processed_Current'], kernel_size=kernel_size)

            # Plot Current on Left Axis
            l1 = ax1.plot(df['timestamps'], filtered_current,
                     label=f"Battery Current (Filtered, k={kernel_size})",
                     color=PLOT_STYLES['colors']['current'],
                     linewidth=PLOT_STYLES['line_width_thin'])

            l2 = ax1.plot(df['timestamps'], df['Permissible_discharge_current'],
                     label="Permissible Discharge Current (A)",
                     color=PLOT_STYLES['colors']['permissible'],
                     linewidth=PLOT_STYLES['line_width_thin'],
                     linestyle='-')

            ax1.set_xlabel("Time")
            ax1.set_ylabel("Current (A)", color=PLOT_STYLES['colors']['current'])
            ax1.tick_params(axis='y', labelcolor=PLOT_STYLES['colors']['current'])
            ax1.grid(True, alpha=PLOT_STYLES['grid_alpha'])

            # Create secondary axis (Temperature)
            ax2 = ax1.twinx()
            
            ntc_cols = [col for col in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
                        if col in df.columns]
            ntc_colors = ["#4a0404", '#1f77b4', '#2ca02c', "#ea954b", "#7c05ec", '#8c564b']
            
            lines_temp = []
            for i, col in enumerate(ntc_cols):
                l = ax2.plot(df['timestamps'], df[col],
                         label=f"{col} (°C)",
                         color=ntc_colors[i % len(ntc_colors)],
                         linewidth=PLOT_STYLES['line_width_thin'],
                         linestyle='--')
                lines_temp.extend(l)

            ax2.set_ylabel("Temperature (°C)", color="#4a0404")
            ax2.tick_params(axis='y', labelcolor="#4a0404")

            # Combine legends
            lines = l1 + l2 + lines_temp
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize='small', ncol=2)

            plt.title(f"Battery Current (Filtered) & Permissible Current vs NTC Temps\nStart: {df['timestamps'].iloc[0]}")
            plt.gcf().autofmt_xdate()
            plt.tight_layout()

            plt.savefig(os.path.join(output_dir, "permissible_vs_battery_current.png"),
                        dpi=PLOT_STYLES['dpi'])
            plt.close()
            print(f"✓ Generated: permissible_vs_battery_current.png (Dual Axis, k={kernel_size})")
            
    except Exception as e:
        print(f"⚠ Error generating permissible current plot: {e}")


def generate_soc_time_plot(df, output_dir):
    """Generate SOC vs time plot."""
    if 'SOC' in df.columns:
        try:
            plt.figure(figsize=PLOT_STYLES['fig_size_wide'])
            plt.plot(df['timestamps'], df['SOC'], 
                     color=PLOT_STYLES['colors']['soc'], linewidth=2.0)
            plt.title('State of Charge (SOC) vs Time')
            plt.xlabel('Time')
            plt.ylabel('SOC (%)')
            plt.grid(True, alpha=PLOT_STYLES['grid_alpha'])
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "soc_time.png"), dpi=PLOT_STYLES['dpi'])
            plt.close()
            print("✓ Generated: soc_time.png")
        except Exception as e:
            print(f"⚠ Error generating SOC vs time plot: {e}")

def generate_ntc_temperatures_plot(df, output_dir):
    """Generate NTC temperatures vs time plot."""
    ntc_cols = [col for col in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4'] 
                if col in df.columns]
    
    if ntc_cols:
        try:
            plt.figure(figsize=PLOT_STYLES['fig_size_wide'])
            colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
            
            for i, col in enumerate(ntc_cols):
                plt.plot(df['timestamps'], df[col], 
                         label=col, linewidth=PLOT_STYLES['line_width_thin'], 
                         color=colors[i % len(colors)])
            
            plt.title(f"NTC Temperature Sensors vs Time\nStart: {df['timestamps'].iloc[0]}")
            plt.xlabel("Time")
            plt.ylabel("Temperature (°C)")
            plt.legend(loc='best')
            plt.grid(True, alpha=PLOT_STYLES['grid_alpha'])
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "ntc_temperatures_vs_time.png"), dpi=PLOT_STYLES['dpi'])
            plt.close()
            print("✓ Generated: ntc_temperatures_vs_time.png")
        except Exception as e:
            print(f"⚠ Error generating NTC temperature plot: {e}")

def generate_soc_vs_voltage_plot(df, output_dir):
    """Generate SOC vs average voltage plot."""
    if 'SOC' in df.columns:
        try:
            df['soc_int'] = df['SOC'].round().astype(int)
            avg_voltage_by_soc = df.groupby('soc_int')['Battery_voltage'].mean().reset_index()
            avg_voltage_by_soc = avg_voltage_by_soc.sort_values('soc_int', ascending=False)
            
            plt.figure(figsize=PLOT_STYLES['fig_size_standard'])
            plt.plot(
                avg_voltage_by_soc['soc_int'],
                avg_voltage_by_soc['Battery_voltage'],
                marker='o',
                color='#d62728',
                linewidth=PLOT_STYLES['line_width_thick'],
                markersize=PLOT_STYLES['marker_size'],
                label='Average Voltage per SOC'
            )
            plt.xlabel("State of Charge (SOC %)", fontsize=12)
            plt.ylabel("Average Battery Voltage (V)", fontsize=12)
            plt.title("SOC vs Average Battery Voltage", fontsize=14)
            plt.legend()
            plt.grid(True, alpha=PLOT_STYLES['grid_alpha'])
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "soc_vs_voltage.png"), dpi=PLOT_STYLES['dpi'])
            plt.close()
            print("✓ Generated: soc_vs_voltage.png")
        except Exception as e:
            print(f"⚠ Error generating SOC vs voltage plot: {e}")

def generate_ride_mode_plots(df, metrics, output_dir):
    """Generate ride mode analysis plots."""
    if 'mode_stats' in metrics:
        mode_stats = metrics['mode_stats']
        
        if 'current' in mode_stats:
            try:
                plt.figure(figsize=(10, 6))
                mode_stats['current'].plot(kind="bar")
                plt.title("Average vs Maximum Battery Current by Ride Mode")
                plt.ylabel("Current (A)")
                plt.xlabel("Ride Mode")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "avg_max_current_by_mode.png"), dpi=150)
                plt.close()
                print("✓ Generated: avg_max_current_by_mode.png")
            except Exception as e:
                print(f"⚠ Error generating current by mode plot: {e}")
        
        if 'power' in mode_stats:
            try:
                plt.figure(figsize=(10, 6))
                mode_stats['power'].plot(kind="bar")
                plt.title("Average vs Maximum Power by Ride Mode")
                plt.ylabel("Power (W)")
                plt.xlabel("Ride Mode")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "avg_max_power_by_mode.png"), dpi=150)
                plt.close()
                print("✓ Generated: avg_max_power_by_mode.png")
            except Exception as e:
                print(f"⚠ Error generating power by mode plot: {e}")
        
        if 'energy' in mode_stats and 'Wh_per_km' in mode_stats['energy'].columns:
            try:
                plt.figure(figsize=(10, 6))
                mode_stats['energy']["Wh_per_km"].plot(kind="bar")
                plt.title("Energy Consumption per km (Wh/km)")
                plt.ylabel("Wh/km")
                plt.xlabel("Ride Mode")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "energy_efficiency_wh_per_km.png"), dpi=150)
                plt.close()
                print("✓ Generated: energy_efficiency_wh_per_km.png")
            except Exception as e:
                print(f"⚠ Error generating energy efficiency plot: {e}")

def generate_temperature_distribution_plots(df, metrics, output_dir):
    """Generate temperature distribution plots."""
    if 'thermal_stats' in metrics:
        temp_cols = [c for c in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4'] 
                     if c in df.columns]
        if temp_cols:
            try:
                df_temp = df[temp_cols].dropna(how="all").copy()
                if len(df_temp) > 0:
                    df_temp["Temp_Max"] = df_temp[temp_cols].max(axis=1)
                    
                    plt.figure(figsize=(10, 6))
                    df_temp["Temp_Max"].hist(bins=50, edgecolor='black')
                    plt.title("Battery Maximum Temperature Distribution")
                    plt.xlabel("Temperature (°C)")
                    plt.ylabel("Frequency")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "battery_max_temperature_distribution.png"), dpi=150)
                    plt.close()
                    print("✓ Generated: battery_max_temperature_distribution.png")
                    
                    df_temp["Temp_Min"] = df_temp[temp_cols].min(axis=1)
                    df_temp["Temp_Delta"] = df_temp["Temp_Max"] - df_temp["Temp_Min"]
                    
                    plt.figure(figsize=(10, 6))
                    df_temp["Temp_Delta"].hist(bins=50, edgecolor='black')
                    plt.title("Battery Thermal Gradient (ΔT)")
                    plt.xlabel("ΔT (°C)")
                    plt.ylabel("Frequency")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "battery_thermal_gradient.png"), dpi=150)
                    plt.close()
                    print("✓ Generated: battery_thermal_gradient.png")
            except Exception as e:
                print(f"⚠ Error generating temperature distribution plots: {e}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

def print_summary_report(df, metrics):
    """Print comprehensive summary report."""
    print("\n" + "="*60)
    print("BATTERY ANALYSIS SUMMARY REPORT (Synced with Batch Tool)")
    print("="*60)
    
    print(f"\n--- Time Range ---")
    print(f"Start: {df['timestamps'].iloc[0]}")
    print(f"End:   {df['timestamps'].iloc[-1]}")
    print(f"Duration: {df['timestamps'].iloc[-1] - df['timestamps'].iloc[0]}")
    print(f"Samples: {len(df):,}")
    
    print(f"\n--- Battery Status ---")
    print(f"Max Battery Current: {metrics['max_current']:.1f} A")
    if metrics['start_soc']:
        print(f"Starting SOC: {metrics['start_soc']:.1f} %")
        print(f"Ending SOC: {metrics['end_soc']:.1f} %")
        print(f"SOC Drop: {metrics['soc_drop']:.1f} %")
    
    print(f"\n--- Energy Metrics ---")
    print(f"Total Discharge:      {metrics['cumulative_wh_discharge']:.0f} Wh")
    print(f"Total Regen (Moving): {metrics['cumulative_wh_regen']:.0f} Wh")
    print(f"Plug-in Charging:     {metrics['cumulative_wh_charger']:.0f} Wh")
    print(f"Net Energy Consumed:  {metrics['net_energy_wh']:.0f} Wh")
    
    if metrics['distance_km']:
        print(f"\n--- Distance & Efficiency ---")
        print(f"Distance Covered: {metrics['distance_km']:.3f} km")
        if metrics['wh_per_km']:
            print(f"Efficiency (Net Energy):   {metrics['wh_per_km']:.1f} Wh/km")
            print(f"Efficiency (Discharge):    {metrics['wh_per_km_discharge']:.1f} Wh/km")
    
    print(f"\n--- Temperature Peaks ---")
    ntc_sensors = ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
    for sensor in ntc_sensors:
        if f'{sensor}_max' in metrics:
            print(f"{sensor}: {metrics[f'{sensor}_max']:.1f} °C (avg: {metrics[f'{sensor}_avg']:.1f} °C)")
    
    # Status distribution
    if 'status_distribution' in metrics:
        print(f"\n=== CHARGE/DISCHARGE STATUS DISTRIBUTION ===")
        for status in [0, 1, 2]:
            stats = metrics['status_distribution'][status]
            print(f"{STATUS_NAMES[status]}: {stats['count']} points ({stats['percentage']:.1f}%)")
    
    # Processed current statistics
    if 'processed_current_stats' in metrics:
        print(f"\n=== PROCESSED CURRENT STATISTICS ===")
        stats = metrics['processed_current_stats']
        print(f"Min: {stats['min']:.2f} A")
        print(f"Max: {stats['max']:.2f} A")
        print(f"Mean: {stats['mean']:.2f} A")
        print(f"Std Dev: {stats['std']:.2f} A")
    
    # Thermal statistics
    if 'thermal_stats' in metrics:
        print(f"\n=== BATTERY THERMAL SENSOR STATS (°C) ===")
        print(metrics['thermal_stats'])
        
        print(f"\n=== PACK LEVEL THERMAL STATS ===")
        pack = metrics['pack_thermal']
        print(f"Average Pack Max Temp: {pack['avg_max_temp']:.1f} °C")
        print(f"Peak Pack Max Temp: {pack['peak_max_temp']:.1f} °C")
        print(f"95th Percentile Pack Temp: {pack['p95_temp']:.1f} °C")
    
    # Ride mode statistics
    if 'mode_stats' in metrics:
        mode_stats = metrics['mode_stats']
        print(f"\n=== RIDE MODE ELECTRICAL STATISTICS ===")
        
        if 'current' in mode_stats:
            print(f"\n--- CURRENT STATS (A) ---")
            print(mode_stats['current'])
        
        if 'power' in mode_stats:
            print(f"\n--- POWER STATS (W) ---")
            print(mode_stats['power'])
        
        if 'energy' in mode_stats and 'Wh_per_km' in mode_stats['energy'].columns:
            print(f"\n--- ENERGY EFFICIENCY (Wh/km) ---")
            print(mode_stats['energy']["Wh_per_km"])
    
    print(f"\n--- Voltage Statistics ---")
    print(f"Min Voltage: {df['Battery_voltage'].min():.2f} V")
    print(f"Max Voltage: {df['Battery_voltage'].max():.2f} V")
    print(f"Avg Voltage: {df['Battery_voltage'].mean():.2f} V")
    
    print(f"\n--- Power Statistics ---")
    print(f"Peak Power: {df['Power_kW'].max():.2f} kW")
    print(f"Average Power: {df['Power_kW'].mean():.2f} kW")
    
    print(f"\n--- Current Limit Analysis ---")
    if 'current_limit_violations' in metrics:
        if metrics['current_limit_violations'] > 0:
            print(f"⚠ Current exceeded limits {metrics['current_limit_violations']:,} times")
            print(f"  ({metrics['current_limit_percent']:.2f}% of samples)")
        else:
            print(f"✓ Current stayed within limits")
    
    print("\n" + "="*60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("BATTERY ANALYSIS TOOL - Enhanced Version")
    print("="*60)
    
    # Setup
    # Setup
    # Initialize Tkinter and hide the main window
    root = tk.Tk()
    root.withdraw()
    
    print("Please select the CSV data file...")
    data_path = filedialog.askopenfilename(
        title="Select Battery Data CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    if not data_path:
        print("No file selected. Exiting.")
        return
        
    output_dir = os.path.dirname(data_path)
    
    print(f"Selected file: {data_path}")
    
    # Load and process data
    print("Loading and processing data...")
    df_raw = load_and_process_data(data_path)
    
    if len(df_raw) == 0:
        print("Error: No valid data after filtering")
        return
    
    # Resample data for analysis
    # NOTE: Spike Filtering happens INSIDE this function now
    print("Resampling data...")
    df_resampled = resample_data(df_raw)
    
    if len(df_resampled) == 0:
        print("Error: No valid data after resampling")
        return
    
    # Calculate metrics
    print("Calculating metrics...")
    df_resampled, metrics = calculate_metrics(df_resampled)
    
    # Generate plots
    print("\nGenerating plots...")
    
    generate_battery_current_heatmap(df_resampled, output_dir)
    generate_power_profile_analysis(df_resampled, output_dir)
    generate_battery_current_status_plot(df_raw, output_dir)  # Use raw data for better resolution on status
    generate_voltage_current_plot(df_resampled, output_dir)
    generate_permissible_current_plot(df_resampled, output_dir)
    generate_soc_time_plot(df_resampled, output_dir)
    generate_ntc_temperatures_plot(df_resampled, output_dir)
    generate_soc_vs_voltage_plot(df_resampled, output_dir)
    generate_ride_mode_plots(df_resampled, metrics, output_dir)
    generate_temperature_distribution_plots(df_resampled, metrics, output_dir)
    
    # Print summary
    print_summary_report(df_resampled, metrics)
    
    # Output confirmation
    print(f"\n📁 All files saved to: {output_dir}")
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()