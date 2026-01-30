import os
import pandas as pd
import matplotlib.pyplot as plt

# Example style dictionary (customize as needed)
PLOT_STYLES = {
    'fig_size_wide': (12, 6),
    'line_width_thin': 1.5,
    'grid_alpha': 0.3,
    'dpi': 150,
    'colors': {
        'current': '#1f77b4',      # Blue
        'permissible': '#ff7f0e'   # Orange
    }
}

def generate_current_ntc_plot(df, output_dir):
    """
    Generate plot of Battery Current, Permissible Current, and NTC temperatures vs Time.
    """
    try:
        # Ensure timestamps column exists
        if 'timestamps' not in df.columns or df.empty:
            print("⚠ No timestamps or empty DataFrame provided.")
            return

        # Create output directory if missing
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=PLOT_STYLES['fig_size_wide'])

        # Plot battery current
        if 'Battery_current1' in df.columns:
            plt.plot(df['timestamps'], df['Battery_current1'],
                     label="Battery Current (A)",
                     color=PLOT_STYLES['colors']['current'],
                     linewidth=PLOT_STYLES['line_width_thin'])

        # Plot permissible discharge current
        if 'Permissible_discharge_current' in df.columns:
            plt.plot(df['timestamps'], df['Permissible_discharge_current'],
                     label="Permissible Discharge Current (A)",
                     color=PLOT_STYLES['colors']['permissible'],
                     linewidth=PLOT_STYLES['line_width_thin'],
                     linestyle='-')

        # Plot NTC sensors
        ntc_cols = [col for col in ['Ntc_Mos', 'Ntc_Com', 'Ntc_1', 'Ntc_2', 'Ntc_3', 'Ntc_4']
                    if col in df.columns]
        ntc_colors = ['#d62728', '#2ca02c', '#9467bd', '#8c564b', '#17becf', '#7f7f7f']

        for i, col in enumerate(ntc_cols):
            plt.plot(df['timestamps'], df[col],
                     label=f"{col} (°C)",
                     color=ntc_colors[i % len(ntc_colors)],
                     linewidth=PLOT_STYLES['line_width_thin'],
                     linestyle='--')

        # Title and labels
        plt.title(f"Battery Current, Permissible Current & NTC Temps\nStart: {df['timestamps'].iloc[0]}")
        plt.xlabel("Time")
        plt.ylabel("Current (A) / Temperature (°C)")
        plt.legend(loc='best')
        plt.grid(True, alpha=PLOT_STYLES['grid_alpha'])
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(output_dir, "current_ntc_vs_time.png")
        plt.savefig(output_path, dpi=PLOT_STYLES['dpi'])
        plt.close()
        print(f"✓ Generated: {output_path}")

    except Exception as e:
        print(f"⚠ Error generating plot: {e}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example: create dummy data
    import numpy as np
    import datetime

    timestamps = pd.date_range(start=datetime.datetime.now(), periods=100, freq='T')
    data = {
        'timestamps': timestamps,
        'Battery_current1': np.random.uniform(10, 50, size=100),
        'Permissible_discharge_current': np.random.uniform(20, 60, size=100),
        'Ntc_Mos': np.random.uniform(25, 40, size=100),
        'Ntc_Com': np.random.uniform(20, 35, size=100),
        'Ntc_1': np.random.uniform(22, 38, size=100),
        'Ntc_2': np.random.uniform(23, 37, size=100),
        'Ntc_3': np.random.uniform(24, 36, size=100),
        'Ntc_4': np.random.uniform(21, 34, size=100),
    }
    df = pd.DataFrame(data)

    # Run plot function
    generate_current_ntc_plot(df, output_dir="plots")