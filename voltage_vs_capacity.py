# import pandas as pd
# import matplotlib.pyplot as plt

# # 1. Load the data
# # Replace 'battery_data.csv' with the actual name of your file
# # If your file is an Excel file, use pd.read_excel('filename.xlsx')
# file_path = 'D:\kushal\Thermal_test_data\charging_session_after_thermal_test.csv' 
# df = pd.read_csv(file_path)

# # 2. Extract the relevant columns based on the headers in your image
# # specific column names from the image: 'Volt(V)' and 'Capacity(mAh)'
# voltage = df['Volt(V)']
# capacity = df['Capacity(mAh)']

# # 3. Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(capacity, voltage, label='Voltage vs Capacity', color='blue', linewidth=1.5)

# # 4. Formatting the plot
# plt.title('Battery Voltage vs. Capacity Curve', fontsize=14)
# plt.xlabel('Capacity (mAh)', fontsize=12)
# plt.ylabel('Voltage (V)', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.legend()

# # 5. Show the plot
# plt.tight_layout()
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# I added 'r' before the string to ensure the path works reliably
file_path = r'D:\kushal\Thermal_test_data\discharge_session_before_test.csv' 
df = pd.read_csv(file_path)

# 2. Extract the relevant columns
voltage = df['Volt(V)']
capacity = df['Capacity(Ah)']

# 3. Create the plot
plt.figure(figsize=(10, 6))
plt.plot(capacity, voltage, label='Voltage vs Capacity', color='blue', linewidth=1.5)

# 4. Formatting the plot
plt.title('Battery Voltage vs. Capacity Curve', fontsize=14)
plt.xlabel('Capacity (mAh)', fontsize=12)
plt.ylabel('Voltage (V)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# --- THE CHANGE IS HERE ---
# This command reverses the x-axis so higher capacity starts on the left
plt.gca().invert_xaxis()
# --------------------------

# 5. Show the plot
plt.tight_layout()
plt.show()