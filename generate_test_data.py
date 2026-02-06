import pandas as pd
import numpy as np
import datetime

# Create dummy data
timestamps = [datetime.datetime(2023, 1, 1, 10, 0, 0) + datetime.timedelta(seconds=i) for i in range(100)]
soh = [90.0] * 100
voltage = [54000.0] * 100 # 54V in mV
current = [10.0] * 100
status = [0] * 100 # Discharge

# Introduce SOH Anomaly
# Drop at index 50
for i in range(50, 100):
    soh[i] = 80.0

# Introduce Filtering Scenarios
# 1. High Voltage (>120V)
voltage[10] = 130000.0 
# 2. Low Voltage (<20V)
voltage[11] = 10000.0
# 3. High Current (>185A)
current[12] = 200.0
# 4. Low Current (<-50A)
current[13] = -60.0
# 5. Charging Status with Positive Current (Should be flipped)
status[14] = 1
current[14] = 20.0 

# Create DataFrame
df = pd.DataFrame({
    'timestamps': timestamps,
    'SOH': soh,
    'Battery_voltage': voltage,
    'Battery_current1': current,
    'Charge_discharge_status': status
})

df.to_csv('test_soh_data.csv', index=False)
print("Created test_soh_data.csv")
