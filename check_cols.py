import pandas as pd
file_path = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\2877D237\1-30-2026\MORNING_RIDE\MULE2_1-30-2026_MORNING_RIDE.csv"
df = pd.read_csv(file_path, nrows=1)
print(list(df.columns))
