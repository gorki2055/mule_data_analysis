import os
import shutil
import time

src = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS\charging_session_analysis.csv"
dst = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS\charging_session_analysis_backup_" + str(int(time.time())) + ".csv"

if os.path.exists(src):
    try:
        os.rename(src, dst)
        print(f"Renamed {src} to {dst}")
    except Exception as e:
        print(f"Rename failed: {e}")
        try:
            os.remove(src) # Try delete if rename fails
            print("Deleted original file")
        except:
            print("Could not rename or delete.")
else:
    print("Source file not found (clean start).")
