import SOC_PLOT
import matplotlib.pyplot as plt
import os

# Mock plt.show to avoid blocking
def mock_show():
    print("Mock plt.show() called.")

plt.show = mock_show

file_path = "c:\\Users\\ADMIN\\Desktop\\MouseWithoutBorders\\report generator\\test_soc_plot.csv"

print(f"Testing SOC_PLOT with {file_path}")
try:
    SOC_PLOT.plot_soc(file_path)
    print("SOC_PLOT.plot_soc executed successfully.")
except Exception as e:
    print(f"SOC_PLOT.plot_soc failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up data file is optional, but good practice if valid
# os.remove(file_path)
