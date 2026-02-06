from soh_batch_plot import process_batch
import os

# Use current directory
cwd = os.getcwd()
print(f"Testing batch plot in: {cwd}")
process_batch(cwd)
