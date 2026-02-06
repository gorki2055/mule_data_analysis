import os
import glob
from pathlib import Path
import report_generator

# Configuration
ROOT_DIR = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\3ACF5792"
OUTPUT_ROOT = r"C:\Users\ADMIN\Desktop\MouseWithoutBorders\LOG\LOG\REPORTS\mule2"
TARGET_DATE = "1-1-2026"  # Set to None to process ALL dates. Example: "1-7-2026"

def find_target_csvs(root_path, target_date=None):
    """
    Crawls the directory structure looking for CSV files in session folders.
    If target_date is provided, only files containing that date string in their path are returned.
    """
    csv_files = []
    
    print(f"Scanning directory: {root_path}")
    
    if target_date:
        print(f" Filtering for date: {target_date}")
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(root, file)
                
                # Check if target_date is in the path (folder name or filename)
                if target_date and target_date not in full_path:
                    continue
                
                # Exclude charging files
                if "charging" in file.lower():
                    print(f"  Skipping charging file: {file}")
                    continue
                
                # Strict Filter: Only process files starting with "MULE" to avoid processing summary/system files
                if not file.upper().startswith("MULE"):
                    print(f"  Skipping non-mule file: {file}")
                    continue
                    
                csv_files.append(full_path)
                print(f"  Found: {file}")
                
    return csv_files

def main():
    print("="*60)
    print("BATCH BATTERY REPORT GENERATOR")
    print("="*60)
    
    # 1. Find all CSV files recursively (filtered by TARGET_DATE if set)
    files = find_target_csvs(ROOT_DIR, target_date=TARGET_DATE)
    
    if not files:
        if TARGET_DATE:
            print(f"❌ No CSV files found matching date '{TARGET_DATE}' in {ROOT_DIR}")
        else:
            print(f"❌ No CSV files found in {ROOT_DIR}")
        return

    print(f"\n✅ Found {len(files)} CSV files to process.")
    
    # 2. Run the report generator
    print("\n🚀 Starting Report Generator...")
    try:
        # We pass the list of files to the modified report_generator
        report_generator.run(input_dir=ROOT_DIR, output_root=OUTPUT_ROOT, file_list=files)
        print("\n🎉 Batch processing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
