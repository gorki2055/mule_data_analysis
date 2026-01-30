
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

class ParquetToCSVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet to CSV Converter")
        self.root.geometry("700x500")

        self.file_paths = []

        # UI Elements
        self.label = tk.Label(root, text="Selected Parquet Files to Convert:", font=("Arial", 12, "bold"))
        self.label.pack(pady=10)

        # Listbox with scrollbar
        self.list_frame = tk.Frame(root)
        self.list_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        self.scrollbar = tk.Scrollbar(self.list_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(self.list_frame, selectmode=tk.EXTENDED, yscrollcommand=self.scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.listbox.yview)

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=15)

        self.add_btn = tk.Button(self.btn_frame, text="Add Files", command=self.add_files, width=15)
        self.add_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(self.btn_frame, text="Clear List", command=self.clear_list, width=15)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.convert_btn = tk.Button(self.btn_frame, text="Convert to CSV", command=self.convert_files, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=15)
        self.convert_btn.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Parquet Files", filetypes=[("Parquet Files", "*.parquet"), ("All Files", "*.*")])
        if files:
            for file in files:
                # Avoid duplicates
                if file not in self.file_paths:
                    self.file_paths.append(file)
                    self.listbox.insert(tk.END, file)
            self.status_label.config(text=f"{len(self.file_paths)} files selected.")

    def clear_list(self):
        self.file_paths = []
        self.listbox.delete(0, tk.END)
        self.status_label.config(text="List cleared.")

    def convert_files(self):
        if not self.file_paths:
            messagebox.showwarning("No Files", "Please select files to convert first.")
            return

        # Optional: Ask for output directory if you want all in one place, or save next to original
        # For now, let's ask for an output directory to keep it clean.
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return

        success_count = 0
        error_count = 0
        
        self.status_label.config(text="Conversion in progress...")
        self.root.update()

        for f in self.file_paths:
            try:
                base_name = os.path.basename(f)
                file_name_without_ext = os.path.splitext(base_name)[0]
                output_path = os.path.join(output_dir, f"{file_name_without_ext}.csv")
                
                df = pd.read_parquet(f)
                df.to_csv(output_path, index=False)
                print(f"Converted {f} to {output_path}")
                success_count += 1
            except Exception as e:
                print(f"Failed to convert {f}: {e}")
                error_count += 1

        self.status_label.config(text=f"Completed. Success: {success_count}, Errors: {error_count}")
        messagebox.showinfo("Conversion Completed", f"Successfully converted {success_count} files.\nErrors: {error_count}\nOutput Directory: {output_dir}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParquetToCSVApp(root)
    root.mainloop()
