
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import os

class ParquetCombinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet File Combiner")
        self.root.geometry("700x500")

        self.file_paths = []

        # UI Elements
        self.label = tk.Label(root, text="Selected Parquet Files to Combine:", font=("Arial", 12, "bold"))
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

        self.combine_btn = tk.Button(self.btn_frame, text="Combine Files", command=self.combine_files, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=15)
        self.combine_btn.pack(side=tk.LEFT, padx=20)
        
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

    def combine_files(self):
        if not self.file_paths:
            messagebox.showwarning("No Files", "Please select files to combine first.")
            return

        if len(self.file_paths) < 2:
            confirm = messagebox.askyesno("Warning", "You have selected only 1 file. Combining usually requires 2+ files. Do you want to proceed?")
            if not confirm:
                return

        output_file = filedialog.asksaveasfilename(defaultextension=".parquet", filetypes=[("Parquet Files", "*.parquet")], title="Save Combined File As")
        if not output_file:
            return

        try:
            self.status_label.config(text="Combining in progress... This may take a while.")
            self.root.update()
            
            # Load files
            dfs = []
            for f in self.file_paths:
                try:
                    df = pd.read_parquet(f)
                    dfs.append(df)
                    print(f"Loaded: {f}, shape: {df.shape}")
                except Exception as e:
                    messagebox.showerror("Error Loading File", f"Could not load file:\n{f}\n\nError: {e}")
                    self.status_label.config(text="Error loading files.")
                    return

            # Concatenate
            combined_df = pd.concat(dfs, ignore_index=True)
            print(f"Combined shape: {combined_df.shape}")
            
            # Save
            combined_df.to_parquet(output_file, index=False)
            
            messagebox.showinfo("Success", f"Files combined successfully!\nSaved to: {output_file}\nTotal rows: {len(combined_df)}")
            self.status_label.config(text="Combine completed successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during combining:\n{str(e)}")
            self.status_label.config(text="Error occurred.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParquetCombinerApp(root)
    root.mainloop()
