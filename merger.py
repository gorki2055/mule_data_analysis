import tkinter as tk
from tkinter import filedialog, messagebox
from asammdf import MDF
import os

class MergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MF4 File Merger")
        self.root.geometry("700x500")

        self.file_paths = []

        # UI Elements
        self.label = tk.Label(root, text="Selected MF4 Files to Merge:", font=("Arial", 12, "bold"))
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

        self.merge_btn = tk.Button(self.btn_frame, text="Merge Files", command=self.merge_files, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=15)
        self.merge_btn.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select MF4 Files", filetypes=[("MF4 Files", "*.mf4"), ("All Files", "*.*")])
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

    def merge_files(self):
        if not self.file_paths:
            messagebox.showwarning("No Files", "Please select files to merge first.")
            return

        if len(self.file_paths) < 2:
            confirm = messagebox.askyesno("Warning", "You have selected only 1 file. Merging usually requires 2+ files. Do you want to proceed?")
            if not confirm:
                return

        output_file = filedialog.asksaveasfilename(defaultextension=".mf4", filetypes=[("MF4 Files", "*.mf4")], title="Save Merged File As")
        if not output_file:
            return

        try:
            self.status_label.config(text="Merging in progress... This may take a while.")
            self.root.update()
            
            # Load files
            mdf_files = []
            for f in self.file_paths:
                try:
                    mdf_files.append(MDF(f))
                except Exception as e:
                    messagebox.showerror("Error Loading File", f"Could not load file:\n{f}\n\nError: {e}")
                    self.status_label.config(text="Error loading files.")
                    return

            # Concatenate
            mdf_combined = MDF.concatenate(mdf_files)
            
            # Save
            mdf_combined.save(output_file)
            
            # Clean up memory
            for m in mdf_files:
                m.close()
            mdf_combined.close()
            
            messagebox.showinfo("Success", f"Files merged successfully!\nSaved to: {output_file}")
            self.status_label.config(text="Merge completed successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during merging:\n{str(e)}")
            self.status_label.config(text="Error occurred.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MergerApp(root)
    root.mainloop()
