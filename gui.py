import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess

def browse_file():
    filename = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("JPEG files", "*.jpeg;*.jpg"), ("All files", "*.*")]
    )
    if filename:
        entry_file_path.delete(0, tk.END)  # Clear current entry
        entry_file_path.insert(0, filename)  # Insert selected file path

def execute_command():
    file_path = entry_file_path.get()
    type_selected = combo_type.get()

    if not file_path or type_selected not in ["long", "short"]:
        messagebox.showerror("Input Error", "Please select a file and a type.")
        return

    command = f"python3 KaptonBatchmode.py -f {file_path} -k {type_selected}"
    
    try:
        # Execute the command
        subprocess.run(command, shell=True, check=True)
        messagebox.showinfo("Success", "Command executed successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Execution Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Kapton Analyzer for 2S module")
root.geometry("600x400")  # Increased width

# File selection
label_file = tk.Label(root, text="Select Input File:")
label_file.pack(pady=5)

entry_file_path = tk.Entry(root, width=50)
entry_file_path.pack(pady=5)

button_browse = tk.Button(root, text="Browse", command=browse_file)
button_browse.pack(pady=5)

# Type selection
label_type = tk.Label(root, text="Select Kapton Type:")
label_type.pack(pady=5)

combo_type = ttk.Combobox(root, values=["long", "short"])
combo_type.pack(pady=5)

# Execute button
button_execute = tk.Button(root, text="Execute Command", command=execute_command)
button_execute.pack(pady=20)

# Contact information
label_contact = tk.Label(root, text="Contact: aloke.kumar.das@cern.ch")
label_contact.pack(pady=10)

# Run the application
root.mainloop()

