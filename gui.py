import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import subprocess
import os
import pandas as pd

def browse_file():
    filename = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[
            ("JPEG files", "*.jpeg *.jpg *.JPEG *.JPG"),
            ("All files", "*.*")
        ]
    )
    if filename:
        entry_file_path.delete(0, tk.END)
        entry_file_path.insert(0, filename)

def show_outputs(image_path, csv_path):
    try:
        img = Image.open(image_path)
        img = img.resize((700, 500), Image.ANTIALIAS)
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk, text="")
        label_image.image = img_tk
    except Exception as e:
        label_image.config(text=f"Failed to load image: {e}")

    try:
        df = pd.read_csv(csv_path)
        text_csv.delete("1.0", tk.END)
        text_csv.insert(tk.END, df.to_string(index=False))
    except Exception as e:
        text_csv.delete("1.0", tk.END)
        text_csv.insert(tk.END, f"Failed to load CSV: {e}")

def execute_command():
    file_path = entry_file_path.get()
    type_selected = combo_type.get()

    if not file_path or type_selected not in ["long", "short"]:
        messagebox.showerror("Input Error", "Please select a file and a type.")
        return

    command = f"python3 /home/tctrl/Desktop/Kapton_analysis/KaptonBatchmode.py -f \"{file_path}\" -k {type_selected}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        messagebox.showinfo("Success", "Analysis executed successfully.")

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        image_output = os.path.join(os.path.dirname(file_path), "PlotsShape", f"{base_name}._sufficientStrips.jpg")
        csv_output = os.path.join(os.path.dirname(file_path), "ResultsShape", f"{base_name}..csv")

        show_outputs(image_output, csv_output)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Execution Error", f"An error occurred: {e}")

# --- GUI SETUP ---
root = tk.Tk()
root.title("Kapton Analyzer for 2S module")
root.geometry("1200x800")

# Controls Frame in one row
frame_controls = tk.Frame(root)
frame_controls.pack(pady=10)

label_file = tk.Label(frame_controls, text="Select Input File:")
label_file.pack(side=tk.LEFT, padx=5)

entry_file_path = tk.Entry(frame_controls, width=50)
entry_file_path.pack(side=tk.LEFT, padx=5)

button_browse = tk.Button(frame_controls, text="Browse", command=browse_file)
button_browse.pack(side=tk.LEFT, padx=5)

label_type = tk.Label(frame_controls, text="Kapton Type:")
label_type.pack(side=tk.LEFT, padx=5)

combo_type = ttk.Combobox(frame_controls, values=["long", "short"], width=10)
combo_type.pack(side=tk.LEFT, padx=5)

button_execute = tk.Button(frame_controls, text="Analyze", command=execute_command)
button_execute.pack(side=tk.LEFT, padx=10)

# Output frames
frame_outputs = tk.Frame(root)
frame_outputs.pack(fill=tk.BOTH, expand=True)

frame_image = tk.LabelFrame(frame_outputs, text="Output Image", width=600, height=500)
frame_image.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
frame_image.pack_propagate(False)

frame_csv = tk.LabelFrame(frame_outputs, text="Output CSV", width=600, height=500)
frame_csv.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
frame_csv.pack_propagate(False)

label_image = tk.Label(frame_image, text="Image will appear here")
label_image.pack(expand=True, fill=tk.BOTH)

text_csv = tk.Text(frame_csv, wrap=tk.NONE)
text_csv.pack(expand=True, fill=tk.BOTH)

label_contact = tk.Label(root, text="Contact: aloke.kumar.das@cern.ch")
label_contact.pack(pady=10)

root.mainloop()
