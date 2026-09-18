import os
import tkinter as tk
from tkinter import filedialog, messagebox


def convert_wav_to_c():
    # 1. Ask user to select the WAV file
    wav_path = filedialog.askopenfilename(
        title="Select WAV File",
        filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")]
    )

    if not wav_path:
        return  # User canceled file selection

    # Extract default name for the output file
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    default_h_name = f"{base_name}.h"

    # 2. Ask user where to save the .h file
    save_path = filedialog.asksaveasfilename(
        title="Save C-Header File As",
        initialfile=default_h_name,
        defaultextension=".h",
        filetypes=[("C Header File", "*.h"), ("All Files", "*.*")]
    )

    if not save_path:
        return  # User canceled save location selection

    try:
        status_label.config(text="Converting audio data...", fg="blue")
        root.update()

        # Sanitize array name for C syntax (replace non-alphanumeric with _)
        array_name = "".join(c if c.isalnum() else "_" for c in base_name)
        if not array_name or array_name[0].isdigit():
            array_name = "audio_" + array_name

        # Read WAV binary data
        with open(wav_path, "rb") as f:
            data = f.read()

        # Write C header file
        with open(save_path, "w") as f:
            f.write("#ifndef AUDIO_DATA_H\n#define AUDIO_DATA_H\n\n")
            f.write("#include <pgmspace.h>\n\n")
            f.write(f"// File size: {len(data)} bytes\n")
            f.write(f"const unsigned int {array_name}_len = {len(data)};\n")
            f.write(f"const unsigned char {array_name}[] PROGMEM = {{\n  ")

            # Write 12 hex bytes per line
            for i, byte in enumerate(data):
                f.write(f"0x{byte:02x}, ")
                if (i + 1) % 12 == 0:
                    f.write("\n  ")

            f.write("\n};\n\n#endif\n")

        status_label.config(text="Conversion successful!", fg="green")
        messagebox.showinfo(
            "Success",
            f"WAV converted successfully!\n\nSaved to:\n{save_path}"
        )

    except Exception as e:
        status_label.config(text="Error converting file.", fg="red")
        messagebox.showerror("Error", f"Failed to convert WAV:\n{str(e)}")


# Setup GUI Window
root = tk.Tk()
root.title("WAV to ESP32 C-Header Converter")
root.geometry("450x220")
root.resizable(False, False)

# Interface components
title_label = tk.Label(root, text="WAV to C-Header Converter", font=("Arial", 14, "bold"))
title_label.pack(pady=15)

info_label = tk.Label(
    root, 
    text="Select a .wav file and pick where to save your ESP32 .h header file.",
    font=("Arial", 9),
    fg="gray",
    wraplength=400
)
info_label.pack(pady=5)

convert_btn = tk.Button(
    root, 
    text="Select WAV & Save C Header", 
    command=convert_wav_to_c, 
    font=("Arial", 11, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=10, 
    pady=8
)
convert_btn.pack(pady=15)

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack(pady=5)

# Launch GUI loop
root.mainloop()