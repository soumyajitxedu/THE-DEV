import tkinter as tk
import pyaudio # Or pyaudiowpatch for system audio
import numpy as np

class StandardModernVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Modern Visualizer (Standard)")
        self.root.geometry("1000x600")
        self.root.configure(bg="#0f0f0f") # Dark background

        # Audio Setup
        self.p = pyaudio.PyAudio()
        self.CHUNK = 1024 * 2
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                                input=True, frames_per_buffer=self.CHUNK)

        # UI: Black Canvas
        self.canvas = tk.Canvas(self.root, width=1000, height=500, 
                               bg="#050505", highlightthickness=0)
        self.canvas.pack(pady=20, padx=20)

        # Start button styled manually
        self.btn = tk.Button(self.root, text="START VISUALIZER", 
                            bg="#00ff88", fg="#000000", font=("Arial", 12, "bold"),
                            command=self.update, borderwidth=0, padx=20, pady=10)
        self.btn.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def update(self):
        try:
            # High-accuracy FFT analysis
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            data_np = np.frombuffer(data, dtype=np.int16)
            fft = np.abs(np.fft.rfft(data_np))[:60] # Get 60 frequency bins
            
            self.canvas.delete("bar")
            
            # Draw modern bars
            w = 1000
            h = 500
            bar_w = w / len(fft)
            
            for i, val in enumerate(fft):
                # Scale values for visibility
                val_scaled = (val / 100000) * h 
                x0 = i * bar_w
                y1 = h
                y0 = h - val_scaled
                
                # Neon gradient effect
                color = self.get_color_gradient(i, len(fft))
                self.canvas.create_rectangle(x0+2, y0, x0+bar_w-2, y1, 
                                           fill=color, outline="", tags="bar")
                
                # Top glow cap
                self.canvas.create_rectangle(x0+2, y0-3, x0+bar_w-2, y0, 
                                           fill="#ffffff", outline="", tags="bar")
            
            self.root.after(10, self.update)
        except Exception as e:
            print(f"Error: {e}")

    def get_color_gradient(self, i, total):
        # Cycles through Electric Blue to Neon Green
        r = 0
        g = int(255 * (i / total))
        b = 255 - g
        return f'#{r:02x}{g:02x}{b:02x}'

    def on_closing(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    StandardModernVisualizer()
