import librosa
import numpy as np
import serial
import time
import threading
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog

# ── File picker ───────────────────────────────────
root = tk.Tk()
root.withdraw()  # hide main window
AUDIO_FILE = filedialog.askopenfilename(
    title="Select Music File",
    filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")]
)

if not AUDIO_FILE:
    print("No file selected. Exiting.")
    exit()

print(f"Selected: {AUDIO_FILE}")

# ── Config ────────────────────────────────────────
COM_PORT = "COM4"
BAUD     = 9600
# ──────────────────────────────────────────────────

print("Loading audio...")
y, sr = librosa.load(AUDIO_FILE, sr=22050)

# ── Beat tracking ─────────────────────────────────
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"BPM: {float(tempo.item()):.1f} | Beats: {len(beat_times)}") # type: ignore

# ── Frequency analysis ────────────────────────────
hop_length = 512
S = np.abs(librosa.stft(y, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=sr)

def band_energy(S, freqs, f_low, f_high, frame):
    idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
    return np.mean(S[idx, frame]) if len(idx) > 0 else 0

beat_commands = []
for bf in beat_frames:
    frame = min(bf, S.shape[1] - 1)
    bass   = band_energy(S, freqs,   20,  250, frame)
    mid    = band_energy(S, freqs,  250, 2000, frame)
    treble = band_energy(S, freqs, 2000, 8000, frame)

    total = bass + mid + treble + 1e-6
    b = bass   / total
    m = mid    / total
    t = treble / total

    if   b > 0.6:             cmd = 0
    elif b > 0.4:             cmd = 1
    elif m > 0.6:             cmd = 2
    elif m > 0.4 and t > 0.3: cmd = 3
    elif t > 0.6:             cmd = 4
    elif b > 0.3 and m > 0.3: cmd = 5
    else:                     cmd = 6
    beat_commands.append(cmd)

print("Analysis done. Connecting Arduino...")
arduino = serial.Serial(COM_PORT, BAUD, timeout=1)
time.sleep(2)
print("Connected!")

# ── Play audio ────────────────────────────────────
def play_audio():
    if sys.platform == "win32":
        subprocess.Popen(
            ["start", "/min", "", AUDIO_FILE],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    elif sys.platform == "darwin":
        subprocess.Popen(["afplay", AUDIO_FILE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["mpg123", AUDIO_FILE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ── Fire beats ────────────────────────────────────
def fire_beats(start_time):
    for i, t in enumerate(beat_times):
        now  = time.time() - start_time
        wait = t - now
        if wait > 0:
            time.sleep(wait)
        cmd = beat_commands[i]
        arduino.write(str(cmd).encode())
        print(f"  t={t:.2f}s | cmd={cmd}")
    print("Song finished.")
    arduino.close()