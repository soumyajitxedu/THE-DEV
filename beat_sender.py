#!/usr/bin/env python3
"""
==============================================================
  MP3 BEAT SYNC - Play MP3 + Sync Arduino LEDs in Real-Time
==============================================================
  HOW IT WORKS:
  1. Load MP3 file
  2. Analyze entire audio for beats/frequencies
  3. Play audio through speakers
  4. Simultaneously send beat data to Arduino via USB Serial
  5. LEDs flash in perfect sync with music!
==============================================================
"""

import numpy as np
import pyaudio
import serial
import serial.tools.list_ports
import time
import sys
import os
import threading
from collections import deque

# ============================================================
# TRY IMPORT LIBRARIES
# ============================================================
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not found. Install with: pip install librosa")

try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("⚠️  pydub not found. Install with: pip install pydub")

try:
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Audio
    'CHUNK'         : 2048,       # Audio chunk size
    'RATE'          : 44100,      # Sample rate
    'CHANNELS'      : 1,          # Mono

    # Beat detection sensitivity
    'BEAT_THRESHOLD': 1.35,       # Lower = more sensitive
    'BEAT_COOLDOWN' : 0.08,       # Seconds between beats

    # Frequency bands → 6 LEDs
    'FREQ_BANDS': [
        (20,   80),    # LED 8  - Sub bass (kick drum thump)
        (80,   300),   # LED 9  - Bass (bass guitar)
        (300,  600),   # LED 10 - Low mid (warmth)
        (600,  2500),  # LED 11 - Mid (vocals/melody)
        (2500, 7000),  # LED 12 - High mid (clarity)
        (7000, 16000), # LED 13 - Treble (hi-hats)
    ],

    # Serial
    'BAUD_RATE'     : 115200,

    # Visualizer in terminal
    'SHOW_VIZ'      : True,
}

# ============================================================
# COLOR / DISPLAY HELPERS
# ============================================================
BAR_CHARS = " ▁▂▃▄▅▆▇█"
MODES = [
    "All Flash",
    "Chase",
    "Alternate",
    "Frequency (Best!)",
    "VU Meter",
    "Random",
]

def cls():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("╔══════════════════════════════════════════════════════╗")
    print("║       🎵  MP3 BEAT SYNC → ARDUINO LEDs  💡          ║")
    print("╚══════════════════════════════════════════════════════╝")

def draw_visualizer(bands, beat, intensity, volume, song_name, elapsed, total):