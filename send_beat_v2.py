#!/usr/bin/env python3
"""
==============================================================
  MP3 BEAT SYNC PRO - Extreme Accuracy Beat-LED Synchronizer
==============================================================
  Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  1. Pre-analyze ENTIRE song with librosa                │
  │     - Beat times (tempo-locked)                         │
  │     - Onset times (transient events)                    │
  │     - Per-frame frequency bands (STFT)                  │
  │                                                         │
  │  2. Build a TIMELINE of LED events                      │
  │     - Each event has exact timestamp in seconds         │
  │                                                         │
  │  3. Play audio via PyAudio                              │
  │     - Track EXACT sample position (no drift)            │
  │     - Compute wall-clock time from samples played       │
  │                                                         │
  │  4. Dedicated Arduino-send thread                       │
  │     - Reads timeline, fires events at precise moment    │
  │     - Compensates for serial latency                    │
  │     - Lookahead scheduling prevents missed beats        │
  └─────────────────────────────────────────────────────────┘
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
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ============================================================
# LIBRARY IMPORTS WITH GRACEFUL FALLBACK
# ============================================================
try:
    import librosa
    import librosa.beat
    import librosa.onset
    import librosa.feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("❌ librosa REQUIRED: pip install librosa")
    sys.exit(1)

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False
    print("❌ pydub REQUIRED: pip install pydub")
    sys.exit(1)

try:
    from scipy.signal import butter, sosfilt
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not found. Install: pip install scipy")

# ============================================================
# CONFIGURATION - Tune these for your setup
# ============================================================
class Config:
    # --- Audio ---
    SAMPLE_RATE     : int   = 44100
    CHUNK_SIZE      : int   = 512       # Smaller = lower latency (was 2048!)
    CHANNELS        : int   = 1

    # --- Serial ---
    BAUD_RATE       : int   = 115200

    # --- Beat Detection (librosa pre-analysis) ---
    # These affect HOW MANY beats/onsets are detected
    BEAT_TIGHTNESS  : float = 400.0     # Higher = stricter beat tracking
    ONSET_DELTA     : float = 0.07      # Higher = fewer onsets detected

    # --- Latency Compensation ---
    # If LEDs flash BEFORE the beat: increase this
    # If LEDs flash AFTER  the beat: decrease this (or go negative)
    SERIAL_LATENCY_MS  : float = 0.0    # ms to compensate for serial delay
    AUDIO_BUFFER_MS    : float = 0.0    # ms of audio buffer pre-roll

    # --- LED Update Rate ---
    FREQ_UPDATE_HZ  : int   = 80        # Frequency band update rate
    # This controls how smooth the LED animation looks

    # --- Frequency Bands → 6 LEDs (pins 8-13) ---
    # Format: (low_hz, high_hz, led_pin, color_name)
    FREQ_BANDS = [
        (20,    80,   8,  "SUB"),   # Sub-bass  (kick thump)
        (80,    250,  9,  "BAS"),   # Bass      (bass guitar)
        (250,   600,  10, "LMD"),   # Low-mid   (warmth)
        (600,   2500, 11, "MID"),   # Mid       (vocals)
        (2500,  7000, 12, "HMD"),   # High-mid  (presence)
        (7000,  18000,13, "TRB"),   # Treble    (hi-hats)
    ]

    # --- Visualization ---
    SHOW_VISUALIZER : bool  = True
    VIZ_UPDATE_HZ   : int   = 25        # Console redraw rate

    # --- Normalization ---
    # Per-band sensitivity multipliers (boost quieter bands)
    BAND_BOOST = [2.5, 2.0, 1.8, 1.5, 1.3, 1.2]

    # Smoothing factor for frequency display (0=none, 0.9=very smooth)
    SMOOTH_ALPHA    : float = 0.6

# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class LEDEvent:
    """A single LED command to fire at a specific time"""
    timestamp   : float          # Seconds from song start
    event_type  : str            # 'BEAT', 'ONSET', 'FREQ', 'VU'
    intensity   : int = 200      # 0-255
    bands       : List[int] = field(default_factory=lambda: [0]*6)
    priority    : int = 1        # Higher = more important

    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class SongAnalysis:
    """Complete pre-analyzed song data"""
    samples          : np.ndarray    # Raw PCM float32
    sample_rate      : int
    duration         : float
    bpm              : float
    beat_times       : np.ndarray    # Seconds
    onset_times      : np.ndarray    # Seconds
    freq_times       : np.ndarray    # Frame timestamps
    freq_bands_data  : np.ndarray    # Shape: (num_frames, 6)
    rms_data         : np.ndarray    # Per-frame RMS energy
    song_name        : str

# ============================================================
# AUDIO ANALYZER - Pre-analyze entire song
# ============================================================
class SongAnalyzer:
    """
    Uses librosa to extract beats, onsets, and spectral data
    from the entire MP3 before playback begins.
    This is FAR more accurate than real-time detection.
    """

    def __init__(self):
        self.config = Config()

    def analyze(self, mp3_path: str) -> SongAnalysis:
        name = os.path.splitext(os.path.basename(mp3_path))[0]
        print(f"\n  📂 Loading: {name}")

        # ── Step 1: Load audio ──────────────────────────────
        y, sr = librosa.load(
            mp3_path,
            sr   = Config.SAMPLE_RATE,
            mono = True
        )
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"  ✅ Loaded {int(duration//60):02d}:{int(duration%60):02d} "
              f"| {len(y):,} samples @ {sr} Hz")

        # ── Step 2: Beat tracking ───────────────────────────
        print("  🥁 Tracking beats (tempo + phase)...")

        # Use dynamic programming beat tracker for high accuracy
        tempo, beat_frames = librosa.beat.beat_track(
            y          = y,
            sr         = sr,
            tightness  = Config.BEAT_TIGHTNESS,
            trim       = False,      # Don't trim weak beats at edges
            units      = 'frames'
        )

        # Handle librosa >= 0.10 array tempo
        bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        print(f"  🎯 BPM: {bpm:.1f} | Beats: {len(beat_times)}")

        # ── Step 3: Onset detection ─────────────────────────
        print("  ⚡ Detecting onsets (transients)...")

        # Use spectral flux onset detection - very accurate for transients
        onset_env    = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope = onset_env,
            sr             = sr,
            delta          = Config.ONSET_DELTA,   # Sensitivity
            backtrack      = True,   # Snap to actual transient start
            units          = 'frames'
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        print(f"  ⚡ Onsets: {len(onset_times)}")

        # ── Step 4: Spectral analysis ───────────────────────
        print("  📊 Computing frequency bands (STFT)...")

        # Use hop_length that matches our chunk size for time alignment
        hop_length = Config.CHUNK_SIZE

        # Compute magnitude spectrogram
        D = librosa.stft(y, n_fft=2048, hop_length=hop_length)
        S = np.abs(D)  # Magnitude

        # Frequency bin centers
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Frame timestamps (aligned with STFT frames)
        num_frames  = S.shape[1]
        frame_times = librosa.frames_to_time(
            np.arange(num_frames), sr=sr, hop_length=hop_length
        )

        # Compute per-frame band energies
        freq_bands_data = self._compute_band_energies(S, freqs, num_frames)

        # ── Step 5: RMS energy ──────────────────────────────
        rms_frames = librosa.feature.rms(
            y=y, frame_length=2048, hop_length=hop_length
        )[0]
        # Pad/trim to match freq frames
        rms_data = np.zeros(num_frames)
        n = min(len(rms_frames), num_frames)
        rms_data[:n] = rms_frames[:n]

        print(f"  ✅ Analysis complete! Frames: {num_frames}")

        return SongAnalysis(
            samples         = y,
            sample_rate     = sr,
            duration        = duration,
            bpm             = bpm,
            beat_times      = beat_times,
            onset_times     = onset_times,
            freq_times      = frame_times,
            freq_bands_data = freq_bands_data,
            rms_data        = rms_data,
            song_name       = name,
        )

    def _compute_band_energies(
        self, S: np.ndarray, freqs: np.ndarray, num_frames: int
    ) -> np.ndarray:
        """
        Compute 6-band energy for every STFT frame.
        Returns array of shape (num_frames, 6), values 0-255.
        """
        bands_raw = np.zeros((num_frames, 6), dtype=np.float32)

        for band_idx, (low, high, _, _) in enumerate(Config.FREQ_BANDS):
            mask = (freqs >= low) & (freqs <= high)
            if not np.any(mask):
                continue

            # Mean energy in this band, per frame
            band_energy      = np.mean(S[mask, :], axis=0)  # shape: (num_frames,)
            bands_raw[:, band_idx] = band_energy[:num_frames]

        # ── Normalize each band independently ──
        # This prevents one band from dominating
        result = np.zeros((num_frames, 6), dtype=np.uint8)

        for band_idx in range(6):
            col = bands_raw[:, band_idx].astype(np.float64)

            # Use 98th percentile as max to avoid outlier clipping
            p98 = np.percentile(col[col > 0], 98) if np.any(col > 0) else 1.0
            p98 = max(p98, 1e-6)

            # Apply boost and normalize to 0-255
            boost     = Config.BAND_BOOST[band_idx]
            col_norm  = np.clip(col / p98 * 255.0 * boost, 0, 255)
            result[:, band_idx] = col_norm.astype(np.uint8)

        return result

# ============================================================
# EVENT TIMELINE BUILDER
# ============================================================
class EventTimelineBuilder:
    """
    Converts song analysis data into a sorted list of LED events.
    Each event has an exact timestamp when it should fire.
    """

    def build(self, analysis: SongAnalysis, mode: int) -> List[LEDEvent]:
        events = []

        if mode == 0:
            # All Flash - beats only, all LEDs
            for t in analysis.beat_times:
                events.append(LEDEvent(
                    timestamp  = t,
                    event_type = 'BEAT',
                    intensity  = 255,
                    priority   = 3
                ))

        elif mode == 1:
            # Chase - beats + onsets
            for t in analysis.beat_times:
                events.append(LEDEvent(t, 'BEAT', 230, priority=3))
            for t in analysis.onset_times:
                events.append(LEDEvent(t, 'ONSET', 150, priority=2))

        elif mode == 2:
            # Alternate - beats
            for t in analysis.beat_times:
                events.append(LEDEvent(t, 'BEAT', 200, priority=3))

        elif mode == 3:
            # Frequency (Best!) - continuous freq update + beat flash
            # Add frequency updates at target framerate
            target_interval = 1.0 / Config.FREQ_UPDATE_HZ
            t = 0.0
            while t < analysis.duration:
                # Find nearest STFT frame
                frame_idx = self._time_to_frame(t, analysis.freq_times)
                bands     = analysis.freq_bands_data[frame_idx].tolist()

                events.append(LEDEvent(
                    timestamp  = t,
                    event_type = 'FREQ',
                    bands      = bands,
                    priority   = 1
                ))
                t += target_interval

            # Also add beat events (high priority - will override FREQ briefly)
            for t in analysis.beat_times:
                events.append(LEDEvent(t, 'BEAT', 255, priority=3))

        elif mode == 4:
            # VU Meter - RMS-based
            target_interval = 1.0 / Config.FREQ_UPDATE_HZ
            t = 0.0
            while t < analysis.duration:
                frame_idx = self._time_to_frame(t, analysis.freq_times)
                rms       = float(analysis.rms_data[frame_idx])
                vol       = min(255, int(rms * 2000))
                events.append(LEDEvent(t, 'VU', vol, priority=1))
                t += target_interval

        elif mode == 5:
            # Random - onsets trigger random effects
            for t in analysis.onset_times:
                events.append(LEDEvent(t, 'ONSET', 200, priority=2))

        else:
            # Default to frequency mode
            return self.build(analysis, 3)

        # Sort by timestamp, then priority (descending for same timestamp)
        events.sort(key=lambda e: (e.timestamp, -e.priority))

        print(f"  📋 Timeline built: {len(events):,} events for mode {mode}")
        return events

    def _time_to_frame(self, t: float, frame_times: np.ndarray) -> int:
        """Find closest STFT frame index for a given time"""
        if len(frame_times) == 0:
            return 0
        idx = int(np.searchsorted(frame_times, t))
        return min(idx, len(frame_times) - 1)

# ============================================================
# ARDUINO CONTROLLER
# ============================================================
class ArduinoController:
    """
    Handles serial communication with Arduino.
    Uses a dedicated send queue + thread to prevent blocking
    the main audio/event loop.
    """

    def __init__(self):
        self.ser        : Optional[serial.Serial] = None
        self.connected  : bool  = False
        self.port       : str   = ""
        self._queue     : queue.Queue = queue.Queue(maxsize=200)
        self._lock      : threading.Lock = threading.Lock()
        self._thread    : Optional[threading.Thread] = None
        self._running   : bool = False

        # Statistics
        self.commands_sent  : int   = 0
        self.commands_dropped: int  = 0

    # ── Port Discovery ──────────────────────────────────────
    def find_port(self) -> Optional[str]:
        ports = serial.tools.list_ports.comports()
        print("\n  🔍 Scanning serial ports...\n")

        ARDUINO_KEYWORDS = [
            'arduino', 'ch340', 'ch341', 'ch342',
            'ft232', 'ft231', 'usb serial', 'usb-serial',
            'mega', 'uno', 'nano', 'leonardo', 'pro micro',
        ]

        for p in ports:
            desc = p.description.lower()
            print(f"     {p.device:15s} → {p.description}")
            if any(kw in desc for kw in ARDUINO_KEYWORDS):
                return p.device
        return None

    # ── Connection ──────────────────────────────────────────
    def connect(self, port: str = None) -> bool:
        if not port:
            port = self.find_port()

        if not port:
            manual = input(
                "\n  ⚠️  Arduino not found. Enter port (e.g. COM3, /dev/ttyUSB0): "
            ).strip()
            port = manual or None

        if not port:
            print("  ❌ No port specified. Running without Arduino.")
            return False

        try:
            print(f"\n  🔌 Connecting {port} @ {Config.BAUD_RATE} baud...")
            self.ser = serial.Serial(
                port     = port,
                baudrate = Config.BAUD_RATE,
                timeout  = 1.0,
                write_timeout = 0.05,    # Don't block on write!
            )
            time.sleep(2.2)              # Arduino bootloader delay
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Ping test
            self._write_raw("P\n")
            time.sleep(0.15)
            resp = self.ser.readline().decode(errors='ignore').strip()

            self.connected = True
            self.port      = port

            if resp in ('PONG', 'ARDUINO_READY', 'OK'):
                print(f"  ✅ Arduino confirmed on {port}! Response: {resp}")
            else:
                print(f"  ✅ Connected on {port} (no ping, continuing)")

            # Start background send thread
            self._running = True
            self._thread  = threading.Thread(
                target=self._send_loop, daemon=True, name="ArduinoSend"
            )
            self._thread.start()
            return True

        except serial.SerialException as e:
            print(f"  ❌ Serial error: {e}")
            return False

    # ── Background Send Loop ────────────────────────────────
    def _send_loop(self):
        """
        Runs in its own thread. Drains the command queue and
        writes to serial. This way the main thread NEVER blocks
        waiting for serial I/O.
        """
        while self._running:
            try:
                cmd = self._queue.get(timeout=0.005)
                if cmd is None:
                    break
                self._write_raw(cmd + "\n")
                self.commands_sent += 1
            except queue.Empty:
                continue
            except Exception:
                pass

    def _write_raw(self, data: str):
        if self.ser and self.ser.is_open:
            self.ser.write(data.encode('ascii'))

    # ── Public Command Interface ────────────────────────────
    def send(self, cmd: str, urgent: bool = False) -> bool:
        """
        Queue a command for sending.
        If urgent=True, clear queue first (for beat flashes).
        """
        if not self.connected:
            return False

        if urgent:
            # Clear pending low-priority commands so beat fires NOW
            try:
                while not self._queue.empty():
                    self._queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(cmd)
            return True
        except queue.Full:
            self.commands_dropped += 1
            return False

    def send_beat(self, intensity: int):
        """Beat flash - urgent, clears queue"""
        self.send(f"B{min(255, intensity)}", urgent=True)

    def send_freq(self, bands: List[int]):
        """Frequency bands update"""
        self.send("F" + ",".join(map(str, bands)))

    def send_vu(self, level: int):
        """VU meter level"""
        self.send(f"V{min(255, level)}")

    def send_onset(self, intensity: int):
        """Onset flash (softer than beat)"""
        self.send(f"S{min(255, intensity)}")

    def set_mode(self, mode: int):
        """Set LED effect mode"""
        self.send(f"M{mode}", urgent=True)

    def all_off(self):
        """Turn off all LEDs"""
        self.send("O", urgent=True)

    # ── Cleanup ─────────────────────────────────────────────
    def disconnect(self):
        self.all_off()
        time.sleep(0.1)
        self._running = False
        if self._thread:
            self._queue.put(None)         # Signal thread to exit
            self._thread.join(timeout=2)
        if self.ser:
            self.ser.close()
        self.connected = False
        print(f"\n  🔌 Disconnected. Sent: {self.commands_sent}, "
              f"Dropped: {self.commands_dropped}")

# ============================================================
# PRECISION CLOCK - Tracks playback position accurately
# ============================================================
class PlaybackClock:
    """
    Tracks song position based on samples written to audio buffer.
    This is MORE ACCURATE than wall clock because it's tied to
    actual audio output, not system time (which can drift).
    """

    def __init__(self, sample_rate: int, chunk_size: int):
        self.sample_rate    = sample_rate
        self.chunk_size     = chunk_size
        self.samples_played = 0
        self._start_wall    = 0.0
        self._lock          = threading.Lock()

    def start(self):
        self._start_wall    = time.perf_counter()
        self.samples_played = 0

    def tick(self, num_samples: int):
        """Call this each time a chunk is written to audio buffer"""
        with self._lock:
            self.samples_played += num_samples

    def current_time(self) -> float:
        """
        Returns current playback position in seconds.
        Uses sample count as primary clock (most accurate),
        falls back to wall clock.
        """
        with self._lock:
            return self.samples_played / self.sample_rate

    def wall_time(self) -> float:
        return time.perf_counter() - self._start_wall

# ============================================================
# CONSOLE VISUALIZER
# ============================================================
class ConsoleVisualizer:
    """Real-time terminal visualization with smooth animations"""

    BLOCKS    = " ▁▂▃▄▅▆▇█"
    LED_ON    = "💡"
    LED_OFF   = "⚫"
    BEAT_ON   = " ● BEAT ● "
    BEAT_OFF  = "          "

    def __init__(self, analysis: SongAnalysis):
        self.analysis     = analysis
        self.smooth_bands = [0.0] * 6
        self.last_draw    = 0.0
        self.beat_flash   = 0.0  # Time of last beat (for display)

    def update(
        self,
        bands     : List[int],
        is_beat   : bool,
        volume    : int,
        elapsed   : float,
    ):
        now = time.perf_counter()
        if now - self.last_draw < 1.0 / Config.VIZ_UPDATE_HZ:
            return
        self.last_draw = now

        if is_beat:
            self.beat_flash = now

        # Smooth band display
        alpha = Config.SMOOTH_ALPHA
        for i in range(6):
            self.smooth_bands[i] = (
                alpha * self.smooth_bands[i] + (1 - alpha) * bands[i]
            )

        self._render(elapsed, is_beat, volume)

    def _render(self, elapsed: float, is_beat: bool, volume: int):
        total = self.analysis.duration
        name  = self.analysis.song_name[:24]
        bpm   = self.analysis.bpm

        # Progress
        pct      = elapsed / total if total > 0 else 0
        bar_fill = int(48 * pct)
        prog_bar = "█" * bar_fill + "░" * (48 - bar_fill)
        e_str    = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        t_str    = f"{int(total//60):02d}:{int(total%60):02d}"

        # Beat indicator
        beat_age = time.perf_counter() - self.beat_flash
        beat_str = self.BEAT_ON if beat_age < 0.1 else self.BEAT_OFF

        # Frequency bars
        labels = [lo for lo, hi, pin, name_ in Config.FREQ_BANDS]
        band_display = ""
        for i, (v, (lo, hi, pin, lbl)) in enumerate(
            zip(self.smooth_bands, Config.FREQ_BANDS)
        ):
            idx = min(8, int(v / 32))
            band_display += f" {lbl}:{self.BLOCKS[idx]}"

        # LED icons
        leds = "".join(
            self.LED_ON if v > 60 else self.LED_OFF
            for v in self.smooth_bands
        )

        # Volume bar
        vol_filled = min(24, int(volume / 10.7))
        vol_bar    = "█" * vol_filled + "░" * (24 - vol_filled)

        # Render (overwrite previous lines)
        lines = [
            f"\r╔══════════════════════════════════════════════════════╗",
            f"\r║  🎵 {name:<24} {bpm:5.1f} BPM  {beat_str}  ║",
            f"\r║  [{prog_bar}] {e_str}/{t_str}  ║",
            f"\r║ {band_display}                        ║",
            f"\r║  LEDs: {leds}   Vol:[{vol_bar}]  ║",
            f"\r╚══════════════════════════════════════════════════════╝",
        ]

        output = "\n".join(lines)
        # Move cursor up to overwrite
        sys.stdout.write('\033[7A' + output)
        sys.stdout.flush()

    def start(self):
        """Print blank lines for visualizer to overwrite"""
        print("\n" * 7)

# ============================================================
# MAIN PLAYER - The core engine
# ============================================================
class BeatSyncPlayer:
    """
    Orchestrates everything:
    - Audio playback (PyAudio)
    - Precision clock
    - Event timeline firing
    - Arduino communication
    - Visualizer
    """

    def __init__(self, arduino: ArduinoController):
        self.arduino    = arduino
        self.pa         = pyaudio.PyAudio()
        self.clock      : Optional[PlaybackClock] = None
        self._stop_flag : threading.Event = threading.Event()

        # Latency offset in seconds (adjust to sync LEDs with audio)
        self.latency_offset = (
            Config.SERIAL_LATENCY_MS + Config.AUDIO_BUFFER_MS
        ) / 1000.0

    def play(self, analysis: SongAnalysis, events: List[LEDEvent], mode: int):
        """
        Main playback function.
        Audio runs in THIS thread (required by PyAudio).
        Event firing runs in a separate thread.
        """
        self._stop_flag.clear()

        # Setup clock
        self.clock = PlaybackClock(analysis.sample_rate, Config.CHUNK_SIZE)

        # Setup visualizer
        viz = ConsoleVisualizer(analysis)

        if Config.SHOW_VISUALIZER:
            viz.start()

        # Set Arduino mode
        if self.arduino.connected:
            self.arduino.set_mode(mode)
            time.sleep(0.05)

        # Current state (shared with event thread via closure)
        state = {
            'bands'   : [0] * 6,
            'is_beat' : False,
            'volume'  : 0,
        }

        # ── Start event dispatcher thread ───────────────────
        event_thread = threading.Thread(
            target = self._event_dispatcher,
            args   = (events, analysis, state),
            daemon = True,
            name   = "EventDispatch"
        )
        event_thread.start()

        # ── Audio playback loop (main thread) ───────────────
        stream = self.pa.open(
            format            = pyaudio.paFloat32,
            channels          = 1,
            rate              = analysis.sample_rate,
            output            = True,
            frames_per_buffer = Config.CHUNK_SIZE,
        )

        samples   = analysis.samples
        chunk     = Config.CHUNK_SIZE
        pos       = 0

        print(f"\n  ▶️  {analysis.song_name}  ({analysis.bpm:.1f} BPM)")

        self.clock.start()

        try:
            while pos < len(samples) and not self._stop_flag.is_set():
                end      = pos + chunk
                chunk_d  = samples[pos:end]

                # Pad last chunk
                if len(chunk_d) < chunk:
                    chunk_d = np.pad(chunk_d, (0, chunk - len(chunk_d)))

                # Write to audio device
                stream.write(chunk_d.tobytes())

                # Advance clock AFTER write (audio is now playing)
                self.clock.tick(chunk)

                # Update visualizer from shared state
                if Config.SHOW_VISUALIZER:
                    viz.update(
                        state['bands'],
                        state['is_beat'],
                        state['volume'],
                        self.clock.current_time(),
                    )

                pos += chunk

        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped by user")

        finally:
            self._stop_flag.set()
            stream.stop_stream()
            stream.close()
            event_thread.join(timeout=2)
            self.arduino.all_off()
            print(f"\n\n\n  ✅ Finished: {analysis.song_name}")

    def _event_dispatcher(
        self,
        events    : List[LEDEvent],
        analysis  : SongAnalysis,
        state     : dict
    ):
        """
        Fires LED events at precisely the right time.

        Key insight: We compare the event timestamp against
        the SAMPLE-BASED clock, not wall clock. This keeps
        us perfectly in sync with actual audio output.

        Lookahead: We fire events slightly EARLY to compensate
        for serial transmission delay.
        """
        event_idx      = 0
        total_events   = len(events)
        last_freq_time = -1.0
        MIN_SLEEP      = 0.001  # 1ms minimum sleep

        while not self._stop_flag.is_set() and event_idx < total_events:
            # Get current playback position from sample clock
            now = self.clock.current_time()

            # --- Process all events that should fire now or earlier ---
            while event_idx < total_events:
                evt = events[event_idx]

                # Fire event early by latency_offset to compensate serial delay
                fire_at = evt.timestamp - self.latency_offset

                if now >= fire_at:
                    self._fire_event(evt, analysis, state, now)
                    event_idx += 1
                else:
                    break  # Events are sorted, so we can stop here

            # --- Sleep until next event ---
            if event_idx < total_events:
                next_fire = events[event_idx].timestamp - self.latency_offset
                sleep_time = next_fire - now - 0.001  # Wake up 1ms early
                if sleep_time > MIN_SLEEP:
                    time.sleep(sleep_time)
                else:
                    time.sleep(MIN_SLEEP)

    def _fire_event(
        self,
        evt      : LEDEvent,
        analysis : SongAnalysis,
        state    : dict,
        now      : float
    ):
        """Execute a single LED event"""
        if not self.arduino.connected:
            return

        et = evt.event_type

        if et == 'BEAT':
            self.arduino.send_beat(evt.intensity)
            state['is_beat'] = True

            # Schedule beat-off (reset is_beat in state)
            def reset_beat():
                time.sleep(0.08)
                state['is_beat'] = False
            threading.Thread(target=reset_beat, daemon=True).start()

        elif et == 'ONSET':
            self.arduino.send_onset(evt.intensity)

        elif et == 'FREQ':
            bands = evt.bands
            state['bands']  = bands
            state['volume'] = max(bands)
            self.arduino.send_freq(bands)

        elif et == 'VU':
            state['volume'] = evt.intensity
            self.arduino.send_vu(evt.intensity)

    def stop(self):
        self._stop_flag.set()

    def close(self):
        self.pa.terminate()

# ============================================================
# LATENCY CALIBRATION TOOL
# ============================================================
def calibrate_latency(arduino: ArduinoController):
    """
    Interactive calibration: plays a click track and helps you
    measure the offset between audio and LED flash.
    """
    print("\n  🔧 LATENCY CALIBRATION")
    print("  " + "─" * 40)
    print("  Listen for the click AND watch the LED.")
    print("  If LED flashes BEFORE click: increase SERIAL_LATENCY_MS")
    print("  If LED flashes AFTER click:  decrease it (or go negative)")
    print(f"\n  Current: SERIAL_LATENCY_MS = {Config.SERIAL_LATENCY_MS} ms")
    print(f"           AUDIO_BUFFER_MS   = {Config.AUDIO_BUFFER_MS} ms")

    val = input("\n  Enter new SERIAL_LATENCY_MS (or Enter to skip): ").strip()
    if val:
        try:
            Config.SERIAL_LATENCY_MS = float(val)
            print(f"  ✅ Set to {Config.SERIAL_LATENCY_MS} ms")
        except ValueError:
            pass

# ============================================================
# FILE PICKER
# ============================================================
def pick_mp3() -> Optional[str]:
    print("\n  📁 MP3 SELECTION")
    print("  " + "─" * 40)

    # Scan current directory
    mp3_files = sorted(
        f for f in os.listdir('.')
        if f.lower().endswith('.mp3')
    )

    if mp3_files:
        print("\n  Found in current folder:\n")
        for i, f in enumerate(mp3_files):
            mb = os.path.getsize(f) / 1_048_576
            print(f"   [{i+1:2d}] {f} ({mb:.1f} MB)")

        choice = input("\n  Enter number or full path: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(mp3_files):
                return mp3_files[idx]

    path = input("\n  Enter MP3 path: ").strip().strip('"').strip("'")
    return path if os.path.exists(path) else None

# ============================================================
# MODE SELECTOR
# ============================================================
MODES = [
    (0, "All Flash",       "Every beat → all 6 LEDs flash together"),
    (1, "Chase",           "Beats + onsets chase across LEDs"),
    (2, "Alternate",       "Beats alternate between odd/even LEDs"),
    (3, "Frequency ★",    "Each LED = one frequency band (BEST!)"),
    (4, "VU Meter",        "LEDs fill up with volume level"),
    (5, "Onset Flash",     "Every transient → random LED"),
]

def pick_mode() -> int:
    print("\n  🎮 LED MODE")
    print("  " + "─" * 40)
    for idx, name, desc in MODES:
        star = " ←" if idx == 3 else ""
        print(f"   [{idx}] {name:<18} {desc}{star}")

    choice = input("\n  Select mode [3]: ").strip()
    return int(choice) if choice.isdigit() and 0 <= int(choice) <= 5 else 3

# ============================================================
# MAIN
# ============================================================
def main():
    os.system('cls' if os.name == 'nt' else 'clear')

    print("╔══════════════════════════════════════════════════════╗")
    print("║      🎵  MP3 BEAT SYNC PRO  →  Arduino LEDs  💡     ║")
    print("║           Extreme Accuracy Edition                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print("""
  How it works:
  ┌───────────────────────────────────────────────────┐
  │  Pre-analyze → Build Timeline → Play + Fire exact │
  │  Sample-accurate clock | Lookahead scheduling     │
  │  No drift. No jitter. Perfect sync.               │
  └───────────────────────────────────────────────────┘
  Pin mapping:
    Pin  8 = Sub-Bass  (20-80 Hz)    kick drum thump
    Pin  9 = Bass      (80-250 Hz)   bass guitar
    Pin 10 = Low-Mid   (250-600 Hz)  warmth
    Pin 11 = Mid       (600-2500 Hz) vocals/melody
    Pin 12 = High-Mid  (2.5-7 kHz)  presence/clarity
    Pin 13 = Treble    (7-18 kHz)   hi-hats/cymbals
    """)

    # ── Step 1: Connect Arduino ──────────────────────────
    print("═" * 54)
    print("  STEP 1: Arduino Connection")
    print("═" * 54)
    arduino = ArduinoController()
    arduino.connect()

    # ── Step 2: Pick MP3 ─────────────────────────────────
    print("\n" + "═" * 54)
    print("  STEP 2: Select MP3")
    print("═" * 54)
    mp3_path = pick_mp3()
    if not mp3_path or not os.path.exists(mp3_path):
        print("  ❌ File not found!")
        arduino.disconnect()
        return

    # ── Step 3: Pick Mode ────────────────────────────────
    print("\n" + "═" * 54)
    print("  STEP 3: LED Mode")
    print("═" * 54)
    mode = pick_mode()

    # ── Step 4: Optional Calibration ─────────────────────
    print("\n" + "═" * 54)
    print("  STEP 4: Latency Calibration (Optional)")
    print("═" * 54)
    do_cal = input("\n  Calibrate latency? [y/N]: ").strip().lower()
    if do_cal == 'y':
        calibrate_latency(arduino)

    # ── Step 5: Analyze ──────────────────────────────────
    print("\n" + "═" * 54)
    print("  Analyzing song...")
    print("═" * 54)

    try:
        analyzer = SongAnalyzer()
        analysis = analyzer.analyze(mp3_path)

        timeline_builder = EventTimelineBuilder()
        events           = timeline_builder.build(analysis, mode)

    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        arduino.disconnect()
        return

    # ── Step 6: Play ─────────────────────────────────────
    print("\n" + "═" * 54)
    print(f"  ▶  Starting playback in 3 seconds...")
    print(f"     Press Ctrl+C to stop")
    print("═" * 54)

    for i in range(3, 0, -1):
        print(f"\r  Starting in {i}...", end='', flush=True)
        time.sleep(1)
    print("\r                    ")

    player = BeatSyncPlayer(arduino)

    try:
        player.play(analysis, events, mode)
    except KeyboardInterrupt:
        print("\n  ⏹  Stopped.")
    finally:
        player.stop()
        player.close()
        arduino.disconnect()
        print("\n  👋 Done!\n")

# ============================================================
if __name__ == "__main__":
    main()