#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           MP3 BEAT SYNC PRO - Single File Edition           ║
║                  Ultra-Accuracy LED Sync                    ║
╚══════════════════════════════════════════════════════════════╝

INSTALL:
    pip install librosa pyaudio pyserial rich pyyaml joblib
        numpy soundfile pydub scipy

USAGE:
    python beat_sync.py
    python beat_sync.py mysong.mp3
    python beat_sync.py mysong.mp3 --mode 3 --port COM3
    python beat_sync.py --calibrate
"""

# ── Standard library ──────────────────────────────────────────────────────────
from __future__ import annotations
import argparse
import hashlib
import math
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Third-party (checked below) ───────────────────────────────────────────────
def _require(pkg: str, install: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"  ❌  Missing: {pkg}  →  pip install {install}")
        sys.exit(1)

np        = _require("numpy",    "numpy")
import numpy as np
pyaudio   = _require("pyaudio",  "pyaudio")
import pyaudio
serial_mod= _require("serial",   "pyserial")
import serial
import serial.tools.list_ports
rich_mod  = _require("rich",     "rich")

from rich.console   import Console
from rich.layout    import Layout
from rich.live      import Live
from rich.panel     import Panel
from rich.prompt    import Confirm, Prompt
from rich.table     import Table
from rich.text      import Text

# Optional but heavily used
try:
    import librosa, librosa.beat, librosa.onset, librosa.feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("  ❌  librosa required  →  pip install librosa"); sys.exit(1)

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("  ⚠️   joblib not found – caching disabled.  pip install joblib")

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit here or pass CLI flags)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Cfg:
    # ── Audio ──────────────────────────────────────────────────────────────
    SAMPLE_RATE     : int   = 44100
    CHUNK           : int   = 256       # Small chunk = low latency
    CHANNELS        : int   = 1
    OUT_DEVICE      : Optional[int] = None

    # ── Serial ─────────────────────────────────────────────────────────────
    BAUD            : int   = 500000    # 500k = ~0.1ms per command
    PORT            : Optional[str] = None
    SERIAL_Q_SIZE   : int   = 600

    # ── Beat Detection ──────────────────────────────────────────────────────
    BEAT_TIGHTNESS  : float = 400.0
    ONSET_DELTA     : float = 0.06
    MIN_BPM         : float = 60.0
    MAX_BPM         : float = 200.0

    # ── Latency (ms) ────────────────────────────────────────────────────────
    # Negative = fire LEDs EARLIER to compensate serial delay
    LATENCY_MS      : float = -8.0

    # ── Frequency Bands → LED pins ──────────────────────────────────────────
    # (low_hz, high_hz, pin, label, boost)
    BANDS           : List[Tuple] = field(default_factory=lambda: [
        (20,    80,    3,  "SUB", 3.2),
        (80,    250,   5,  "BAS", 2.6),
        (250,   600,   6,  "LMD", 2.1),
        (600,   2500,  9,  "MID", 1.8),
        (2500,  7000,  10, "HMD", 1.5),
        (7000,  18000, 11, "TRB", 1.3),
    ])

    # ── LED Behaviour ───────────────────────────────────────────────────────
    LED_HZ          : int   = 100       # How often freq bands are sent
    SMOOTH_ALPHA    : float = 0.55      # Display smoothing (0=raw)
    BEAT_MS         : int   = 75        # Beat flash hold (ms)
    ONSET_MS        : int   = 45

    # ── AGC (Auto Gain Control) ─────────────────────────────────────────────
    AGC_ENABLED     : bool  = True
    AGC_WINDOW_S    : float = 5.0       # Rolling window for percentile
    AGC_PERCENTILE  : float = 98.0
    AGC_FLOOR       : float = 0.001

    # ── BPM Breathing ───────────────────────────────────────────────────────
    BREATHING       : bool  = True
    BREATH_THRESH   : float = 0.015     # RMS below this = quiet section
    BREATH_PEAK     : int   = 55        # Max brightness

    # ── Cache ───────────────────────────────────────────────────────────────
    CACHE_DIR       : str   = "./beat_cache"
    CACHE_VER       : str   = "4.0"

    # ── Visualizer ──────────────────────────────────────────────────────────
    VIZ             : bool  = True
    VIZ_HZ          : int   = 30

    # Colours for each band in the terminal
    COLORS          : List[str] = field(default_factory=lambda: [
        "red", "dark_orange", "yellow", "green", "cyan", "blue"
    ])


CFG = Cfg()   # Global singleton – tweak above or via CLI

# ══════════════════════════════════════════════════════════════════════════════
#  EVENT BUS  – lightweight pub/sub
# ══════════════════════════════════════════════════════════════════════════════
class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event: str, cb: Callable):
        with self._lock:
            self._subs.setdefault(event, []).append(cb)

    def emit(self, event: str, **kw):
        with self._lock:
            cbs = list(self._subs.get(event, []))
        for cb in cbs:
            try: cb(**kw)
            except Exception: pass

    # Fire without acquiring lock (hot path inside audio callback)
    def emit_fast(self, event: str, **kw):
        for cb in self._subs.get(event, []):
            try: cb(**kw)
            except Exception: pass


BUS = EventBus()

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SongAnalysis:
    samples         : np.ndarray
    sr              : int
    duration        : float
    name            : str
    bpm             : float
    beat_times      : np.ndarray
    onset_times     : np.ndarray
    onset_strength  : np.ndarray
    frame_times     : np.ndarray
    band_data       : np.ndarray   # (frames, 6) uint8  0-255
    rms_data        : np.ndarray   # (frames,)   float32
    quiet_mask      : np.ndarray   # (frames,)   bool
    version         : str = "4.0"


@dataclass
class LEDEvent:
    t    : float           # Timestamp (seconds)
    kind : str             # BEAT | ONSET | FREQ | VU | BREATHE
    val  : int   = 0       # intensity / vu level
    bands: List[int] = field(default_factory=lambda: [0]*6)
    pri  : int   = 1       # 0=highest

    def __lt__(self, o):
        return (self.t, self.pri) < (o.t, o.pri)

# ══════════════════════════════════════════════════════════════════════════════
#  SONG ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
class SongAnalyzer:

    # ── Cache helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _hash(path: str) -> str:
        h = hashlib.md5()
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            h.update(f.read(min(65536, size)))
            if size > 65536:
                f.seek(-65536, 2)
                h.update(f.read(65536))
        return h.hexdigest()[:16]

    def _cache_path(self, mp3: str) -> str:
        os.makedirs(CFG.CACHE_DIR, exist_ok=True)
        return os.path.join(CFG.CACHE_DIR,
                            f"{self._hash(mp3)}_v{CFG.CACHE_VER}.pkl")

    def _load_cache(self, mp3: str) -> Optional[SongAnalysis]:
        if not HAS_JOBLIB:
            return None
        p = self._cache_path(mp3)
        if os.path.exists(p):
            try:
                obj = joblib.load(p)
                if isinstance(obj, SongAnalysis) and obj.version == CFG.CACHE_VER:
                    return obj
            except Exception:
                pass
        return None

    def _save_cache(self, mp3: str, a: SongAnalysis):
        if not HAS_JOBLIB:
            return
        try:
            joblib.dump(a, self._cache_path(mp3), compress=3)
        except Exception:
            pass

    # ── Audio loading ──────────────────────────────────────────────────────
    def _load_audio(self, path: str) -> np.ndarray:
        """Try soundfile → pydub → librosa (fastest to slowest)."""
        if HAS_SF:
            try:
                y, orig_sr = sf.read(path, dtype="float32", always_2d=False)
                if y.ndim == 2:
                    y = y.mean(axis=1)
                if orig_sr != CFG.SAMPLE_RATE:
                    y = librosa.resample(y, orig_sr=orig_sr,
                                         target_sr=CFG.SAMPLE_RATE)
                return y
            except Exception:
                pass
        if HAS_PYDUB:
            try:
                seg = (AudioSegment.from_file(path)
                       .set_channels(1)
                       .set_frame_rate(CFG.SAMPLE_RATE)
                       .set_sample_width(2))
                raw = np.frombuffer(seg.raw_data, dtype=np.int16)
                return raw.astype(np.float32) / 32768.0
            except Exception:
                pass
        y, _ = librosa.load(path, sr=CFG.SAMPLE_RATE, mono=True)
        return y

    # ── Beat tracking ──────────────────────────────────────────────────────
    def _beats(self, y: np.ndarray):
        tempo, frames = librosa.beat.beat_track(
            y=y, sr=CFG.SAMPLE_RATE,
            tightness=CFG.BEAT_TIGHTNESS, trim=False)
        bpm = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
        bpm = float(np.clip(bpm, CFG.MIN_BPM, CFG.MAX_BPM))
        return bpm, librosa.frames_to_time(frames, sr=CFG.SAMPLE_RATE)

    # ── Onset detection ────────────────────────────────────────────────────
    def _onsets(self, y: np.ndarray):
        env = librosa.onset.onset_strength(
            y=y, sr=CFG.SAMPLE_RATE, aggregate=np.median)
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=CFG.SAMPLE_RATE,
            delta=CFG.ONSET_DELTA, backtrack=True, units="frames")
        times = librosa.frames_to_time(frames, sr=CFG.SAMPLE_RATE)
        if len(frames):
            raw = env[np.clip(frames, 0, len(env)-1)]
            mx  = raw.max()
            strength = raw / mx if mx > 0 else raw
        else:
            strength = np.array([])
        return times, strength

    # ── Spectrogram + AGC normalisation ───────────────────────────────────
    def _spectrogram(self, y: np.ndarray):
        hop   = CFG.CHUNK
        n_fft = max(2048, hop * 8)
        S     = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        freqs = librosa.fft_frequencies(sr=CFG.SAMPLE_RATE, n_fft=n_fft)
        nf    = S.shape[1]
        ftimes= librosa.frames_to_time(np.arange(nf),
                                        sr=CFG.SAMPLE_RATE, hop_length=hop)

        nb     = len(CFG.BANDS)
        raw    = np.zeros((nf, nb), dtype=np.float32)
        for i, (lo, hi, *_) in enumerate(CFG.BANDS):
            mask = (freqs >= lo) & (freqs <= hi)
            if mask.any():
                raw[:, i] = S[mask, :].mean(axis=0)

        # AGC: rolling percentile per band
        win   = max(1, int(CFG.AGC_WINDOW_S * CFG.SAMPLE_RATE / hop))
        out   = np.zeros_like(raw)
        for b in range(nb):
            col   = raw[:, b].astype(np.float64)
            boost = CFG.BANDS[b][4]
            norm  = np.zeros_like(col)

            if CFG.AGC_ENABLED:
                for i in range(nf):
                    slc = col[max(0, i-win):i+1]
                    nz  = slc[slc > 0]
                    p   = np.percentile(nz, CFG.AGC_PERCENTILE) \
                          if len(nz) > 3 else CFG.AGC_FLOOR
                    p   = max(p, CFG.AGC_FLOOR)
                    norm[i] = col[i] / p * boost
            else:
                p = np.percentile(col[col > 0], 98) if col.any() else 1.0
                norm = col / max(p, 1e-6) * boost

            out[:, b] = np.clip(norm, 0.0, 1.0)

        return ftimes, (out * 255).astype(np.uint8)

    # ── RMS energy ────────────────────────────────────────────────────────
    def _energy(self, y: np.ndarray, nf: int):
        hop  = CFG.CHUNK
        rms  = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
        out  = np.zeros(nf, dtype=np.float32)
        n    = min(len(rms), nf)
        out[:n] = rms[:n]
        quiet = out < CFG.BREATH_THRESH
        return out, quiet

    # ── Public entry point ────────────────────────────────────────────────
    def analyze(self, path: str) -> SongAnalysis:
        name = os.path.splitext(os.path.basename(path))[0]

        cached = self._load_cache(path)
        if cached:
            console.print(f"  [cyan]⚡ Cache hit:[/cyan] {name}")
            return cached

        console.print(f"\n  [bold]🔬 Analyzing:[/bold] {name}")
        t0 = time.perf_counter()

        console.print("    Loading audio...")
        y = self._load_audio(path)
        dur = len(y) / CFG.SAMPLE_RATE

        console.print("    Tracking beats...")
        bpm, beat_t = self._beats(y)

        console.print("    Detecting onsets...")
        onset_t, onset_s = self._onsets(y)

        console.print("    Computing spectrogram (AGC)...")
        frame_t, band_data = self._spectrogram(y)

        console.print("    Computing energy...")
        rms, quiet = self._energy(y, len(frame_t))

        elapsed = time.perf_counter() - t0
        console.print(
            f"  [green]✅ Done in {elapsed:.1f}s[/green] | "
            f"BPM={bpm:.1f} | Beats={len(beat_t)} | "
            f"Onsets={len(onset_t)} | "
            f"Duration={int(dur//60):02d}:{int(dur%60):02d}"
        )

        a = SongAnalysis(
            samples=y, sr=CFG.SAMPLE_RATE, duration=dur, name=name,
            bpm=bpm, beat_times=beat_t,
            onset_times=onset_t, onset_strength=onset_s,
            frame_times=frame_t, band_data=band_data,
            rms_data=rms, quiet_mask=quiet,
        )
        self._save_cache(path, a)
        return a


# ══════════════════════════════════════════════════════════════════════════════
#  TIMELINE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
class TimelineBuilder:

    def build(self, a: SongAnalysis, mode: int) -> List[LEDEvent]:
        evts: List[LEDEvent] = []

        # Decimate frequency frames to target LED_HZ
        frames_per_s = CFG.SAMPLE_RATE / CFG.CHUNK
        step = max(1, int(frames_per_s / CFG.LED_HZ))

        # ── Frequency / VU events ──────────────────────────────────────────
        if mode == 3:
            for i in range(0, len(a.frame_times), step):
                evts.append(LEDEvent(
                    t=float(a.frame_times[i]), kind="FREQ", pri=1,
                    bands=a.band_data[i].tolist()))

        elif mode == 4:
            for i in range(0, len(a.frame_times), step):
                vol = min(255, int(float(a.rms_data[i]) * 3000))
                evts.append(LEDEvent(
                    t=float(a.frame_times[i]), kind="VU", val=vol, pri=1))

        # ── Beat events ────────────────────────────────────────────────────
        if mode in (0, 1, 2, 3):
            for t in a.beat_times:
                evts.append(LEDEvent(t=float(t), kind="BEAT", val=255, pri=0))

        # ── Onset events ───────────────────────────────────────────────────
        if mode in (1, 5):
            for t, s in zip(a.onset_times, a.onset_strength):
                evts.append(LEDEvent(
                    t=float(t), kind="ONSET",
                    val=int(np.clip(s * 220, 80, 220)), pri=1))

        # ── BPM Breathing (quiet sections only) ───────────────────────────
        if CFG.BREATHING:
            dt    = 1.0 / CFG.LED_HZ
            period= 60.0 / a.bpm
            t     = 0.0
            while t < a.duration:
                fi = int(np.searchsorted(a.frame_times, t))
                fi = min(fi, len(a.quiet_mask) - 1)
                if a.quiet_mask[fi]:
                    phase  = (t % period) / period
                    breath = int((math.sin(phase * 2 * math.pi) * 0.5 + 0.5)
                                 * CFG.BREATH_PEAK)
                    bands  = [breath] * len(CFG.BANDS)
                    evts.append(LEDEvent(
                        t=t, kind="BREATHE", val=breath, bands=bands, pri=2))
                t += dt

        evts.sort()
        console.print(f"  [cyan]📋 Timeline:[/cyan] {len(evts):,} events")
        return evts


# ══════════════════════════════════════════════════════════════════════════════
#  ARDUINO CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════
class Arduino:
    """
    Priority-queue-backed serial controller.
    Lower priority number = sent first (beat=0, freq=2).
    """

    P_BEAT   = 0
    P_ONSET  = 1
    P_FREQ   = 2
    P_VU     = 3
    P_BREATH = 4

    def __init__(self):
        self.ser      : Optional[serial.Serial] = None
        self.connected= False
        self._q       = queue.PriorityQueue(maxsize=CFG.SERIAL_Q_SIZE)
        self._seq     = 0
        self._seq_lk  = threading.Lock()
        self._running = False
        self._thread  : Optional[threading.Thread] = None
        self.sent = self.dropped = 0

        # Subscribe to bus
        BUS.subscribe("beat",    lambda **k: self._enq(self.P_BEAT,
                                    f"B{k['val']}", flush=True))
        BUS.subscribe("onset",   lambda **k: self._enq(self.P_ONSET,
                                    f"S{k['val']}"))
        BUS.subscribe("freq",    lambda **k: self._enq(self.P_FREQ,
                                    "F" + ",".join(map(str, k['bands']))))
        BUS.subscribe("vu",      lambda **k: self._enq(self.P_VU,
                                    f"V{k['val']}"))
        BUS.subscribe("breathe", lambda **k: self._enq(self.P_BREATH,
                                    "F" + ",".join(map(str, k['bands']))))
        BUS.subscribe("mode",    lambda **k: self._enq(self.P_BEAT,
                                    f"M{k['val']}", flush=True))
        BUS.subscribe("stop",    lambda **k: self._enq(self.P_BEAT,
                                    "O", flush=True))

    # ── Connection ─────────────────────────────────────────────────────────
    def connect(self, port: str = None) -> bool:
        port = port or CFG.PORT or self._autodetect()
        if not port:
            port = Prompt.ask(
                "\n  ⚠️  Arduino not found. Enter port (COM3 / /dev/ttyUSB0)",
                default="")
        if not port:
            console.print("  [yellow]⚠️  No port – audio only[/yellow]")
            return False
        try:
            console.print(f"  🔌 Connecting [cyan]{port}[/cyan] "
                          f"@ [cyan]{CFG.BAUD}[/cyan] baud...")
            self.ser = serial.Serial(
                port=port, baudrate=CFG.BAUD,
                timeout=0.5, write_timeout=0.02)
            time.sleep(2.2)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self._write("P\n")
            time.sleep(0.15)
            resp = self.ser.readline().decode(errors="ignore").strip()
            self.connected = True
            tag = f"[green]{resp}[/green]" if resp else "no ping"
            console.print(f"  [green]✅ Connected![/green] ({tag})")
            self._running = True
            self._thread  = threading.Thread(
                target=self._loop, daemon=True, name="ArduinoSend")
            self._thread.start()
            return True
        except serial.SerialException as e:
            console.print(f"  [red]❌ Serial error: {e}[/red]")
            return False

    def _autodetect(self) -> Optional[str]:
        KW = ["arduino","ch340","ch341","ch342","ft232","ft231",
              "usb serial","usb-serial","mega","uno","nano","leonardo"]
        console.print("\n  🔍 Scanning ports...")
        for p in serial.tools.list_ports.comports():
            console.print(f"     [dim]{p.device:15s} → {p.description}[/dim]")
            if any(k in p.description.lower() for k in KW):
                return p.device
        return None

    # ── Background send loop ───────────────────────────────────────────────
    def _loop(self):
        while self._running:
            try:
                _, _, cmd = self._q.get(timeout=0.005)
                if cmd is None:
                    break
                self._write(cmd + "\n")
                self.sent += 1
            except queue.Empty:
                continue
            except serial.SerialTimeoutException:
                self.dropped += 1
            except Exception:
                pass

    def _write(self, data: str):
        if self.ser and self.ser.is_open:
            self.ser.write(data.encode("ascii"))

    # ── Enqueue helper ─────────────────────────────────────────────────────
    def _enq(self, pri: int, cmd: str, flush: bool = False):
        if not self.connected:
            return
        if flush:
            try:
                while True: self._q.get_nowait()
            except queue.Empty:
                pass
        with self._seq_lk:
            seq = self._seq; self._seq += 1
        try:
            self._q.put_nowait((pri, seq, cmd))
        except queue.Full:
            self.dropped += 1

    # ── Cleanup ────────────────────────────────────────────────────────────
    def disconnect(self):
        BUS.emit("stop")
        time.sleep(0.12)
        self._running = False
        try: self._q.put_nowait((0, 0, None))
        except queue.Full: pass
        if self._thread:
            self._thread.join(timeout=2)
        if self.ser:
            self.ser.close()
        console.print(f"  📊 Serial: sent=[green]{self.sent}[/green] "
                      f"dropped=[yellow]{self.dropped}[/yellow]")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE CLOCK
# ══════════════════════════════════════════════════════════════════════════════
class HWClock:
    """
    Tracks playback position via sample counter.
    Updated by the PyAudio callback (C-level thread) –
    zero drift, no wall-clock jitter.
    """
    def __init__(self, sr: int):
        self._sr      = sr
        self._samples = 0
        self._lock    = threading.Lock()

    def reset(self):
        with self._lock:
            self._samples = 0

    def tick(self, n: int):
        with self._lock:
            self._samples += n

    def now(self) -> float:
        with self._lock:
            return self._samples / self._sr


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYBACK ENGINE  (PyAudio callback mode)
# ══════════════════════════════════════════════════════════════════════════════
class PlaybackEngine:

    def __init__(self):
        self._pa      = pyaudio.PyAudio()
        self._clock   = HWClock(CFG.SAMPLE_RATE)
        self._stop    = threading.Event()
        self._data    : Optional[np.ndarray] = None
        self._pos     = 0
        self._pos_lk  = threading.Lock()
        self._stream  : Optional[pyaudio.Stream] = None

    # ── PyAudio callback (runs in high-priority C thread) ─────────────────
    def _callback(self, in_data, frame_count, time_info, status):
        with self._pos_lk:
            pos = self._pos
            end = pos + frame_count
            raw = self._data

        if raw is None or pos >= len(raw):
            self._stop.set()
            return (bytes(frame_count * 4), pyaudio.paComplete)

        chunk = raw[pos:end]
        if len(chunk) < frame_count:
            chunk = np.pad(chunk, (0, frame_count - len(chunk)))

        with self._pos_lk:
            self._pos += frame_count

        self._clock.tick(frame_count)
        return (chunk.tobytes(), pyaudio.paContinue)

    # ── Event dispatcher (dedicated thread) ───────────────────────────────
    def _dispatch(self, events: List[LEDEvent], a: SongAnalysis):
        """
        Fires BUS events at the precise moment the clock says so.
        Latency compensation: events fire LATENCY_MS early.
        Beat events suppress FREQ for 60ms afterward (avoids flicker).
        """
        offset          = CFG.LATENCY_MS / 1000.0  # Usually negative
        beat_quiet_until= 0.0
        BEAT_QUIET_S    = 0.06
        last_bands      = None
        idx             = 0
        total           = len(events)
        SPIN_THRESH     = 0.0005   # <0.5ms → busy-spin instead of sleep

        while not self._stop.is_set() and idx < total:
            now = self._clock.now()

            # Fire all due events
            while idx < total:
                ev      = events[idx]
                fire_at = ev.t + offset
                if now < fire_at:
                    break

                k = ev.kind
                if k == "BEAT":
                    BUS.emit_fast("beat", val=ev.val)
                    beat_quiet_until = now + BEAT_QUIET_S

                elif k == "ONSET":
                    if now > beat_quiet_until:
                        BUS.emit_fast("onset", val=ev.val)

                elif k == "FREQ":
                    if now > beat_quiet_until:
                        if ev.bands != last_bands:
                            BUS.emit_fast("freq", bands=ev.bands)
                            last_bands = ev.bands

                elif k == "VU":
                    BUS.emit_fast("vu", val=ev.val)

                elif k == "BREATHE":
                    if now > beat_quiet_until:
                        BUS.emit_fast("breathe", val=ev.val, bands=ev.bands)

                idx += 1

            # Sleep or spin until next event
            if idx < total:
                gap = events[idx].t + offset - self._clock.now()
                if gap > SPIN_THRESH:
                    time.sleep(gap - SPIN_THRESH)

    # ── Main play method ──────────────────────────────────────────────────
    def play(self, a: SongAnalysis, events: List[LEDEvent], mode: int,
             viz_start_fn: Optional[Callable] = None):

        self._stop.clear()
        self._data = a.samples.astype(np.float32)
        self._pos  = 0
        self._clock.reset()

        BUS.emit("mode", val=mode)
        time.sleep(0.05)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=CFG.CHANNELS,
            rate=CFG.SAMPLE_RATE,
            output=True,
            frames_per_buffer=CFG.CHUNK,
            output_device_index=CFG.OUT_DEVICE,
            stream_callback=self._callback,
        )

        dt = threading.Thread(
            target=self._dispatch, args=(events, a),
            daemon=True, name="Dispatch")
        dt.start()

        self._stream.start_stream()

        if viz_start_fn:
            viz_start_fn(self._clock.now)

        console.print(f"\n  [bold green]▶  {a.name}[/bold green]  "
                      f"[dim]{a.bpm:.1f} BPM[/dim]  "
                      f"[dim]Ctrl+C to stop[/dim]\n")

        try:
            while self._stream.is_active() and not self._stop.is_set():
                time.sleep(0.05)
        except KeyboardInterrupt:
            console.print("\n  [yellow]⏹  Stopped[/yellow]")
        finally:
            self._stop.set()
            self._stream.stop_stream()
            self._stream.close()
            dt.join(timeout=2)
            BUS.emit("stop")

    def close(self):
        self._pa.terminate()

    @property
    def clock(self) -> HWClock:
        return self._clock


# ══════════════════════════════════════════════════════════════════════════════
#  RICH VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    BLOCKS  = " ▁▂▃▄▅▆▇█"
    FULL    = "█"
    EMPTY   = "░"
    BAR_H   = 10     # Rows of the spectrogram

    def __init__(self, a: SongAnalysis):
        self._a       = a
        self._bands   = [0.0] * len(CFG.BANDS)
        self._smooth  = [0.0] * len(CFG.BANDS)
        self._beat_t  = 0.0
        self._vol     = 0
        self._elapsed = 0.0
        self._lock    = threading.Lock()
        self._running = False
        self._thread  : Optional[threading.Thread] = None

        BUS.subscribe("beat",    self._on_beat)
        BUS.subscribe("freq",    self._on_freq)
        BUS.subscribe("vu",      self._on_vu)
        BUS.subscribe("breathe", self._on_breathe)
        BUS.subscribe("stop",    self._on_stop)

    # ── Callbacks ──────────────────────────────────────────────────────────
    def _on_beat(self, **_):
        with self._lock:
            self._beat_t = time.perf_counter()

    def _on_freq(self, bands, **_):
        with self._lock:
            self._bands = [b / 255.0 for b in bands]
            self._vol   = max(bands)

    def _on_vu(self, val, **_):
        with self._lock:
            self._vol   = val
            n = int(val / 255.0 * len(self._bands))
            self._bands = [1.0 if i < n else 0.0
                           for i in range(len(self._bands))]

    def _on_breathe(self, bands, **_):
        with self._lock:
            self._bands = [b / 255.0 for b in bands]

    def _on_stop(self, **_):
        self._running = False

    # ── Render ─────────────────────────────────────────────────────────────
    def _frame(self) -> Layout:
        with self._lock:
            bands   = list(self._bands)
            vol     = self._vol
            elapsed = self._elapsed
            beat_on = (time.perf_counter() - self._beat_t) < 0.10

        # Smooth
        α = CFG.SMOOTH_ALPHA
        sm = self._smooth
        for i in range(len(bands)):
            sm[i] = α * sm[i] + (1 - α) * bands[i]
        self._smooth = sm

        a    = self._a
        cols = CFG.COLORS

        # ── Header ─────────────────────────────────────────────────────────
        beat_txt = Text("  ● BEAT ●  ", style="bold red") \
                   if beat_on else Text("            ")
        hdr = Table.grid(padding=(0, 2))
        hdr.add_column(style="bold cyan")
        hdr.add_column(style="white")
        hdr.add_column()
        hdr.add_row("🎵", a.name, beat_txt)
        hdr.add_row("BPM", f"{a.bpm:.1f}", "")

        # ── Progress ───────────────────────────────────────────────────────
        total   = a.duration
        pct     = min(elapsed / total, 1.0) if total else 0
        filled  = int(52 * pct)
        e_s = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        t_s = f"{int(total//60):02d}:{int(total%60):02d}"
        prog= Text(
            f"[{self.FULL*filled}{self.EMPTY*(52-filled)}] {e_s}/{t_s}",
            style="cyan")

        # ── Spectrogram bars ───────────────────────────────────────────────
        grid = Table.grid(padding=(0, 1))
        for _ in CFG.BANDS:
            grid.add_column(justify="center", min_width=7)

        # Label row
        grid.add_row(*[
            Text(b[3], style=f"bold {cols[i]}")
            for i, b in enumerate(CFG.BANDS)
        ])

        # Bar rows (top to bottom)
        for row in range(self.BAR_H, 0, -1):
            cells = []
            thresh = row / self.BAR_H
            for i, v in enumerate(sm):
                if v >= thresh:
                    cells.append(Text(self.FULL * 5,
                                      style=f"bold {cols[i]}"))
                else:
                    cells.append(Text(self.EMPTY * 5, style="dim"))
            grid.add_row(*cells)

        # Value row
        grid.add_row(*[
            Text(f"{int(v*255):3d}", style=cols[i])
            for i, v in enumerate(sm)
        ])

        # Hz row
        grid.add_row(*[
            Text(f"{b[0]:.0f}Hz", style="dim")
            for b in CFG.BANDS
        ])

        # ── Volume ─────────────────────────────────────────────────────────
        vp = vol / 255.0
        vf = int(52 * vp)
        vc = "green" if vp < 0.6 else "yellow" if vp < 0.85 else "red"
        vol_txt = Text(
            f"[{self.FULL*vf}{self.EMPTY*(52-vf)}]", style=vc)

        # ── Assemble ───────────────────────────────────────────────────────
        layout = Layout()
        layout.split_column(
            Layout(Panel(hdr,     border_style="cyan",  title="MP3 Beat Sync Pro"), size=5),
            Layout(Panel(prog,    border_style="blue",  title="Progress"),           size=3),
            Layout(Panel(grid,    border_style="green", title="🎚  Bands → LEDs"),  size=self.BAR_H + 5),
            Layout(Panel(vol_txt, border_style=vc,      title="Volume"),             size=3),
        )
        return layout

    # ── Background render loop ─────────────────────────────────────────────
    def start(self, clock_fn: Callable[[], float]):
        """clock_fn() → current playback time in seconds"""
        self._running = True
        hz = CFG.VIZ_HZ

        def _run():
            with Live(
                self._frame(),
                console=Console(),
                refresh_per_second=hz,
                screen=True,
            ) as live:
                while self._running:
                    with self._lock:
                        self._elapsed = clock_fn()
                    live.update(self._frame())
                    time.sleep(1.0 / hz)

        self._thread = threading.Thread(target=_run, daemon=True, name="Viz")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE PICKER
# ══════════════════════════════════════════════════════════════════════════════
def pick_file(hint: Optional[str] = None) -> Optional[str]:
    if hint and os.path.exists(hint):
        return hint

    mp3s = sorted(f for f in os.listdir(".") if f.lower().endswith(".mp3"))
    if mp3s:
        t = Table(title="MP3 files in current folder", border_style="blue")
        t.add_column("#",    style="cyan", width=4)
        t.add_column("File", style="white")
        t.add_column("MB",   style="dim",  width=8)
        for i, f in enumerate(mp3s, 1):
            t.add_row(str(i), f, f"{os.path.getsize(f)/1e6:.1f}")
        console.print(t)
        ch = Prompt.ask("Number or full path")
        if ch.isdigit():
            idx = int(ch) - 1
            if 0 <= idx < len(mp3s):
                return mp3s[idx]
        return ch if os.path.exists(ch) else None

    path = Prompt.ask("MP3 path").strip().strip('"').strip("'")
    return path if os.path.exists(path) else None


MODES = [
    "All Flash",
    "Chase + Onsets",
    "Alternate",
    "Frequency ★ (recommended)",
    "VU Meter",
    "Onset Sparkle",
]

def pick_mode(default: int = 3) -> int:
    t = Table(title="LED Modes", border_style="blue")
    t.add_column("#",    style="cyan", width=4)
    t.add_column("Mode", style="white")
    for i, m in enumerate(MODES):
        mark = " ←" if i == default else ""
        t.add_row(str(i), m + mark)
    console.print(t)
    ch = Prompt.ask("Mode", default=str(default))
    try:
        v = int(ch)
        return v if 0 <= v < len(MODES) else default
    except ValueError:
        return default


# ══════════════════════════════════════════════════════════════════════════════
#  CALIBRATION HELPER
# ══════════════════════════════════════════════════════════════════════════════
def calibrate():
    console.print(Panel(
        "[bold]Latency Calibration[/bold]\n\n"
        "Play a click track or any song with a sharp kick drum.\n"
        "Watch whether LEDs flash BEFORE or AFTER the beat.\n\n"
        f"Current offset: [cyan]{CFG.LATENCY_MS} ms[/cyan]\n\n"
        "[dim]Negative = LEDs fire earlier (compensates serial delay)\n"
        "Typical sweet-spot: -5 ms to -15 ms[/dim]",
        border_style="yellow",
    ))
    val = Prompt.ask("New LATENCY_MS", default=str(CFG.LATENCY_MS))
    try:
        CFG.LATENCY_MS = float(val)
        console.print(f"  ✅ Set to [cyan]{CFG.LATENCY_MS} ms[/cyan]")
    except ValueError:
        console.print("  ⚠️  Invalid – keeping current value")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── CLI args ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="MP3 Beat Sync Pro")
    parser.add_argument("mp3",         nargs="?",  help="MP3 file path")
    parser.add_argument("--mode",      type=int,   default=3,
                        help="LED mode 0-5 (default: 3 = Frequency)")
    parser.add_argument("--port",      default=None,
                        help="Serial port (e.g. COM3 / /dev/ttyUSB0)")
    parser.add_argument("--baud",      type=int,   default=CFG.BAUD)
    parser.add_argument("--latency",   type=float, default=CFG.LATENCY_MS,
                        help="Latency offset in ms (negative = fire early)")
    parser.add_argument("--no-viz",    action="store_true",
                        help="Disable terminal visualizer")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Ignore cached analysis")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive latency calibration")
    args = parser.parse_args()

    # Apply CLI overrides
    if args.port:    CFG.PORT       = args.port
    if args.baud:    CFG.BAUD       = args.baud
    if args.latency: CFG.LATENCY_MS = args.latency
    if args.no_viz:  CFG.VIZ        = False
    if args.no_cache and HAS_JOBLIB:
        # Monkey-patch cache loader to always return None
        SongAnalyzer._load_cache = lambda self, p: None   # type: ignore

    # ── Banner ──────────────────────────────────────────────────────────────
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel(
        Text.assemble(
            ("MP3 Beat Sync Pro\n",          "bold cyan"),
            ("Ultra-Accuracy Single-File Edition\n\n", "cyan"),
            ("PyAudio Callback  │  ", "dim"),
            ("Sample-Accurate Clock  │  ",   "dim"),
            ("500k Baud Serial\n",            "dim"),
            ("AGC Normalisation  │  ",        "dim"),
            ("BPM Breathing  │  ",            "dim"),
            ("Disk Cache  │  ",               "dim"),
            ("Rich UI",                       "dim"),
        ),
        title="🎵 LED Music Sync",
        border_style="cyan", padding=(1, 4),
    ))

    # ── Calibration mode ────────────────────────────────────────────────────
    if args.calibrate:
        calibrate()
        return

    # ── Arduino ─────────────────────────────────────────────────────────────
    console.rule("[bold cyan]① Arduino")
    arduino = Arduino()
    arduino.connect(args.port)

    # ── MP3 ─────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]② Music File")
    mp3 = pick_file(args.mp3)
    if not mp3 or not os.path.exists(mp3):
        console.print("  [red]❌ File not found![/red]")
        arduino.disconnect()
        return

    # ── Mode ────────────────────────────────────────────────────────────────
    console.rule("[bold cyan]③ LED Mode")
    mode = pick_mode(args.mode)

    # ── Latency ─────────────────────────────────────────────────────────────
    console.rule("[bold cyan]④ Latency")
    if Confirm.ask(
        f"  Adjust latency? (current: [cyan]{CFG.LATENCY_MS} ms[/cyan])",
        default=False
    ):
        calibrate()

    # ── Analyse ─────────────────────────────────────────────────────────────
    console.rule("[bold cyan]⑤ Analysis")
    try:
        song = SongAnalyzer().analyze(mp3)
    except Exception as e:
        console.print(f"  [red]❌ Analysis failed: {e}[/red]")
        import traceback; traceback.print_exc()
        arduino.disconnect()
        return

    events = TimelineBuilder().build(song, mode)

    # ── Countdown ───────────────────────────────────────────────────────────
    console.rule("[bold green]▶  Starting")
    for i in range(3, 0, -1):
        console.print(f"\r  [bold yellow]Starting in {i}...[/bold yellow]",
                      end="")
        time.sleep(1)
    console.print()

    # ── Visualizer ──────────────────────────────────────────────────────────
    engine = PlaybackEngine()
    viz    = None
    if CFG.VIZ:
        viz = Visualizer(song)

    # ── Play ────────────────────────────────────────────────────────────────
    def start_viz(clock_fn):
        if viz:
            viz.start(clock_fn)

    try:
        engine.play(song, events, mode, viz_start_fn=start_viz)
    except KeyboardInterrupt:
        console.print("\n  [yellow]⏹  Stopped[/yellow]")
    finally:
        engine.stop()   if hasattr(engine, "stop")  else None
        if viz: viz.stop()
        engine.close()
        arduino.disconnect()
        console.print("\n  [bold green]👋 Done![/bold green]\n")


if __name__ == "__main__":
    main()