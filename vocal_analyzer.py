"""
VocalIQ - Advanced Vocal Analysis & Coaching System



Analyzes pitch, vibrato, breath, belting, resonance, dynamics, and more.
Optional reference song comparison: melody accuracy, rhythm accuracy, note-by-note deviation.
Requires: pip install librosa sounddevice numpy scipy matplotlib rich
"""

import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.columns import Columns
from rich import box
from dataclasses import dataclass, field
from typing import Optional
import warnings
import time
import os
import sys
import json
import argparse
import hashlib
import tempfile
import shutil
import urllib.request

warnings.filterwarnings("ignore")

console = Console()

# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class VocalMetrics:
    # Core pitch
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_min: float = 0.0
    f0_max: float = 0.0
    f0_range_semitones: float = 0.0
    pitch_accuracy_pct: float = 0.0
    intonation_score: float = 0.0       # 0–100
    pitch_stability_score: float = 0.0  # 0–100

    # Vibrato
    vibrato_present: bool = False
    vibrato_rate_hz: float = 0.0        # ideal: 5–7 Hz
    vibrato_extent_semitones: float = 0.0  # ideal: 0.5–1 semitone
    vibrato_regularity: float = 0.0     # 0–100
    vibrato_onset_delay_s: float = 0.0  # how long after note start
    vibrato_score: float = 0.0          # 0–100

    # Breath & airflow
    breath_support_score: float = 0.0   # 0–100
    breath_pressure_consistency: float = 0.0
    phrase_lengths_s: list = field(default_factory=list)
    avg_phrase_length_s: float = 0.0
    breath_noise_ratio: float = 0.0     # breathiness indicator
    subglottal_pressure_estimate: float = 0.0

    # Resonance & tone
    resonance_score: float = 0.0        # 0–100
    singer_formant_strength: float = 0.0  # ~2500–3500 Hz cluster
    nasality_score: float = 0.0         # 0–100 (low = good generally)
    brightness_score: float = 0.0       # spectral centroid-based

    # Belting
    belting_detected: bool = False
    belting_chest_mix: float = 0.0      # ratio chest-like energy
    belting_efficiency_score: float = 0.0
    chest_voice_range: tuple = (0.0, 0.0)
    head_voice_range: tuple = (0.0, 0.0)
    mix_voice_detected: bool = False

    # Dynamics
    dynamic_range_db: float = 0.0
    rms_mean_db: float = 0.0
    rms_std_db: float = 0.0
    dynamic_control_score: float = 0.0  # 0–100
    crescendo_detected: bool = False
    decrescendo_detected: bool = False

    # Articulation & diction
    onset_sharpness: float = 0.0        # note attack crispness
    release_cleanness: float = 0.0
    spectral_flux_mean: float = 0.0

    # Register & passaggio
    passaggio_events: int = 0
    register_breaks: int = 0
    smoothness_through_break: float = 0.0

    # Rhythm & tempo
    note_duration_consistency: float = 0.0
    rhythmic_accuracy: float = 0.0

    # Reference song comparison (populated only when --reference is passed)
    ref_melody_accuracy: float = -1.0       # -1 = not computed; 0–100 otherwise
    ref_rhythm_accuracy: float = -1.0
    ref_pitch_deviation_cents: float = 0.0  # mean absolute cents from reference melody
    ref_missed_notes_pct: float = 0.0       # % of ref notes not matched
    ref_early_late_ms: float = 0.0          # mean timing offset (+ = late, – = early)
    ref_worst_moments: list = field(default_factory=list)  # [(time_s, desc), ...]

    # Overall
    overall_score: float = 0.0
    voiced_fraction: float = 0.0


@dataclass
class Tip:
    category: str
    severity: str   # "critical", "warning", "info", "praise"
    headline: str
    detail: str
    exercise: str


# ─────────────────────────────────────────────────────────────
# AUDIO RECORDING
# ─────────────────────────────────────────────────────────────

class AudioRecorder:
    def __init__(self, sr: int = 44100, channels: int = 1):
        self.sr = sr
        self.channels = channels

    def record(self, duration: float) -> np.ndarray:
        console.print(f"\n[bold red]● REC[/bold red] Recording for [bold]{duration}s[/bold]... sing now!\n")
        audio = sd.rec(int(duration * self.sr), samplerate=self.sr,
                       channels=self.channels, dtype='float32')
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("[cyan]Recording...", total=int(duration * 10))
            for _ in range(int(duration * 10)):
                time.sleep(0.1)
                progress.advance(task)
        sd.wait()
        console.print("[green]✓ Recording complete.[/green]\n")
        return audio.flatten()

    def list_devices(self):
        console.print(sd.query_devices())


# ─────────────────────────────────────────────────────────────
# PITCH ANALYSIS ENGINE
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# NOTE SEGMENTER  — foundation for per-note analysis
# ─────────────────────────────────────────────────────────────

class NoteSegmenter:
    """
    Groups voiced frames into individual sustained notes.
    A 'note' is a contiguous voiced region where the median pitch
    doesn't jump more than 1.5 semitones from frame to frame.
    Returns a list of Note dicts with rich per-note stats.
    """

    MIN_NOTE_FRAMES = 8     # ~47 ms at hop=256/44100 — shorter = noise
    MAX_PITCH_JUMP  = 1.5   # semitones between adjacent frames within one note

    def segment(self, f0: np.ndarray, times: np.ndarray,
                y: np.ndarray, sr: int, hop: int = 256) -> list[dict]:
        voiced = ~np.isnan(f0) & (f0 > 0)
        midi   = np.full(len(f0), np.nan)
        midi[voiced] = librosa.hz_to_midi(f0[voiced])

        notes = []
        i = 0
        while i < len(f0):
            if not voiced[i]:
                i += 1
                continue
            # start of a note
            j = i + 1
            while j < len(f0) and voiced[j]:
                if not np.isnan(midi[j]) and not np.isnan(midi[j-1]):
                    if abs(midi[j] - midi[j-1]) > self.MAX_PITCH_JUMP:
                        break
                j += 1
            if j - i >= self.MIN_NOTE_FRAMES:
                seg_midi  = midi[i:j]
                seg_f0    = f0[i:j]
                valid_seg = seg_midi[~np.isnan(seg_midi)]
                if len(valid_seg) >= self.MIN_NOTE_FRAMES:
                    t_start = float(times[i])
                    t_end   = float(times[min(j-1, len(times)-1)])
                    # intonation: deviation of segment median from nearest semitone
                    median_midi  = float(np.median(valid_seg))
                    target_midi  = round(median_midi)
                    cents_off    = (median_midi - target_midi) * 100
                    # stability: std of midi within note (excludes natural vibrato)
                    note_std     = float(np.std(valid_seg))
                    # slope: pitch drift within the note (cents/frame)
                    if len(valid_seg) > 4:
                        slope, *_ = stats.linregress(
                            np.arange(len(valid_seg)), valid_seg * 100
                        )
                    else:
                        slope = 0.0
                    notes.append({
                        "start_s":    t_start,
                        "end_s":      t_end,
                        "dur_s":      t_end - t_start,
                        "median_midi": median_midi,
                        "target_midi": target_midi,
                        "cents_off":  float(cents_off),
                        "note_std":   note_std,
                        "drift_slope": float(slope),  # positive = going sharp, negative = going flat
                        "note_name":  librosa.midi_to_note(int(round(median_midi))),
                        "frame_start": i,
                        "frame_end":   j,
                    })
            i = j
        return notes


class PitchAnalyzer:
    """
    Precision pitch analysis.
    Uses pYIN for frame-level f0, then NoteSegmenter for per-note stats.
    All scoring is done over sustained notes, not raw frames, which
    avoids penalising transitions and scoops that are stylistic.
    """

    def __init__(self, sr: int = 44100, fmin: float = 60.0, fmax: float = 1200.0):
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = 256
        self.segmenter  = NoteSegmenter()

    def extract(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns f0 (Hz), voiced_flag, times arrays."""
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=np.nan,
            beta_parameters=(2, 18),      # tighter prior on voicing — reduces false voiced
            boltzmann_parameter=2.0,
        )
        times = librosa.times_like(f0, sr=self.sr, hop_length=self.hop_length)
        return f0, voiced_flag, times

    def extract_notes(self, f0: np.ndarray, times: np.ndarray,
                      y: np.ndarray) -> list[dict]:
        return self.segmenter.segment(f0, times, y, self.sr, self.hop_length)

    def note_name(self, freq: float) -> str:
        if freq <= 0 or np.isnan(freq):
            return "unknown"
        midi  = librosa.hz_to_midi(freq)
        note  = librosa.midi_to_note(int(round(midi)))
        cents = (midi - round(midi)) * 100
        sign  = "+" if cents >= 0 else ""
        return f"{note} ({sign}{cents:.0f}c)"

    def semitone_range(self, f0: np.ndarray) -> float:
        valid = f0[~np.isnan(f0) & (f0 > 0)]
        if len(valid) < 2:
            return 0.0
        return float(12 * np.log2(valid.max() / valid.min()))

    def intonation_score(self, f0: np.ndarray, notes: list[dict] = None) -> float:
        """
        Per-note intonation: score the median pitch of each sustained note
        against the nearest semitone, weighted by note duration.
        This is far more accurate than per-frame scoring because it ignores
        the natural glide at note boundaries.
        """
        if notes:
            weighted_devs, weights = [], []
            for n in notes:
                if n["dur_s"] < 0.1:
                    continue
                weighted_devs.append(abs(n["cents_off"]))
                weights.append(n["dur_s"])
            if weighted_devs:
                avg_dev = float(np.average(weighted_devs, weights=weights))
                return float(np.clip(100 - (avg_dev / 40) * 100, 0, 100))

        # fallback: frame-level
        valid = f0[~np.isnan(f0) & (f0 > 0)]
        if len(valid) < 10:
            return 50.0
        midi = librosa.hz_to_midi(valid)
        cents_dev = np.abs((midi - np.round(midi)) * 100)
        return float(np.clip(100 - (cents_dev.mean() / 40) * 100, 0, 100))

    def pitch_stability_score(self, f0: np.ndarray, notes: list[dict] = None) -> float:
        """
        Per-note stability: standard deviation of pitch within each sustained note.
        A note is stable if it stays within ~20 cents throughout.
        Vibrato is excluded by measuring std within 200ms windows.
        """
        if notes:
            stds, weights = [], []
            for n in notes:
                if n["dur_s"] < 0.15:
                    continue
                stds.append(n["note_std"] * 100)   # semitones → cents
                weights.append(n["dur_s"])
            if stds:
                avg_std = float(np.average(stds, weights=weights))
                return float(np.clip(100 - (avg_std / 30) * 100, 0, 100))

        valid = f0[~np.isnan(f0) & (f0 > 0)]
        if len(valid) < 10:
            return 50.0
        diffs = np.diff(librosa.hz_to_midi(valid))
        jitter = np.abs(diffs).mean()
        return float(np.clip(100 - jitter * 25, 0, 100))

    def flat_sharp_bias(self, notes: list[dict]) -> str:
        """Detect if the voice consistently leans flat, sharp, or neither."""
        if not notes:
            return "neutral"
        offsets = [n["cents_off"] for n in notes if n["dur_s"] > 0.2]
        if not offsets:
            return "neutral"
        mean_off = float(np.mean(offsets))
        if mean_off < -12:
            return f"flat by ~{abs(mean_off):.0f} cents on average"
        if mean_off > 12:
            return f"sharp by ~{mean_off:.0f} cents on average"
        return "centred"

    def find_problem_notes(self, notes: list[dict], threshold_cents: float = 35.0) -> list[dict]:
        """Return notes with the worst intonation, sorted by severity."""
        bad = [n for n in notes if abs(n["cents_off"]) >= threshold_cents and n["dur_s"] > 0.15]
        return sorted(bad, key=lambda n: abs(n["cents_off"]), reverse=True)[:6]


# ─────────────────────────────────────────────────────────────
# VIBRATO ANALYZER
# ─────────────────────────────────────────────────────────────

class VibratoAnalyzer:
    """
    Accurate vibrato analysis using three independent estimators that must agree.

    Rate: estimated via (1) autocorrelation peak, (2) zero-crossing rate on
    the bandpassed signal, and (3) FFT peak — median of the three is used.

    Extent: measured as mean peak-to-peak amplitude of the Hilbert envelope
    (not RMS, which underestimates by ~30%).

    Naturalness: checks that the vibrato is present on long notes but NOT on
    short notes, which distinguishes real vibrato from tremolo or wobble.
    """
    RATE_MIN = 4.0
    RATE_MAX = 9.5

    def analyze(self, f0: np.ndarray, times: np.ndarray, sr: int,
                hop: int = 256, notes: list = None) -> dict:
        valid_mask = ~np.isnan(f0) & (f0 > 0)
        if valid_mask.sum() < 60:
            return self._empty()

        idx       = np.arange(len(f0))
        valid_idx = idx[valid_mask]
        f0_interp = np.interp(idx, valid_idx, f0[valid_mask])

        median_f0 = np.median(f0_interp[valid_mask])
        f0_cents  = 1200 * np.log2(np.clip(f0_interp / (median_f0 + 1e-9), 1e-6, None))

        frame_rate = sr / hop

        # Bandpass 4–9.5 Hz
        nyq = frame_rate / 2
        b, a = signal.butter(4, [self.RATE_MIN / nyq, self.RATE_MAX / nyq], btype='band')
        f0_bp = signal.filtfilt(b, a, f0_cents)

        # ── Estimator 1: autocorrelation ──────────────────────
        ac       = np.correlate(f0_bp, f0_bp, mode='full')[len(f0_bp)-1:]
        ac      /= ac[0] + 1e-9
        lo_lag   = int(frame_rate / self.RATE_MAX)
        hi_lag   = int(frame_rate / self.RATE_MIN)
        if hi_lag >= len(ac):
            return self._empty()
        ac_peak  = np.argmax(ac[lo_lag:hi_lag]) + lo_lag
        rate_ac  = frame_rate / ac_peak if ac_peak > 0 else 0.0

        # ── Estimator 2: zero-crossing rate ───────────────────
        zc       = np.where(np.diff(np.sign(f0_bp)))[0]
        if len(zc) > 2:
            zc_rate = (len(zc) / 2) / (len(f0_bp) / frame_rate)
        else:
            zc_rate = 0.0

        # ── Estimator 3: FFT peak ─────────────────────────────
        fft_vals = np.abs(np.fft.rfft(f0_bp, n=len(f0_bp) * 4))
        fft_freq = np.fft.rfftfreq(len(f0_bp) * 4, d=1.0 / frame_rate)
        fft_mask = (fft_freq >= self.RATE_MIN) & (fft_freq <= self.RATE_MAX)
        if fft_mask.any():
            fft_rate = float(fft_freq[fft_mask][np.argmax(fft_vals[fft_mask])])
        else:
            fft_rate = 0.0

        valid_rates = [r for r in [rate_ac, zc_rate, fft_rate]
                       if self.RATE_MIN <= r <= self.RATE_MAX]
        if not valid_rates:
            return self._empty()
        vibrato_rate = float(np.median(valid_rates))

        # ── Extent: peak-to-peak via Hilbert ──────────────────
        analytic     = signal.hilbert(f0_bp)
        env          = np.abs(analytic)
        # peak-to-peak = 2 * mean amplitude; convert cents → semitones
        extent_st    = float(env.mean() * 2 / 100)

        # ── Regularity: coefficient of variation of envelope ──
        regularity   = float(np.clip(
            100 - (env.std() / (env.mean() + 1e-9)) * 80, 0, 100
        ))

        # ── Presence: must exceed meaningful amplitude ─────────
        present = (
            self.RATE_MIN <= vibrato_rate <= self.RATE_MAX
            and extent_st > 0.12
            and env.mean() > 5.0          # at least 5 cents of oscillation
        )

        # ── Naturalness: vibrato on long notes, less on short ──
        naturalness = 100.0
        if notes:
            long_notes  = [n for n in notes if n["dur_s"] >= 0.6]
            short_notes = [n for n in notes if n["dur_s"] < 0.3]
            if long_notes and short_notes:
                long_std  = np.mean([n["note_std"] for n in long_notes])
                short_std = np.mean([n["note_std"] for n in short_notes])
                # Natural vibrato: long notes much more oscillatory than short
                ratio = long_std / (short_std + 1e-9)
                naturalness = float(np.clip(ratio * 40, 0, 100))

        onset_delay = self._detect_onset(env, frame_rate)
        score       = self._compute_score(vibrato_rate, extent_st, regularity, naturalness, present)

        return {
            "present":            present,
            "rate_hz":            vibrato_rate,
            "rate_ac":            rate_ac,
            "rate_zc":            zc_rate,
            "rate_fft":           fft_rate,
            "extent_semitones":   extent_st,
            "regularity":         regularity,
            "naturalness":        naturalness,
            "onset_delay_s":      onset_delay,
            "score":              score,
            "f0_cents":           f0_cents,
            "f0_bp":              f0_bp,
        }

    def _detect_onset(self, env: np.ndarray, frame_rate: float) -> float:
        threshold = env.max() * 0.25
        above     = env > threshold
        onset     = np.argmax(above) if above.any() else len(env)
        return float(onset / frame_rate)

    def _compute_score(self, rate, extent, regularity, naturalness, present) -> float:
        if not present:
            return 0.0
        rate_score   = 100 - min(abs(rate - 6.0) / 1.5, 1.0) * 55
        extent_score = 100 - min(abs(extent - 0.65) / 0.35, 1.0) * 55
        return float(np.clip(
            rate_score * 0.30 + extent_score * 0.30 + regularity * 0.20 + naturalness * 0.20,
            0, 100
        ))

    def _empty(self) -> dict:
        return {"present": False, "rate_hz": 0.0, "rate_ac": 0.0, "rate_zc": 0.0,
                "rate_fft": 0.0, "extent_semitones": 0.0, "regularity": 0.0,
                "naturalness": 100.0, "onset_delay_s": 0.0, "score": 0.0,
                "f0_cents": np.array([]), "f0_bp": np.array([])}


# ─────────────────────────────────────────────────────────────
# BREATH & AIRFLOW ANALYZER
# ─────────────────────────────────────────────────────────────

def detect_phrases(voiced_mask: np.ndarray, hop_time: float,
                   max_gap_s: float = 0.30, min_phrase_s: float = 0.25) -> list[tuple[float, float]]:
    """
    Group voiced frames into sung phrases, returned as (start_s, end_s).
    Short unvoiced gaps (consonants, quick glottal stops) are bridged so a phrase
    only ends where the singer actually pauses, usually to breathe.
    """
    runs = []
    in_run, start = False, 0
    for i, v in enumerate(voiced_mask):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            runs.append([start, i])
    if in_run:
        runs.append([start, len(voiced_mask)])

    max_gap = int(round(max_gap_s / hop_time))
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= max_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a * hop_time, b * hop_time) for a, b in merged if (b - a) * hop_time >= min_phrase_s]


class BreathAnalyzer:
    """
    Detects breath noise, phrase length, pressure consistency.
    Breathiness estimated via HNR (Harmonic-to-Noise Ratio proxy).
    """

    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        # Segment into voiced / unvoiced / silent
        rms = librosa.feature.rms(y=y, hop_length=256)[0]
        voiced_mask = ~np.isnan(f0)

        # Phrase detection: voiced regions, bridging consonant-length gaps
        hop_time = 256 / sr
        phrases = detect_phrases(voiced_mask, hop_time)
        phrase_lengths = [b - a for a, b in phrases]

        # Breathiness: ratio of noise floor between harmonics
        hnr = self._estimate_hnr(y, sr)
        breath_noise_ratio = float(np.clip(1.0 - hnr / 30.0, 0.0, 1.0))

        # Support score: consistent RMS during voiced frames
        voiced_rms = rms[voiced_mask[:len(rms)]] if voiced_mask[:len(rms)].sum() > 5 else rms
        rms_cv = voiced_rms.std() / (voiced_rms.mean() + 1e-9)
        support_score = float(np.clip(100 - rms_cv * 200, 0, 100))

        # Subglottal pressure proxy: very rough – relates to RMS in the lower partials
        pressure_estimate = float(np.clip(voiced_rms.mean() * 1000, 0, 100))

        avg_phrase = float(np.mean(phrase_lengths)) if phrase_lengths else 0.0

        return {
            "support_score": support_score,
            "pressure_consistency": float(np.clip(100 - rms_cv * 150, 0, 100)),
            "phrase_lengths_s": phrase_lengths,
            "phrases": phrases,
            "avg_phrase_length_s": avg_phrase,
            "breath_noise_ratio": breath_noise_ratio,
            "subglottal_pressure_estimate": pressure_estimate,
        }

    def _estimate_hnr(self, y: np.ndarray, sr: int, frame_len: int = 2048) -> float:
        """Rough HNR estimate via cepstrum."""
        frame = y[:frame_len] if len(y) >= frame_len else y
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        cepstrum = np.abs(np.fft.irfft(np.log(spectrum + 1e-9)))

        min_lag = int(sr / 500)
        max_lag = int(sr / 60)
        if max_lag >= len(cepstrum):
            return 15.0

        peak = cepstrum[min_lag:max_lag].max()
        noise_floor = np.percentile(cepstrum[min_lag:max_lag], 20)
        hnr_db = 10 * np.log10((peak + 1e-9) / (noise_floor + 1e-9))
        return float(np.clip(hnr_db, 0, 40))


# ─────────────────────────────────────────────────────────────
# RESONANCE & TONE ANALYZER
# ─────────────────────────────────────────────────────────────

class ResonanceAnalyzer:
    """
    Analyzes formants, singer's formant cluster, nasality, brightness.
    Singer's formant: strong energy concentration around 2500–3500 Hz.
    """
    SINGER_FORMANT_LOW = 2500
    SINGER_FORMANT_HIGH = 3500

    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        # STFT for spectral analysis
        D = librosa.stft(y, n_fft=4096, hop_length=256)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

        # Singer's formant strength
        sf_mask = (freqs >= self.SINGER_FORMANT_LOW) & (freqs <= self.SINGER_FORMANT_HIGH)
        ref_mask = (freqs >= 500) & (freqs <= 2000)
        sf_energy = S[sf_mask].mean()
        ref_energy = S[ref_mask].mean()
        sf_ratio = sf_energy / (ref_energy + 1e-9)
        sf_score = float(np.clip(sf_ratio * 300, 0, 100))

        # Spectral centroid → brightness
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
        brightness = float(np.clip((centroid.mean() - 500) / 2000 * 100, 0, 100))

        # Nasality proxy: energy in 800–1800 Hz anti-formant region
        nasal_mask = (freqs >= 800) & (freqs <= 1200)
        nasal_energy = S[nasal_mask].mean()
        nasality = float(np.clip(nasal_energy / (ref_energy + 1e-9) * 150, 0, 100))

        # Overall resonance score
        resonance_score = float(np.clip(sf_score * 0.5 + brightness * 0.3 + (100 - nasality) * 0.2, 0, 100))

        return {
            "resonance_score": resonance_score,
            "singer_formant_strength": sf_score,
            "nasality_score": nasality,
            "brightness_score": brightness,
        }


# ─────────────────────────────────────────────────────────────
# BELTING & REGISTER ANALYZER
# ─────────────────────────────────────────────────────────────

class BeltingAnalyzer:
    """
    Identifies chest, mix, and head voice registers.
    Belting: sustained high intensity above passaggio with chest-like spectral tilt.
    """
    # Typical female passaggio ~E4–G4, male ~D3–F3 (approximate MIDI)
    FEMALE_PASSAGGIO_MIDI = (64, 67)
    MALE_PASSAGGIO_MIDI = (50, 53)

    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray, times: np.ndarray) -> dict:
        D = librosa.stft(y, n_fft=2048, hop_length=256)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        valid = ~np.isnan(f0)
        if valid.sum() < 10:
            return self._empty()

        f0_valid = f0[valid]
        midi_valid = librosa.hz_to_midi(f0_valid)

        # Detect voice type (crude: median f0)
        median_midi = np.median(midi_valid)
        is_female = median_midi > 56  # roughly tenor/soprano boundary

        passaggio = self.FEMALE_PASSAGGIO_MIDI if is_female else self.MALE_PASSAGGIO_MIDI
        above_passaggio = (midi_valid > passaggio[1]).mean()

        # Spectral tilt: chest voice has more low-frequency energy
        low_mask = freqs < 800
        high_mask = freqs > 800
        spectral_tilt = S[low_mask].mean() / (S[high_mask].mean() + 1e-9)
        # High tilt = chest-heavy; low tilt = heady
        chest_mix = float(np.clip(spectral_tilt / 3.0 * 100, 0, 100))

        # RMS above passaggio
        rms = librosa.feature.rms(y=y, hop_length=256)[0]
        rms_mean = rms.mean()

        belting = (
            above_passaggio > 0.3
            and chest_mix > 50
            and rms_mean > 0.03
        )

        efficiency = float(np.clip(chest_mix * 0.4 + (100 - above_passaggio * 100) * 0.3 + 50, 0, 100))

        # Register ranges
        chest_idx = midi_valid < passaggio[0]
        head_idx = midi_valid > passaggio[1]
        chest_range = (float(midi_valid[chest_idx].min()) if chest_idx.any() else 0,
                       float(midi_valid[chest_idx].max()) if chest_idx.any() else 0)
        head_range = (float(midi_valid[head_idx].min()) if head_idx.any() else 0,
                      float(midi_valid[head_idx].max()) if head_idx.any() else 0)

        mix_detected = (midi_valid > passaggio[0]).any() and (midi_valid < passaggio[1] + 3).any()

        passaggio_crossings = int(np.sum(np.diff((midi_valid > passaggio[1]).astype(int)) != 0))
        register_breaks = self._detect_breaks(f0, sr)

        return {
            "belting_detected": belting,
            "belting_chest_mix": chest_mix,
            "belting_efficiency_score": efficiency,
            "chest_voice_range": chest_range,
            "head_voice_range": head_range,
            "mix_voice_detected": bool(mix_detected),
            "passaggio_events": passaggio_crossings,
            "register_breaks": register_breaks,
            "smoothness_through_break": float(np.clip(100 - register_breaks * 20, 0, 100)),
            "is_female_voice": is_female,
        }

    def _detect_breaks(self, f0: np.ndarray, sr: int, hop: int = 256) -> int:
        """Count sudden large jumps in f0 that indicate register breaks."""
        valid = ~np.isnan(f0)
        midi = np.full(len(f0), np.nan)
        midi[valid] = librosa.hz_to_midi(f0[valid])
        diffs = np.abs(np.diff(midi))
        diffs_clean = diffs[~np.isnan(diffs)]
        breaks = int((diffs_clean > 5).sum())  # >5 semitone jump = break
        return breaks

    def _empty(self):
        return {"belting_detected": False, "belting_chest_mix": 0.0,
                "belting_efficiency_score": 0.0, "chest_voice_range": (0, 0),
                "head_voice_range": (0, 0), "mix_voice_detected": False,
                "passaggio_events": 0, "register_breaks": 0,
                "smoothness_through_break": 100.0, "is_female_voice": True}


# ─────────────────────────────────────────────────────────────
# DYNAMICS ANALYZER
# ─────────────────────────────────────────────────────────────

class DynamicsAnalyzer:
    def analyze(self, y: np.ndarray, sr: int) -> dict:
        rms = librosa.feature.rms(y=y, hop_length=256)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-9)

        dynamic_range = float(rms_db.max() - rms_db.min())
        rms_mean_db = float(rms_db.mean())
        rms_std_db = float(rms_db.std())

        # Detect crescendo/decrescendo
        n = len(rms_db)
        first_half = rms_db[:n // 2].mean()
        second_half = rms_db[n // 2:].mean()
        crescendo = bool(second_half - first_half > 3)
        decrescendo = bool(first_half - second_half > 3)

        # Control score: reward range, penalize excessive spikiness
        spike_penalty = np.abs(np.diff(rms_db)).mean()
        control = float(np.clip(
            min(dynamic_range, 40) / 40 * 70 + (1 - spike_penalty / 20) * 30,
            0, 100
        ))

        return {
            "dynamic_range_db": dynamic_range,
            "rms_mean_db": rms_mean_db,
            "rms_std_db": rms_std_db,
            "dynamic_control_score": control,
            "crescendo_detected": crescendo,
            "decrescendo_detected": decrescendo,
        }


# ─────────────────────────────────────────────────────────────
# ARTICULATION ANALYZER
# ─────────────────────────────────────────────────────────────

class ArticulationAnalyzer:
    def analyze(self, y: np.ndarray, sr: int) -> dict:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256)
        if len(onset_frames) < 2:
            return {"onset_sharpness": 50.0, "release_cleanness": 50.0,
                    "spectral_flux_mean": 0.0, "note_duration_consistency": 50.0}

        # Onset sharpness: energy rise rate around detected onsets
        rms = librosa.feature.rms(y=y, hop_length=256)[0]
        sharpness_vals = []
        for f in onset_frames:
            if f + 5 < len(rms):
                rise = rms[f + 3] - rms[max(f - 2, 0)]
                sharpness_vals.append(rise)

        onset_sharpness = float(np.clip(np.mean(sharpness_vals) * 1000, 0, 100)) if sharpness_vals else 50.0

        # Spectral flux
        S = np.abs(librosa.stft(y, hop_length=256))
        flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)).mean()
        sf_mean = float(flux)

        # Note duration consistency
        durations = np.diff(onset_frames)
        dur_cv = durations.std() / (durations.mean() + 1e-9) if len(durations) > 1 else 1.0
        consistency = float(np.clip(100 - dur_cv * 50, 0, 100))

        return {
            "onset_sharpness": onset_sharpness,
            "release_cleanness": 70.0,  # placeholder – would need offset detection
            "spectral_flux_mean": sf_mean,
            "note_duration_consistency": consistency,
        }


# ─────────────────────────────────────────────────────────────
# REFERENCE SONG ANALYZER
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SONG FETCHER  — search and download any online song
# ─────────────────────────────────────────────────────────────

SONGS_CACHE_DIR = os.path.expanduser("~/.vocaliq/songs")


class SongFetcher:
    """
    Lets the user pick any song from YouTube by name or URL
    and downloads its audio for use as a reference track.

    Requires: pip install yt-dlp

    Songs are cached in ~/.vocaliq/songs/ by a hash of the query/URL
    so the same song is never re-downloaded.

    Usage:
        fetcher = SongFetcher()
        path = fetcher.get("Shape of You Ed Sheeran")   # search by name
        path = fetcher.get("https://youtu.be/JGwWNGJdvx8")  # direct URL
    """

    def __init__(self):
        os.makedirs(SONGS_CACHE_DIR, exist_ok=True)
        self._check_ytdlp()

    # ── public API ──────────────────────────────────────────

    def search_and_pick(self, query: str) -> Optional[str]:
        """
        Search YouTube for `query`, show top 5 results, let the user pick one,
        confirm it is actually the song they were singing, then download it.
        Loops back to the search if the user says it is the wrong version.
        Returns None if the user cancels at any point.
        """
        while True:
            console.print(f"\n[cyan]Searching for:[/cyan] [bold]{query}[/bold]\n")
            results = self._search(query, max_results=5)
            if not results:
                console.print("[red]No results found. Check your internet connection.[/red]")
                return None

            self._print_results(results)
            choice = self._ask_choice(len(results))
            if choice is None:
                return None

            picked = results[choice]

            # Confirm this is actually the song the user sang
            confirmed = self._confirm_song(picked)
            if confirmed is True:
                return self._download(picked["url"], picked["title"])
            elif confirmed == "retry":
                # User wants to search again with a new query
                new_query = input("\n  Search again with a different term: ").strip()
                if not new_query:
                    return None
                query = new_query
                continue
            elif confirmed == "pick_again":
                # Show the same results again so they can choose differently
                continue
            else:
                # cancelled
                return None

    def get(self, url_or_query: str) -> Optional[str]:
        """
        If `url_or_query` looks like a URL, fetch its details, confirm with the user,
        then download. Otherwise search for it interactively.
        """
        if url_or_query.startswith("http://") or url_or_query.startswith("https://"):
            console.print(f"\n[cyan]Fetching song info from URL...[/cyan]")
            info = self._info(url_or_query)
            if not info:
                console.print("[red]Could not fetch song info from that URL.[/red]")
                return None

            entry = {
                "title":    info.get("title", "Unknown title"),
                "uploader": info.get("uploader", "Unknown artist"),
                "url":      url_or_query,
                "duration": self._fmt_duration(info.get("duration", 0)),
                "views":    self._fmt_views(info.get("view_count", 0)),
                "description": (info.get("description") or "")[:300],
            }
            confirmed = self._confirm_song(entry)
            if confirmed is True:
                return self._download(url_or_query, entry["title"])
            elif confirmed in ("retry", "pick_again"):
                query = input("\n  Search for the correct song instead: ").strip()
                return self.search_and_pick(query) if query else None
            return None

        return self.search_and_pick(url_or_query)

    # ── search ──────────────────────────────────────────────

    def _search(self, query: str, max_results: int = 5) -> list[dict]:
        try:
            import yt_dlp
        except ImportError:
            console.print("[red]yt-dlp not installed. Run: pip install yt-dlp[/red]")
            return []

        ydl_opts = {
            "quiet":         True,
            "no_warnings":   True,
            "extract_flat":  True,
            "default_search": f"ytsearch{max_results}",
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(query, download=False)
            except Exception as e:
                console.print(f"[red]Search failed: {e}[/red]")
                return []

        entries = info.get("entries", []) if info else []
        results = []
        for entry in entries[:max_results]:
            if not entry:
                continue
            duration = entry.get("duration", 0)
            results.append({
                "title":       entry.get("title", "Unknown"),
                "url":         entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                "uploader":    entry.get("uploader", ""),
                "duration":    self._fmt_duration(duration),
                "views":       self._fmt_views(entry.get("view_count", 0)),
                "description": (entry.get("description") or "")[:300],
            })
        return results

    def _info(self, url: str) -> Optional[dict]:
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception:
            return None

    # ── display ─────────────────────────────────────────────

    def _print_results(self, results: list[dict]):
        t = Table(box=box.SIMPLE_HEAD, show_header=True,
                  header_style="bold cyan", border_style="bright_black")
        t.add_column("#",        justify="right", min_width=3,  style="bold cyan")
        t.add_column("Title",    min_width=42,    style="white")
        t.add_column("Channel",  min_width=18,    style="dim")
        t.add_column("Duration", justify="right", min_width=8,  style="yellow")
        t.add_column("Views",    justify="right", min_width=8,  style="dim")
        for i, r in enumerate(results, 1):
            t.add_row(str(i), r["title"], r["uploader"], r["duration"], r["views"])
        console.print(t)

    def _ask_choice(self, n: int) -> Optional[int]:
        console.print(
            f"[bold white]Pick a song (1 to {n}), or press Enter to cancel:[/bold white]"
        )
        raw = input("  > ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        console.print("[yellow]Invalid choice.[/yellow]")
        return None

    def _confirm_song(self, entry: dict) -> "bool | str":
        """
        Show the user exactly what song was selected and ask them to confirm
        it is the version they were actually singing.

        Returns:
          True        — confirmed, proceed with download
          "pick_again"— show the same result list again
          "retry"     — search with a new query
          False       — cancel entirely
        """
        console.print()
        console.print(Panel(
            f"[bold white]{entry['title']}[/bold white]\n"
            f"[dim]Channel / Artist:[/dim]  {entry.get('uploader', 'Unknown')}\n"
            f"[dim]Duration:[/dim]          {entry.get('duration', '?')}\n"
            f"[dim]Views:[/dim]             {entry.get('views', '?')}\n"
            + (
                f"\n[dim]Description:[/dim]\n"
                f"[dim italic]{entry['description'][:280]}[/dim italic]"
                if entry.get("description") else ""
            ),
            title="[bold cyan]Is this the song you were singing?[/bold cyan]",
            border_style="cyan",
        ))
        console.print(
            "  [bold cyan]Y[/bold cyan]  Yes, that is it\n"
            "  [bold yellow]P[/bold yellow]  No, let me pick a different one from the list\n"
            "  [bold yellow]S[/bold yellow]  No, let me search with different words\n"
            "  [bold red]C[/bold red]  Cancel  (just analyse my voice without a song)\n"
        )
        while True:
            raw = input("  Y / P / S / C: ").strip().lower()
            if raw in ("y", "yes", ""):
                return True
            if raw in ("p", "pick"):
                return "pick_again"
            if raw in ("s", "search"):
                return "retry"
            if raw in ("c", "cancel", "n", "no"):
                return False
            console.print("  [yellow]Just type Y, P, S, or C.[/yellow]")

    # ── download ────────────────────────────────────────────

    def _download(self, url: str, title: str) -> Optional[str]:
        """Download audio to the cache and return the file path."""
        cache_key  = hashlib.md5(url.encode()).hexdigest()[:12]
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60]
        dest_path  = os.path.join(SONGS_CACHE_DIR, f"{safe_title}_{cache_key}.mp3")

        if os.path.exists(dest_path):
            console.print(f"[green]Using cached audio:[/green] {os.path.basename(dest_path)}")
            return dest_path

        try:
            import yt_dlp
        except ImportError:
            console.print("[red]yt-dlp not installed. Run: pip install yt-dlp[/red]")
            return None

        console.print(f"\n[cyan]Downloading audio:[/cyan] {title}")

        tmp_dir  = tempfile.mkdtemp()
        tmp_base = os.path.join(tmp_dir, "audio")

        ydl_opts = {
            "format":            "bestaudio/best",
            "outtmpl":           tmp_base + ".%(ext)s",
            "quiet":             True,
            "no_warnings":       True,
            "postprocessors": [{
                "key":            "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "progress_hooks": [self._progress_hook],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            console.print(f"[red]Download failed: {e}[/red]")
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        tmp_mp3 = tmp_base + ".mp3"
        if not os.path.exists(tmp_mp3):
            # yt-dlp may have named it differently
            for f in os.listdir(tmp_dir):
                if f.endswith(".mp3"):
                    tmp_mp3 = os.path.join(tmp_dir, f)
                    break
            else:
                console.print("[red]Downloaded file not found.[/red]")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None

        shutil.move(tmp_mp3, dest_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        console.print(f"[green]Downloaded and saved:[/green] {os.path.basename(dest_path)}")
        return dest_path

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _progress_hook(d: dict):
        if d["status"] == "downloading":
            pct = d.get("_percent_str", "").strip()
            speed = d.get("_speed_str", "").strip()
            console.print(f"  [dim]{pct}  {speed}[/dim]", end="\r")
        elif d["status"] == "finished":
            console.print("  [green]Download complete. Converting to MP3...[/green]")

    @staticmethod
    def _fmt_duration(seconds) -> str:
        if not seconds:
            return "?"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    @staticmethod
    def _fmt_views(n) -> str:
        if not n:
            return ""
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        if n >= 1_000:
            return f"{n/1_000:.0f}K"
        return str(n)

    @staticmethod
    def _check_ytdlp():
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            console.print(
                "[yellow]Note: yt-dlp is not installed. Song search and download will not work.\n"
                "Install it with: pip install yt-dlp[/yellow]"
            )

    # ── cache management ────────────────────────────────────

    @staticmethod
    def list_cached() -> list[dict]:
        if not os.path.exists(SONGS_CACHE_DIR):
            return []
        files = [f for f in os.listdir(SONGS_CACHE_DIR) if f.endswith(".mp3")]
        result = []
        for f in sorted(files):
            path  = os.path.join(SONGS_CACHE_DIR, f)
            size  = os.path.getsize(path)
            mtime = os.path.getmtime(path)
            result.append({
                "filename": f,
                "path":     path,
                "size_mb":  round(size / 1e6, 1),
                "date":     time.strftime("%Y-%m-%d", time.localtime(mtime)),
            })
        return result

    @staticmethod
    def clear_cache():
        if os.path.exists(SONGS_CACHE_DIR):
            shutil.rmtree(SONGS_CACHE_DIR)
            os.makedirs(SONGS_CACHE_DIR, exist_ok=True)
            console.print("[green]Song cache cleared.[/green]")


class ReferenceSongAnalyzer:
    """
    Compares the sung vocal against a reference recording of the song.

    Pipeline:
      1. Isolate the melody in the reference (harmonic–percussive separation plus
         a low-frequency cut; a lightweight stand-in for a neural vocal splitter).
      2. Extract f0 from both at 22.05 kHz. Reference contours are cached on disk
         by file hash, so repeat practice on the same song is fast.
      3. Subsequence-DTW on a pitch-class cost finds *where* in the song the
         singer's take belongs and aligns tempo, even if they sang only a verse
         or sang an octave away from the original.
      4. Score note accuracy, missed notes and onset timing over the matched part,
         and locate the worst moments on the singer's own timeline.
    """

    SR = 22050
    HOP = 512
    DTW_FPS = 10            # alignment resolution (frames per second)
    MAX_REF_S = 420         # analyse at most 7 minutes of reference audio
    HIT_CENTS = 50          # within a quarter tone counts as "on the note"
    BAD_CENTS = 80

    def __init__(self, sr: int = 44100, cache_dir: Optional[str] = None):
        self.sr = sr
        self.cache_dir = cache_dir

    # ── public entry point ──────────────────────────────────

    def analyze(self, y_vocal: np.ndarray, ref_path: str, sr_vocal: Optional[int] = None) -> dict:
        sr_vocal = sr_vocal or self.sr
        y_voc = y_vocal if sr_vocal == self.SR else librosa.resample(y_vocal, orig_sr=sr_vocal, target_sr=self.SR)

        console.print(f"[cyan]Loading reference:[/cyan] {ref_path}")
        f0_ref, onsets_ref = self._reference_features(ref_path)
        f0_voc = self._extract_f0(y_voc)

        midi_ref = self._to_midi(f0_ref)
        midi_voc = self._to_midi(f0_voc)
        fps = self.SR / self.HOP

        console.print("[cyan]Finding your part of the song...[/cyan]")
        ref_idx, octave_shift = self._align(midi_voc, midi_ref)
        ref_on_voc = midi_ref[ref_idx] + octave_shift        # reference melody on the singer's timeline

        voc_voiced = ~np.isnan(midi_voc)
        ref_voiced = ~np.isnan(ref_on_voc)
        both = voc_voiced & ref_voiced
        cent_dev = np.full(len(midi_voc), np.nan)
        cent_dev[both] = (midi_voc[both] - ref_on_voc[both]) * 100

        if both.sum() == 0:
            melody_accuracy, mean_dev = 0.0, 0.0
        else:
            abs_dev = np.abs(cent_dev[both])
            melody_accuracy = float((abs_dev < self.HIT_CENTS).mean() * 100)
            mean_dev = float(abs_dev.mean())
        missed_pct = float((ref_voiced & ~voc_voiced).sum() / max(ref_voiced.sum(), 1) * 100)

        ref_start_s = float(ref_idx[0] / fps)
        ref_end_s = float(ref_idx[-1] / fps)
        rhythm = self._rhythm_accuracy(y_voc, onsets_ref, ref_idx, fps)
        spans = self._worst_moments(cent_dev, ref_on_voc, fps)

        # Compact contours for charting (≈ 20 points per second)
        step = max(1, int(round(fps / 20)))
        t = np.arange(len(midi_voc))[::step] / fps
        return {
            "melody_accuracy": melody_accuracy,
            "pitch_deviation_cents": mean_dev,
            "missed_notes_pct": missed_pct,
            "early_late_ms": rhythm["early_late_ms"],
            "rhythm_accuracy": rhythm["rhythm_accuracy"],
            "worst_moments": [(t, desc) for t, desc, _ in spans],       # [(time_s, description)]
            "worst_spans": [(t, end, desc) for t, desc, end in spans],  # [(start_s, end_s, description)]
            "ref_start_s": ref_start_s,
            "ref_end_s": ref_end_s,
            "octave_shift": int(octave_shift // 12),
            "contour": {
                "t": t.round(3).tolist(),
                "voice": _nan_to_none(midi_voc[::step]),
                "reference": _nan_to_none(ref_on_voc[::step]),
            },
        }

    # ── reference features (cached) ─────────────────────────

    def _reference_features(self, ref_path: str) -> tuple[np.ndarray, np.ndarray]:
        cache_file = None
        if self.cache_dir:
            with open(ref_path, "rb") as f:
                digest = hashlib.sha1(f.read()).hexdigest()
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, f"{digest}-v2.npz")
            if os.path.exists(cache_file):
                data = np.load(cache_file)
                return data["f0"], data["onsets"]

        y_ref, _ = librosa.load(ref_path, sr=self.SR, mono=True, duration=self.MAX_REF_S)
        console.print(f"[green]✓ Reference loaded ({len(y_ref)/self.SR:.1f}s)[/green]")
        y_melody = self._isolate_melody(y_ref)
        f0 = self._extract_f0(y_melody)
        onsets = librosa.onset.onset_detect(y=y_melody, sr=self.SR, hop_length=self.HOP, units='time')
        if cache_file:
            np.savez_compressed(cache_file, f0=f0, onsets=onsets)
        return f0, onsets

    def _isolate_melody(self, y: np.ndarray) -> np.ndarray:
        """Keep sustained tonal material and drop kick/bass energy below ~180 Hz."""
        D = librosa.stft(y, n_fft=2048, hop_length=self.HOP)
        H, _ = librosa.decompose.hpss(D, margin=3.0)
        freqs = librosa.fft_frequencies(sr=self.SR, n_fft=2048)
        H[freqs < 180, :] *= 0.1
        return librosa.istft(H, hop_length=self.HOP, length=len(y))

    def _extract_f0(self, y: np.ndarray) -> np.ndarray:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'),
            sr=self.SR, hop_length=self.HOP, fill_na=np.nan,
        )
        return f0

    @staticmethod
    def _to_midi(f0: np.ndarray) -> np.ndarray:
        out = np.full(len(f0), np.nan)
        valid = ~np.isnan(f0) & (f0 > 0)
        out[valid] = librosa.hz_to_midi(f0[valid])
        return out

    # ── alignment ────────────────────────────────────────────

    def _align(self, midi_voc: np.ndarray, midi_ref: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Returns (ref_index for every vocal frame, octave shift in semitones).
        The cost is octave-invariant (pitch class distance), so a tenor singing a
        soprano's song still aligns; the octave offset is resolved afterwards.
        """
        fps = self.SR / self.HOP
        step = max(1, int(round(fps / self.DTW_FPS)))
        v = self._downsample(midi_voc, step)
        r = self._downsample(midi_ref, step)

        vv, rv = ~np.isnan(v), ~np.isnan(r)
        diff = np.abs(np.subtract.outer(np.nan_to_num(v), np.nan_to_num(r)))
        pc_dist = np.abs(((diff + 6) % 12) - 6)                  # 0..6 semitones
        # Within ~half a semitone counts as "the same note" (tiny constant keeps a mild preference
        # for diagonal steps). Without the dead zone, a singer who is consistently a bit sharp or
        # flat pays on every cell, and DTW "saves" cost by squeezing the reference, drifting the
        # alignment behind the singer.
        matched = np.clip(pc_dist - 0.6, 0.0, 4.0) + 0.02
        cost = np.where(np.logical_and.outer(vv, rv), matched, 0.0)
        cost = np.where(np.logical_xor.outer(vv, rv), 2.0, cost)  # singing vs. silence mismatch

        subseq = len(v) < len(r)
        _, wp = librosa.sequence.dtw(C=cost, subseq=subseq, backtrack=True)
        wp = wp[::-1]                                             # ascending order
        # One reference position per vocal (down-sampled) frame: average duplicates
        vi, first = np.unique(wp[:, 0], return_index=True)
        ri = np.array([wp[wp[:, 0] == k, 1].mean() for k in vi])
        full_v = np.arange(len(midi_voc)) / step
        ref_idx = np.round(np.interp(full_v, vi, ri) * step).astype(int)
        ref_idx = np.clip(ref_idx, 0, len(midi_ref) - 1)

        # Resolve octave: most common whole-octave offset between aligned voiced frames
        both = ~np.isnan(midi_voc) & ~np.isnan(midi_ref[ref_idx])
        if both.sum() >= 5:
            octs = np.round((midi_voc[both] - midi_ref[ref_idx][both]) / 12.0)
            vals, counts = np.unique(octs, return_counts=True)
            octave_shift = float(vals[np.argmax(counts)] * 12)
        else:
            octave_shift = 0.0
        return ref_idx, octave_shift

    @staticmethod
    def _downsample(midi: np.ndarray, step: int) -> np.ndarray:
        n = len(midi) // step
        if n == 0:
            return midi.copy()
        blocks = midi[: n * step].reshape(n, step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(blocks, axis=1)

    # ── scoring ──────────────────────────────────────────────

    def _worst_moments(self, cent_dev: np.ndarray, ref_on_voc: np.ndarray, fps: float) -> list:
        bad = np.nan_to_num(np.abs(cent_dev)) > self.BAD_CENTS
        min_len = int(0.15 * fps)
        events, i = [], 0
        while i < len(bad):
            if not bad[i]:
                i += 1
                continue
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i >= min_len:
                seg = cent_dev[i:j]
                dev = float(np.nanmean(seg))
                mid = (i + j) // 2
                ref_note = librosa.midi_to_note(int(round(ref_on_voc[mid]))) if not np.isnan(ref_on_voc[mid]) else "?"
                t = float(i / fps)
                direction = "sharp" if dev > 0 else "flat"
                events.append((t, f"{abs(dev):.0f}¢ {direction} on {ref_note} at {t:.1f}s", float(j / fps), abs(dev)))
            i = j
        events.sort(key=lambda e: -e[3] * (e[2] - e[0]))   # worst = most off for longest
        return sorted([(t, desc, end) for t, desc, end, _ in events[:8]])

    def _rhythm_accuracy(self, y_voc: np.ndarray, onsets_ref: np.ndarray, ref_idx: np.ndarray, fps: float) -> dict:
        """Map reference onsets onto the singer's timeline via the alignment, then match within 150 ms."""
        onsets_voc = librosa.onset.onset_detect(y=y_voc, sr=self.SR, hop_length=self.HOP, units='time')
        ref_t = ref_idx / fps
        lo, hi = ref_t[0], ref_t[-1]
        in_part = onsets_ref[(onsets_ref >= lo) & (onsets_ref <= hi)]
        if len(in_part) == 0 or len(onsets_voc) == 0:
            return {"rhythm_accuracy": 50.0, "early_late_ms": 0.0}
        voc_t = np.arange(len(ref_idx)) / fps
        order = np.argsort(ref_t, kind="stable")
        expected = np.interp(in_part, ref_t[order], voc_t[order])   # where each reference onset should fall for the singer

        offsets, used = [], set()
        for t_exp in expected:
            d = np.abs(onsets_voc - t_exp)
            k = int(np.argmin(d))
            if d[k] < 0.15 and k not in used:
                offsets.append((onsets_voc[k] - t_exp) * 1000)
                used.add(k)
        if not offsets:
            return {"rhythm_accuracy": 0.0, "early_late_ms": 0.0}
        offsets = np.array(offsets)
        match_rate = len(offsets) / len(expected)
        return {
            "rhythm_accuracy": float(np.clip(match_rate * 100 - np.abs(offsets).mean() / 2, 0, 100)),
            "early_late_ms": float(offsets.mean()),
        }


def _nan_to_none(arr) -> list:
    """JSON-friendly list: NaN → None, floats rounded to 2 dp."""
    return [None if (v is None or np.isnan(v)) else round(float(v), 2) for v in np.asarray(arr, dtype=float)]


# ─────────────────────────────────────────────────────────────
# DEEP ANALYSIS MODULES
# ─────────────────────────────────────────────────────────────

class VowelFormantAnalyzer:
    """
    Tracks F1/F2 formants frame-by-frame using LPC.
    Detects vowel modification issues, spread vowels, swallowed vowels.
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray, hop: int = 256) -> dict:
        frame_len = 2048
        results = {"f1_mean": 0.0, "f2_mean": 0.0,
                   "vowel_modification_score": 50.0,
                   "spread_vowel_ratio": 0.0,
                   "covered_vowel_ratio": 0.0,
                   "f1_f2_trajectory": []}

        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop)
        voiced_mask = ~np.isnan(f0)
        f1_vals, f2_vals = [], []

        for i in range(min(frames.shape[1], len(voiced_mask))):
            if not voiced_mask[i]:
                continue
            frame = frames[:, i] * np.hanning(frame_len)
            try:
                # LPC order ~2 + sr/1000
                order = int(2 + sr / 1000)
                a = librosa.lpc(frame, order=order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                freqs = np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi))
                freqs = np.sort(freqs[freqs > 80])
                if len(freqs) >= 2:
                    f1_vals.append(freqs[0])
                    f2_vals.append(freqs[1])
            except Exception:
                pass

        if len(f1_vals) < 5:
            return results

        f1 = np.array(f1_vals)
        f2 = np.array(f2_vals)
        results["f1_mean"] = float(f1.mean())
        results["f2_mean"] = float(f2.mean())

        # Spread vowels: high F2 (>2200 Hz) — can indicate jaw tension / "ee"-ification
        results["spread_vowel_ratio"] = float((f2 > 2200).mean())
        # Covered/swallowed vowels: very low F1 (<300 Hz) — too much tongue bunching
        results["covered_vowel_ratio"] = float((f1 < 300).mean())

        # Modification score: reward F1 in 400–900 Hz, F2 in 1000–2200 Hz (open, resonant range)
        f1_ok = ((f1 >= 400) & (f1 <= 900)).mean()
        f2_ok = ((f2 >= 1000) & (f2 <= 2200)).mean()
        results["vowel_modification_score"] = float((f1_ok * 0.5 + f2_ok * 0.5) * 100)

        results["f1_f2_trajectory"] = list(zip(f1_vals[:200], f2_vals[:200]))
        return results


class LaryngealAnalyzer:
    """
    Estimates laryngeal height from spectral characteristics.
    High larynx → raised spectral centroid in upper mid-range, brightened nasality.
    Low larynx → darker timbre, deeper formants.
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        D = librosa.stft(y, n_fft=2048, hop_length=256)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Laryngeal height proxy: ratio of 2–4 kHz energy to 500 Hz–2 kHz energy
        high_mask = (freqs >= 2000) & (freqs <= 4000)
        mid_mask  = (freqs >= 500)  & (freqs <= 2000)
        high_e = S[high_mask].mean()
        mid_e  = S[mid_mask].mean()
        ratio  = high_e / (mid_e + 1e-9)

        # Classify: ratio > 0.45 → likely high larynx; < 0.2 → low larynx
        if ratio > 0.45:
            position = "high"
            height_score = float(np.clip(100 - (ratio - 0.45) / 0.3 * 60, 0, 100))
        elif ratio < 0.2:
            position = "low"
            height_score = float(np.clip(100 - (0.2 - ratio) / 0.15 * 40, 0, 100))
        else:
            position = "neutral"
            height_score = 90.0

        # Pharyngeal space proxy: low-mid energy ratio
        low_mask = (freqs >= 150) & (freqs <= 500)
        pharyngeal_space = float(S[low_mask].mean() / (S[mid_mask].mean() + 1e-9))

        return {
            "laryngeal_position": position,
            "laryngeal_height_score": height_score,
            "high_larynx_ratio": float(ratio),
            "pharyngeal_space_score": float(np.clip(pharyngeal_space * 200, 0, 100)),
        }


class OnsetTypeAnalyzer:
    """
    Classifies each note onset as:
      - hard_glottal: abrupt, high energy spike (H: attack ratio > 0.8)
      - breathy_onset: gradual rise with noise floor (HNR spike precedes pitch)
      - balanced: clean, immediate tone without spike
    """
    def analyze(self, y: np.ndarray, sr: int, hop: int = 256) -> dict:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units='frames')
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]

        if len(onsets) < 2:
            return {"hard_glottal_pct": 0.0, "breathy_onset_pct": 0.0,
                    "balanced_onset_pct": 100.0, "onset_type_score": 70.0}

        hard, breathy, balanced = 0, 0, 0
        for f in onsets:
            pre  = rms[max(f - 3, 0):f].mean() if f > 0 else 0
            peak = rms[f:min(f + 3, len(rms))].mean()
            post = rms[min(f + 3, len(rms) - 1):min(f + 8, len(rms))].mean()
            if pre < 1e-5:
                ratio = 0.0
            else:
                ratio = peak / (pre + 1e-9)

            rise_rate = (peak - pre) / (pre + 1e-9)

            if ratio > 4.0:
                hard += 1
            elif rise_rate < 0.5 and peak < pre * 1.5:
                breathy += 1
            else:
                balanced += 1

        total = len(onsets)
        h_pct = hard    / total * 100
        b_pct = breathy / total * 100
        bl_pct = balanced / total * 100

        score = float(np.clip(bl_pct * 0.6 + b_pct * 0.4 + (100 - h_pct) * 0.3, 0, 100))

        return {
            "hard_glottal_pct": h_pct,
            "breathy_onset_pct": b_pct,
            "balanced_onset_pct": bl_pct,
            "onset_type_score": score,
            "total_onsets": total,
        }


class PhraseContourAnalyzer:
    """
    Splits the performance into individual phrases and analyses each:
      - Does pitch drift flat/sharp toward phrase end?
      - Is there a consistent scooping pattern at phrase starts?
      - Are phrase climaxes in the right place dynamically?
    """
    def analyze(self, f0: np.ndarray, times: np.ndarray,
                y: np.ndarray, sr: int, hop: int = 256) -> dict:
        voiced = ~np.isnan(f0) & (f0 > 0)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_t = librosa.times_like(rms, sr=sr, hop_length=hop)

        # Segment phrases: voiced regions
        phrases = []
        in_p = False
        p_start = 0
        for i, v in enumerate(voiced):
            if v and not in_p:
                in_p = True; p_start = i
            elif not v and in_p:
                in_p = False
                if i - p_start > 10:
                    phrases.append((p_start, i))
        if in_p and len(f0) - p_start > 10:
            phrases.append((p_start, len(f0)))

        phrase_stats = []
        end_flat_count = 0
        scoop_count = 0
        total_phrases = len(phrases)

        for start, end in phrases:
            seg = f0[start:end]
            seg_valid = seg[~np.isnan(seg) & (seg > 0)]
            if len(seg_valid) < 5:
                continue

            midi_seg = librosa.hz_to_midi(seg_valid)
            t_norm = np.linspace(0, 1, len(midi_seg))

            # Pitch trend within phrase: linear regression
            slope, intercept, r, p_val, _ = stats.linregress(t_norm, midi_seg)

            # Flat ending: negative slope in last 30% of phrase
            last_third = midi_seg[int(len(midi_seg) * 0.7):]
            if len(last_third) > 3:
                end_slope, *_ = stats.linregress(np.arange(len(last_third)), last_third)
                if end_slope < -0.08:
                    end_flat_count += 1

            # Scoop at start: pitch rises sharply in first 15%
            first_15 = midi_seg[:max(int(len(midi_seg) * 0.15), 3)]
            if len(first_15) > 2:
                start_slope, *_ = stats.linregress(np.arange(len(first_15)), first_15)
                if start_slope > 0.3:
                    scoop_count += 1

            # Dynamic climax position: where is the loudest point?
            p_rms_idx = np.searchsorted(rms_t, times[start])
            p_rms_end = np.searchsorted(rms_t, times[min(end, len(times) - 1)])
            seg_rms = rms[p_rms_idx:p_rms_end]
            climax_pos = float(np.argmax(seg_rms) / (len(seg_rms) + 1e-9)) if len(seg_rms) > 0 else 0.5

            phrase_stats.append({
                "pitch_slope": float(slope),
                "climax_position": climax_pos,
                "length_frames": end - start,
            })

        end_flat_ratio = end_flat_count / (total_phrases + 1e-9)
        scoop_ratio    = scoop_count    / (total_phrases + 1e-9)

        # Climax distribution: ideal = 0.6–0.8 position (rule of thirds / golden ratio)
        if phrase_stats:
            climax_positions = [p["climax_position"] for p in phrase_stats]
            ideal_climax = float(np.mean([abs(c - 0.7) for c in climax_positions]))
        else:
            ideal_climax = 0.5

        phrase_shaping_score = float(np.clip(100 - end_flat_ratio * 60 - scoop_ratio * 40 - ideal_climax * 80, 0, 100))

        return {
            "total_phrases": total_phrases,
            "end_flat_ratio": end_flat_ratio,
            "scoop_ratio": scoop_ratio,
            "phrase_shaping_score": phrase_shaping_score,
            "phrase_stats": phrase_stats,
            "avg_climax_position": float(np.mean([p["climax_position"] for p in phrase_stats])) if phrase_stats else 0.5,
        }


class TensionAnalyzer:
    """
    Estimates muscular tension signatures from audio:
      - Jaw tension: excess energy in the 1–3 kHz band with narrow vowels
      - Throat constriction: reduced pharyngeal space, raised formants
      - Tongue tension: excess sibilance and high-frequency edge
      - Neck tension: abrupt vibrato rate, pitch instability on high notes
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        D = librosa.stft(y, n_fft=2048, hop_length=256)
        S = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Jaw tension: 1–3 kHz dominance relative to 200–1000 Hz
        jaw_band  = (freqs >= 1000) & (freqs <= 3000)
        open_band = (freqs >= 200)  & (freqs <= 1000)
        jaw_ratio = S[jaw_band].mean() / (S[open_band].mean() + 1e-9)
        jaw_tension = float(np.clip((jaw_ratio - 0.3) / 0.4 * 100, 0, 100))

        # Throat constriction: spectral centroid on voiced frames
        voiced = ~np.isnan(f0)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)[0]
        voiced_centroid = centroid[voiced[:len(centroid)]] if voiced[:len(centroid)].sum() > 5 else centroid
        constriction = float(np.clip((voiced_centroid.mean() - 1000) / 2000 * 100, 0, 100))

        # Tongue tension: sibilance ratio (5–8 kHz)
        sibilant_band = (freqs >= 5000) & (freqs <= 8000)
        tongue_tension = float(np.clip(S[sibilant_band].mean() / (S[open_band].mean() + 1e-9) * 200, 0, 100))

        # High-note tension: pitch stability on highest 20% of notes
        valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
        if len(valid_f0) > 10:
            high_threshold = np.percentile(valid_f0, 80)
            high_mask = (f0 > high_threshold) & ~np.isnan(f0)
            high_f0 = f0[high_mask]
            if len(high_f0) > 5:
                high_jitter = np.abs(np.diff(librosa.hz_to_midi(high_f0))).mean()
                neck_tension = float(np.clip(high_jitter * 25, 0, 100))
            else:
                neck_tension = 50.0
        else:
            neck_tension = 50.0

        overall_tension = float((jaw_tension + constriction + tongue_tension + neck_tension) / 4)

        return {
            "jaw_tension": jaw_tension,
            "throat_constriction": constriction,
            "tongue_tension": tongue_tension,
            "neck_tension": neck_tension,
            "overall_tension": overall_tension,
        }


class IntervalAccuracyAnalyzer:
    """
    Analyses pitch interval accuracy — how precisely the voice lands
    on intervals (2nds, 3rds, 5ths, octaves) relative to the semitone grid.
    Exposes which interval types cause the most trouble.
    """
    INTERVAL_NAMES = {
        1: "minor 2nd", 2: "major 2nd", 3: "minor 3rd", 4: "major 3rd",
        5: "perfect 4th", 6: "tritone", 7: "perfect 5th",
        8: "minor 6th", 9: "major 6th", 10: "minor 7th", 11: "major 7th",
        12: "octave",
    }

    def analyze(self, f0: np.ndarray) -> dict:
        valid = ~np.isnan(f0) & (f0 > 0)
        if valid.sum() < 20:
            return {"interval_accuracy": {}, "worst_interval": "unknown",
                    "largest_leap_semitones": 0.0, "leap_accuracy": 50.0}

        midi = librosa.hz_to_midi(f0[valid])

        # Find note-boundary frames: large jumps between sustained pitches
        diffs = np.abs(np.diff(midi))
        note_boundaries = np.where(diffs > 0.5)[0]

        interval_errors = {}  # interval_size_semitones → list of cent deviations
        large_leaps = []

        for i in note_boundaries:
            if i + 1 >= len(midi):
                continue
            interval_semitones = round(abs(midi[i + 1] - midi[i]))
            if interval_semitones == 0:
                continue

            # Accuracy: how close to the ideal semitone grid did we land?
            target_arrival = round(midi[i + 1])
            actual_arrival = midi[i + 1]
            cent_error = abs(actual_arrival - target_arrival) * 100

            interval_errors.setdefault(interval_semitones, []).append(cent_error)
            if interval_semitones >= 5:
                large_leaps.append((interval_semitones, cent_error))

        # Summarise
        interval_accuracy = {}
        for semitones, errors in interval_errors.items():
            name = self.INTERVAL_NAMES.get(semitones, f"{semitones}st")
            interval_accuracy[name] = {
                "avg_error_cents": float(np.mean(errors)),
                "occurrences": len(errors),
                "worst_cents": float(np.max(errors)),
            }

        worst_interval = max(interval_accuracy, key=lambda k: interval_accuracy[k]["avg_error_cents"]) \
            if interval_accuracy else "unknown"

        largest_leap = max([l[0] for l in large_leaps], default=0)
        leap_acc = 100 - float(np.mean([l[1] for l in large_leaps])) if large_leaps else 80.0

        return {
            "interval_accuracy": interval_accuracy,
            "worst_interval": worst_interval,
            "largest_leap_semitones": float(largest_leap),
            "leap_accuracy": float(np.clip(leap_acc, 0, 100)),
        }


class PitchEndingAnalyzer:
    """
    Checks phrase endings: do notes resolve cleanly or trail off flat?
    Also detects over-use of scoops, fall-offs, and note bending.
    """
    def analyze(self, f0: np.ndarray, times: np.ndarray) -> dict:
        voiced = ~np.isnan(f0) & (f0 > 0)
        midi = np.full(len(f0), np.nan)
        midi[voiced] = librosa.hz_to_midi(f0[voiced])

        # Find phrase ends: last voiced frame before silence gap
        endings = []
        for i in range(1, len(voiced)):
            if voiced[i - 1] and not voiced[i]:
                # Look at the last 0.2s of this phrase
                lookback = max(0, i - int(0.2 * 44100 / 256))
                seg = midi[lookback:i]
                seg_clean = seg[~np.isnan(seg)]
                if len(seg_clean) < 3:
                    continue
                end_slope, *_ = stats.linregress(np.arange(len(seg_clean)), seg_clean)
                endings.append(end_slope)

        if not endings:
            return {"trail_off_ratio": 0.0, "clean_ending_ratio": 1.0,
                    "ending_score": 75.0, "avg_ending_slope": 0.0}

        endings = np.array(endings)
        trail_off = (endings < -0.15).mean()
        sharp_end  = (endings > 0.15).mean()
        clean      = 1.0 - trail_off - sharp_end

        score = float(np.clip(clean * 100 - trail_off * 40, 0, 100))

        return {
            "trail_off_ratio": float(trail_off),
            "clean_ending_ratio": float(clean),
            "ending_score": score,
            "avg_ending_slope": float(endings.mean()),
        }


class SupportConsistencyAnalyzer:
    """
    Tracks whether breath support degrades across the duration of a recording.
    Splits recording into thirds and compares RMS consistency + pitch stability.
    Also flags held notes that go flat mid-note.
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray, hop: int = 256) -> dict:
        n = len(y)
        third = n // 3
        segments = [y[:third], y[third:2 * third], y[2 * third:]]

        rms_means = []
        for seg in segments:
            rms = librosa.feature.rms(y=seg, hop_length=hop)[0]
            rms_means.append(float(rms.mean()))

        # Stamina: does RMS decay toward the end?
        if rms_means[0] > 0:
            end_decay = (rms_means[0] - rms_means[2]) / (rms_means[0] + 1e-9)
        else:
            end_decay = 0.0

        stamina_score = float(np.clip(100 - end_decay * 150, 0, 100))

        # Held-note flat drift: within a sustained note (>0.5s), does pitch drop?
        held_note_flat = 0
        held_note_total = 0
        voiced = ~np.isnan(f0) & (f0 > 0)
        in_note = False
        note_start = 0

        for i, v in enumerate(voiced):
            if v and not in_note:
                in_note = True; note_start = i
            elif not v and in_note:
                in_note = False
                dur = (i - note_start) * hop / sr
                if dur > 0.4:
                    held_note_total += 1
                    seg = f0[note_start:i]
                    seg_midi = librosa.hz_to_midi(seg[~np.isnan(seg) & (seg > 0)])
                    if len(seg_midi) > 5:
                        slope, *_ = stats.linregress(np.arange(len(seg_midi)), seg_midi)
                        if slope < -0.05:
                            held_note_flat += 1

        flat_rate = held_note_flat / (held_note_total + 1e-9)

        return {
            "stamina_score": stamina_score,
            "rms_thirds": rms_means,
            "end_decay_ratio": float(end_decay),
            "held_note_flat_rate": float(flat_rate),
            "held_note_total": held_note_total,
        }


class AgilityAnalyzer:
    """
    Detects fast vocal runs (melisma) and rates how clearly each note
    in the run lands. A run is a passage where pitch changes faster than
    ~8 notes per second with each note distinct enough to hear.
    """
    MIN_NOTES_IN_RUN = 4
    MIN_NPS          = 6.0   # notes per second to qualify as a run

    def analyze(self, f0: np.ndarray, times: np.ndarray,
                sr: int, hop: int = 256, notes: list = None) -> dict:
        if notes is None or len(notes) < self.MIN_NOTES_IN_RUN:
            return self._empty()

        frame_rate = sr / hop
        runs       = []
        run_buf    = []

        for i, note in enumerate(notes):
            if i == 0:
                run_buf = [note]
                continue
            gap = note["start_s"] - notes[i-1]["end_s"]
            if gap < 0.08:          # notes close together — part of run
                run_buf.append(note)
            else:
                if len(run_buf) >= self.MIN_NOTES_IN_RUN:
                    runs.append(run_buf)
                run_buf = [note]
        if len(run_buf) >= self.MIN_NOTES_IN_RUN:
            runs.append(run_buf)

        if not runs:
            return self._empty()

        run_stats = []
        for run in runs:
            dur   = run[-1]["end_s"] - run[0]["start_s"]
            nps   = len(run) / (dur + 1e-9)
            if nps < self.MIN_NPS:
                continue
            # Clarity: how close is each note to the nearest semitone?
            deviations = [abs(n["cents_off"]) for n in run]
            clarity    = float(np.clip(100 - np.mean(deviations) / 30 * 100, 0, 100))
            # Evenness: are note durations roughly equal?
            durs       = [n["dur_s"] for n in run]
            evenness   = float(np.clip(
                100 - (np.std(durs) / (np.mean(durs) + 1e-9)) * 80, 0, 100
            ))
            run_stats.append({
                "start_s":  run[0]["start_s"],
                "nps":      nps,
                "notes":    len(run),
                "clarity":  clarity,
                "evenness": evenness,
                "score":    (clarity * 0.6 + evenness * 0.4),
            })

        if not run_stats:
            return self._empty()

        overall = float(np.mean([r["score"] for r in run_stats]))
        fastest = max(run_stats, key=lambda r: r["nps"])

        return {
            "run_count":     len(run_stats),
            "overall_score": overall,
            "fastest_nps":   fastest["nps"],
            "fastest_start": fastest["start_s"],
            "avg_clarity":   float(np.mean([r["clarity"] for r in run_stats])),
            "avg_evenness":  float(np.mean([r["evenness"] for r in run_stats])),
            "runs":          run_stats,
        }

    def _empty(self):
        return {"run_count": 0, "overall_score": 0.0, "fastest_nps": 0.0,
                "fastest_start": 0.0, "avg_clarity": 0.0, "avg_evenness": 0.0,
                "runs": []}


class LegatoAnalyzer:
    """
    Measures how smoothly notes connect to each other.
    Legato = no gap between notes, no glottal click, smooth pitch transition.
    Staccato patterns and unintentional gaps are flagged separately.
    """
    def analyze(self, f0: np.ndarray, times: np.ndarray,
                y: np.ndarray, sr: int, hop: int = 256,
                notes: list = None) -> dict:
        if notes is None or len(notes) < 3:
            return self._empty()

        gaps        = []   # silence gaps between consecutive notes
        transitions = []   # cents jump at note boundaries

        for i in range(1, len(notes)):
            gap = notes[i]["start_s"] - notes[i-1]["end_s"]
            if gap > 0:
                gaps.append(gap)
            # pitch jump between end of last note and start of this one
            jump_cents = abs(notes[i]["median_midi"] - notes[i-1]["median_midi"]) * 100
            transitions.append(jump_cents)

        avg_gap   = float(np.mean(gaps)) if gaps else 0.0
        gap_ratio = len([g for g in gaps if g > 0.04]) / (len(notes) - 1)

        # Smoothness of transitions (large leaps are fine; sudden glottal pops are not)
        rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_t    = librosa.times_like(rms, sr=sr, hop_length=hop)
        glottal_attacks = 0
        for note in notes[1:]:
            frame = int(note["start_s"] * sr / hop)
            if frame + 3 < len(rms) and frame > 0:
                pre  = rms[max(frame-2, 0):frame].mean()
                peak = rms[frame:frame+3].mean()
                if pre < 1e-5 and peak > 0.05:
                    glottal_attacks += 1

        glottal_ratio = glottal_attacks / max(len(notes) - 1, 1)

        legato_score = float(np.clip(
            100 - gap_ratio * 50 - avg_gap * 200 - glottal_ratio * 30,
            0, 100
        ))

        return {
            "legato_score":    legato_score,
            "avg_gap_s":       avg_gap,
            "gap_ratio":       gap_ratio,
            "glottal_attacks": glottal_attacks,
            "glottal_ratio":   glottal_ratio,
            "note_count":      len(notes),
        }

    def _empty(self):
        return {"legato_score": 75.0, "avg_gap_s": 0.0, "gap_ratio": 0.0,
                "glottal_attacks": 0, "glottal_ratio": 0.0, "note_count": 0}


class TwangAnalyzer:
    """
    Detects 'twang' — the bright, narrow-epilaryngeal-tube quality used in
    belting, musical theatre, country, and R&B. Measurable as elevated energy
    in the 2–4 kHz 'twang formant' cluster relative to the rest of the spectrum.
    Twang = efficient, safe high-note production. Absence in belting = strain risk.
    """
    TWANG_LO = 2000
    TWANG_HI = 4000

    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        D     = librosa.stft(y, n_fft=2048, hop_length=256)
        S     = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        twang_mask  = (freqs >= self.TWANG_LO) & (freqs <= self.TWANG_HI)
        body_mask   = (freqs >= 300)  & (freqs < self.TWANG_LO)

        twang_e = S[twang_mask].mean()
        body_e  = S[body_mask].mean()

        ratio = twang_e / (body_e + 1e-9)

        # Twang: high ratio is good for projection and safe belting
        # Very high ratio with no body = too bright / thin
        if ratio > 0.6:
            quality = "strong"
        elif ratio > 0.35:
            quality = "moderate"
        elif ratio > 0.18:
            quality = "weak"
        else:
            quality = "absent"

        score = float(np.clip(ratio / 0.6 * 100, 0, 100))

        return {
            "twang_quality": quality,
            "twang_ratio":   float(ratio),
            "twang_score":   score,
        }


class ProjectionAnalyzer:
    """
    Estimates how well the voice projects and fills space.
    Projection = sufficient energy in the 2–4 kHz singer's formant region,
    adequate overall RMS, and a healthy harmonic-to-noise ratio.
    A well-projecting voice sounds present and forward without a microphone.
    """
    def analyze(self, y: np.ndarray, sr: int, m: "VocalMetrics") -> dict:
        rms     = float(np.sqrt(np.mean(y ** 2)))
        rms_db  = float(20 * np.log10(rms + 1e-9))

        D     = librosa.stft(y, n_fft=2048, hop_length=256)
        S     = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Presence band: 1–5 kHz
        pres_mask = (freqs >= 1000) & (freqs <= 5000)
        low_mask  = freqs < 500
        presence  = S[pres_mask].mean() / (S[low_mask].mean() + 1e-9)

        # Harmonic richness: how many harmonics are audible
        voiced_rms = float(librosa.feature.rms(y=y, hop_length=256)[0].mean())

        # Combine into projection index
        projection = float(np.clip(
            presence * 30
            + min(voiced_rms * 500, 40)
            + m.singer_formant_strength * 0.3,
            0, 100
        ))

        if projection >= 75:
            label = "strong projection"
        elif projection >= 50:
            label = "moderate projection"
        elif projection >= 30:
            label = "weak projection"
        else:
            label = "very little projection"

        return {
            "projection_score": projection,
            "projection_label": label,
            "presence_ratio":   float(presence),
        }


class ToneConsistencyAnalyzer:
    """
    Checks whether the voice's tonal colour (timbre) stays consistent
    across the recording. Inconsistent tone = switching between chest-heavy,
    heady, and mixed timbre without musical intent.
    Uses spectral centroid variance as a proxy for timbre stability.
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        hop      = 256
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        voiced   = ~np.isnan(f0) & (f0 > 0)
        voiced_c = centroid[voiced[:len(centroid)]] if voiced[:len(centroid)].sum() > 5 else centroid

        if len(voiced_c) < 10:
            return {"tone_consistency": 75.0, "centroid_cv": 0.0, "centroid_mean": 0.0}

        cv    = voiced_c.std() / (voiced_c.mean() + 1e-9)
        score = float(np.clip(100 - cv * 150, 0, 100))

        # Detect if the voice gets noticeably brighter or darker over time
        thirds = np.array_split(voiced_c, 3)
        means  = [t.mean() for t in thirds if len(t) > 0]
        drift  = (max(means) - min(means)) / (np.mean(means) + 1e-9) if means else 0.0

        return {
            "tone_consistency":   score,
            "centroid_cv":        float(cv),
            "centroid_mean":      float(voiced_c.mean()),
            "tone_drift":         float(drift),
            "tone_drift_label":   "stable" if drift < 0.15 else "drifting" if drift < 0.35 else "very inconsistent",
        }


class VocalDistortionAnalyzer:
    """
    Detects intentional vocal distortion/grit — the rough, gravelly quality used
    in rock, metal, soul, and R&B. Measured via aperiodic noise energy between
    harmonics and spectral irregularity.
    Distinguishes intentional distortion from accidental strain-related noise.
    """
    def analyze(self, y: np.ndarray, sr: int, f0: np.ndarray) -> dict:
        hop   = 256
        D     = librosa.stft(y, n_fft=2048, hop_length=hop)
        S     = np.abs(D)

        # HNR-based approach: distortion = aperiodic energy between harmonics
        voiced   = ~np.isnan(f0) & (f0 > 0)
        if voiced.sum() < 20:
            return self._empty()

        # Spectral irregularity (Krimphoff measure)
        log_S = np.log(S + 1e-9)
        irreg = np.mean(np.abs(np.diff(log_S, axis=0)))

        # Aperiodicity proxy: RMS of residual after harmonic removal
        y_harm   = librosa.effects.harmonic(y, margin=4.0)
        y_resid  = y - y_harm[:len(y)]
        harm_rms = float(np.sqrt(np.mean(y_harm ** 2)))
        resid_rms = float(np.sqrt(np.mean(y_resid ** 2)))
        dist_ratio = resid_rms / (harm_rms + 1e-9)

        present  = dist_ratio > 0.08
        level    = (
            "heavy"    if dist_ratio > 0.35 else
            "moderate" if dist_ratio > 0.18 else
            "light"    if dist_ratio > 0.08 else
            "none"
        )
        score = float(np.clip(dist_ratio * 200, 0, 100))

        return {
            "distortion_present": present,
            "distortion_level":   level,
            "distortion_ratio":   float(dist_ratio),
            "distortion_score":   score,
        }

    def _empty(self):
        return {"distortion_present": False, "distortion_level": "none",
                "distortion_ratio": 0.0, "distortion_score": 0.0}


class RhythmicFeelAnalyzer:
    """
    Analyses rhythmic feel beyond just note duration consistency.
    Detects:
      - Whether the singer rushes or drags overall
      - Groove: consistent micro-timing patterns (e.g. slightly behind the beat)
      - Rhythmic freedom vs metronomic rigidity
    """
    def analyze(self, y: np.ndarray, sr: int, notes: list = None) -> dict:
        if not notes or len(notes) < 6:
            return self._empty()

        hop = 256
        # Detect beat from audio
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

        if len(beat_times) < 4:
            return self._empty()

        # Compare note onsets to nearest beats
        note_starts = np.array([n["start_s"] for n in notes])
        offsets_ms  = []
        for t in note_starts:
            nearest_beat = beat_times[np.argmin(np.abs(beat_times - t))]
            off_ms = (t - nearest_beat) * 1000
            if abs(off_ms) < 300:   # only count near-beat notes
                offsets_ms.append(off_ms)

        if not offsets_ms:
            return self._empty()

        offsets_ms   = np.array(offsets_ms)
        mean_offset  = float(offsets_ms.mean())   # + = behind beat, - = ahead
        spread       = float(offsets_ms.std())    # low = consistent, high = loose

        rushes  = (offsets_ms < -60).mean()
        drags   = (offsets_ms > 60).mean()

        feel = (
            "rushes consistently"  if mean_offset < -50 else
            "drags consistently"   if mean_offset > 50  else
            "behind the beat (pocket)" if 20 < mean_offset <= 50 else
            "slightly ahead of the beat" if -50 <= mean_offset < -20 else
            "right on the beat"
        )

        groove_score = float(np.clip(
            100 - abs(mean_offset) / 2 - spread / 3 - rushes * 40 - drags * 40,
            0, 100
        ))

        return {
            "tempo_bpm":      float(tempo),
            "mean_offset_ms": mean_offset,
            "feel":           feel,
            "groove_score":   groove_score,
            "rushes_ratio":   float(rushes),
            "drags_ratio":    float(drags),
            "timing_spread":  spread,
        }

    def _empty(self):
        return {"tempo_bpm": 0.0, "mean_offset_ms": 0.0, "feel": "unknown",
                "groove_score": 50.0, "rushes_ratio": 0.0, "drags_ratio": 0.0,
                "timing_spread": 0.0}


class DictionAnalyzer:
    """
    Analyses consonant clarity and diction intelligibility.
    Measures:
      - Sibilance balance (s/sh sounds — too much = hissy, too little = muddy)
      - Consonant-to-vowel energy ratio
      - Articulation sharpness of plosives (p/b/t/d/k/g)
    """
    def analyze(self, y: np.ndarray, sr: int) -> dict:
        hop   = 256
        D     = librosa.stft(y, n_fft=2048, hop_length=hop)
        S     = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Sibilance: 5–10 kHz energy
        sib_mask  = (freqs >= 5000) & (freqs <= 10000)
        mid_mask  = (freqs >= 500)  & (freqs <= 4000)
        sib_ratio = float(S[sib_mask].mean() / (S[mid_mask].mean() + 1e-9))

        # Plosive detection: short high-energy bursts in broadband signal
        rms     = librosa.feature.rms(y=y, hop_length=hop)[0]
        onsets  = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop)
        plosive_count = 0
        for f in onsets:
            if f + 2 < len(rms) and f > 0:
                pre  = rms[max(f-2, 0)]
                peak = rms[f]
                if pre < 0.02 and peak > 0.06:   # sudden burst = plosive
                    plosive_count += 1

        # Zero-crossing rate: high = consonant-rich, low = vowel-heavy
        zcr       = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        zcr_mean  = float(zcr.mean())

        if sib_ratio > 0.4:
            sib_quality = "too sibilant (hissy)"
        elif sib_ratio > 0.15:
            sib_quality = "balanced"
        else:
            sib_quality = "under-sibilant (muddy consonants)"

        diction_score = float(np.clip(
            (1 - abs(sib_ratio - 0.25) / 0.25) * 50
            + min(zcr_mean * 500, 30)
            + min(plosive_count * 3, 20),
            0, 100
        ))

        return {
            "diction_score":  diction_score,
            "sib_ratio":      sib_ratio,
            "sib_quality":    sib_quality,
            "plosive_count":  plosive_count,
            "zcr_mean":       zcr_mean,
        }


class ExpressionAnalyzer:
    """
    Measures overall expressive variety — does the singing tell a story,
    or does everything sound the same? Looks at dynamic shape, pitch contour
    variety, and timing flexibility across phrases.
    """
    def analyze(self, y: np.ndarray, sr: int,
                f0: np.ndarray, notes: list = None) -> dict:
        hop  = 256
        rms  = librosa.feature.rms(y=y, hop_length=hop)[0]

        # Dynamic variety: variance of phrase-level RMS
        phrase_rms  = []
        voiced      = ~np.isnan(f0) & (f0 > 0)
        in_phrase   = False
        buf         = []
        for i, v in enumerate(voiced):
            t_rms = min(i, len(rms) - 1)
            if v:
                if not in_phrase:
                    in_phrase = True
                    buf = []
                buf.append(rms[t_rms])
            elif in_phrase:
                in_phrase = False
                if buf:
                    phrase_rms.append(np.mean(buf))

        dynamic_variety = float(np.std(phrase_rms) / (np.mean(phrase_rms) + 1e-9)) if phrase_rms else 0.0

        # Pitch contour variety: how many different shapes do phrases take
        contour_variety = 0.0
        if notes and len(notes) >= 4:
            midis   = [n["median_midi"] for n in notes]
            slopes  = np.diff(midis)
            # Count direction changes — more = more expressive melodic movement
            sign_changes = np.sum(np.diff(np.sign(slopes)) != 0)
            contour_variety = float(np.clip(sign_changes / len(notes) * 100, 0, 100))

        # Timing variety: some phrases faster, some slower (rubato feel)
        if notes and len(notes) >= 4:
            note_durs = [n["dur_s"] for n in notes]
            timing_variety = float(np.std(note_durs) / (np.mean(note_durs) + 1e-9))
        else:
            timing_variety = 0.0

        expression_score = float(np.clip(
            dynamic_variety * 40
            + contour_variety * 0.35
            + timing_variety * 25,
            0, 100
        ))

        return {
            "expression_score":    expression_score,
            "dynamic_variety":     dynamic_variety,
            "contour_variety":     contour_variety,
            "timing_variety":      timing_variety,
            "phrase_count":        len(phrase_rms),
        }


class RootCauseEngine:
    """
    Correlates multiple metrics to identify underlying root causes
    rather than just reporting surface symptoms.
    Returns a ranked list of (root_cause, confidence, affected_symptoms).
    """

    def analyze(self, m: VocalMetrics, deep: dict) -> list[dict]:
        causes = []

        tension = deep.get("tension", {})
        breath  = deep.get("breath_detail", {})
        formant = deep.get("formants", {})
        larynx  = deep.get("larynx", {})
        phrase  = deep.get("phrase", {})
        support = deep.get("support", {})

        # ── ROOT CAUSE 1: Breath pressure collapse ──
        breath_issues = []
        if m.breath_support_score < 65:        breath_issues.append("weak support score")
        if m.breath_noise_ratio > 0.4:         breath_issues.append("breathiness")
        if support.get("held_note_flat_rate", 0) > 0.35: breath_issues.append("notes going flat mid-hold")
        if phrase.get("end_flat_ratio", 0) > 0.4:        breath_issues.append("phrases ending flat")
        if m.intonation_score < 72 and m.breath_support_score < 70:
            breath_issues.append("pitch drift")

        if len(breath_issues) >= 2:
            causes.append({
                "root": "Breath pressure collapse",
                "confidence": min(0.5 + len(breath_issues) * 0.1, 0.95),
                "symptoms": breath_issues,
                "fix_priority": 1,
            })

        # ── ROOT CAUSE 2: Laryngeal tension ──
        tension_issues = []
        ot = tension.get("overall_tension", 0)
        if tension.get("jaw_tension", 0) > 55:       tension_issues.append("jaw tension")
        if tension.get("throat_constriction", 0) > 55: tension_issues.append("throat constriction")
        if tension.get("neck_tension", 0) > 55:       tension_issues.append("neck/high-note tension")
        if m.vibrato_rate_hz > 7.5 and m.vibrato_present: tension_issues.append("fast tense vibrato")
        if larynx.get("laryngeal_position") == "high": tension_issues.append("high larynx")
        if m.pitch_stability_score < 55:               tension_issues.append("pitch jitter")

        if len(tension_issues) >= 2:
            causes.append({
                "root": "Laryngeal / muscular tension",
                "confidence": min(0.4 + len(tension_issues) * 0.12, 0.92),
                "symptoms": tension_issues,
                "fix_priority": 1,
            })

        # ── ROOT CAUSE 3: Resonance space issues ──
        res_issues = []
        if m.nasality_score > 60:                       res_issues.append("excess nasality")
        if m.singer_formant_strength < 45:              res_issues.append("weak ring/carrying power")
        if larynx.get("laryngeal_position") == "high":  res_issues.append("high larynx compresses pharynx")
        if formant.get("covered_vowel_ratio", 0) > 0.3: res_issues.append("swallowed/covered vowels")
        if formant.get("spread_vowel_ratio", 0) > 0.4:  res_issues.append("spread/bright vowels")

        if len(res_issues) >= 2:
            causes.append({
                "root": "Resonance space / vowel formation",
                "confidence": min(0.45 + len(res_issues) * 0.1, 0.9),
                "symptoms": res_issues,
                "fix_priority": 2,
            })

        # ── ROOT CAUSE 4: Register coordination ──
        reg_issues = []
        if m.register_breaks > 1:                        reg_issues.append("audible register breaks")
        if not m.mix_voice_detected and m.f0_range_semitones > 8: reg_issues.append("no mix voice through passaggio")
        if m.belting_detected and m.belting_efficiency_score < 60: reg_issues.append("forced belt")
        if m.vibrato_score < 40 and m.vibrato_present:   reg_issues.append("vibrato only in one register")

        if len(reg_issues) >= 2:
            causes.append({
                "root": "Register / passaggio coordination",
                "confidence": min(0.4 + len(reg_issues) * 0.12, 0.9),
                "symptoms": reg_issues,
                "fix_priority": 2,
            })

        # ── ROOT CAUSE 5: Ear training / pitch memory ──
        ear_issues = []
        if m.intonation_score < 65:                      ear_issues.append("consistent pitch inaccuracy")
        if m.ref_melody_accuracy >= 0 and m.ref_melody_accuracy < 70: ear_issues.append("melody deviation from reference")
        interval = deep.get("intervals", {})
        worst_int = interval.get("worst_interval", "")
        if worst_int and interval.get("interval_accuracy", {}).get(worst_int, {}).get("avg_error_cents", 0) > 40:
            ear_issues.append(f"poor {worst_int} accuracy")

        if len(ear_issues) >= 2:
            causes.append({
                "root": "Ear training / pitch memory",
                "confidence": min(0.4 + len(ear_issues) * 0.12, 0.88),
                "symptoms": ear_issues,
                "fix_priority": 2,
            })

        # ── ROOT CAUSE 6: Phrasing / musical expression ──
        phrase_issues = []
        if m.dynamic_range_db < 12:                       phrase_issues.append("flat dynamics")
        if phrase.get("phrase_shaping_score", 100) < 55:  phrase_issues.append("poor phrase shaping")
        if phrase.get("avg_climax_position", 0.5) < 0.4:  phrase_issues.append("climax too early in phrase")
        if phrase.get("scoop_ratio", 0) > 0.4:            phrase_issues.append("scooping on note starts")
        ending = deep.get("endings", {})
        if ending.get("trail_off_ratio", 0) > 0.4:        phrase_issues.append("notes trailing off at phrase ends")

        if len(phrase_issues) >= 2:
            causes.append({
                "root": "Phrasing / musical shaping",
                "confidence": min(0.4 + len(phrase_issues) * 0.1, 0.85),
                "symptoms": phrase_issues,
                "fix_priority": 3,
            })

        causes.sort(key=lambda c: (-c["confidence"], c["fix_priority"]))
        return causes


# ─────────────────────────────────────────────────────────────
# TIP ENGINE
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# SINGER PROFILE  — personalisation layer
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# USER ACCOUNTS & ONBOARDING
# ─────────────────────────────────────────────────────────────

ACCOUNTS_DIR = os.path.expanduser("~/.vocaliq/users")

GENRE_CHOICES = [
    "Pop", "R&B / Soul", "Musical Theatre", "Classical / Opera",
    "Jazz", "Rock", "Gospel / Choir", "Folk / Singer-Songwriter",
    "Country", "Electronic / EDM", "Other",
]

GOAL_CHOICES = [
    "Improve pitch accuracy",
    "Build vocal range",
    "Develop vibrato",
    "Reduce vocal strain / tension",
    "Prepare for a performance or audition",
    "Improve breath control",
    "Sing higher / belt better",
    "Sound more professional / polished",
    "Improve resonance and tone quality",
    "Just have fun and track progress",
]

KNOWN_ISSUE_CHOICES = [
    "None that I know of",
    "My pitch goes flat when I'm tired",
    "I strain or feel tension on high notes",
    "I lose my voice quickly",
    "My voice cracks / breaks registers",
    "I sound breathy",
    "I have been told I go sharp",
    "I've had nodules or other vocal injury",
    "I get very nervous performing",
    "My vibrato is unstable or missing",
]


@dataclass
class UserAccount:
    """Everything we know about this singer from signup and history."""
    username:          str  = ""
    display_name:      str  = ""
    created_at:        str  = ""
    last_session:      str  = ""
    session_count:     int  = 0

    # Onboarding answers
    years_singing:     str  = "less than 1 year"   # "less than 1 year" / "1-3" / "3-7" / "7+"
    takes_lessons:     bool = False
    genres:            list = field(default_factory=list)
    goals:             list = field(default_factory=list)
    known_issues:      list = field(default_factory=list)
    practice_days_pw:  int  = 3         # days per week
    self_assessed_level: str = "beginner"  # beginner / intermediate / advanced

    # Streaks
    practice_streak:   int  = 0    # consecutive calendar days with a session
    session_streak:    int  = 0    # consecutive sessions completed (no day limit)
    longest_streak:    int  = 0    # all-time best practice streak
    last_session_date: str  = ""   # "YYYY-MM-DD" for daily streak tracking

    # Per-area improvement streaks (consecutive sessions each area improved)
    area_streaks:      dict = field(default_factory=dict)   # {"Pitch": 2, "Breath": 0, ...}
    area_levels:       dict = field(default_factory=dict)   # last qualitative level per area

    # Session history (internal, not shown as scores)
    session_history:   list = field(default_factory=list)   # [{"date": .., "areas_improved": [...], "issues": [...]}, ...]
    recurring_issues:  list = field(default_factory=list)
    improved_areas:    list = field(default_factory=list)


class AccountManager:
    """
    Handles creating, loading, saving, and listing user accounts.
    All data is stored as JSON files under ~/.vocaliq/users/.
    """

    def __init__(self):
        os.makedirs(ACCOUNTS_DIR, exist_ok=True)

    # ── persistence ─────────────────────────────────────────

    def _path(self, username: str) -> str:
        safe = "".join(c for c in username.lower() if c.isalnum() or c in "_-")
        return os.path.join(ACCOUNTS_DIR, f"{safe}.json")

    def exists(self, username: str) -> bool:
        return os.path.exists(self._path(username))

    def save(self, account: UserAccount):
        with open(self._path(account.username), "w") as f:
            json.dump(account.__dict__, f, indent=2)

    def load(self, username: str) -> Optional[UserAccount]:
        path = self._path(username)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        a = UserAccount()
        for k, v in data.items():
            if hasattr(a, k):
                setattr(a, k, v)
        return a

    def list_accounts(self) -> list[str]:
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(ACCOUNTS_DIR)
            if f.endswith(".json")
        ]

    def update_session(self, account: UserAccount, area_scores: dict,
                       issues: list[str], strengths: list[str]):
        """
        Update streaks and history after each session.
        area_scores: {"Pitch": 72.0, "Breath": 55.0, ...}  (used internally only, never shown)
        """
        apply_session_to_account(account, area_scores, issues)
        self.save(account)


def apply_session_to_account(account: UserAccount, area_scores: dict, issues: list[str],
                             when: Optional[str] = None) -> UserAccount:
    """
    Fold one completed session into the account's streaks, per-area levels and
    recurring issues. `when` is "YYYY-MM-DD HH:MM" (defaults to now), which lets
    callers replay stored history, e.g. the web app rebuilding an account from its database.
    """
    now   = when or time.strftime("%Y-%m-%d %H:%M")
    today = now[:10]

    account.session_count += 1
    account.last_session   = now

    # ── Practice streak (calendar days) ─────────────────
    if account.last_session_date:
        from datetime import datetime, timedelta
        last = datetime.strptime(account.last_session_date, "%Y-%m-%d")
        curr = datetime.strptime(today, "%Y-%m-%d")
        gap  = (curr - last).days
        if gap == 1:
            account.practice_streak += 1
        elif gap == 0:
            pass  # same day, don't increment
        else:
            account.practice_streak = 1  # streak broken
    else:
        account.practice_streak = 1

    account.last_session_date = today
    account.session_streak   += 1
    account.longest_streak    = max(account.longest_streak, account.practice_streak)

    # ── Per-area improvement streaks ─────────────────────
    LEVEL_THRESHOLDS = [
        (85, "excellent"),
        (70, "strong"),
        (55, "getting there"),
        (40, "developing"),
        (0,  "needs work"),
    ]

    def score_to_level(s: float) -> str:
        for threshold, label in LEVEL_THRESHOLDS:
            if s >= threshold:
                return label
        return "needs work"

    area_streaks = account.area_streaks or {}
    area_levels  = account.area_levels  or {}
    improved_now = []

    for area, score in area_scores.items():
        new_level = score_to_level(score)
        old_level = area_levels.get(area, "")
        level_order = {lbl: i for i, (_, lbl) in enumerate(reversed(LEVEL_THRESHOLDS))}
        improved = (old_level and
                    level_order.get(new_level, 0) > level_order.get(old_level, 0))

        if improved:
            area_streaks[area] = area_streaks.get(area, 0) + 1
            improved_now.append(area)
        else:
            area_streaks[area] = 0

        area_levels[area] = new_level

    account.area_streaks   = area_streaks
    account.area_levels    = area_levels
    account.improved_areas = improved_now

    # ── Recurring issues ─────────────────────────────────
    prev_issues = []
    if account.session_history:
        prev_issues = account.session_history[-1].get("issues", [])
    account.recurring_issues = [i for i in issues if i in prev_issues]

    # ── Session history (no scores stored) ───────────────
    account.session_history.append({
        "date":           now,
        "areas_improved": improved_now,
        "issues":         issues,
    })
    account.session_history = account.session_history[-20:]
    return account


class OnboardingQuestionnaire:
    """
    Interactive CLI onboarding for new accounts.
    Asks friendly, plain-English questions and returns a filled UserAccount.
    """

    def run(self, username: str) -> UserAccount:
        account = UserAccount()
        account.username   = username
        account.created_at = time.strftime("%Y-%m-%d %H:%M")

        console.print()
        console.print(Panel(
            "[bold cyan]Quick setup[/bold cyan]  —  just a few questions so your tips are written for you.\n"
            "[dim]Takes about a minute. Only happens once.[/dim]",
            border_style="cyan",
        ))

        account.display_name = self._ask(
            "What's your name or nickname?",
            default=username.capitalize(),
        )

        account.years_singing = self._choice(
            "How long have you been singing?",
            ["Less than a year", "1 to 3 years", "3 to 7 years", "More than 7 years"],
        )

        account.takes_lessons = self._yes_no("Are you taking singing lessons right now?")

        account.self_assessed_level = self._choice(
            "How would you describe yourself as a singer?",
            ["Total beginner", "I know the basics",
             "I sing for people sometimes", "I perform regularly"],
        )

        account.genres = self._multi_choice(
            "What kind of music do you sing? Pick as many as you like.",
            GENRE_CHOICES,
        )

        account.goals = self._multi_choice(
            "What are you trying to get better at? Pick up to 3.",
            GOAL_CHOICES,
            max_picks=3,
        )

        days_str = self._choice(
            "How often do you sing or practise in a typical week?",
            ["Rarely", "Once or twice", "Three or four times", "Almost every day"],
        )
        account.practice_days_pw = {
            "Rarely": 0, "Once or twice": 1,
            "Three or four times": 3, "Almost every day": 6,
        }.get(days_str, 3)

        account.known_issues = self._multi_choice(
            "Has anyone ever told you something about your voice, or do you notice anything yourself?",
            KNOWN_ISSUE_CHOICES,
        )

        console.print()
        console.print(Panel(
            f"[bold green]All set, {account.display_name}![/bold green]\n"
            "Your profile is saved. Your tips from now on will be written just for you.",
            border_style="green",
        ))
        return account

    # ── helpers ─────────────────────────────────────────────

    def _ask(self, question: str, default: str = "") -> str:
        console.print(f"\n[bold white]{question}[/bold white]")
        if default:
            console.print(f"  [dim]Press Enter to use \"{default}\"[/dim]")
        val = input("  > ").strip()
        return val if val else default

    def _yes_no(self, question: str) -> bool:
        console.print(f"\n[bold white]{question}[/bold white]")
        console.print("  [dim]Type y or n[/dim]")
        val = input("  > ").strip().lower()
        return val.startswith("y")

    def _choice(self, question: str, options: list[str]) -> str:
        console.print(f"\n[bold white]{question}[/bold white]")
        for i, opt in enumerate(options, 1):
            console.print(f"  [cyan]{i}[/cyan]  {opt}")
        while True:
            val = input("  Type a number: ").strip()
            if val.isdigit() and 1 <= int(val) <= len(options):
                return options[int(val) - 1]
            console.print("  [yellow]Just type one of the numbers above.[/yellow]")

    def _multi_choice(self, question: str, options: list[str],
                       max_picks: int = 99) -> list[str]:
        console.print(f"\n[bold white]{question}[/bold white]")
        if max_picks < 99:
            console.print(f"  [dim]Type up to {max_picks} numbers with spaces between them, like: 1 3 5[/dim]")
        else:
            console.print("  [dim]Type the numbers you want, with spaces between them, like: 1 3 5[/dim]")
        for i, opt in enumerate(options, 1):
            console.print(f"  [cyan]{i}[/cyan]  {opt}")
        while True:
            raw = input("  > ").strip()
            picks = []
            valid = True
            for tok in raw.split():
                if tok.isdigit() and 1 <= int(tok) <= len(options):
                    picks.append(options[int(tok) - 1])
                else:
                    valid = False
                    break
            if valid and picks and len(picks) <= max_picks:
                return picks
            console.print(f"  [yellow]Please pick up to {max_picks} numbers from the list above.[/yellow]")


def account_login_flow(manager: AccountManager,
                       questionnaire: OnboardingQuestionnaire,
                       force_new: bool = False) -> UserAccount:
    """
    Runs the account selection / creation flow and returns a loaded account.
    Called once at the start of main().
    """
    existing = manager.list_accounts()

    console.print()
    if existing and not force_new:
        console.print(Panel(
            "[bold white]Who's singing today?[/bold white]",
            border_style="cyan",
        ))
        options = existing + ["Create a new profile"]
        for i, name in enumerate(options, 1):
            console.print(f"  [cyan]{i}[/cyan]  {name}")
        while True:
            raw = input("\n  Type a number: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                chosen = options[int(raw) - 1]
                break
            console.print("  [yellow]Just type one of the numbers above.[/yellow]")

        if chosen != "Create a new profile":
            account = manager.load(chosen)
            if account:
                streak_msg = (
                    f"  [yellow]{account.practice_streak} day streak![/yellow]  "
                    if account.practice_streak >= 2 else ""
                )
                last = account.last_session or "first time"
                console.print(f"\n[green]Hey {account.display_name}![/green]  {streak_msg}Last time: {last}\n")
                return account
    else:
        console.print(Panel(
            "[bold white]Let's set up your profile.[/bold white]",
            border_style="cyan",
        ))

    while True:
        username = input("  Pick a username (letters and numbers only): ").strip()
        if not username:
            continue
        if manager.exists(username):
            console.print(f"  [yellow]\"{username}\" is already taken. Try a different one.[/yellow]")
        else:
            break

    account = questionnaire.run(username)
    manager.save(account)
    return account


@dataclass
class SingerProfile:
    """All singer-specific facts needed to write personalised tips."""
    # Identity & account
    voice_type:     str  = "unknown"
    skill_level:    str  = "developing"
    skill_label:    str  = "developing singer"
    display_name:   str  = ""
    genres:         list = field(default_factory=list)
    goals:          list = field(default_factory=list)
    known_issues:   list = field(default_factory=list)
    takes_lessons:  bool = False
    years_singing:  str  = ""
    practice_days:  int  = 3
    session_number:    int  = 1
    practice_streak:   int  = 0
    session_streak:    int  = 0
    longest_streak:    int  = 0
    area_streaks:      dict = field(default_factory=dict)
    area_levels:       dict = field(default_factory=dict)
    recurring_issues:  list = field(default_factory=list)
    improved_areas:    list = field(default_factory=list)

    # Strongest & weakest areas (category names)
    top_strengths: list = field(default_factory=list)
    top_weaknesses: list = field(default_factory=list)
    single_priority: str = ""             # the ONE thing to focus on first

    # Pitch specifics
    mean_note: str = ""                   # e.g. "A4"
    range_desc: str = ""                  # e.g. "just over an octave (13 semitones)"
    intonation_pattern: str = ""          # "consistently flat", "consistently sharp", "unstable"
    worst_pitch_moments: list = field(default_factory=list)   # [(time_s, desc), ...]

    # Vibrato specifics
    vib_rate_delta: float = 0.0           # how far from ideal (signed)
    vib_extent_delta: float = 0.0
    vib_ideal_rate_range: tuple = (5.0, 7.0)

    # Breath specifics
    phrase_len_grade: str = ""            # "very short", "short", "adequate", "strong"
    breath_pattern: str = ""             # "collapses at phrase end", "inconsistent pressure", "good"

    # Register
    passaggio_note_lo: str = ""
    passaggio_note_hi: str = ""

    # Tension
    dominant_tension: str = ""           # "jaw", "throat", "tongue", "neck", or ""

    # Interval trouble
    worst_interval: str = ""
    worst_interval_cents: float = 0.0

    # Cross-linked problems (root causes already computed)
    connected_issues: list = field(default_factory=list)  # [(issue_a, issue_b, link_desc), ...]

    # Encouragement hook (something genuinely good to mention first)
    opening_praise: str = ""


class PersonalizationEngine:
    """
    Builds a SingerProfile from VocalMetrics + deep analysis dict.
    This profile is injected into every tip so they read as if written
    specifically for this person rather than copied from a textbook.
    """

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def build(self, m: VocalMetrics, belt: dict, deep: dict,
              account: "UserAccount" = None) -> SingerProfile:
        p = SingerProfile()
        vprofile = deep.get("voice_profile", {})

        # ── Account fields ───────────────────────────────────
        if account:
            p.display_name    = account.display_name or account.username
            p.genres          = account.genres
            p.goals           = account.goals
            p.known_issues    = account.known_issues
            p.takes_lessons   = account.takes_lessons
            p.years_singing   = account.years_singing
            p.practice_days   = account.practice_days_pw
            p.session_number   = account.session_count + 1
            p.practice_streak  = account.practice_streak
            p.session_streak   = account.session_streak
            p.longest_streak   = account.longest_streak
            p.area_streaks     = account.area_streaks or {}
            p.area_levels      = account.area_levels  or {}
            p.recurring_issues = account.recurring_issues or []
            p.improved_areas   = account.improved_areas  or []

        # ── Voice type ──────────────────────────────────────
        p.voice_type = deep.get("voice_type", "unknown")
        p.vib_ideal_rate_range = vprofile.get("vib_rate", (5.0, 7.0))

        pass_lo, pass_hi = vprofile.get("passaggio", (50, 67))
        p.passaggio_note_lo = self._midi_name(pass_lo)
        p.passaggio_note_hi = self._midi_name(pass_hi)

        # ── Skill level ─────────────────────────────────────
        s = m.overall_score
        if s >= 88:
            p.skill_level, p.skill_label = "professional", "professional-level singer"
        elif s >= 75:
            p.skill_level, p.skill_label = "advanced", "advanced singer"
        elif s >= 60:
            p.skill_level, p.skill_label = "intermediate", "intermediate singer"
        elif s >= 40:
            p.skill_level, p.skill_label = "developing", "developing singer"
        else:
            p.skill_level, p.skill_label = "novice", "beginner singer"

        # ── Mean note & range ────────────────────────────────
        if m.f0_mean > 0:
            midi_mean = librosa.hz_to_midi(m.f0_mean)
            p.mean_note = self._midi_name(int(round(midi_mean)))
        r = m.f0_range_semitones
        if r < 5:
            p.range_desc = f"very narrow ({r:.0f} semitones, less than a 4th)"
        elif r < 8:
            p.range_desc = f"about a 5th–6th ({r:.0f} semitones)"
        elif r < 13:
            p.range_desc = f"just under an octave ({r:.0f} semitones)"
        elif r < 16:
            p.range_desc = f"just over an octave ({r:.0f} semitones)"
        else:
            p.range_desc = f"a wide {r:.0f} semitones ({r/12:.1f} octaves)"

        # ── Intonation pattern ───────────────────────────────
        # Look at phrase contour slopes to tell if drift is flat, sharp, or random
        phrase_stats = deep.get("phrase", {}).get("phrase_stats", [])
        if phrase_stats:
            slopes = [ps["pitch_slope"] for ps in phrase_stats]
            mean_slope = float(np.mean(slopes))
            if mean_slope < -0.06:
                p.intonation_pattern = "consistently flat (especially toward phrase ends)"
            elif mean_slope > 0.06:
                p.intonation_pattern = "consistently sharp (pushing pitch up)"
            elif m.pitch_stability_score < 55:
                p.intonation_pattern = "unstable and fluctuating rather than drifting in one direction"
            else:
                p.intonation_pattern = "mostly centred with occasional drift"
        else:
            p.intonation_pattern = "variable"

        # ── Worst pitch moments ──────────────────────────────
        p.worst_pitch_moments = m.ref_worst_moments[:4] if m.ref_worst_moments else []

        # ── Vibrato deltas from ideal ────────────────────────
        ideal_rate_mid = sum(p.vib_ideal_rate_range) / 2
        p.vib_rate_delta = m.vibrato_rate_hz - ideal_rate_mid
        p.vib_extent_delta = m.vibrato_extent_semitones - 0.7  # ideal centre

        # ── Breath ──────────────────────────────────────────
        avg = m.avg_phrase_length_s
        if avg < 2:
            p.phrase_len_grade = "very short"
        elif avg < 3.5:
            p.phrase_len_grade = "shorter than ideal"
        elif avg < 6:
            p.phrase_len_grade = "adequate"
        else:
            p.phrase_len_grade = "strong"

        end_flat = deep.get("phrase", {}).get("end_flat_ratio", 0)
        supp_decay = deep.get("support", {}).get("end_decay_ratio", 0)
        if end_flat > 0.4 or supp_decay > 0.3:
            p.breath_pattern = "collapses toward phrase ends"
        elif m.breath_support_score < 65:
            p.breath_pattern = "inconsistent sub-glottal pressure"
        else:
            p.breath_pattern = "reasonably consistent"

        # ── Dominant tension source ──────────────────────────
        tension = deep.get("tension", {})
        tension_vals = {
            "jaw":    tension.get("jaw_tension", 0),
            "throat": tension.get("throat_constriction", 0),
            "tongue": tension.get("tongue_tension", 0),
            "neck":   tension.get("neck_tension", 0),
        }
        highest = max(tension_vals, key=tension_vals.get)
        if tension_vals[highest] > 50:
            p.dominant_tension = highest

        # ── Interval trouble ─────────────────────────────────
        intervals = deep.get("intervals", {})
        p.worst_interval = intervals.get("worst_interval", "")
        worst_int_data = intervals.get("interval_accuracy", {}).get(p.worst_interval, {})
        p.worst_interval_cents = worst_int_data.get("avg_error_cents", 0)

        # ── Cross-linked problems ────────────────────────────
        roots = deep.get("root_causes", [])
        for cause in roots:
            if len(cause["symptoms"]) >= 2:
                p.connected_issues.append((
                    cause["symptoms"][0],
                    cause["symptoms"][1],
                    cause["root"],
                ))

        # ── Strengths & weaknesses ───────────────────────────
        categories = {
            "Intonation":  m.intonation_score,
            "Vibrato":     m.vibrato_score if m.vibrato_present else 0,
            "Breath":      m.breath_support_score,
            "Resonance":   m.resonance_score,
            "Dynamics":    m.dynamic_control_score,
            "Stability":   m.pitch_stability_score,
            "Passaggio":   m.smoothness_through_break,
        }
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        p.top_strengths  = [c for c, v in sorted_cats if v >= 70][:2]
        p.top_weaknesses = [c for c, v in sorted_cats[::-1] if v < 65][:3]
        p.single_priority = p.top_weaknesses[0] if p.top_weaknesses else "expression"

        # ── Opening praise line ──────────────────────────────
        if p.top_strengths:
            strength_str = " and ".join(p.top_strengths).lower()
            p.opening_praise = (
                f"Your {strength_str} "
                + ("stand out as clear strengths." if len(p.top_strengths) > 1 else "stands out as a clear strength.")
            )
        elif m.overall_score >= 55:
            p.opening_praise = "You have a solid foundation to build on."
        else:
            p.opening_praise = "Every expert was once a beginner. These tips will give you a clear path forward."

        return p

    def _midi_name(self, midi: int) -> str:
        return f"{self.NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


# ─────────────────────────────────────────────────────────────
# TIP ENGINE
# ─────────────────────────────────────────────────────────────

class TipEngine:
    """
    Generates detailed, prioritized, personalised coaching tips.
    Uses VocalMetrics, deep analysis dict, and SingerProfile so every
    tip names exact measurements, references the singer's voice type,
    and reads as written specifically for them.
    """

    def __init__(self):
        self.personalizer = PersonalizationEngine()

    # ── praise pool ──────────────────────────────────────────
    # Every tip gets one of these appended so the feedback never feels cold.
    # They rotate so the user sees a different one each time.

    PRAISE_POOL = [
        "You're doing great by even taking the time to analyse and improve. Most singers never do.",
        "Small consistent steps like this are exactly how real progress happens. Keep at it.",
        "The fact that you're paying attention to this detail already puts you ahead of most singers.",
        "Everyone starts somewhere. You're heading in exactly the right direction.",
        "Honestly, just showing up and working on this stuff is half the battle. Well done.",
        "This is the kind of thing that separates good singers from great ones. You're on the right track.",
        "Progress in singing is rarely overnight, but it IS consistent. You're building something real here.",
        "Each session you do this, your ear and your voice get a little more connected. That compounds over time.",
        "The singers you admire went through these exact same drills. You're on the same path.",
        "Be patient with yourself. This stuff takes repetition, not perfection.",
        "Noticing a problem is literally step one to fixing it. You've already done the hard part.",
        "Your voice is an instrument that gets better the more carefully you listen to it. You're listening.",
        "It takes courage to analyse your own voice honestly. That's something worth recognising.",
        "Stick with it. The breakthroughs in singing always come when you least expect them.",
        "You're building muscle memory right now. Your future self is going to thank you for this.",
        "Every great singer has a list of things they're still working on. This is just yours.",
        "The best practice is honest practice. That's exactly what you're doing.",
        "This is one of those things that feels tricky now and then one day just clicks. You're close.",
        "Singing improvement is about showing up consistently more than anything else. You're here. That counts.",
        "Trust the process. The exercises here are used by professional singers around the world.",
    ]

    def __init__(self):
        self.personalizer  = PersonalizationEngine()
        self._praise_index = 0

    def _next_praise(self) -> str:
        """Return the next praise line from the pool, cycling through without repeating."""
        line = self.PRAISE_POOL[self._praise_index % len(self.PRAISE_POOL)]
        self._praise_index += 1
        return line

    # ── tip-writing helpers ──────────────────────────────────

    @staticmethod
    def _steps(*steps: str) -> str:
        """Format a numbered exercise step list in a friendly, readable way."""
        lines = ["\n\nHere is how to practise this:"]
        for i, s in enumerate(steps, 1):
            lines.append(f"  {i}. {s}")
        return "\n".join(lines)

    @staticmethod
    def _goal_note(p: "SingerProfile", tip_area: str) -> str:
        goal_map = {
            "pitch":      "Improve pitch accuracy",
            "intonation": "Improve pitch accuracy",
            "vibrato":    "Develop vibrato",
            "breath":     "Improve breath control",
            "range":      "Build vocal range",
            "belt":       "Sing higher and belt better",
            "resonance":  "Improve resonance and tone quality",
            "tension":    "Reduce vocal strain and tension",
            "passaggio":  "Sing higher and belt better",
        }
        matched = goal_map.get(tip_area.lower(), "")
        if matched and matched in p.goals:
            return f"\n\nGood news: this directly addresses one of your own goals, which was \"{matched}\". You picked the right thing to work on."
        return ""

    @staticmethod
    def _progress_note(p: "SingerProfile", area: str) -> str:
        streak = p.area_streaks.get(area, 0)
        if area in p.recurring_issues:
            return f"\n\nHeads up: this has come up in a few of your recent sessions. That actually makes it easier to fix because it tells you exactly what to drill. One consistent week on the exercise below and you will very likely break the pattern."
        if streak >= 3:
            return f"\n\nYou have been improving in {area.lower()} for {streak} sessions in a row. That is a real streak. Keep the momentum going."
        if streak == 2:
            return f"\n\nThis improved since last session too. You are building a streak here. Two in a row and counting."
        if area in p.improved_areas:
            return f"\n\nThis improved since your last session. That is a great sign. One more improvement and you have got a streak going."
        return ""

    @staticmethod
    def _genre_context(p: "SingerProfile", tip_area: str) -> str:
        if not p.genres:
            return ""
        primary = p.genres[0].lower()
        context_map = {
            "vibrato": {
                "classical / opera": "For classical and opera singers, vibrato is expected on almost every sustained note. It is part of the core sound.",
                "pop":               "In pop music, vibrato is usually saved for the end of long notes or emotional high points rather than used all the time.",
                "r&b / soul":        "In R&B and soul, vibrato tends to be expressive and personal. It still needs to feel controlled though, not accidental.",
                "musical theatre":   "In musical theatre it depends on the role. Traditional legit roles use vibrato freely while contemporary styles often favour a straighter tone.",
            },
            "breath": {
                "classical / opera": "Classical and opera singers typically sustain phrases for eight to twelve seconds without a breath. Building that capacity is a huge part of the training.",
                "pop":               "Pop phrasing is more conversational, but running out of air mid-phrase still sounds weak even through a microphone.",
            },
            "resonance": {
                "classical / opera": "Classical singers specifically develop a bright ring in the voice called the singer's formant so the voice carries over an orchestra without amplification.",
                "pop":               "Even in pop with a microphone, good resonance gives your voice a fullness and warmth that makes it much more pleasant to listen to.",
            },
        }
        cat = context_map.get(tip_area.lower(), {})
        for g, note in cat.items():
            if g in primary:
                return f"\n\nFor {p.genres[0]} singers specifically: {note}"
        return ""

    @staticmethod
    def _session_context(p: "SingerProfile") -> str:
        parts = []
        if p.practice_streak >= 7:
            parts.append(f"You are on a {p.practice_streak} day practice streak. That kind of consistency is genuinely rare and it is paying off.")
        elif p.practice_streak >= 3:
            parts.append(f"You have practised {p.practice_streak} days in a row. Keep that going.")
        elif p.practice_streak == 1 and p.session_streak > 1:
            parts.append("Welcome back. Getting back in is the hardest part and you did it.")

        hot_streaks = [(area, n) for area, n in p.area_streaks.items() if n >= 2]
        if hot_streaks:
            area, n = max(hot_streaks, key=lambda x: x[1])
            parts.append(f"Your {area.lower()} has been improving for {n} sessions in a row.")

        return (" " + " ".join(parts)) if parts else ""

    @staticmethod
    def _clean(text: str) -> str:
        """
        Remove dashes used as separators and strip any numeric score references.
        """
        import re
        # em/en dash with surrounding spaces
        text = re.sub(r'\s+[—–]\s+', ', ', text)
        text = re.sub(r'[—–]', ', ', text)
        # hyphen as separator
        text = re.sub(r' - ', ', ', text)
        # remove score annotations like "(score: 72/100)", "(72/100)", "72/100", "score: 72/100"
        text = re.sub(r'\(score:\s*\d+/100\)', '', text)
        text = re.sub(r'\(\d+/100\)', '', text)
        text = re.sub(r'\bscore[:\s]+\d+/100\b', '', text)
        text = re.sub(r'\b\d+/100\b', '', text)
        # remove "Aim for X in Y weeks" score goal lines
        text = re.sub(r'Aim for \d+ in \d+ weeks[^.]*\.', '', text)
        # tidy up any double spaces or comma-space-period left behind
        text = re.sub(r',\s*\.', '.', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\(\s*\)', '', text)
        return text.strip()

    def generate(self, m: VocalMetrics, belt: dict, deep: dict = None,
                 account: "UserAccount" = None) -> list[Tip]:
        tips = []
        deep = deep or {}

        # Build personalisation profile first
        p = self.personalizer.build(m, belt, deep, account=account)

        tension   = deep.get("tension", {})
        formants  = deep.get("formants", {})
        larynx    = deep.get("larynx", {})
        phrase    = deep.get("phrase", {})
        onsets    = deep.get("onsets", {})
        intervals = deep.get("intervals", {})
        support   = deep.get("support", {})
        endings   = deep.get("endings", {})
        roots     = deep.get("root_causes", [])

        vt   = p.voice_type.title() if p.voice_type != "unknown" else "your voice type"
        sk   = p.skill_label
        pri  = p.single_priority.lower()

        # ── PERSONALISED OPENING CARD ──────────────────────────
        strengths_line = f"{p.opening_praise}  " if p.opening_praise else ""
        weaknesses_line = (
            f"Your top priority right now is [bold]{pri}[/bold]. "
            if p.single_priority else ""
        )
        connected_line = ""
        if p.connected_issues:
            a, b, root = p.connected_issues[0]
            connected_line = f"Worth knowing: your {a} and {b} are likely connected. Both trace back to {root.lower()}, so fixing that one thing should help both."

        # Build richer greeting when account data is present
        name_greeting = f"Hey {p.display_name}!" if p.display_name else f"Hi!"
        if p.session_number == 1:
            session_line = " Welcome to your very first VocalIQ session!"
        else:
            session_line = f" Session {p.session_number}." + self._session_context(p)
        genre_line = (
            f" You sing {', '.join(p.genres[:2])}."
            if p.genres else ""
        )
        goal_line = (
            f" Your main goal is: {p.goals[0].lower()}."
            if p.goals else ""
        )
        recurring_line = (
            f"\n\nSomething to watch: {', '.join(p.recurring_issues[:2])} "
            + ("have" if len(p.recurring_issues) > 1 else "has")
            + " come up in your last few sessions. Today is a good day to really focus on that."
            if p.recurring_issues else ""
        )
        # Build streak highlights
        streak_lines = []
        if p.practice_streak >= 2:
            streak_lines.append(f"{p.practice_streak} day practice streak")
        hot = [(a, n) for a, n in p.area_streaks.items() if n >= 2]
        for area, n in sorted(hot, key=lambda x: -x[1])[:2]:
            streak_lines.append(f"{area.lower()} improving {n} sessions in a row")
        streak_highlight = (
            "\n\nStreaks: " + ", ".join(streak_lines) + "."
            if streak_lines else ""
        )
        improved_line = (
            f"\n\nGlowing up since last session: {', '.join(p.improved_areas[:2])}. One more and that is a streak!"
            if p.improved_areas and not any(a in p.area_streaks and p.area_streaks[a] >= 2 for a in p.improved_areas) else ""
        )
        known_issue_line = ""
        for issue in p.known_issues:
            if "flat" in issue.lower() and p.intonation_pattern and "flat" in p.intonation_pattern:
                known_issue_line = f"\n\nYou mentioned \"{issue}\" when you signed up, and we are seeing exactly that today. The exercises below are designed to address this specifically."
                break
            elif "strain" in issue.lower() and (tension.get("overall_tension", 0) > 50 or tension.get("neck_tension", 0) > 50):
                known_issue_line = f"\n\nYou mentioned \"{issue}\" when you signed up. Today's analysis confirms tension is present. The tips below address it directly."
                break

        tips.append(Tip(
            "Your Personal Summary", "info",
            f"{name_greeting} Here's your personalised session report",
            f"{name_greeting}{session_line}{genre_line}{goal_line}\n\n"
            f"What we found today:\n"
            f"  • You sing around {p.mean_note} with a range of {p.range_desc}.\n"
            f"  • Your pitch pattern: {p.intonation_pattern}.\n"
            f"  • Breath: {p.breath_pattern}. Phrase length is {p.phrase_len_grade}.\n"
            f"  • {strengths_line}\n"
            f"  • Top priority right now: {pri}."
            f"{connected_line and chr(10) + chr(10) + connected_line}"
            f"{streak_highlight}{recurring_line}{improved_line}{known_issue_line}",
            f"Before your next practice:\n"
            f"  1. Warm up for 5–10 minutes (humming, lip trills, gentle scales).\n"
            f"  2. Spend 10 minutes specifically on {pri}.\n"
            f"  3. Record yourself singing the same phrase you sang today.\n"
            f"  4. Compare the recording to today's report and listen for the changes.\n"
            + (f"\nPractice tip: You said you practice {p.practice_days} days a week. That is "
               + ("great. Consistency is everything." if p.practice_days >= 4 else
                  "a solid start. Even adding one more day will speed up your progress.")
               if p.practice_days else ""),
        ))

        # ── ROOT CAUSE SUMMARY ──────────────────────────────────
        for cause in roots[:3]:
            if cause["confidence"] >= 0.65:
                symptom_list = ", ".join(cause["symptoms"])
                name_addr = f"{p.display_name}, " if p.display_name else ""
                tips.append(Tip(
                    "Root Cause — Most Important Fix", "critical",
                    f"The #1 thing to fix: {cause['root']}  ({cause['confidence']*100:.0f}% confident)",
                    f"{name_addr}instead of trying to fix everything at once, the analysis found one "
                    f"underlying cause that's driving multiple problems at the same time: "
                    f"{cause['root'].lower()}.\n\n"
                    f"Things being caused by this:\n"
                    + "\n".join(f"  • {s}" for s in cause["symptoms"])
                    + f"\n\nWhat this means in plain English: if you fix this one thing, "
                    f"all the symptoms above will start improving on their own. "
                    f"This is worth prioritising above everything else in your next few sessions.",
                    self._root_cause_exercise(cause["root"]),
                ))

        # ── PITCH / INTONATION ─────────────────────────────────
        problem_notes   = deep.get("problem_notes", [])
        flat_sharp_bias = deep.get("flat_sharp_bias", "centred")

        # Build a specific problem-note callout
        if problem_notes:
            worst = problem_notes[0]
            direction = "flat" if worst["cents_off"] < 0 else "sharp"
            specific_note_line = (
                f"\n\nThe most off-pitch note in this recording was {worst['note_name']} "
                f"at {worst['start_s']:.1f}s, which landed {abs(worst['cents_off']):.0f} cents {direction}. "
                f"That is the specific note to drill first."
            )
            if len(problem_notes) > 1:
                others = ", ".join(
                    f"{n['note_name']} ({abs(n['cents_off']):.0f}c {'flat' if n['cents_off'] < 0 else 'sharp'})"
                    for n in problem_notes[1:4]
                )
                specific_note_line += f" Other problem spots: {others}."
        else:
            specific_note_line = ""

        bias_line = (
            f"\n\nPattern: your voice is {flat_sharp_bias} across sustained notes. "
            + ("Try imagining the pitch slightly higher as you approach each note." if "flat" in flat_sharp_bias
               else "Try releasing more breath pressure so you don't push up." if "sharp" in flat_sharp_bias
               else "")
        ) if flat_sharp_bias not in ("centred", "neutral") else ""

        gn = self._goal_note(p, "pitch")
        pn = self._progress_note(p, "Pitch")
        if m.intonation_score < 60:
            tension_link = (
                f"\n\nAlso: we detected {p.dominant_tension} tension in your body — "
                "relaxing that area often immediately improves pitch accuracy."
                if p.dominant_tension else ""
            )
            breath_link = (
                "\n\nYour breath is collapsing toward the end of phrases — "
                "as air runs out, pitch follows it down. Fixing your breath support "
                "will likely fix a large part of this pitch issue at the same time."
                if "phrase end" in p.breath_pattern else ""
            )
            tips.append(Tip(
                "Pitch Accuracy", "critical",
                f"Your pitch is {p.intonation_pattern}",
                f"Pitch accuracy is what makes a note sound right versus off. Right now your pitch is "
                f"{p.intonation_pattern}, meaning notes are landing in the wrong place or drifting away.\n\n"
                f"Your centre pitch is around {p.mean_note}."
                f"{specific_note_line}{bias_line}"
                f"{breath_link}{tension_link}{gn}{pn}",
                self._steps(
                    "Open a free tuner app on your phone such as GuiteTune or insTuner.",
                    f"Sing {p.mean_note} on a long 'ah' and hold it for 10 full seconds.",
                    "Watch the needle. Keep it in the green zone the whole time. Do not force it. Just breathe and retry.",
                    f"Once that is steady, practise the specific problem notes found above, one at a time.",
                    "Then do a slow 5-note scale. Record it and listen back.",
                )
            ))
        elif m.intonation_score < 80:
            tips.append(Tip(
                "Pitch Accuracy", "warning",
                f"Almost there — small pitch wobble  (score: {m.intonation_score:.0f}/100)",
                f"Good news: your pitch is mostly landing in the right place around {p.mean_note}. "
                f"But it {p.intonation_pattern}, usually on longer notes or toward the end of a phrase.\n\n"
                f"For a {sk} this is a fine-tuning issue. The next level of polish is right here."
                f"{specific_note_line}{bias_line}{gn}{pn}",
                self._steps(
                    "Sing a phrase you know well — then immediately hum the same phrase.",
                    "Humming often centres better because there's less to think about. Notice the difference.",
                    "Now sing it again, but aim for that same relaxed centred feeling from the hum.",
                    "Mark any note you felt go flat and drill just that note with a 'ng' glide (siren) up and down.",
                )
            ))
        else:
            tips.append(Tip(
                "Pitch Accuracy", "praise",
                f"Excellent pitch accuracy!  (score: {m.intonation_score:.0f}/100)",
                f"Your pitch is consistently landing right on the note — centred around {p.mean_note} "
                f"and staying there. This is genuinely hard to develop, and you're doing it really well."
                f"{pn}",
                self._steps(
                    "Keep this sharp by practising over a slightly detuned drone (10 cents sharp).",
                    "Sing your phrase over it. When you remove the drone, your pitch will feel even more precise.",
                )
            ))

        if m.pitch_stability_score < 55:
            tension_cause = f"\n\nThe {p.dominant_tension} tension we also detected is almost certainly making this worse." if p.dominant_tension else ""
            tips.append(Tip(
                "Pitch Stability", "critical",
                f"Your pitch is shaking/wobbling uncontrollably  (score: {m.pitch_stability_score:.0f}/100)",
                f"This is different from vibrato (which is controlled and beautiful). "
                f"What we're seeing is an irregular wobble — the pitch jumps around quickly in a way that sounds "
                f"unstable or 'nervous.'\n\n"
                f"The most common causes are:\n"
                f"  1. Physical tension in the jaw, throat, or neck\n"
                f"  2. Inconsistent air pressure — the breath is uneven\n"
                f"  3. Trying too hard — the voice tightens under pressure"
                f"{tension_cause}",
                self._steps(
                    "Before you sing: do a full yawn. Open your mouth wide and let everything feel heavy and loose.",
                    "Now hum on a single note — jaw completely loose, no forcing.",
                    "Imagine you are blowing a candle flame without blowing it out. Steady, gentle airstream.",
                    "Gradually add more voice while keeping that same loose, steady feeling.",
                    "Record yourself and compare — the wobble should reduce as tension leaves.",
                )
            ))

        # ── VIBRATO ────────────────────────────────────────────
        ideal_lo, ideal_hi = p.vib_ideal_rate_range
        if not m.vibrato_present:
            tension_note = (
                f" The {p.dominant_tension} tension detected is a likely suppressor — releasing it often lets vibrato emerge naturally."
                if p.dominant_tension else ""
            )
            tips.append(Tip(
                "Vibrato", "info",
                f"No vibrato detected — straight tone throughout",
                f"For a {vt}, vibrato is expected on sustained notes to add warmth and carrying power. "
                f"Its absence may be intentional (straight-tone style) or tension-related.{tension_note}",
                f"Exercise for {sk}: 'messa di voce' on {p.mean_note} — swell from pp to ff and back. "
                "Vibrato often emerges on the release. Don't force it; allow it by relaxing the throat."
            ))
        else:
            rate   = m.vibrato_rate_hz
            extent = m.vibrato_extent_semitones

            if rate < ideal_lo:
                delta_str = f"{ideal_lo - rate:.1f} Hz below the {vt} ideal minimum ({ideal_lo} Hz)"
                tips.append(Tip(
                    "Vibrato — Rate", "warning",
                    f"Vibrato rate too slow: {rate:.1f} Hz  ({delta_str})",
                    f"Your vibrato is oscillating at {rate:.1f} Hz — {delta_str}. "
                    f"For a {vt}, this sounds like a 'wobble' rather than controlled vibrato. "
                    "Usually caused by low breath pressure or a heavy (pressed) laryngeal mechanism.",
                    f"Exercise: Staccato on {p.mean_note} at ♩=120, then transition to legato — notice if rate stabilises. "
                    "Think 'appoggio': lean your breath into the note and keep the ribcage expanded."
                ))
            elif rate > ideal_hi:
                delta_str = f"{rate - ideal_hi:.1f} Hz above the {vt} ideal ({ideal_hi} Hz)"
                tension_link = f" This aligns with the {p.dominant_tension} tension detected." if p.dominant_tension else ""
                tips.append(Tip(
                    "Vibrato — Rate", "warning",
                    f"Vibrato rate too fast: {rate:.1f} Hz  ({delta_str})",
                    f"At {rate:.1f} Hz your vibrato is {delta_str}. It sounds edgy or 'bleaty'.{tension_link}",
                    f"Exercise: Open a 'hot potato' space in the back of your mouth. Sing 'ah' on {p.mean_note} "
                    "with the jaw fully dropped and record the difference."
                ))
            else:
                tips.append(Tip(
                    "Vibrato — Rate", "praise",
                    f"Vibrato rate is ideal for a {vt}: {rate:.1f} Hz  (target {ideal_lo}–{ideal_hi} Hz)",
                    f"Your {rate:.1f} Hz vibrato sits perfectly in the sweet spot for your voice type. "
                    f"This is a hallmark of a well-trained {vt}.",
                    "Focus next on consistent onset — bring it in within 0.5s of the note start on every long note."
                ))

            if extent < 0.4:
                tips.append(Tip(
                    "Vibrato — Extent", "warning",
                    f"Vibrato width too narrow: {extent:.2f} semitones  (ideal for {vt}: 0.5–1.0 st)",
                    f"Your vibrato oscillates only {extent:.2f} semitones — about half the ideal width for a {vt}. "
                    "This sounds tight or almost straight-tone despite the oscillation being present.",
                    f"Exercise: Sing 'vee-vee-vee' on {p.mean_note}, then blend into legato — the extent should open up. "
                    "Also try a 'hooty owl' sound to widen the pharyngeal resonator."
                ))
            elif extent > 1.5:
                tips.append(Tip(
                    "Vibrato — Extent", "warning",
                    f"Vibrato width too wide: {extent:.2f} semitones  (ideal for {vt}: 0.5–1.0 st)",
                    f"At {extent:.2f} semitones your vibrato is {extent - 1.0:.2f} st wider than ideal for a {vt}, "
                    "causing pitch to sound unstable to listeners.",
                    "Exercise: Sing into a tuner and watch the needle — keep swings within ±50¢. "
                    "'Ng' consonant helps stabilise the soft palate and narrow the oscillation."
                ))

            if m.vibrato_onset_delay_s > 1.5:
                tips.append(Tip(
                    "Vibrato — Onset Delay", "info",
                    f"Vibrato enters {m.vibrato_onset_delay_s:.1f}s after note start  (ideal: <0.8s for {vt})",
                    f"For a {vt}, vibrato should begin within 0.3–0.8s. Yours waits {m.vibrato_onset_delay_s:.1f}s — "
                    "making it sound like you're 'finding' the vibrato rather than having it ready.",
                    "Exercise: On every long note in your current repertoire, engage breath support at the moment of onset — "
                    "think 'vibrato is already waiting inside the note before I open my mouth.'"
                ))

        # ── BREATH ─────────────────────────────────────────────
        gn_b = self._goal_note(p, "breath")
        pn_b = self._progress_note(p, "Breath")
        if m.breath_support_score < 60:
            pitch_link = (
                "\n\nImportant: this is also likely why your pitch is drifting. "
                "When breath pressure drops, pitch goes flat. Fix the breath, fix the pitch."
                if m.intonation_score < 72 else ""
            )
            vib_link = (
                "\n\nIt's also causing your vibrato to wobble — vibrato needs steady air pressure under it."
                if m.vibrato_present and m.vibrato_rate_hz < 5 else ""
            )
            tips.append(Tip(
                "Breath Support", "critical",
                f"Not enough breath support under your voice  (score: {m.breath_support_score:.0f}/100)",
                f"Think of breath support like the foundation of a building. Without it, everything built on top "
                f"(pitch, tone, vibrato, power) becomes shaky.\n\n"
                f"Right now your breath is {p.breath_pattern}. This means the air pressure "
                f"underneath your voice is uneven — it's either too weak or it drops away too quickly."
                f"{pitch_link}{vib_link}{gn_b}{pn_b}",
                self._steps(
                    "Stand up straight. Place one hand on your belly and one on your side ribs.",
                    "Breathe in slowly for 4 counts. Feel your sides and belly expand outward — like a balloon inflating.",
                    "Now hiss on 'sss' for as long as you can. Keep those ribs expanded — don't let them collapse.",
                    "Do this 3 times, aiming to last 20–30 seconds on the hiss.",
                    "Now sing a short phrase using the same expanded feeling. Notice if your tone feels stronger.",
                    f"Goal: hold {p.mean_note} for 10 seconds without the pitch dropping. Time yourself.",
                )
            ))

        if m.breath_noise_ratio > 0.5:
            style_note = (
                "\n\nNote: if you're going for a breathy pop/R&B style intentionally, "
                "some of this is fine — but it should be a choice, not a limitation."
                if any("r&b" in g.lower() or "pop" in g.lower() for g in p.genres) else ""
            )
            tips.append(Tip(
                "Breathiness — Too Much Air", "warning",
                f"Too much air is escaping with your voice  (breathiness: {m.breath_noise_ratio:.2f})",
                f"When you sing, air should flow through your vocal cords in a controlled way. "
                f"Right now, extra air is leaking out — you can hear it as a 'breathy' or 'airy' quality.\n\n"
                f"This wastes your air faster, reduces how far your voice carries, and makes the "
                f"tone sound softer and less defined than it could be.{style_note}{gn_b}",
                self._steps(
                    f"Hum 'nng' on {p.mean_note}. Feel a gentle buzzing in your nose and lips.",
                    "Now open your mouth slowly into 'ah' while keeping that same buzz feeling.",
                    "The buzzing feeling means your vocal cords are closing cleanly — that's what you want.",
                    "Repeat: 'nng → ah → nng → ah' 10 times on the same pitch.",
                    "Try speaking a sentence in a clear, firm voice — then sing with that same firmness.",
                )
            ))

        if m.avg_phrase_length_s < 3.0 and m.avg_phrase_length_s > 0:
            target = 5.0 if p.skill_level in ("intermediate", "advanced", "professional") else 4.0
            tips.append(Tip(
                "Phrase Length — Running Out of Air", "warning",
                f"Your phrases are short ({m.avg_phrase_length_s:.1f}s average) — you're running out of air",
                f"A 'phrase' is one unbroken stretch of singing between breaths. Yours average "
                f"{m.avg_phrase_length_s:.1f} seconds, which is shorter than ideal.\n\n"
                f"This usually means either:\n"
                f"  • You're not taking in enough air before you start\n"
                f"  • You're burning through the air too fast (tension wastes air)\n"
                f"  • Your posture is restricting your lung capacity{gn_b}",
                self._steps(
                    "Stand against a wall, feet 15 cm forward. Back stays on the wall.",
                    "Take a big 'back breath' — feel your back push into the wall as you inhale.",
                    "Sing your phrase. When you finish, stay in position and inhale the same way again.",
                    "Try to add just 2 more words to the phrase before breathing. Then 2 more.",
                    f"Target: {target:.0f} seconds in one phrase within 2 weeks.",
                )
            ))

        # ── RESONANCE ──────────────────────────────────────────
        if m.singer_formant_strength < 40:
            larynx_link = " Your larynx is sitting high, compressing the pharyngeal resonator." \
                if larynx.get("laryngeal_position") == "high" else ""
            tips.append(Tip(
                "Resonance — Singer's Formant", "warning",
                f"Weak singer's formant for a {vt}  ({m.singer_formant_strength:.0f}/100)",
                f"The 2500–3500 Hz energy cluster is what makes a {vt} ring over a band or orchestra. "
                f"Yours scores {m.singer_formant_strength:.0f}/100.{larynx_link}",
                f"Exercise for {vt}: Scales on 'nay' or 'gee' — feel the buzz forward in your mask (between nose and upper lip). "
                "Alternate between a 'hooty' sound and that bright 'nay' — the mix of both is your target resonance."
            ))

        if m.nasality_score > 65:
            tips.append(Tip(
                "Resonance — Nasality", "warning",
                f"Excess nasality for a {vt}  ({m.nasality_score:.0f}/100)",
                f"Your resonance is routing too much through the nasal cavity. For a {vt} this makes the tone sound "
                f"pinched or honky, and limits the chest/pharyngeal warmth that defines your voice type.",
                "Exercise: Pinch your nose while sustaining 'ah' — if the tone changes significantly, nasality is present. "
                "Practice raising the soft palate: imagine you're just about to yawn while you sing."
            ))

        # ── BELTING ────────────────────────────────────────────
        if belt.get("belting_detected"):
            if m.belting_efficiency_score < 60:
                tips.append(Tip(
                    "Belting", "critical",
                    f"Belt detected above {p.passaggio_note_hi} — strain indicators present  ({m.belting_efficiency_score:.0f}/100)",
                    f"As a {vt}, your passaggio sits around {p.passaggio_note_lo}–{p.passaggio_note_hi}. "
                    "You're sustaining chest-heavy production above that without enough twang/resonance support. "
                    "This pattern risks vocal fatigue and fold trauma over repeated sessions.",
                    f"Exercise: Speak-sing the phrase first at normal speech level. Add a 'bratty witch' quality "
                    "(narrows the epilaryngeal tube — makes belting sustainable). Then remove the exaggeration but keep the placement. "
                    "Never belt when your voice is tired or dry."
                ))
            else:
                tips.append(Tip(
                    "Belting", "praise",
                    f"Efficient belt above {p.passaggio_note_hi} for a {vt}  ({m.belting_efficiency_score:.0f}/100)",
                    f"You're carrying chest resonance above your {vt} passaggio with good efficiency. "
                    "Forward placement is working well.",
                    "Monitor fatigue after high-belt sessions. Balance with soft legit/mix singing — "
                    "your voice needs the contrast for longevity."
                ))

        if belt.get("register_breaks", 0) > 2:
            tips.append(Tip(
                "Register / Passaggio", "critical",
                f"{belt.get('register_breaks')} register breaks through {p.passaggio_note_lo}–{p.passaggio_note_hi}",
                f"Your {vt} passaggio is between {p.passaggio_note_lo} and {p.passaggio_note_hi}. "
                f"You're breaking {belt.get('register_breaks')} times — the laryngeal muscles aren't yet coordinating "
                "a smooth chest-to-head transition.",
                f"Exercise: Pianissimo siren on 'ng' from below {p.passaggio_note_lo} to above {p.passaggio_note_hi} and back. "
                "Any break means you went too loud — restart softer. Do this 10× daily until seamless."
            ))

        # ── DYNAMICS ───────────────────────────────────────────
        if m.dynamic_range_db < 10:
            tips.append(Tip(
                "Dynamics", "warning",
                f"Flat dynamics — only {m.dynamic_range_db:.1f} dB range  (aim for 20+ dB as a {vt})",
                f"Your {m.dynamic_range_db:.1f} dB dynamic range is narrow for a {sk}. "
                "Musical storytelling lives in contrast — without it, even accurate pitch sounds monotonous.",
                "Exercise: Mark your current song with three dynamic levels (p / mf / f). "
                "In practice, exaggerate them 3× beyond what feels natural — you'll likely land at just right when recorded."
            ))

        if m.dynamic_control_score < 60:
            tips.append(Tip(
                "Dynamic Control", "warning",
                f"Abrupt dynamic changes — shaping needs work  ({m.dynamic_control_score:.0f}/100)",
                f"For a {vt} at {sk} level, phrase shaping should feel like drawing a bow across strings — "
                "continuous, intentional. Right now the volume jumps rather than flows.",
                f"Exercise: Messa di voce on {p.mean_note}: pp→ff→pp on one breath. "
                "This is the foundational {vt} dynamic control exercise. 5 repetitions per practice session."
            ))

        # ── ARTICULATION ───────────────────────────────────────
        if m.onset_sharpness < 35:
            tips.append(Tip(
                "Articulation — Attacks", "info",
                f"Soft, scoopy note attacks  ({m.onset_sharpness:.0f}/100)",
                f"Your notes are starting below the target and sliding up. For a {vt} this blurs melodic clarity. "
                + ("At your level this is one of the finer details — and worth the effort." if p.skill_level in ("intermediate", "advanced") else ""),
                f"Exercise: Attack {p.mean_note} with a crisp 'ha' — the pitch must be right on target from frame one. "
                "Use a tuner to verify you're not scooping. Staccato scales at ♩=100."
            ))

        if m.note_duration_consistency < 60:
            tips.append(Tip(
                "Rhythm / Timing", "warning",
                f"Uneven note lengths  ({m.note_duration_consistency:.0f}/100)",
                f"Note durations vary significantly. As a {sk} this suggests either rhythmic insecurity "
                "or uncontrolled rubato rather than intentional expression.",
                "Exercise: Sing at 70% tempo with a metronome. Confirm every beat lands. "
                "Then add intentional rubato back with purpose — not accident."
            ))

        # ── TENSION ────────────────────────────────────────────
        jaw = tension.get("jaw_tension", 0)
        throat = tension.get("throat_constriction", 0)
        tongue = tension.get("tongue_tension", 0)
        neck = tension.get("neck_tension", 0)

        if jaw > 65:
            tips.append(Tip(
                "Tension — Jaw", "critical",
                f"High jaw tension detected (score: {jaw:.0f}/100)",
                "Excess energy in the 1–3 kHz range indicates the jaw and buccal muscles are clenching. "
                "This narrows the oral resonator, brightens the tone artificially, and reduces vowel clarity. "
                "Jaw tension is one of the most common causes of a 'tight' or 'pinched' sound.",
                "Exercise: Massage your jaw joint (TMJ) with two fingers before singing. Practice the "
                "'dropped jaw' on every vowel — the bottom teeth should be 2 finger-widths below the top. "
                "Sing scales on 'mah' with a pencil sideways between your teeth to force the jaw open."
            ))
        elif jaw > 45:
            tips.append(Tip(
                "Tension — Jaw", "warning",
                f"Moderate jaw tension ({jaw:.0f}/100) — may restrict resonance",
                "Some jaw tension is present, limiting vowel space and resonance. "
                "On higher notes especially, the jaw tends to close as pitch rises — fight this instinct.",
                "Exercise: 'Yawn-drop' before every phrase. As pitch rises, consciously open wider — "
                "counterintuitive but essential. Check in a mirror: is your jaw position changing on high notes?"
            ))

        if throat > 65:
            tips.append(Tip(
                "Tension — Throat Constriction", "critical",
                f"Significant throat constriction detected (score: {throat:.0f}/100)",
                "The spectral balance indicates the pharyngeal and supraglottal spaces are narrowed — "
                "your throat is 'squeezing' the sound. This causes a thin, bright, or strained tone, "
                "reduces pitch range, and is a primary risk factor for vocal fatigue and nodules.",
                "Exercise: 'Sigh of relief' — breathe in and release a relaxed, open sigh. Notice how "
                "the throat feels open. Try to maintain that openness while singing. The 'hot potato' "
                "sensation (space at the back of the mouth) is your target. Avoid swallowing mid-phrase."
            ))

        if neck > 70:
            tips.append(Tip(
                "Tension — High-Note Neck Tension", "warning",
                f"Pitch instability increases on high notes (neck tension: {neck:.0f}/100)",
                "Your pitch jitter is significantly higher on upper-register notes, suggesting you're "
                "squeezing or 'reaching' for high pitches rather than supporting through them. "
                "This makes high notes sound strained and unpredictable.",
                "Exercise: Approach high notes from above (portamento downward from above the target). "
                "This neutralises the 'reaching' reflex. Also: practice lip trills all the way through "
                "the top of your range — the lip vibration prevents laryngeal squeezing."
            ))

        if tongue > 65:
            tips.append(Tip(
                "Tension — Tongue", "warning",
                f"Tongue tension affecting tone clarity (score: {tongue:.0f}/100)",
                "Excess sibilance and high-frequency harshness suggest the tongue root is raised or retracted, "
                "or the tongue tip is too tense. This causes consonants to sound hissy and vowels to lose "
                "their warmth.",
                "Exercise: Practice 'ng-ah' transitions slowly — 'ng' forces the tongue dorsum down and "
                "back, then 'ah' opens it forward. Repeat on scales. Also: say 'la-la-la' rapidly to "
                "loosen the tongue tip before singing."
            ))

        # ── LARYNGEAL HEIGHT ────────────────────────────────────
        lp = larynx.get("laryngeal_position", "neutral")
        ls = larynx.get("laryngeal_height_score", 80)
        if lp == "high" and ls < 65:
            tips.append(Tip(
                "Laryngeal Height — Too High", "warning",
                "High larynx detected — compressing your resonant space",
                "A raised larynx shortens the vocal tract, producing a brighter, thinner tone with less "
                "carrying power. It also makes vibrato faster and less controlled, and raises the passaggio. "
                "Common causes: singing anxiously, swallowing reflex on high notes, imitation of pop singers "
                "who work with microphones rather than acoustic projection.",
                "Exercise: Place your finger on your larynx (Adam's apple) as you sing an ascending scale. "
                "It should rise only slightly. If it jumps up, practise the scale while consciously keeping "
                "the larynx low — think of yawning, or of the feeling just before a yawn. "
                "The Italian 'chiaroscuro' technique (bright-dark balance) is the goal."
            ))
        elif lp == "low" and ls < 65:
            tips.append(Tip(
                "Laryngeal Height — Too Low", "info",
                "Larynx sitting unusually low — over-darkened tone",
                "An artificially depressed larynx creates a 'hooty', over-covered tone that lacks brightness "
                "and can sound affected. This is sometimes done intentionally (operatic/classical style) but "
                "in pop, musical theatre, or R&B contexts it sounds unnatural.",
                "Exercise: Sing on 'nay' or 'nyah' — the bright vowel naturally raises the larynx to a "
                "neutral position. Blend between the 'nay' brightness and your normal tone. "
                "Aim for the Italian 'chiaroscuro': both bright AND warm simultaneously."
            ))

        # ── VOWEL FORMATION ─────────────────────────────────────
        spread = formants.get("spread_vowel_ratio", 0)
        covered = formants.get("covered_vowel_ratio", 0)
        vow_score = formants.get("vowel_modification_score", 70)

        if spread > 0.45:
            tips.append(Tip(
                "Vowel Formation — Spread Vowels", "warning",
                f"Spread vowel quality detected ({spread*100:.0f}% of voiced frames)",
                "High F2 formant values indicate you're 'spreading' many vowels — tongue high and forward, "
                "lips stretched. This creates a bright, shouty or 'screamy' quality and restricts the "
                "resonating column. Especially problematic on open vowels ('ah', 'oh') in upper register.",
                "Exercise: Sing 'ah' and feel the back of your throat open (soft palate raised, pharynx wide). "
                "The lips should be slightly rounded, not spread. On high notes, all vowels should 'modify' "
                "toward a rounder, more open shape. Practice 'ah–aw–oh' transitions on ascending scales."
            ))

        if covered > 0.35:
            tips.append(Tip(
                "Vowel Formation — Over-covered Vowels", "warning",
                f"Swallowed/over-covered vowel quality ({covered*100:.0f}% of frames)",
                "Very low F1 values suggest the tongue is bunched up or the jaw is too closed, "
                "causing vowels to sound 'swallowed' or muffled. Lyric clarity suffers significantly.",
                "Exercise: Exaggerate vowel shapes in front of a mirror. 'AH' = wide open jaw. "
                "'EE' = smile but keep space in the back. 'OO' = rounded but not collapsing. "
                "Record spoken vowels, then sung vowels, and compare the clarity."
            ))

        if vow_score < 55 and spread <= 0.45 and covered <= 0.35:
            tips.append(Tip(
                "Vowel Formation", "warning",
                f"Inconsistent vowel formation (score: {vow_score:.0f}/100)",
                "Your vowel formants vary widely, suggesting inconsistent tongue and jaw positioning "
                "across different pitches and vowels. This causes uneven tone quality across phrases.",
                "Exercise: Practice each vowel individually on a sustained pitch: 'ah-eh-ee-oh-oo' on one "
                "breath, focusing on keeping the throat open and only moving what needs to move. "
                "Record and listen for consistency of tone colour."
            ))

        # ── PHRASE CONTOUR & SHAPING ────────────────────────────
        ps = phrase.get("phrase_shaping_score", 75)
        end_flat = phrase.get("end_flat_ratio", 0)
        scoop = phrase.get("scoop_ratio", 0)
        climax_pos = phrase.get("avg_climax_position", 0.7)

        if end_flat > 0.45:
            tips.append(Tip(
                "Phrasing — Endings Going Flat", "critical",
                f"Pitch drops at the end of phrases ({end_flat*100:.0f}% of phrases affected)",
                "More than a third of your phrases end with a downward pitch slide. This is almost always "
                "caused by breath pressure running out — as air depletes, sub-glottal pressure drops, and "
                "pitch follows. It also causes notes to sound like they're 'falling off a cliff' musically.",
                "Exercise: Sing a phrase and, at the LAST note, consciously engage your core (think 'support "
                "through the end'). Imagine the last note is the most important one. Practice holding the "
                "final pitch of each phrase steady for 2 extra beats after you'd normally release it."
            ))

        if scoop > 0.45:
            tips.append(Tip(
                "Phrasing — Scooping on Note Starts", "warning",
                f"Pitch scooping detected on note onsets ({scoop*100:.0f}% of phrases)",
                "Notes are starting below their target pitch and rising up — called 'scooping.' "
                "This sounds stylistically casual or unprepared, and blurs melodic clarity. "
                "It often results from the voice not being 'on the breath' at onset.",
                "Exercise: Practice attacking notes with an 'h' consonant ('ha') at the exact target pitch. "
                "Use a tuner to see if you're starting on the note or below it. "
                "Think of the pitch as already being inside you before you open your mouth."
            ))

        if climax_pos < 0.45:
            tips.append(Tip(
                "Phrasing — Climax Placed Too Early", "info",
                f"Phrases peak too early (avg climax at {climax_pos*100:.0f}% of phrase length)",
                "Musical phrases are most compelling when they build toward a climax around 60–75% of the "
                "way through. If you peak early, the phrase 'dies' before it ends and loses tension.",
                "Exercise: Mark each phrase with its climax note. Make sure you arrive there by building "
                "gradually — don't give everything away in the first bar. Think 'hold back, hold back, NOW.'"
            ))

        if ps < 55:
            tips.append(Tip(
                "Phrasing — Overall Shaping", "warning",
                f"Phrase shaping needs work (score: {ps:.0f}/100)",
                "Multiple phrase shaping issues detected together: scooping, early peaks, flat endings, "
                "and/or flat dynamics. This makes the singing sound more like notes being placed than "
                "music being shaped.",
                "Exercise: Speak the lyric of one phrase dramatically first — no pitch, just inflection. "
                "Then sing it, keeping the same inflectional shape. The spoken version reveals where the "
                "natural musical architecture lives."
            ))

        # ── NOTE ENDINGS ────────────────────────────────────────
        trail_off = endings.get("trail_off_ratio", 0)
        end_score = endings.get("ending_score", 75)

        if trail_off > 0.4:
            tips.append(Tip(
                "Note Endings / Releases", "warning",
                f"Notes trailing off on {trail_off*100:.0f}% of phrase endings",
                "Pitch slides downward at the end of notes, creating an unintentional fall-off effect. "
                "While stylistic fall-offs are valid in some genres (R&B, jazz), random trailing-off "
                "makes the voice sound weak or unsupported.",
                "Exercise: On held notes, imagine a horizontal arrow pointing straight ahead — the pitch "
                "stays level until you consciously decide to release it. Practice 'clean cut' releases: "
                "the note stops without any portamento downward. Use a tuner to confirm."
            ))

        # ── ONSET TYPES ──────────────────────────────────────────
        hard_glottal = onsets.get("hard_glottal_pct", 0)
        breathy_onset = onsets.get("breathy_onset_pct", 0)
        onset_score = onsets.get("onset_type_score", 70)

        if hard_glottal > 40:
            tips.append(Tip(
                "Onset Type — Hard Glottal Attacks", "critical",
                f"Hard glottal attacks on {hard_glottal:.0f}% of note starts",
                "A hard glottal attack (sometimes called a 'glottal stop') is when the vocal folds slam "
                "together before tone begins, causing a clicking 'pop' at the note start. Occasional "
                "hard attacks are stylistic; frequent use causes vocal fold trauma and fatigue over time.",
                "Exercise: Replace hard attacks with an aspirated 'h' onset ('ha', 'ho') temporarily "
                "to reset the habit. Then work toward a 'balanced onset': airflow and fold closure begin "
                "simultaneously. Imagine the note starting from a gentle exhale, not a push."
            ))

        if breathy_onset > 40:
            tips.append(Tip(
                "Onset Type — Breathy Onsets", "info",
                f"Breathy/delayed onsets on {breathy_onset:.0f}% of note starts",
                "Notes are starting with air before tone — the folds are slow to close at onset. "
                "This can be stylistic (pop, R&B breathiness) but if unintentional, it wastes air "
                "and produces an unfocused attack.",
                "Exercise: Practice 'mm' humming into a vowel: 'mm-ah'. The hum gives the folds a "
                "head start in finding closure. Gradually shorten the 'mm' until the attack is clean "
                "and immediate."
            ))

        # ── INTERVAL ACCURACY ───────────────────────────────────
        worst_int = intervals.get("worst_interval", "")
        int_data = intervals.get("interval_accuracy", {})
        leap_acc = intervals.get("leap_accuracy", 80)

        if worst_int and worst_int != "unknown":
            worst_err = int_data.get(worst_int, {}).get("avg_error_cents", 0)
            count = int_data.get(worst_int, {}).get("occurrences", 0)
            if worst_err > 35 and count >= 3:
                ref_songs = {
                    "minor 2nd": "'Joy to the World' (descending half-step)",
                    "major 2nd": "'Happy Birthday' (opening)",
                    "minor 3rd": "'Smoke on the Water' (riff)",
                    "major 3rd": "'When the Saints Go Marching In'",
                    "perfect 4th": "'Here Comes the Bride'",
                    "perfect 5th": "'Twinkle Twinkle Little Star'",
                    "major 6th": "'My Bonnie Lies Over the Ocean'",
                    "octave": "'Somewhere Over the Rainbow'",
                }
                ref = ref_songs.get(worst_int, "a familiar song that features this interval")
                tips.append(Tip(
                    f"Interval Accuracy — {worst_int.title()}",
                    "warning",
                    f"Recurring {worst_int} inaccuracy: avg {worst_err:.0f}¢ off  ({count} occurrences)",
                    f"Your {vt} voice is consistently missing {worst_int}s by {worst_err:.0f}¢. "
                    f"This appeared {count} times — it's a pattern in your ear, not a one-off mistake. "
                    f"As a {sk}, drilling this specific interval will have an immediate impact on your intonation score.",
                    f"Exercise: Use a piano — play the root, sing the {worst_int}, check with a tuner. "
                    f"Anchor this interval by humming {ref} before each practice session. "
                    "Your brain will borrow that reference to calibrate."
                ))

        if leap_acc < 65 and intervals.get("largest_leap_semitones", 0) >= 5:
            tips.append(Tip(
                "Large Leaps", "warning",
                f"Large interval leaps landing inaccurately (accuracy: {leap_acc:.0f}/100)",
                f"Jumps of a 5th or larger are being mis-targeted. Leaps are the hardest interval type "
                "to sing in tune because the larynx must change register or muscle configuration rapidly.",
                "Exercise: Portamento (slow glide) drills — sing the leap slowly as a continuous glide, "
                "then progressively faster until it becomes a clean leap. Always overshoot slightly "
                "in practice so the real performance lands correctly."
            ))

        # ── STAMINA & CONSISTENCY ───────────────────────────────
        stam = support.get("stamina_score", 80)
        flat_rate = support.get("held_note_flat_rate", 0)
        decay = support.get("end_decay_ratio", 0)

        if stam < 60:
            rms_thirds = support.get("rms_thirds", [0, 0, 0])
            decay_pct = abs(support.get("end_decay_ratio", 0)) * 100
            tips.append(Tip(
                "Stamina", "warning",
                f"Vocal energy drops {decay_pct:.0f}% toward the end of this session  ({stam:.0f}/100)",
                f"For a {sk}, energy should stay consistent across a full song. "
                f"Your volume dropped {decay_pct:.0f}% in the final third — pitch and tone quality followed. "
                "This points to insufficient warm-up, accumulated tension, or breath management breaking down under fatigue.",
                "Exercise: Build a 10-minute warm-up into every session before serious singing. "
                "Cool down with 5 minutes of gentle humming after each run-through. "
                "Hydrate with room-temperature water throughout — cold water tightens the folds."
            ))

        if flat_rate > 0.4 and support.get("held_note_total", 0) >= 5:
            tips.append(Tip(
                "Sustained Notes", "warning",
                f"{flat_rate*100:.0f}% of your held notes go flat mid-note  ({support.get('held_note_total', 0)} notes checked)",
                f"This is distinct from phrase-end flat — pitch drops during the held note itself. "
                f"For a {vt} this is a classic appoggio breakdown: the ribcage collapses mid-note and "
                "sub-glottal pressure falls, pulling pitch down with it.",
                f"Exercise: Hold {p.mean_note} with one hand on your ribcage. Feel it stay expanded for the "
                "ENTIRE note — like hugging a barrel. Drill 5s holds, then 8s, then 12s. "
                "Do NOT let the ribcage collapse before you intentionally release."
            ))

        # ── REFERENCE SONG ─────────────────────────────────────
        if m.ref_melody_accuracy >= 0:
            acc = m.ref_melody_accuracy
            dev = m.ref_pitch_deviation_cents
            missed = m.ref_missed_notes_pct
            offset = m.ref_early_late_ms

            if acc < 50:
                tips.append(Tip(
                    "Melody Accuracy (vs. Reference)", "critical",
                    f"Singing a different melody — only {acc:.0f}% of notes match",
                    f"Average deviation of {dev:.0f} cents from the reference melody. More than half your notes "
                    "are landing on the wrong pitch entirely. This suggests you haven't fully memorized the melody "
                    "or are improvising without intending to.",
                    "Exercise: Slow down the reference track to 70% speed (use an app like Amazing Slow Downer). "
                    "Sing each phrase in isolation, matching the reference note-for-note. Then practice without "
                    "the reference. Record yourself and compare phrase by phrase."
                ))
            elif acc < 75:
                tips.append(Tip(
                    "Melody Accuracy (vs. Reference)", "warning",
                    f"Some melody drift — {acc:.0f}% note accuracy, avg {dev:.0f}¢ off",
                    "Most notes are in the right ballpark but with consistent sharp or flat tendencies, "
                    "or occasional wrong note choices. This is common when a melody has chromatic turns or "
                    "large leaps you haven't fully internalised.",
                    "Exercise: Identify the specific phrases where you drift (see 'worst moments' below). "
                    "Isolate those intervals and drill them: sing the interval, play it on piano, sing again. "
                    "Use a tuner during practice to see the exact cent deviation."
                ))
            else:
                tips.append(Tip(
                    "Melody Accuracy (vs. Reference)", "praise",
                    f"Strong melody accuracy! {acc:.0f}% note match",
                    f"You're landing {acc:.0f}% of reference notes correctly with only {dev:.0f}¢ average deviation. "
                    "This shows solid song learning and pitch memory.",
                    "Now focus on stylistic nuance — subtle scoops, ornaments, and phrasing that match the "
                    "original artist's intent or your own artistic interpretation."
                ))

            if missed > 25:
                tips.append(Tip(
                    "Missed Notes (vs. Reference)", "warning",
                    f"{missed:.0f}% of reference notes were not sung",
                    "A significant portion of the melody went unsung — either you went silent when the reference "
                    "had notes, or your breath management ran out mid-phrase.",
                    "Exercise: Print or display the lyrics with each note marked. Go through the song and highlight "
                    "every place you fall silent. Practice those transitions specifically — ensure you have enough "
                    "air to complete each phrase before taking a breath."
                ))

            if abs(offset) > 80:
                direction = "late" if offset > 0 else "early"
                tips.append(Tip(
                    "Rhythmic Timing (vs. Reference)", "warning",
                    f"Consistently singing {direction} ({abs(offset):.0f}ms offset)",
                    f"On average you enter {abs(offset):.0f}ms {direction} compared to the reference. "
                    + ("Singing late often means waiting for the reference instead of anticipating the beat. "
                       if offset > 0 else
                       "Singing early suggests rushing — often caused by anxiety or insufficient breath planning. "),
                    "Exercise: Practice with a metronome at half tempo. Clap the rhythm of the melody before "
                    "singing it. Then sing with the click track and consciously place every note on the beat. "
                    "Record and zoom into the waveform to see your offset visually."
                ))
            elif m.ref_rhythm_accuracy >= 0:
                if m.ref_rhythm_accuracy >= 80:
                    tips.append(Tip(
                        "Rhythmic Timing (vs. Reference)", "praise",
                        f"Excellent rhythmic accuracy ({m.ref_rhythm_accuracy:.0f}/100)",
                        "Your note onsets closely match the reference track's timing. This is a sign of strong "
                        "internal pulse and good song preparation.",
                        "Continue developing this by practicing rubato pieces where you intentionally bend time "
                        "while keeping ensemble awareness."
                    ))

            if m.ref_worst_moments:
                worst_list = "\n".join(f"  • {desc}" for _, desc in m.ref_worst_moments[:5])
                tips.append(Tip(
                    "Specific Problem Moments (vs. Reference)", "info",
                    "Top pitch deviation events detected",
                    f"These moments had the largest deviation from the reference melody:\n{worst_list}",
                    "Drill each of these moments individually. Isolate the 2–3 note run containing the error, "
                    "loop it, and practice until it feels automatic before putting it back in context."
                ))

        # ── OVERALL ────────────────────────────────────────────
        if m.overall_score >= 85:
            strengths_str = " and ".join(p.top_strengths) if p.top_strengths else "multiple areas"
            streak_note = (
                f" You are on a {p.practice_streak} day streak right now."
                if p.practice_streak >= 3 else ""
            )
            tips.append(Tip(
                "Overall", "praise",
                f"Outstanding session for a {vt}",
                f"You are demonstrating professional-level control in {strengths_str}.{streak_note} "
                f"This is what consistent practice builds toward.",
                "Focus next on interpretation and style. The technical foundation is there. "
                f"Find repertoire that challenges your {p.voice_type.lower() if p.voice_type != 'unknown' else 'voice type'} "
                "and start performing it for real audiences."
            ))
        elif m.overall_score >= 70:
            streak_note = (
                f" A {p.practice_streak} day practice streak is clearly working."
                if p.practice_streak >= 3 else ""
            )
            tips.append(Tip(
                "Overall", "praise",
                f"Really solid session for a {vt}",
                f"{p.opening_praise}{streak_note} "
                f"You are making real progress. Your clearest path forward right now is focusing on {pri}.",
                f"Keep showing up. Record yourself singing the same phrase once a week and listen back. "
                f"You will hear your own improvement and that feeling is the best motivation there is."
            ))

        # Sort: critical first, then warning, info, praise — but root-cause always leads
        # ── Deduplication ─────────────────────────────────────
        # Each tip has a natural "key" derived from its category.
        # We keep only the highest-severity tip per key.
        # This guarantees no two tips address the same thing.
        sev_rank = {"critical": 0, "warning": 1, "info": 2, "praise": 3}
        seen_keys: dict[str, Tip] = {}   # key → best tip so far

        for tip in tips:
            # Normalise the key: lower-case category, strip trailing qualifiers
            import re as _re
            key = _re.sub(r'\s+(rate|extent|onset|delay|type|control|accuracy)$',
                          '', tip.category.lower().strip())
            key = _re.sub(r'[^a-z0-9 ]', '', key).strip()
            if key not in seen_keys:
                seen_keys[key] = tip
            else:
                existing = seen_keys[key]
                if sev_rank.get(tip.severity, 9) < sev_rank.get(existing.severity, 9):
                    seen_keys[key] = tip  # replace with higher-severity version

        # Sort: summary first, then by severity
        summary_cats = {"your personal summary", "root cause analysis",
                        "root cause , most important fix"}
        ordered_unique = sorted(
            seen_keys.values(),
            key=lambda t: (
                0 if t.category.lower().strip() in summary_cats else 1,
                sev_rank.get(t.severity, 9),
            )
        )

        # Clean dashes, strip score numbers, append a unique praise line
        self._praise_index = 0
        cleaned = []
        for tip in ordered_unique:
            praise = self._next_praise()
            cleaned.append(Tip(
                category = self._clean(tip.category),
                severity = tip.severity,
                headline = self._clean(tip.headline),
                detail   = self._clean(tip.detail),
                exercise = self._clean(tip.exercise) + f"\n\n{praise}",
            ))
        return cleaned

    def _root_cause_exercise(self, root: str) -> str:
        exercises = {
            "Breath pressure collapse":
                "Daily appoggio drill: stand with your back against a wall, feet 15cm forward. "
                "Inhale and feel your sides and back expand. Hiss on 'sss' for 20 seconds while "
                "keeping that expansion. Gradually extend to 40s. Apply this posture to every phrase.",
            "Laryngeal / muscular tension":
                "Full-body release sequence before every practice: roll down the spine, shake out hands, "
                "yawn 5 times slowly, massage jaw, blow through loose lips. Then hum a scale pianissimo "
                "with no effort — if any note feels pushed, restart softer.",
            "Resonance space / vowel formation":
                "Chiaroscuro exercise: sing 'oo' very round and dark, then 'ee' very bright — feel "
                "both extremes. Now find the midpoint where both colours coexist simultaneously. "
                "This is your target resonance. Practice all vowels finding this balance.",
            "Register / passaggio coordination":
                "Pianissimo siren: 'ng' from your lowest note to your highest and back, with ZERO "
                "break — if you feel a flip, go softer. Do this 10 times daily until the transition "
                "is seamless. Then repeat with 'mum', then 'wee-oh', before applying to repertoire.",
            "Ear training / pitch memory":
                "Daily interval training: use an app (Functional Ear Trainer, TonedEar) for 10 minutes. "
                "Also: sing along to a well-tuned instrument (piano or guitar) and match every pitch. "
                "Record yourself and compare with the instrument track to locate drift patterns.",
            "Phrasing / musical shaping":
                "Conduct your own singing: use your free arm to physically conduct phrase shapes — "
                "swell up to the climax, taper down at the end. The body gesture drives the musical "
                "intent. Record and listen: can you hear the shape you drew in the air?",
        }
        return exercises.get(root,
            "Work with a vocal coach to address this root cause with targeted exercises "
            "specific to your voice type and repertoire.")


# ─────────────────────────────────────────────────────────────
# MAIN ANALYZER ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# AUDIO PREPROCESSOR
# ─────────────────────────────────────────────────────────────

class AudioPreprocessor:
    """
    Full cleaning pipeline applied to every input before analysis.

    Steps (in order):
      1. Resample to target_sr
      2. Stereo → mono
      3. DC offset removal
      4. Clipping detection + cubic-interpolation repair
      5. Spectral subtraction noise reduction  ← main upgrade
      6. Adaptive soft noise gate (frame-level gain)
      7. Trim leading/trailing silence
      8. Loudness normalisation to −18 LUFS
      9. Duration check
    """

    CLIP_THRESHOLD = 0.995
    MIN_DURATION_S = 3.0
    TARGET_LUFS    = -18.0
    # Noise estimation: use the quietest N seconds at the start or end
    NOISE_SAMPLE_S = 0.5

    # ── public entry ────────────────────────────────────────

    def process(self, y: np.ndarray, sr: int,
                target_sr: int = 44100) -> tuple[np.ndarray, dict]:
        report: dict = {}

        # 1. Resample
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            report["resampled"] = f"{sr} Hz → {target_sr} Hz"
            sr = target_sr

        # 2. Stereo → mono
        if y.ndim == 2:
            y = y.mean(axis=0)
            report["stereo_collapsed"] = True

        y = y.astype(np.float32)

        # 3. DC offset
        dc = float(y.mean())
        if abs(dc) > 0.005:
            y -= dc
            report["dc_offset_removed"] = f"{dc:.4f}"

        # 4. Clipping repair
        clip_mask  = np.abs(y) >= self.CLIP_THRESHOLD
        clip_ratio = float(clip_mask.mean())
        report["clipping_ratio"] = clip_ratio
        if clip_ratio > 0.001:
            y = self._repair_clipping(y, clip_mask)
            report["clipping_repaired"] = True

        # 5. Estimate noise profile & decide how aggressively to clean
        noise_profile, noise_floor_db = self._estimate_noise_profile(y, sr)
        report["noise_floor_db"] = float(noise_floor_db)

        if noise_floor_db > -50:
            # Spectral subtraction removes stationary background noise
            nr_strength = min(1.0, (noise_floor_db + 60) / 30)   # 0 at -60 dB, 1 at -30 dB
            y = self._spectral_subtract(y, sr, noise_profile, strength=nr_strength)
            report["spectral_subtraction"] = True
            report["nr_strength"] = round(nr_strength, 2)

        if noise_floor_db > -40:
            # Soft noise gate on top for any residual noise bursts
            y = self._soft_gate(y, sr, noise_floor_db)
            report["noise_gated"] = True

        # 6. Re-estimate noise floor after cleaning (for quality reporting)
        _, report["noise_floor_db_after"] = self._estimate_noise_profile(y, sr)

        # 7. Trim silence
        try:
            y_trimmed, trim_idx = librosa.effects.trim(
                y, top_db=35, frame_length=2048, hop_length=256)
            if len(y_trimmed) > sr * self.MIN_DURATION_S:
                y = y_trimmed
                report["trimmed_s"] = round(float(trim_idx[0] / sr), 2)
        except Exception:
            pass

        # 8. Loudness normalise
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms < 1e-7:
            report["silent_input"] = True
        else:
            target_rms  = 10 ** (self.TARGET_LUFS / 20)
            gain        = min(target_rms / rms, 30.0)
            y           = np.clip(y * gain, -1.0, 1.0)
            report["gain_applied_db"] = round(float(20 * np.log10(gain + 1e-9)), 1)

        # 9. Duration
        duration = len(y) / sr
        report["duration_s"] = round(duration, 2)
        if duration < self.MIN_DURATION_S:
            report["too_short"] = True

        return y, report

    # ── noise profile estimation ─────────────────────────────

    def _estimate_noise_profile(self, y: np.ndarray,
                                sr: int) -> tuple[np.ndarray, float]:
        """
        Estimate a per-bin noise power spectrum from the quietest region.
        Strategy:
          - Compute RMS per 512-sample frame.
          - Identify the quietest 10% of frames (these are likely noise-only).
          - Average their magnitude spectra → noise profile.
        Falls back to first NOISE_SAMPLE_S if too little signal variance.
        """
        hop      = 512
        n_fft    = 2048
        rms      = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_db   = librosa.amplitude_to_db(rms + 1e-9)
        floor_db = float(np.percentile(rms_db, 10))

        # Collect quiet frames
        quiet_threshold = floor_db + 6          # up to 6 dB above the floor
        quiet_frames    = np.where(rms_db <= quiet_threshold)[0]

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop)   # (freq, time)

        if len(quiet_frames) >= 5:
            noise_mag = np.abs(D[:, quiet_frames]).mean(axis=1)
        else:
            # Fall back: first NOISE_SAMPLE_S
            n_frames = max(1, int(self.NOISE_SAMPLE_S * sr / hop))
            noise_mag = np.abs(D[:, :n_frames]).mean(axis=1)

        return noise_mag, floor_db

    # ── spectral subtraction ─────────────────────────────────

    def _spectral_subtract(self, y: np.ndarray, sr: int,
                            noise_profile: np.ndarray,
                            strength: float = 0.8) -> np.ndarray:
        """
        Wiener-inspired spectral subtraction:
          1. STFT the signal.
          2. For each frame, subtract scaled noise profile from magnitude.
          3. Half-wave rectify (never go below a small floor to avoid musical noise).
          4. Reconstruct phase via original angles.
          5. ISTFT back to time domain.

        strength (0–1): how aggressively to subtract. 1 = full subtraction.
        """
        hop   = 512
        n_fft = 2048

        D         = librosa.stft(y, n_fft=n_fft, hop_length=hop)
        mag       = np.abs(D)
        phase     = np.angle(D)

        # Scale noise profile to match number of frequency bins
        if len(noise_profile) != mag.shape[0]:
            noise_profile = np.interp(
                np.linspace(0, 1, mag.shape[0]),
                np.linspace(0, 1, len(noise_profile)),
                noise_profile,
            )

        # Over-subtraction factor α: higher = more aggressive, more musical noise risk
        alpha = 1.0 + strength * 1.5

        # Spectral floor β: keeps a small residual to avoid complete silence artefacts
        beta  = 0.05

        noise_col   = noise_profile[:, np.newaxis]          # broadcast over time
        mag_clean   = mag - alpha * noise_col
        mag_floor   = beta * noise_col
        mag_clean   = np.maximum(mag_clean, mag_floor)

        # Smooth across time to reduce musical noise (median filter per-bin)
        # Use a short 3-frame median — cheap and effective
        from scipy.ndimage import median_filter
        mag_clean = median_filter(mag_clean, size=(1, 3))

        D_clean   = mag_clean * np.exp(1j * phase)
        y_clean   = librosa.istft(D_clean, hop_length=hop, length=len(y))
        return y_clean.astype(np.float32)

    # ── soft gate ───────────────────────────────────────────

    def _soft_gate(self, y: np.ndarray, sr: int,
                   noise_floor_db: float) -> np.ndarray:
        """
        Frame-level gain curve that fades out frames close to the noise floor.
        Uses a smooth sigmoid to avoid clicks.
        """
        hop           = 512
        rms           = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_db        = librosa.amplitude_to_db(rms + 1e-9)
        threshold_db  = noise_floor_db + 14.0

        # Sigmoid-shaped gain: 0 at noise floor, 1 at threshold+10
        x    = (rms_db - noise_floor_db) / max(threshold_db - noise_floor_db, 1.0)
        gain = 1.0 / (1.0 + np.exp(-6.0 * (x - 0.5)))
        gain = np.convolve(gain, np.ones(7) / 7, mode='same')  # smooth

        gain_full = np.interp(
            np.arange(len(y)),
            np.arange(len(gain)) * hop + hop // 2,
            gain,
        )
        return (y * gain_full).astype(np.float32)

    # ── clipping repair ─────────────────────────────────────

    def _repair_clipping(self, y: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        idx       = np.arange(len(y))
        unclipped = ~mask
        if unclipped.sum() < 10:
            return y
        return np.interp(idx, idx[unclipped], y[unclipped]).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# RECORDING QUALITY CHECKER
# ─────────────────────────────────────────────────────────────

class RecordingQualityChecker:
    """
    Assesses whether the cleaned audio is good enough to analyse reliably.

    Returns a `decision`:
      "pass"   → analysis proceeds normally
      "warn"   → analysis proceeds with caveats shown to user
      "retry"  → quality is so bad that re-recording is the right call
                 (only meaningful for live mic input; file inputs get "warn" at worst)
    Also returns specific, actionable fix_hints so the user knows exactly what to change.
    """

    # Thresholds
    SNR_RETRY  = 10    # dB  — below this, analysis is essentially random
    SNR_WARN   = 22    # dB  — below this, warn but proceed
    NF_RETRY   = -25   # dB  — noise floor after cleaning still above this = retry
    NF_WARN    = -38   # dB
    CLIP_RETRY = 0.08  # 8 % clipped even after repair = retry
    CLIP_WARN  = 0.01
    DUR_RETRY  = 2.0   # s
    DUR_WARN   = 7.0   # s
    SIL_WARN   = 0.80  # 80% silence

    def check(self, y: np.ndarray, sr: int, prep_report: dict,
              is_live: bool = True) -> dict:
        issues    = []   # things that might warrant a retry
        warnings  = []   # degraded but usable
        fix_hints = []   # how to fix each problem

        rms    = librosa.feature.rms(y=y, hop_length=256)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-9)
        dur    = prep_report.get("duration_s", len(y) / sr)

        # Use post-cleaning noise floor if available
        nf = prep_report.get("noise_floor_db_after",
             prep_report.get("noise_floor_db", -60.0))
        snr_db = float(rms_db.max() - nf)

        # ── Silent / no signal ──────────────────────────────
        if prep_report.get("silent_input"):
            issues.append("No audio signal detected at all.")
            fix_hints.append("Check that your microphone is plugged in, unmuted, and selected as the input device.")

        # ── Duration ────────────────────────────────────────
        if dur < self.DUR_RETRY:
            issues.append(f"Recording is only {dur:.1f}s — too short for reliable analysis.")
            fix_hints.append("Sing for at least 10 seconds to give all analyzers enough data.")
        elif dur < self.DUR_WARN:
            warnings.append(f"Short recording ({dur:.1f}s) — some analyses need more data.")

        # ── Clipping ────────────────────────────────────────
        clip_ratio = prep_report.get("clipping_ratio", 0.0)
        if clip_ratio > self.CLIP_RETRY:
            issues.append(f"Severe clipping: {clip_ratio*100:.1f}% of samples are distorted even after repair.")
            fix_hints.append("Lower your microphone gain by 6–10 dB, or move 10 cm further from the mic.")
        elif clip_ratio > self.CLIP_WARN:
            warnings.append(f"Mild clipping ({clip_ratio*100:.2f}%) — partially repaired.")

        # ── Noise floor / SNR ───────────────────────────────
        if nf > self.NF_RETRY:
            issues.append(f"Noise floor is still very high ({nf:.0f} dB) after cleaning.")
            fix_hints.append("Move to a quieter room, close windows, turn off fans/AC, and position the mic closer to your mouth.")
        elif nf > self.NF_WARN:
            warnings.append(f"Moderate background noise (floor: {nf:.0f} dB after cleaning).")
            fix_hints.append("For best results: close the room, use a directional (cardioid) microphone.")

        if snr_db < self.SNR_RETRY:
            issues.append(f"Signal-to-noise ratio is only {snr_db:.0f} dB — voice is barely above background noise.")
            fix_hints.append("Sing louder or move the mic 10–15 cm closer. Your voice should be clearly audible above any room noise.")
        elif snr_db < self.SNR_WARN:
            warnings.append(f"Moderate SNR ({snr_db:.0f} dB) — pitch and resonance scores may be slightly off.")

        # ── Silence ratio ────────────────────────────────────
        silence_ratio = float((rms_db < rms_db.max() - 35).mean())
        if silence_ratio > self.SIL_WARN:
            warnings.append(f"Recording is {silence_ratio*100:.0f}% silence — the voice isn't being captured consistently.")
            fix_hints.append("Check for mic cutouts. Try singing a continuous phrase rather than single notes with long gaps.")

        # ── Very quiet input ─────────────────────────────────
        gain_db = prep_report.get("gain_applied_db", 0)
        if gain_db > 22:
            warnings.append(f"Input was very quiet — had to boost {gain_db:.0f} dB.")
            fix_hints.append("Increase microphone gain in your system settings, or move closer to the mic.")

        # ── Noise reduction applied? tell the user ───────────
        nr_strength = prep_report.get("nr_strength", 0)
        if nr_strength > 0.5:
            warnings.append(
                f"Heavy noise reduction was applied (strength {nr_strength:.0f}×). "
                "Breath and resonance scores may be slightly affected."
            )

        # ── Decision ────────────────────────────────────────
        if issues and is_live:
            decision = "retry"
        elif issues:
            decision = "warn"
        elif warnings:
            decision = "warn"
        else:
            decision = "pass"

        quality_score = max(0, 100 - len(issues) * 28 - len(warnings) * 7
                            - max(0, self.SNR_WARN - snr_db) * 1.5)

        return {
            "decision":      decision,
            "quality_score": int(quality_score),
            "snr_db":        round(snr_db, 1),
            "noise_floor_db": round(nf, 1),
            "issues":        issues,
            "warnings":      warnings,
            "fix_hints":     fix_hints,
            "duration_s":    dur,
            "silence_ratio": silence_ratio,
            "is_live":       is_live,
        }


# ─────────────────────────────────────────────────────────────
# VOICE TYPE CLASSIFIER & ADAPTIVE CONFIG
# ─────────────────────────────────────────────────────────────

# Each voice type carries its own analysis parameters so every downstream
# analyzer uses thresholds that match the actual voice rather than one-size-fits-all defaults.
VOICE_TYPE_PROFILES = {
    #  name               median_midi  range_midi  passaggio   fmin    fmax   ideal_vibrato_rate
    "soprano":   dict(median_lo=65, median_hi=79, passaggio=(67, 72), fmin=220, fmax=1400, vib_rate=(5.5, 7.5)),
    "mezzo":     dict(median_lo=60, median_hi=68, passaggio=(62, 67), fmin=175, fmax=1100, vib_rate=(5.0, 7.0)),
    "alto":      dict(median_lo=55, median_hi=65, passaggio=(58, 63), fmin=150, fmax=900,  vib_rate=(5.0, 7.0)),
    "tenor":     dict(median_lo=52, median_hi=63, passaggio=(52, 57), fmin=130, fmax=800,  vib_rate=(5.5, 7.5)),
    "baritone":  dict(median_lo=45, median_hi=57, passaggio=(48, 53), fmin=100, fmax=700,  vib_rate=(5.0, 7.0)),
    "bass":      dict(median_lo=36, median_hi=50, passaggio=(43, 48), fmin=73,  fmax=600,  vib_rate=(4.5, 6.5)),
    "unknown":   dict(median_lo=40, median_hi=80, passaggio=(50, 67), fmin=65,  fmax=1400, vib_rate=(5.0, 7.0)),
}


class VoiceTypeClassifier:
    """
    Classifies voice type (soprano/mezzo/alto/tenor/baritone/bass) from
    a first-pass pYIN extraction.  Returns the profile dict so every
    downstream analyzer can use voice-specific thresholds.
    """

    def classify(self, y: np.ndarray, sr: int) -> tuple[str, dict]:
        # Cheap first-pass pitch extraction at low resolution
        f0, voiced, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=512,
            fill_na=np.nan,
        )
        valid = f0[~np.isnan(f0) & (f0 > 0)]
        if len(valid) < 10:
            return "unknown", VOICE_TYPE_PROFILES["unknown"]

        median_midi = float(np.median(librosa.hz_to_midi(valid)))
        range_midi  = float(librosa.hz_to_midi(valid.max()) - librosa.hz_to_midi(valid.min()))

        # Score each voice type by how close the median lands
        best_type  = "unknown"
        best_score = float("inf")
        for vtype, profile in VOICE_TYPE_PROFILES.items():
            if vtype == "unknown":
                continue
            mid_target = (profile["median_lo"] + profile["median_hi"]) / 2
            score = abs(median_midi - mid_target)
            if score < best_score:
                best_score = score
                best_type  = vtype

        profile = VOICE_TYPE_PROFILES[best_type].copy()
        profile["detected_median_midi"] = median_midi
        profile["detected_range_midi"]  = range_midi

        return best_type, profile

    def report(self, voice_type: str, profile: dict) -> str:
        midi_name = lambda m: f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][int(m)%12]}{int(m)//12-1}"
        lo  = midi_name(profile["median_lo"])
        hi  = midi_name(profile["median_hi"])
        p_lo = midi_name(profile["passaggio"][0])
        p_hi = midi_name(profile["passaggio"][1])
        voice_desc = {
            "soprano":  "high female voice",
            "mezzo":    "middle female voice",
            "alto":     "lower female voice",
            "tenor":    "high male voice",
            "baritone": "middle male voice",
            "bass":     "low male voice",
        }.get(voice_type, "")
        desc_str = f"  [dim]({voice_desc})[/dim]" if voice_desc else ""
        return (
            f"Sounds like a [bold cyan]{voice_type.title()}[/bold cyan]{desc_str}  "
            f"[dim]Typical range: {lo} to {hi}. The tricky zone (passaggio) is around {p_lo} to {p_hi}.[/dim]"
        )


class NeedsRetryError(Exception):
    """Raised when recording quality is too poor for reliable analysis."""
    def __init__(self, quality: dict):
        self.quality = quality
        issues_str = " | ".join(quality.get("issues", []))
        super().__init__(f"Recording quality too low: {issues_str}")


class VocalIQ:
    def __init__(self, sr: int = 44100, ref_cache_dir: Optional[str] = None):
        self.sr = sr
        self.preprocessor         = AudioPreprocessor()
        self.quality_checker      = RecordingQualityChecker()
        self.voice_classifier     = VoiceTypeClassifier()
        self.pitch_analyzer       = PitchAnalyzer(sr=sr)
        self.vibrato_analyzer     = VibratoAnalyzer()
        self.breath_analyzer      = BreathAnalyzer()
        self.resonance_analyzer   = ResonanceAnalyzer()
        self.belting_analyzer     = BeltingAnalyzer()
        self.dynamics_analyzer    = DynamicsAnalyzer()
        self.articulation_analyzer = ArticulationAnalyzer()
        self.reference_analyzer   = ReferenceSongAnalyzer(sr=sr, cache_dir=ref_cache_dir)
        self.formant_analyzer     = VowelFormantAnalyzer()
        self.larynx_analyzer      = LaryngealAnalyzer()
        self.onset_analyzer       = OnsetTypeAnalyzer()
        self.phrase_analyzer      = PhraseContourAnalyzer()
        self.tension_analyzer     = TensionAnalyzer()
        self.interval_analyzer    = IntervalAccuracyAnalyzer()
        self.ending_analyzer      = PitchEndingAnalyzer()
        self.support_analyzer     = SupportConsistencyAnalyzer()
        self.root_cause_engine    = RootCauseEngine()
        self.tip_engine           = TipEngine()

    def analyze(self, y: np.ndarray, sr_in: int = None,
                reference_path: Optional[str] = None,
                is_live: bool = False,
                account: "UserAccount" = None,
                on_stage=None) -> tuple[VocalMetrics, list[Tip], dict]:
        """on_stage(description) is called as each analysis step starts (for progress UIs)."""

        sr_in = sr_in or self.sr

        # ── 1. Preprocess ───────────────────────────────────
        console.print(Panel("[bold white]Getting your audio ready...[/bold white]", border_style="bright_black"))
        y, prep_report = self.preprocessor.process(y, sr_in, target_sr=self.sr)

        # ── 2. Quality check ────────────────────────────────
        quality = self.quality_checker.check(y, self.sr, prep_report, is_live=is_live)
        self._print_quality_report(quality, prep_report)

        if quality["decision"] == "retry" and is_live:
            raise NeedsRetryError(quality)
        elif quality["decision"] in ("retry", "warn") and prep_report.get("silent_input"):
            raise ValueError("Silent input — no audio signal detected. Check your microphone.")

        # ── 3. Voice type detection ──────────────────────────
        console.print("[cyan]Working out what kind of voice you have...[/cyan]")
        voice_type, vprofile = self.voice_classifier.classify(y, self.sr)
        console.print(self.voice_classifier.report(voice_type, vprofile))

        # Reconfigure pitch analyzer with voice-specific fmin/fmax
        self.pitch_analyzer.fmin = vprofile["fmin"]
        self.pitch_analyzer.fmax = vprofile["fmax"]

        # Reconfigure belting analyzer passaggio
        self.belting_analyzer.FEMALE_PASSAGGIO_MIDI = tuple(vprofile["passaggio"]) \
            if voice_type in ("soprano", "mezzo", "alto") else self.belting_analyzer.FEMALE_PASSAGGIO_MIDI
        self.belting_analyzer.MALE_PASSAGGIO_MIDI = tuple(vprofile["passaggio"]) \
            if voice_type in ("tenor", "baritone", "bass") else self.belting_analyzer.MALE_PASSAGGIO_MIDI

        console.print()

        with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}")) as p:
            t = p.add_task("Listening to your pitch...")

            def stage(desc: str):
                p.update(t, description=desc)
                if on_stage:
                    on_stage(desc)

            stage("Listening to your pitch...")
            f0, voiced_flag, times = self.pitch_analyzer.extract(y)
            stage("Finding each note...")
            notes = self.pitch_analyzer.extract_notes(f0, times, y)
            stage("Checking vibrato...")
            vib = self.vibrato_analyzer.analyze(f0, times, self.sr, notes=notes)
            stage("Checking breath...")
            breath = self.breath_analyzer.analyze(y, self.sr, f0)
            stage("Checking tone and resonance...")
            res = self.resonance_analyzer.analyze(y, self.sr, f0)
            stage("Checking high notes and registers...")
            belt = self.belting_analyzer.analyze(y, self.sr, f0, times)
            stage("Checking volume and expression...")
            dyn = self.dynamics_analyzer.analyze(y, self.sr)
            stage("Checking how you start and end notes...")
            art = self.articulation_analyzer.analyze(y, self.sr)
            stage("Checking vowel shapes...")
            formants = self.formant_analyzer.analyze(y, self.sr, f0)
            stage("Checking throat position...")
            larynx = self.larynx_analyzer.analyze(y, self.sr, f0)
            stage("Checking how cleanly notes begin...")
            onsets = self.onset_analyzer.analyze(y, self.sr)
            stage("Checking phrase shape...")
            phrase = self.phrase_analyzer.analyze(f0, times, y, self.sr)
            stage("Checking tension...")
            tension = self.tension_analyzer.analyze(y, self.sr, f0)
            stage("Checking interval accuracy...")
            intervals = self.interval_analyzer.analyze(f0)
            stage("Checking note endings...")
            endings = self.ending_analyzer.analyze(f0, times)
            stage("Checking how your energy holds up...")
            support = self.support_analyzer.analyze(y, self.sr, f0)
            stage("Writing your tips...")

        valid_f0 = f0[~np.isnan(f0)]
        voiced_fraction = voiced_flag.mean() if len(voiced_flag) > 0 else 0.0

        m = VocalMetrics()

        # Pitch
        if len(valid_f0) > 0:
            m.f0_mean = float(valid_f0.mean())
            m.f0_std = float(valid_f0.std())
            m.f0_min = float(valid_f0.min())
            m.f0_max = float(valid_f0.max())
        m.f0_range_semitones    = self.pitch_analyzer.semitone_range(f0)
        m.intonation_score      = self.pitch_analyzer.intonation_score(f0, notes)
        m.pitch_stability_score = self.pitch_analyzer.pitch_stability_score(f0, notes)
        m.voiced_fraction       = float(voiced_fraction)

        # Vibrato
        m.vibrato_present = vib["present"]
        m.vibrato_rate_hz = vib["rate_hz"]
        m.vibrato_extent_semitones = vib["extent_semitones"]
        m.vibrato_regularity = vib["regularity"]
        m.vibrato_onset_delay_s = vib["onset_delay_s"]
        m.vibrato_score = vib["score"]

        # Breath
        m.breath_support_score = breath["support_score"]
        m.breath_pressure_consistency = breath["pressure_consistency"]
        m.phrase_lengths_s = breath["phrase_lengths_s"]
        m.avg_phrase_length_s = breath["avg_phrase_length_s"]
        m.breath_noise_ratio = breath["breath_noise_ratio"]
        m.subglottal_pressure_estimate = breath["subglottal_pressure_estimate"]

        # Resonance
        m.resonance_score = res["resonance_score"]
        m.singer_formant_strength = res["singer_formant_strength"]
        m.nasality_score = res["nasality_score"]
        m.brightness_score = res["brightness_score"]

        # Belting
        m.belting_detected = belt["belting_detected"]
        m.belting_chest_mix = belt["belting_chest_mix"]
        m.belting_efficiency_score = belt["belting_efficiency_score"]
        m.chest_voice_range = belt["chest_voice_range"]
        m.head_voice_range = belt["head_voice_range"]
        m.mix_voice_detected = belt["mix_voice_detected"]
        m.passaggio_events = belt["passaggio_events"]
        m.register_breaks = belt["register_breaks"]
        m.smoothness_through_break = belt["smoothness_through_break"]

        # Dynamics
        m.dynamic_range_db = dyn["dynamic_range_db"]
        m.rms_mean_db = dyn["rms_mean_db"]
        m.rms_std_db = dyn["rms_std_db"]
        m.dynamic_control_score = dyn["dynamic_control_score"]
        m.crescendo_detected = dyn["crescendo_detected"]
        m.decrescendo_detected = dyn["decrescendo_detected"]

        # Articulation
        m.onset_sharpness = art["onset_sharpness"]
        m.release_cleanness = art["release_cleanness"]
        m.spectral_flux_mean = art["spectral_flux_mean"]
        m.note_duration_consistency = art["note_duration_consistency"]
        m.rhythmic_accuracy = art["note_duration_consistency"]

        # Reference song comparison
        ref_data = None
        if reference_path:
            console.print()
            console.print(Panel("[bold yellow]Comparing your vocal to the reference song...[/bold yellow]", border_style="yellow"))
            if on_stage:
                on_stage("Comparing you with the original song...")
            ref_data = self.reference_analyzer.analyze(y, reference_path, sr_vocal=self.sr)
            m.ref_melody_accuracy = ref_data["melody_accuracy"]
            m.ref_rhythm_accuracy = ref_data["rhythm_accuracy"]
            m.ref_pitch_deviation_cents = ref_data["pitch_deviation_cents"]
            m.ref_missed_notes_pct = ref_data["missed_notes_pct"]
            m.ref_early_late_ms = ref_data["early_late_ms"]
            m.ref_worst_moments = ref_data["worst_moments"]

        # Overall score — weighted sum (weights total 1.0), blending in
        # reference accuracy when available
        if m.ref_melody_accuracy >= 0:
            m.overall_score = float(np.sum([
                m.intonation_score * 0.14,
                m.vibrato_score * 0.08,
                m.breath_support_score * 0.12,
                m.resonance_score * 0.10,
                m.dynamic_control_score * 0.08,
                m.pitch_stability_score * 0.08,
                m.smoothness_through_break * 0.07,
                m.ref_melody_accuracy * 0.20,
                m.ref_rhythm_accuracy * 0.13,
            ]))
        else:
            m.overall_score = float(np.sum([
                m.intonation_score * 0.20,
                m.vibrato_score * 0.12,
                m.breath_support_score * 0.18,
                m.resonance_score * 0.15,
                m.dynamic_control_score * 0.12,
                m.pitch_stability_score * 0.12,
                m.smoothness_through_break * 0.11,
            ]))

        flat_sharp_bias = self.pitch_analyzer.flat_sharp_bias(notes)
        problem_notes   = self.pitch_analyzer.find_problem_notes(notes)

        deep = {
            "formants":        formants,
            "larynx":          larynx,
            "onsets":          onsets,
            "phrase":          phrase,
            "tension":         tension,
            "intervals":       intervals,
            "endings":         endings,
            "support":         support,
            "breath_detail":   breath,
            "voice_type":      voice_type,
            "voice_profile":   vprofile,
            "quality":         quality,
            "prep":            prep_report,
            "notes":           notes,
            "flat_sharp_bias": flat_sharp_bias,
            "problem_notes":   problem_notes,
        }
        deep["root_causes"] = self.root_cause_engine.analyze(m, deep)

        tips = self.tip_engine.generate(m, belt, deep, account=account)

        # Prepend recording quality warnings as tips
        for issue in quality.get("issues", []):
            tips.insert(0, Tip("Recording Quality", "critical",
                               "Recording quality issue — analysis accuracy reduced",
                               issue, "Re-record in a quieter environment with the mic 15–30cm from your mouth."))
        for warn in quality.get("warnings", []):
            tips.append(Tip("Recording Quality", "info",
                            "Recording quality note", warn,
                            "For best results: quiet room, condenser or dynamic mic, gain set so peaks reach -12 dB."))

        extra = {"f0": f0, "times": times, "vib": vib, "voiced_flag": voiced_flag, "notes": notes,
                 "y": y, "ref_data": ref_data, "belt": belt, "deep": deep,
                 "voice_type": voice_type, "vprofile": vprofile, "quality": quality}
        return m, tips, extra

    def _print_quality_report(self, quality: dict, prep: dict):
        decision = quality.get("decision", "pass")
        dur      = quality["duration_s"]

        status_text = {
            "pass":  "[bright_green]Audio sounds good[/bright_green]",
            "warn":  "[yellow]Audio is a bit noisy but we can work with it[/yellow]",
            "retry": "[red]The audio is too noisy to get good results[/red]",
        }.get(decision, "Checking audio...")

        lines = [f"{status_text}   {dur:.1f}s recorded"]

        # Plain-language cleanup summary
        cleaned = []
        if prep.get("spectral_subtraction"):
            cleaned.append("background noise reduced")
        if prep.get("clipping_repaired"):
            cleaned.append("distortion repaired")
        if prep.get("gain_applied_db", 0) > 4:
            cleaned.append("volume boosted")
        if prep.get("trimmed_s"):
            cleaned.append("silence trimmed from the start")
        if cleaned:
            lines.append(f"  [dim]We cleaned it up: {', '.join(cleaned)}.[/dim]")

        for issue in quality.get("issues", []):
            lines.append(f"  [red]Problem: {issue}[/red]")
        for warn in quality.get("warnings", []):
            lines.append(f"  [yellow]Note: {warn}[/yellow]")
        for hint in quality.get("fix_hints", []):
            lines.append(f"  [cyan]Tip: {hint}[/cyan]")

        border = {"pass": "green", "warn": "yellow", "retry": "red"}.get(decision, "white")
        console.print(Panel("\n".join(lines), title="[bold]Recording check[/bold]",
                            border_style=border))


# ─────────────────────────────────────────────────────────────
# EXPORT (for apps and web front-ends)
# ─────────────────────────────────────────────────────────────

def area_scores_from_metrics(m: VocalMetrics) -> dict:
    """Per-area scores keyed by the area names the tip engine uses for streaks."""
    return {
        "Pitch":      m.intonation_score,
        "Breath":     m.breath_support_score,
        "Vibrato":    m.vibrato_score,
        "Resonance":  m.resonance_score,
        "Dynamics":   m.dynamic_control_score,
        "Stability":  m.pitch_stability_score,
        "Passaggio":  m.smoothness_through_break,
    }


def root_cause_issues(extra: dict, min_confidence: float = 0.6) -> list[str]:
    """Names of the confident root causes for a session (used to spot recurring issues)."""
    return [c["root"] for c in extra.get("deep", {}).get("root_causes", []) if c["confidence"] >= min_confidence]


def export_session_series(m: VocalMetrics, extra: dict, sr: int = 44100, fps: float = 40.0) -> dict:
    """
    Compact, JSON-serialisable time series for drawing a session's charts in a
    browser or mobile app: the pitch contour, detected notes, phrases (breaths),
    loudness, vibrato summary and, when present, the reference-song comparison.
    All times are in seconds on the timeline of `extra["y"]` (the cleaned audio).
    """
    f0, times, y = extra["f0"], extra["times"], extra["y"]
    deep = extra.get("deep", {})
    vprofile = extra.get("vprofile", {}) or {}
    hop_t = float(times[1] - times[0]) if len(times) > 1 else 256 / sr

    step = max(1, int(round((1 / hop_t) / fps)))
    midi = np.full(len(f0), np.nan)
    voiced = ~np.isnan(f0) & (f0 > 0)
    midi[voiced] = librosa.hz_to_midi(f0[voiced])

    rms_hop = 1024
    rms = librosa.feature.rms(y=y, hop_length=rms_hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)

    series = {
        "version": 1,
        "duration_s": round(len(y) / sr, 3),
        "pitch": {"dt": round(hop_t * step, 5), "midi": _nan_to_none(midi[::step])},
        "loudness": {"dt": round(rms_hop / sr, 5), "db": [round(float(v), 1) for v in np.maximum(rms_db, -60)]},
        "notes": [
            {"s": round(n["start_s"], 3), "e": round(n["end_s"], 3), "midi": round(n["median_midi"], 2),
             "target": int(n["target_midi"]), "cents": round(n["cents_off"], 1), "name": n["note_name"]}
            for n in extra.get("notes", [])
        ],
        "phrases": [[round(a, 3), round(b, 3)] for a, b in deep.get("breath_detail", {}).get("phrases", [])],
        "problem_notes": [
            {"s": round(n["start_s"], 3), "e": round(n["end_s"], 3), "name": n["note_name"], "cents": round(n["cents_off"], 1)}
            for n in deep.get("problem_notes", [])
        ],
        "passaggio": list(vprofile.get("passaggio", (50, 67))),
        "voice_type": extra.get("voice_type", "unknown"),
        "vibrato": {
            "present": bool(m.vibrato_present),
            "rate_hz": round(m.vibrato_rate_hz, 2),
            "extent_st": round(m.vibrato_extent_semitones, 3),
            "regularity": round(m.vibrato_regularity, 1),
            "ideal_rate": list(vprofile.get("vib_rate", (5.0, 7.0))),
            "ideal_extent": [0.5, 1.0],
        },
    }

    ref = extra.get("ref_data")
    if ref:
        series["reference"] = {
            "contour": ref["contour"],
            "worst": [{"s": round(a, 2), "e": round(b, 2), "text": d} for a, b, d in ref.get("worst_spans", [])],
            "ref_start_s": round(ref["ref_start_s"], 2),
            "ref_end_s": round(ref["ref_end_s"], 2),
            "octave_shift": ref.get("octave_shift", 0),
        }
    return series


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# VISUAL COACH  — all annotated coaching charts
# ─────────────────────────────────────────────────────────────

BG       = '#0d0d0d'
PANEL_BG = '#141428'
AX_BG    = '#1a1a2e'
C_GOOD   = '#00d4ff'
C_WARN   = '#ffd93d'
C_BAD    = '#ff6b6b'
C_GREEN  = '#6bcb77'
C_PURPLE = '#c77dff'
C_GRAY   = '#aaaaaa'

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')
    ax.tick_params(colors=C_GRAY, labelsize=8)
    ax.grid(alpha=0.12, color='white', linewidth=0.5)
    if title:
        ax.set_title(title, color='white', fontsize=9, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_GRAY, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_GRAY, fontsize=8)


def _midi_yticks(ax, midi_arr):
    """Add note-name Y-axis labels (C3, D4, …) instead of raw MIDI numbers."""
    valid = midi_arr[~np.isnan(midi_arr)]
    if len(valid) == 0:
        return
    lo = int(np.floor(valid.min())) - 1
    hi = int(np.ceil(valid.max())) + 1
    ticks = [m for m in range(lo, hi + 1) if m % 2 == 0]
    labels = [f"{NOTE_NAMES[m % 12]}{m // 12 - 1}" for m in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7, color=C_GRAY)
    ax.set_ylim(lo - 0.5, hi + 0.5)


class VisualCoach:
    """
    Generates five annotated coaching charts saved as PNGs:
      1. vocaliq_pitch_map.png      — colour-coded pitch accuracy + correction arrows
      2. vocaliq_vibrato.png        — vibrato waveform with ideal band + gauge
      3. vocaliq_breath_register.png— phrase map + register colour timeline
      4. vocaliq_dynamics.png       — dynamics curve + shaping guide
      5. vocaliq_overview.png       — radar + score bars + spectrogram
      6. vocaliq_reference.png      — reference overlay + cent-deviation heatmap (if ref)
    """

    HOP = 256

    def __init__(self, m: VocalMetrics, extra: dict, sr: int):
        self.m   = m
        self.ex  = extra
        self.sr  = sr
        self.y   = extra["y"]
        self.f0  = extra["f0"]
        self.times = extra["times"]
        self.vib = extra["vib"]

    # ── 1. Pitch accuracy map ───────────────────────────────

    def chart_pitch(self):
        f0, times, m = self.f0, self.times, self.m
        valid = ~np.isnan(f0) & (f0 > 0)
        midi_all = np.full(len(f0), np.nan)
        midi_all[valid] = librosa.hz_to_midi(f0[valid])

        # Cents deviation from nearest semitone
        cents_dev = np.full(len(f0), np.nan)
        cents_dev[valid] = (midi_all[valid] - np.round(midi_all[valid])) * 100

        fig, axes = plt.subplots(3, 1, figsize=(16, 10), facecolor=BG,
                                  gridspec_kw={'height_ratios': [3, 1.2, 0.7], 'hspace': 0.45})
        fig.suptitle("Pitch Accuracy Map", color='white', fontsize=13, fontweight='bold', y=0.97)

        # — Top: colour-coded pitch dots —
        ax = axes[0]
        _style_ax(ax, ylabel='Note')

        # Background semitone grid stripes
        if valid.any():
            lo = int(np.floor(midi_all[valid].min())) - 1
            hi = int(np.ceil(midi_all[valid].max())) + 1
            for midi_n in range(lo, hi + 1):
                color = '#1e1e35' if midi_n % 2 == 0 else '#222240'
                ax.axhspan(midi_n - 0.5, midi_n + 0.5, color=color, linewidth=0)
                ax.axhline(midi_n, color='#2a2a4a', linewidth=0.4, zorder=1)
                label = f"{NOTE_NAMES[midi_n % 12]}{midi_n // 12 - 1}"
                ax.text(-0.005 * times[-1], midi_n, label, ha='right', va='center',
                        color='#6677aa', fontsize=6.5)

        # Colour scatter by deviation severity
        def dev_color(dev):
            a = np.abs(dev)
            if a < 20:  return C_GOOD
            if a < 45:  return C_WARN
            return C_BAD

        colors_pts = np.array([dev_color(d) if not np.isnan(d) else '#444' for d in cents_dev])
        ax.scatter(times[valid], midi_all[valid], s=2.5, c=colors_pts[valid], zorder=3)

        # Correction arrows: at each large-deviation region, draw an arrow pointing toward semitone centre
        labeled_arrow = False
        in_bad = False
        bad_start_idx = 0
        for i in range(len(cents_dev)):
            is_bad = valid[i] and abs(cents_dev[i]) > 45
            if is_bad and not in_bad:
                in_bad = True
                bad_start_idx = i
            elif not is_bad and in_bad:
                in_bad = False
                mid = (bad_start_idx + i) // 2
                if valid[mid] and not np.isnan(cents_dev[mid]):
                    t_mid = times[mid]
                    y_mid = midi_all[mid]
                    target = round(y_mid)
                    direction = 1 if target > y_mid else -1
                    arrow_kw = dict(arrowstyle='->', color=C_WARN,
                                    lw=1.4, mutation_scale=12)
                    ax.annotate('', xy=(t_mid, target - direction * 0.1),
                                 xytext=(t_mid, y_mid),
                                 arrowprops=arrow_kw, zorder=5)
                    label = f"{'go higher' if direction > 0 else 'go lower'}"
                    ax.text(t_mid, (y_mid + target) / 2, label, ha='left',
                            va='center', color=C_WARN, fontsize=6.5,
                            bbox=dict(fc=BG, ec='none', pad=1))
                    labeled_arrow = True

        _midi_yticks(ax, midi_all)
        ax.set_xlim(0, times[-1])
        ax.set_xlabel('Time (s)', color=C_GRAY, fontsize=8)

        # Legend
        from matplotlib.lines import Line2D
        legend_els = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GOOD,  label='In tune (< 20¢)', ms=6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_WARN,  label='Slightly off (20–45¢)', ms=6),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=C_BAD,   label='Off (> 45¢)', ms=6),
        ]
        ax.legend(handles=legend_els, loc='upper right', fontsize=7,
                  facecolor='#111133', labelcolor='white', framealpha=0.8)

        # — Middle: cents deviation bar —
        ax2 = axes[1]
        _style_ax(ax2, title='Cents deviation from target note  (0 = perfectly in tune)',
                  xlabel='Time (s)', ylabel='Cents (¢)')
        ax2.axhline(0,   color='white',  linewidth=0.8, zorder=3)
        ax2.axhspan(-20,  20, color='#003322', alpha=0.4, label='±20¢ "in tune" zone')
        ax2.axhspan(-50, -20, color='#332200', alpha=0.25)
        ax2.axhspan( 20,  50, color='#332200', alpha=0.25)
        ax2.axhspan(-100, -50, color='#330000', alpha=0.25)
        ax2.axhspan(  50,  100, color='#330000', alpha=0.25)

        bar_colors = [dev_color(d) if not np.isnan(d) else '#333' for d in cents_dev]
        ax2.bar(times, np.nan_to_num(cents_dev), width=self.HOP / self.sr,
                color=bar_colors, alpha=0.85, zorder=2)
        ax2.set_xlim(0, times[-1])
        ax2.set_ylim(-105, 105)
        ax2.set_yticks([-100, -50, -20, 0, 20, 50, 100])
        ax2.tick_params(colors=C_GRAY, labelsize=7)

        # Annotation boxes
        mean_dev = float(np.nanmean(np.abs(cents_dev)))
        ax2.text(0.99, 0.88, f"Mean |dev|: {mean_dev:.0f}¢\nIntonation: {m.intonation_score:.0f}/100",
                 transform=ax2.transAxes, ha='right', va='top', color='white',
                 fontsize=8, bbox=dict(fc='#111133', ec='#334', pad=4))

        # — Bottom: legend strip for zones —
        ax3 = axes[2]
        ax3.set_facecolor(BG)
        ax3.axis('off')
        zones = [('< 20¢ — in tune', C_GOOD), ('20–45¢ — slight drift', C_WARN), ('> 45¢ — needs work', C_BAD)]
        for xi, (label, col) in enumerate(zones):
            ax3.add_patch(plt.Rectangle((xi * 0.34 + 0.01, 0.1), 0.02, 0.8,
                                         transform=ax3.transAxes, color=col, clip_on=False))
            ax3.text(xi * 0.34 + 0.045, 0.5, label, transform=ax3.transAxes,
                     va='center', color='white', fontsize=8)

        fig.savefig('vocaliq_pitch_map.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_pitch_map.png[/green]')
        plt.show()

    # ── 2. Vibrato chart ────────────────────────────────────

    def chart_vibrato(self):
        m, vib, times = self.m, self.vib, self.times
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), facecolor=BG,
                                  gridspec_kw={'width_ratios': [3, 1]})
        fig.suptitle('Vibrato Analysis', color='white', fontsize=13, fontweight='bold')

        # — Left: vibrato waveform with ideal-band overlay —
        ax = axes[0]
        _style_ax(ax, title='Pitch oscillation (relative to median, in cents)',
                  xlabel='Time (s)', ylabel='Cents from median pitch')

        if vib.get("f0_cents") is not None and len(vib["f0_cents"]) > 10:
            vc = vib["f0_cents"]
            vt = times[:len(vc)]

            # Ideal vibrato extent band (±70¢ = ±0.7 semitones)
            ax.axhspan(-70, 70, color='#003322', alpha=0.35, label='Ideal extent ±70¢')
            ax.axhspan(-120, -70, color='#330000', alpha=0.2, label='Too wide')
            ax.axhspan( 70,  120, color='#330000', alpha=0.2)
            ax.axhline(0, color='white', linewidth=0.6, linestyle='--')

            ax.plot(vt, vc, color='#aaaacc', linewidth=0.7, alpha=0.6, label='Raw f0 (cents)')
            if vib.get("f0_bp") is not None and len(vib["f0_bp"]) == len(vc):
                ax.plot(vt, vib["f0_bp"], color=C_GOOD, linewidth=1.4, label='Vibrato band (4–9 Hz)')

            ax.set_xlim(0, vt[-1])
            ax.set_ylim(-130, 130)
            ax.set_yticks([-100, -70, -50, 0, 50, 70, 100])
            ax.tick_params(colors=C_GRAY, labelsize=7)

            # Annotate onset delay
            onset = m.vibrato_onset_delay_s
            if onset < vt[-1]:
                ax.axvline(onset, color=C_WARN, linewidth=1.2, linestyle=':', zorder=4)
                ax.text(onset + 0.05, 100, f'Vibrato\nstarts\n{onset:.1f}s', color=C_WARN,
                        fontsize=7, va='top')

            leg = ax.legend(fontsize=7.5, facecolor='#111133', labelcolor='white', framealpha=0.85)

        else:
            ax.text(0.5, 0.5, 'No vibrato detected\n(straight tone or very narrow oscillation)',
                    transform=ax.transAxes, ha='center', va='center',
                    color=C_WARN, fontsize=11)

        # — Right: gauges for rate and extent —
        ax2 = axes[1]
        ax2.set_facecolor(AX_BG)
        ax2.axis('off')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#2a2a4a')

        def gauge(ax, cx, cy, r, value, vmin, vmax, ideal_min, ideal_max,
                  label, unit, fmt='.1f'):
            theta_start = np.deg2rad(210)
            theta_end   = np.deg2rad(-30)
            theta_range = theta_start - theta_end

            # Background arc
            t_bg = np.linspace(theta_end, theta_start, 200)
            ax.plot(cx + r * np.cos(t_bg), cy + r * np.sin(t_bg),
                    color='#2a2a4a', linewidth=10, solid_capstyle='round', zorder=1)

            # Ideal zone arc (green)
            frac_lo = (ideal_min - vmin) / (vmax - vmin)
            frac_hi = (ideal_max - vmin) / (vmax - vmin)
            t_ideal = np.linspace(
                theta_start - frac_lo * theta_range,
                theta_start - frac_hi * theta_range, 100)
            ax.plot(cx + r * np.cos(t_ideal), cy + r * np.sin(t_ideal),
                    color='#005522', linewidth=10, solid_capstyle='butt', zorder=2)

            # Value arc
            frac = np.clip((value - vmin) / (vmax - vmin), 0, 1)
            col = C_GOOD if ideal_min <= value <= ideal_max else C_WARN if abs(value - (ideal_min + ideal_max) / 2) < (ideal_max - ideal_min) else C_BAD
            t_val = np.linspace(theta_start, theta_start - frac * theta_range, 100)
            ax.plot(cx + r * np.cos(t_val), cy + r * np.sin(t_val),
                    color=col, linewidth=6, solid_capstyle='round', zorder=3)

            # Needle
            angle = theta_start - frac * theta_range
            ax.annotate('', xy=(cx + r * 0.72 * np.cos(angle), cy + r * 0.72 * np.sin(angle)),
                        xytext=(cx, cy),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2, mutation_scale=14))

            ax.text(cx, cy - r * 0.55, f'{value:{fmt}} {unit}', ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')
            ax.text(cx, cy - r * 0.78, label, ha='center', va='center', color=C_GRAY, fontsize=8)
            ax.text(cx - r * 0.85, cy - r * 0.55, f'{vmin}', ha='center', color='#555', fontsize=7)
            ax.text(cx + r * 0.85, cy - r * 0.55, f'{vmax}', ha='center', color='#555', fontsize=7)
            ax.text(cx, cy + r * 0.35,
                    f'Ideal: {ideal_min}–{ideal_max} {unit}',
                    ha='center', color=C_GREEN, fontsize=7)

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        gauge(ax2, 0.5, 0.77, 0.22,
              value=m.vibrato_rate_hz, vmin=3, vmax=10,
              ideal_min=5, ideal_max=7,
              label='Vibrato Rate', unit='Hz')

        gauge(ax2, 0.5, 0.25, 0.22,
              value=m.vibrato_extent_semitones, vmin=0, vmax=2,
              ideal_min=0.5, ideal_max=1.0,
              label='Vibrato Extent', unit='st', fmt='.2f')

        fig.savefig('vocaliq_vibrato.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_vibrato.png[/green]')
        plt.show()

    # ── 3. Breath & register map ────────────────────────────

    def chart_breath_register(self, belt_data: dict):
        m, y, sr = self.m, self.y, self.sr
        f0, times = self.f0, self.times

        fig, axes = plt.subplots(3, 1, figsize=(16, 9), facecolor=BG,
                                  gridspec_kw={'height_ratios': [2, 1, 1], 'hspace': 0.5})
        fig.suptitle('Breath & Register Map', color='white', fontsize=13, fontweight='bold')

        # — Top: register colour strip over pitch contour —
        ax = axes[0]
        _style_ax(ax, title='Register Timeline  (colour = voice register)',
                  xlabel='Time (s)', ylabel='Note')

        valid = ~np.isnan(f0) & (f0 > 0)
        midi_all = np.full(len(f0), np.nan)
        midi_all[valid] = librosa.hz_to_midi(f0[valid])

        is_female = belt_data.get('is_female_voice', True)
        pass_lo, pass_hi = (64, 67) if is_female else (50, 53)

        # Colour each point by register
        reg_colors = []
        for mv in midi_all:
            if np.isnan(mv):
                reg_colors.append('#333')
            elif mv < pass_lo:
                reg_colors.append('#ff9f43')   # chest — orange
            elif mv > pass_hi:
                reg_colors.append('#54a0ff')   # head  — blue
            else:
                reg_colors.append('#5f27cd')   # mix   — purple

        ax.scatter(times[valid], midi_all[valid], s=3, c=[reg_colors[i] for i in np.where(valid)[0]], zorder=3)

        # Passaggio band
        ax.axhspan(pass_lo, pass_hi, color='#5f27cd', alpha=0.12, label=f'Passaggio ({pass_lo}–{pass_hi} MIDI)')
        ax.axhline(pass_lo, color='#5f27cd', linewidth=0.7, linestyle='--')
        ax.axhline(pass_hi, color='#5f27cd', linewidth=0.7, linestyle='--')

        # Register break markers
        if belt_data.get('register_breaks', 0) > 0:
            midi_s = midi_all[~np.isnan(midi_all)]
            times_s = times[~np.isnan(midi_all)]
            diffs = np.abs(np.diff(midi_s))
            break_pts = np.where(diffs > 5)[0]
            for bp in break_pts[:6]:
                ax.axvline(times_s[bp], color=C_BAD, linewidth=1.2, alpha=0.7, zorder=4)
                ax.text(times_s[bp] + 0.05, pass_hi + 0.5, 'BREAK', color=C_BAD, fontsize=6.5)

        _midi_yticks(ax, midi_all)
        ax.set_xlim(0, times[-1])

        from matplotlib.patches import Patch
        legend_els = [
            Patch(color='#ff9f43', label='Chest voice'),
            Patch(color='#5f27cd', label='Mix / passaggio'),
            Patch(color='#54a0ff', label='Head / falsetto'),
        ]
        ax.legend(handles=legend_els, loc='upper right', fontsize=7.5,
                  facecolor='#111133', labelcolor='white', framealpha=0.85)

        # — Middle: phrase / breath map —
        ax2 = axes[1]
        _style_ax(ax2, title='Phrase Length Map  (each bar = one phrase, gap = breath)',
                  xlabel='Time (s)', ylabel='')

        rms = librosa.feature.rms(y=y, hop_length=self.HOP)[0]
        rms_t = librosa.times_like(rms, sr=sr, hop_length=self.HOP)
        voiced_mask = ~np.isnan(f0)
        voiced_rms  = np.interp(rms_t, times, voiced_mask.astype(float))
        voiced_binary = voiced_rms > 0.5

        # Build phrases
        phrases = []
        in_p = False
        p_start = 0.0
        for i, v in enumerate(voiced_binary):
            t = rms_t[i]
            if v and not in_p:
                in_p = True; p_start = t
            elif not v and in_p:
                in_p = False
                dur = t - p_start
                if dur > 0.3:
                    phrases.append((p_start, dur))
        if in_p:
            phrases.append((p_start, rms_t[-1] - p_start))

        for pi, (t_start, dur) in enumerate(phrases):
            col = C_GOOD if dur >= 4 else C_WARN if dur >= 2 else C_BAD
            ax2.barh(0, dur, left=t_start, height=0.5, color=col, alpha=0.85, zorder=2)
            if dur > 0.8:
                ax2.text(t_start + dur / 2, 0, f'{dur:.1f}s', ha='center', va='center',
                         color='white', fontsize=7, fontweight='bold')

        # Breath moment labels
        for i in range(1, len(phrases)):
            prev_end = phrases[i - 1][0] + phrases[i - 1][1]
            next_start = phrases[i][0]
            gap_mid = (prev_end + next_start) / 2
            ax2.text(gap_mid, -0.38, '↓breath', ha='center', color='#7788aa', fontsize=6.5)

        ax2.set_xlim(0, times[-1])
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_yticks([])
        ax2.axhline(0, color='#333', linewidth=0.5)

        from matplotlib.patches import Patch as P2
        leg2 = [P2(color=C_GOOD, label='Long (≥4s — great)'),
                P2(color=C_WARN, label='Medium (2–4s)'),
                P2(color=C_BAD,  label='Short (<2s — breath issue)')]
        ax2.legend(handles=leg2, loc='upper right', fontsize=7,
                   facecolor='#111133', labelcolor='white')

        # — Bottom: RMS energy with breathiness overlay —
        ax3 = axes[2]
        _style_ax(ax3, title='Energy & Breathiness',
                  xlabel='Time (s)', ylabel='dB / Breathiness')

        rms_db = librosa.amplitude_to_db(rms + 1e-9)
        ax3.fill_between(rms_t, rms_db, rms_db.min(), alpha=0.45, color=C_GREEN, label='Volume (dB)')
        ax3.plot(rms_t, rms_db, color=C_GREEN, linewidth=0.8)

        # Breathiness indicator: spectral flatness (higher = breathier)
        sfm = librosa.feature.spectral_flatness(y=y, hop_length=self.HOP)[0]
        sfm_t = librosa.times_like(sfm, sr=sr, hop_length=self.HOP)
        sfm_scaled = sfm / (sfm.max() + 1e-9) * (rms_db.max() - rms_db.min()) + rms_db.min()
        ax3.plot(sfm_t, sfm_scaled, color=C_WARN, linewidth=0.8, alpha=0.7, label='Breathiness (higher = breathy)')

        ax3.set_xlim(0, times[-1])
        ax3.legend(fontsize=7, facecolor='#111133', labelcolor='white')

        fig.savefig('vocaliq_breath_register.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_breath_register.png[/green]')
        plt.show()

    # ── 4. Dynamics coaching chart ──────────────────────────

    def chart_dynamics(self):
        m, y, sr = self.m, self.y, self.sr
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), facecolor=BG,
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.45})
        fig.suptitle('Dynamics Coaching', color='white', fontsize=13, fontweight='bold')

        rms = librosa.feature.rms(y=y, hop_length=self.HOP)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-9)
        rms_t  = librosa.times_like(rms, sr=sr, hop_length=self.HOP)

        ax = axes[0]
        _style_ax(ax, title='Volume over time with shaping guide',
                  xlabel='Time (s)', ylabel='Volume (dB)')

        # Dynamic zones
        top = rms_db.max()
        ax.axhspan(top - 6,  top,      color='#1a3300', alpha=0.35, label='Forte zone')
        ax.axhspan(top - 18, top - 6,  color='#1a2200', alpha=0.25, label='Mezzo-forte')
        ax.axhspan(top - 35, top - 18, color='#111a00', alpha=0.20, label='Mezzo-piano')

        ax.fill_between(rms_t, rms_db, rms_db.min(), alpha=0.4, color=C_GREEN)
        ax.plot(rms_t, rms_db, color=C_GREEN, linewidth=1.2, label='Your dynamics')

        # Smooth "ideal phrase shape" guide using a rolling Gaussian envelope
        window = max(1, int(sr / self.HOP * 1.5))
        kernel = signal.windows.gaussian(window * 4 + 1, std=window)
        kernel /= kernel.sum()
        ideal_env = np.convolve(rms_db, kernel, mode='same')
        ax.plot(rms_t, ideal_env, color=C_WARN, linewidth=1.4, linestyle='--',
                label='Smoothed target shape')

        # Spike annotations
        spikes = np.where(np.abs(rms_db - ideal_env) > 8)[0]
        annotated = set()
        for sp in spikes:
            bucket = sp // 20
            if bucket in annotated:
                continue
            annotated.add(bucket)
            diff = rms_db[sp] - ideal_env[sp]
            label = 'too loud' if diff > 0 else 'too soft'
            col   = C_BAD if diff > 0 else C_WARN
            ax.annotate(label,
                        xy=(rms_t[sp], rms_db[sp]),
                        xytext=(rms_t[sp], rms_db[sp] + (5 if diff > 0 else -8)),
                        color=col, fontsize=6.5,
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
                        ha='center')

        ax.set_xlim(0, rms_t[-1])
        ax.legend(fontsize=7.5, facecolor='#111133', labelcolor='white', framealpha=0.85)

        # Stats box
        ax.text(0.01, 0.97,
                f'Dynamic range: {m.dynamic_range_db:.1f} dB\n'
                f'Control score: {m.dynamic_control_score:.0f}/100\n'
                f'{"Crescendo ↑" if m.crescendo_detected else ""}'
                f'{"  Decrescendo ↓" if m.decrescendo_detected else ""}',
                transform=ax.transAxes, va='top', color='white',
                fontsize=8, bbox=dict(fc='#111133', ec='#334', pad=4))

        # — Bottom: dynamic shape recommendation strip —
        ax2 = axes[1]
        ax2.set_facecolor(AX_BG)
        ax2.axis('off')

        tips_txt = []
        if m.dynamic_range_db < 10:
            tips_txt.append('▲ Widen your dynamic range — try starting phrases softer, peaking in the middle')
        if m.dynamic_control_score < 60:
            tips_txt.append('▲ Smooth out sudden volume jumps — shape phrases like a wave, not a staircase')
        if not m.crescendo_detected and not m.decrescendo_detected:
            tips_txt.append('▲ Add intentional crescendo and decrescendo for musical expression')
        if not tips_txt:
            tips_txt.append('✓ Dynamics look well-shaped. Keep using contrast to tell the story of the song.')

        for i, t in enumerate(tips_txt[:2]):
            ax2.text(0.01, 0.75 - i * 0.45, t, transform=ax2.transAxes,
                     color=C_WARN if t.startswith('▲') else C_GREEN,
                     fontsize=8.5, va='top', wrap=True)

        fig.savefig('vocaliq_dynamics.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_dynamics.png[/green]')
        plt.show()

    # ── 5. Overview dashboard ───────────────────────────────

    def chart_overview(self):
        m, y, sr = self.m, self.y, self.sr
        fig = plt.figure(figsize=(18, 11), facecolor=BG)
        fig.suptitle(f'VocalIQ — Overview Dashboard   Overall: {m.overall_score:.0f}/100',
                     color='white', fontsize=14, fontweight='bold', y=0.98)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

        # 5a. Radar
        ax_rad = fig.add_subplot(gs[0, 0], projection='polar')
        ax_rad.set_facecolor(AX_BG)
        cats  = ['Intonation', 'Vibrato', 'Breath', 'Resonance', 'Dynamics', 'Stability', 'Passaggio']
        vals  = [m.intonation_score, m.vibrato_score, m.breath_support_score,
                 m.resonance_score, m.dynamic_control_score, m.pitch_stability_score,
                 m.smoothness_through_break]
        N     = len(cats)
        ang   = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        ang  += ang[:1]
        vp    = vals + vals[:1]
        ax_rad.set_theta_offset(np.pi / 2); ax_rad.set_theta_direction(-1)
        ax_rad.plot(ang, vp, 'o-', lw=2, color=C_GOOD)
        ax_rad.fill(ang, vp, alpha=0.22, color=C_GOOD)
        ax_rad.set_xticks(ang[:-1]); ax_rad.set_xticklabels(cats, color='white', fontsize=7)
        ax_rad.set_ylim(0, 100); ax_rad.set_yticks([25, 50, 75])
        ax_rad.set_yticklabels(['25', '50', '75'], color='#666', fontsize=5)
        ax_rad.grid(color='#444', alpha=0.5)
        ax_rad.set_title('Score Radar', color='white', pad=14, fontsize=9)

        # 5b. Score bars
        ax_bar = fig.add_subplot(gs[0, 1])
        _style_ax(ax_bar, title='Category Scores')
        labels = ['Intonation', 'Vibrato', 'Breath', 'Resonance', 'Dynamics', 'Stability', 'Passaggio']
        bar_cols = [C_GOOD if v >= 75 else C_WARN if v >= 50 else C_BAD for v in vals]
        bars = ax_bar.barh(labels, vals, color=bar_cols, height=0.55)
        ax_bar.set_xlim(0, 110)
        ax_bar.axvline(75, color='#445', linestyle='--', linewidth=0.9)
        ax_bar.text(75.5, -0.7, 'target', color='#666', fontsize=7)
        ax_bar.tick_params(colors=C_GRAY, labelsize=8)
        for bar, v in zip(bars, vals):
            ax_bar.text(v + 1.5, bar.get_y() + bar.get_height() / 2,
                        f'{v:.0f}', va='center', color='white', fontsize=8)

        # 5c. Overall grade circle
        ax_g = fig.add_subplot(gs[0, 2])
        ax_g.set_facecolor(AX_BG); ax_g.axis('off')
        score = m.overall_score
        grade_col = C_GOOD if score >= 75 else C_WARN if score >= 55 else C_BAD
        grade_letter = 'A' if score >= 90 else 'B' if score >= 80 else 'C' if score >= 70 else 'D' if score >= 55 else 'F'

        theta = np.linspace(0, 2 * np.pi, 300)
        ax_g.plot(np.cos(theta), np.sin(theta), color='#2a2a4a', linewidth=15)
        frac = score / 100
        t_arc = np.linspace(np.pi / 2, np.pi / 2 - frac * 2 * np.pi, 300)
        ax_g.plot(np.cos(t_arc), np.sin(t_arc), color=grade_col, linewidth=15, solid_capstyle='round')
        ax_g.text(0, 0.08, f'{score:.0f}', ha='center', va='center',
                  color='white', fontsize=36, fontweight='bold')
        ax_g.text(0, -0.32, 'out of 100', ha='center', color=C_GRAY, fontsize=9)
        ax_g.text(0, -0.62, f'Grade: {grade_letter}', ha='center', color=grade_col, fontsize=14, fontweight='bold')
        ax_g.set_xlim(-1.5, 1.5); ax_g.set_ylim(-1.3, 1.3)
        ax_g.set_title('Overall Score', color='white', fontsize=9)

        # 5d. Spectrogram
        ax_spec = fig.add_subplot(gs[1, :2])
        _style_ax(ax_spec, title='Spectrogram (0–4kHz)', xlabel='Time (s)', ylabel='Hz')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=self.HOP)) + 1e-9, ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=self.HOP, x_axis='time', y_axis='hz',
                                  ax=ax_spec, cmap='magma')
        ax_spec.set_ylim(0, 4000)
        ax_spec.tick_params(colors=C_GRAY, labelsize=7)

        # 5e. Key stats text panel
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.set_facecolor(AX_BG); ax_stats.axis('off')
        ax_stats.set_title('Key Stats', color='white', fontsize=9)

        def stat_line(y_pos, label, value, good):
            col = C_GOOD if good else C_WARN
            ax_stats.text(0.05, y_pos, label, transform=ax_stats.transAxes,
                          color=C_GRAY, fontsize=8, va='center')
            ax_stats.text(0.95, y_pos, value, transform=ax_stats.transAxes,
                          color=col, fontsize=8, va='center', ha='right', fontweight='bold')

        stat_line(0.92, 'Mean pitch',         f'{m.f0_mean:.0f} Hz', True)
        stat_line(0.82, 'Range',              f'{m.f0_range_semitones:.1f} st', m.f0_range_semitones > 8)
        stat_line(0.72, 'Vibrato rate',       f'{m.vibrato_rate_hz:.1f} Hz' if m.vibrato_present else 'none',
                  5.0 <= m.vibrato_rate_hz <= 7.0)
        stat_line(0.62, 'Avg phrase length',  f'{m.avg_phrase_length_s:.1f}s', m.avg_phrase_length_s >= 3.5)
        stat_line(0.52, 'Breathiness',        f'{m.breath_noise_ratio:.2f}', m.breath_noise_ratio < 0.4)
        stat_line(0.42, 'Dynamic range',      f'{m.dynamic_range_db:.1f} dB', m.dynamic_range_db >= 15)
        stat_line(0.32, "Singer's formant",   f'{m.singer_formant_strength:.0f}/100', m.singer_formant_strength >= 55)
        stat_line(0.22, 'Register breaks',    str(m.register_breaks), m.register_breaks == 0)
        if m.ref_melody_accuracy >= 0:
            stat_line(0.12, 'Melody accuracy', f'{m.ref_melody_accuracy:.0f}%', m.ref_melody_accuracy >= 75)
            stat_line(0.04, 'Rhythm accuracy', f'{m.ref_rhythm_accuracy:.0f}/100', m.ref_rhythm_accuracy >= 75)

        fig.savefig('vocaliq_overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_overview.png[/green]')
        plt.show()

    # ── 6. Reference comparison chart ──────────────────────

    def chart_reference(self):
        m, ref_data = self.m, self.ex.get('ref_data')
        if not ref_data:
            return

        contour = ref_data['contour']
        t_al   = np.asarray(contour['t'], dtype=float)
        midi_r = np.array([np.nan if v is None else v for v in contour['reference']], dtype=float)
        midi_v = np.array([np.nan if v is None else v for v in contour['voice']], dtype=float)

        valid_r = ~np.isnan(midi_r)
        valid_v = ~np.isnan(midi_v)
        both    = valid_r & valid_v

        fig, axes = plt.subplots(3, 1, figsize=(16, 11), facecolor=BG,
                                  gridspec_kw={'height_ratios': [3, 1.5, 1], 'hspace': 0.5})
        fig.suptitle('Reference Song Comparison', color='white', fontsize=13, fontweight='bold')

        # — Top: pitch overlay —
        ax = axes[0]
        _style_ax(ax, title='Your Vocal (red) vs. Reference Melody (blue)',
                  xlabel='Time (s)', ylabel='Note')

        ax.scatter(t_al[valid_r], midi_r[valid_r], s=2, c='#6699ff', alpha=0.55, label='Reference')
        ax.scatter(t_al[valid_v], midi_v[valid_v], s=2, c=C_BAD,     alpha=0.65, label='Your vocal')

        # Deviation fill between the two
        if both.any():
            ax.fill_between(t_al[both], midi_r[both], midi_v[both],
                            alpha=0.18, color=C_WARN, label='Deviation gap')

        # Worst-moment markers with labels
        for t_bad, desc in ref_data.get('worst_moments', [])[:6]:
            ax.axvline(t_bad, color=C_WARN, linewidth=1.0, alpha=0.7, zorder=4)
            ax.text(t_bad + 0.05, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 80,
                    desc.split(' at ')[0], color=C_WARN, fontsize=6, rotation=90, va='top')

        _midi_yticks(ax, midi_r)
        ax.set_xlim(0, t_al[-1])
        ax.legend(fontsize=7.5, facecolor='#111133', labelcolor='white', framealpha=0.85, loc='upper right')

        ax.text(0.01, 0.97,
                f'Melody accuracy: {m.ref_melody_accuracy:.0f}%\n'
                f'Avg deviation: {m.ref_pitch_deviation_cents:.0f}¢\n'
                f'Missed notes: {m.ref_missed_notes_pct:.0f}%',
                transform=ax.transAxes, va='top', color='white',
                fontsize=8, bbox=dict(fc='#111133', ec='#334', pad=4))

        # — Middle: cent-deviation heatmap timeline —
        ax2 = axes[1]
        _style_ax(ax2, title='Pitch deviation from reference (¢)  — red = sharp, blue = flat',
                  xlabel='Time (s)', ylabel='Cents (¢)')

        cent_dev = np.full(len(midi_r), np.nan)
        if both.any():
            cent_dev[both] = (midi_v[both] - midi_r[both]) * 100

        # Colour: positive (sharp) = red, negative (flat) = blue
        pos_mask = ~np.isnan(cent_dev) & (cent_dev >= 0)
        neg_mask = ~np.isnan(cent_dev) & (cent_dev < 0)
        ax2.bar(t_al[pos_mask], cent_dev[pos_mask],
                width=self.HOP / self.sr, color=C_BAD,  alpha=0.75, label='Sharp')
        ax2.bar(t_al[neg_mask], cent_dev[neg_mask],
                width=self.HOP / self.sr, color='#54a0ff', alpha=0.75, label='Flat')
        ax2.axhline(0, color='white', linewidth=0.8)
        ax2.axhspan(-20, 20, color='#003322', alpha=0.3, label='±20¢ in-tune zone')
        ax2.set_ylim(-120, 120)
        ax2.set_yticks([-100, -50, -20, 0, 20, 50, 100])
        ax2.tick_params(colors=C_GRAY, labelsize=7)
        ax2.set_xlim(0, t_al[-1])
        ax2.legend(fontsize=7, facecolor='#111133', labelcolor='white', loc='upper right')

        # — Bottom: timing offset —
        ax3 = axes[2]
        _style_ax(ax3, title=f'Timing  |  avg offset: {m.ref_early_late_ms:+.0f}ms  '
                              f'(+ = late, – = early)  |  Rhythm score: {m.ref_rhythm_accuracy:.0f}/100',
                  xlabel='Time (s)', ylabel='')
        ax3.axis('off')
        direction = 'late' if m.ref_early_late_ms > 0 else 'early'
        col = C_WARN if abs(m.ref_early_late_ms) > 80 else C_GOOD
        ax3.text(0.5, 0.6,
                 f'You sing {abs(m.ref_early_late_ms):.0f}ms {direction} on average.',
                 transform=ax3.transAxes, ha='center', va='center', color=col, fontsize=11)
        advice = ('Try anticipating the beat — sing slightly ahead of where you think the note is.'
                  if m.ref_early_late_ms > 80 else
                  'Try holding back slightly — let the backing track lead you.'
                  if m.ref_early_late_ms < -80 else
                  'Timing is tight — great rhythmic feel!')
        ax3.text(0.5, 0.2, advice, transform=ax3.transAxes, ha='center', va='center',
                 color='white', fontsize=9, style='italic')

        fig.savefig('vocaliq_reference.png', dpi=150, bbox_inches='tight', facecolor=BG)
        console.print('[green]✓ Saved vocaliq_reference.png[/green]')
        plt.show()

    # ── public entry: render everything ────────────────────

    def render_all(self, belt_data: dict):
        console.print('\n[bold cyan]Generating coaching charts…[/bold cyan]')
        self.chart_pitch()
        self.chart_vibrato()
        self.chart_breath_register(belt_data)
        self.chart_dynamics()
        self.chart_overview()
        if self.ex.get('ref_data'):
            self.chart_reference()


def plot_results(m: VocalMetrics, extra: dict, sr: int, belt_data: dict):
    VisualCoach(m, extra, sr).render_all(belt_data)


# ─────────────────────────────────────────────────────────────
# RICH CLI DISPLAY
# ─────────────────────────────────────────────────────────────

LEVEL_LABELS = [
    (85, "excellent",    "bright_green", "★★★"),
    (70, "strong",       "green",        "★★☆"),
    (55, "getting there","yellow",       "★☆☆"),
    (40, "developing",   "dark_orange",  "◐☆☆"),
    (0,  "needs work",   "red",          "○☆☆"),
]

def _score_to_level(s: float) -> tuple[str, str, str]:
    for threshold, label, color, stars in LEVEL_LABELS:
        if s >= threshold:
            return label, color, stars
    return "needs work", "red", "○☆☆"


def display_results(m: VocalMetrics, tips: list[Tip], pa: PitchAnalyzer,
                    extra: dict = None, account: "UserAccount" = None):

    voice_type = (extra or {}).get("voice_type", "unknown")
    quality    = (extra or {}).get("quality", {})
    deep       = (extra or {}).get("deep", {})

    # ── Header panel ──────────────────────────────────────────
    console.print()

    # ── Header ────────────────────────────────────────────────
    vt_line = f"Voice: [bold cyan]{voice_type.title()}[/bold cyan]   " if voice_type != "unknown" else ""
    q_level, q_col, _ = _score_to_level(quality.get("quality_score", 100))
    q_line  = f"Recording: [{q_col}]{q_level}[/{q_col}]"
    streak_parts = []
    if account:
        if account.practice_streak >= 2:
            streak_parts.append(f"[yellow]{account.practice_streak} day streak[/yellow]")
        for area, n in sorted((account.area_streaks or {}).items(), key=lambda x: -x[1])[:2]:
            if n >= 2:
                streak_parts.append(f"[green]{area.lower()} on a roll ({n} sessions)[/green]")
    streak_line = "   " + "   ".join(streak_parts) if streak_parts else ""

    console.print(Panel(
        f"{vt_line}{q_line}{streak_line}",
        title="[bold cyan]Done![/bold cyan]",
        border_style="cyan",
    ))

    # ── Streak board ──────────────────────────────────────────
    if account:
        fire = lambda n: "🔥" * min(n, 5) if n >= 2 else ("" if n == 0 else "day 1")
        sb = Table(title="Your Progress", box=box.SIMPLE_HEAD, show_header=True,
                   header_style="bold cyan", border_style="bright_black")
        sb.add_column("What",       style="bold white", min_width=30)
        sb.add_column("Right now",  justify="center",   min_width=24)
        sb.add_column("Where you are", justify="right", min_width=16)

        sb.add_row(
            "Days practised in a row",
            f"[yellow]{account.practice_streak} day{'s' if account.practice_streak != 1 else ''}[/yellow]  "
            f"{fire(account.practice_streak)}  "
            f"[dim](best ever: {account.longest_streak})[/dim]",
            "",
        )
        sb.add_row("Total sessions", f"[cyan]{account.session_count}[/cyan]", "")
        for area, n in sorted((account.area_streaks or {}).items(), key=lambda x: -x[1]):
            level = (account.area_levels or {}).get(area, "")
            lev_label, lev_col, lev_stars = "not yet rated", "dim", ""
            if level:
                for thr, lbl, col, stars in LEVEL_LABELS:
                    if lbl == level:
                        lev_label, lev_col, lev_stars = lbl, col, stars
                        break
            streak_str = (
                f"[green]getting better for {n} sessions[/green]  {fire(n)}"
                if n >= 1 else "[dim]no run yet[/dim]"
            )
            sb.add_row(area, streak_str, f"[{lev_col}]{lev_label}[/{lev_col}]  {lev_stars}")
        console.print(sb)

    # ── Snapshot table ────────────────────────────────────────
    t = Table(title="Quick snapshot of your singing", box=box.ROUNDED, show_header=True,
              header_style="bold cyan", border_style="bright_black")
    t.add_column("Area",   style="bold white", min_width=24)
    t.add_column("What we found", style="white", min_width=40)
    t.add_column("How it is",     justify="right", min_width=16)

    def row(area, detail, score=None):
        if score is not None:
            lbl, col, stars = _score_to_level(score)
            level_str = f"[{col}]{lbl}[/{col}]  {stars}"
        else:
            level_str = ""
        t.add_row(area, detail, level_str)

    row("Pitch",
        f"Your centre note is around {pa.note_name(m.f0_mean)}.  "
        f"You sang across {m.f0_range_semitones:.0f} semitones.")
    row("Are notes landing in tune?",
        "How closely each note hits the right pitch.", m.intonation_score)
    row("Are held notes staying steady?",
        "Does the pitch wobble while you hold a note.", m.pitch_stability_score)
    t.add_section()
    row("Vibrato",
        "Yes, detected" if m.vibrato_present else "Not detected in this recording.")
    if m.vibrato_present:
        rate_ok = "good" if 5 <= m.vibrato_rate_hz <= 7 else ("a bit slow" if m.vibrato_rate_hz < 5 else "a bit fast")
        row("Vibrato speed",
            f"{m.vibrato_rate_hz:.1f} wobbles per second  ({rate_ok}, ideal is 5 to 7).",
            m.vibrato_score)
    t.add_section()
    row("Breath",
        f"Average phrase before a breath: {m.avg_phrase_length_s:.1f} seconds.",
        m.breath_support_score)
    row("Breathiness",
        "How airy your tone is. Less is usually stronger.",
        int((1 - m.breath_noise_ratio) * 100))
    t.add_section()
    row("Tone quality",
        "How full, warm, and carrying your voice sounds.", m.resonance_score)
    row("Nasality",
        "How much sound is coming through your nose. Lower is usually better.",
        int(100 - m.nasality_score))
    t.add_section()
    row("High notes and register",
        f"Register breaks: {m.register_breaks}.  "
        f"Mix voice {'detected' if m.mix_voice_detected else 'not detected'}.",
        m.smoothness_through_break)
    if m.belting_detected:
        row("Belting", "Belting detected above your passaggio.", m.belting_efficiency_score)
    t.add_section()
    row("Loudness and expression",
        f"You used {m.dynamic_range_db:.0f} dB of volume range.", m.dynamic_control_score)
    row("How cleanly notes start",
        "Are attacks crisp or scoopy.", m.onset_sharpness)
    row("Rhythm",
        "How evenly you hold note lengths.", m.note_duration_consistency)

    if m.ref_melody_accuracy >= 0:
        t.add_section()
        row("Matching the song melody",
            f"You matched {m.ref_melody_accuracy:.0f}% of the notes in the reference.",
            m.ref_melody_accuracy)
        dev = m.ref_pitch_deviation_cents
        row("How far off were missed notes",
            f"When notes were off, they were about {dev:.0f} cents away from the right pitch.")
        direction = "behind the beat" if m.ref_early_late_ms > 0 else "ahead of the beat"
        row("Timing vs the song",
            f"You were {abs(m.ref_early_late_ms):.0f} milliseconds {direction} on average.",
            m.ref_rhythm_accuracy)
        if m.ref_worst_moments:
            for _, desc in m.ref_worst_moments[:3]:
                t.add_row("Trickiest spot", desc, "")

    console.print(t)

    # ── Tips ──────────────────────────────────────────────────
    console.print()
    severity_styles = {
        "critical": ("bold red",    "✖"),
        "warning":  ("bold yellow", "⚠"),
        "info":     ("bold cyan",   "ℹ"),
        "praise":   ("bold green",  "★"),
    }

    console.print(Panel(
        "[bold white]Here are your tips.[/bold white]  "
        "[dim]Read them top to bottom — the most important ones come first.[/dim]",
        border_style="magenta",
    ))

    for i, tip in enumerate(tips, 1):
        color, icon = severity_styles.get(tip.severity, ("white", "●"))
        sev_label = {
            "critical": "Fix this first",
            "warning":  "Worth working on",
            "info":     "Good to know",
            "praise":   "Well done",
        }.get(tip.severity, "")
        console.print(Panel(
            f"[{color}]{icon}  {tip.headline}[/{color}]\n\n"
            f"[white]{tip.detail}[/white]\n\n"
            f"[dim cyan]How to practise this:[/dim cyan]\n[italic]{tip.exercise}[/italic]",
            title=f"[bold white]{i}.  {tip.category}[/bold white]"
                  + (f"  [dim]({sev_label})[/dim]" if sev_label else ""),
            border_style=color.replace("bold ", ""),
            padding=(0, 2),
        ))

    console.print()


# ─────────────────────────────────────────────────────────────
# SESSION ARCHIVE
# ─────────────────────────────────────────────────────────────

SESSIONS_DIR = os.path.expanduser("~/.vocaliq/sessions")


class SessionArchive:
    """
    Saves and retrieves past singing sessions for a user.

    Each session is stored as a folder:
        ~/.vocaliq/sessions/<username>/<timestamp>_<label>/
            audio.wav       — the raw cleaned recording
            tips.txt        — plain-text version of all tips
            tips.json       — machine-readable tips
            snapshot.json   — key vocal metrics
            meta.json       — date, voice type, song, streaks, label

    Users can browse sessions, re-read tips from any past session,
    and play back their audio to hear how they've changed over time.
    """

    def __init__(self, username: str):
        self.username = username
        self.base_dir = os.path.join(SESSIONS_DIR, username)
        os.makedirs(self.base_dir, exist_ok=True)

    # ── save ────────────────────────────────────────────────

    def save(self,
             y: np.ndarray,
             sr: int,
             tips: list,
             m: "VocalMetrics",
             extra: dict,
             account: "UserAccount",
             reference_title: str = "") -> str:
        """
        Save a complete session. Returns the session folder path.
        Asks the user for an optional short label (e.g. 'warmup' or 'chorus').
        """
        timestamp = time.strftime("%Y-%m-%d_%H-%M")
        label = self._ask_label()
        folder_name = f"{timestamp}_{label}" if label else timestamp
        session_dir = os.path.join(self.base_dir, folder_name)
        os.makedirs(session_dir, exist_ok=True)

        # Save audio
        audio_path = os.path.join(session_dir, "audio.wav")
        self._save_wav(y, sr, audio_path)

        # Save tips as plain text (easy to read later)
        tips_txt_path = os.path.join(session_dir, "tips.txt")
        self._save_tips_txt(tips, tips_txt_path)

        # Save tips as JSON
        tips_json_path = os.path.join(session_dir, "tips.json")
        tips_data = [
            {"category": t.category, "severity": t.severity,
             "headline": t.headline, "detail": t.detail, "exercise": t.exercise}
            for t in tips
        ]
        with open(tips_json_path, "w") as f:
            json.dump(tips_data, f, indent=2)

        # Save snapshot of key metrics (no scores, just readable facts)
        snapshot = {
            "centre_note":       extra.get("deep", {}).get("flat_sharp_bias", ""),
            "f0_mean_hz":        round(m.f0_mean, 1),
            "range_semitones":   round(m.f0_range_semitones, 1),
            "vibrato_present":   m.vibrato_present,
            "vibrato_rate_hz":   round(m.vibrato_rate_hz, 2),
            "vibrato_extent_st": round(m.vibrato_extent_semitones, 2),
            "avg_phrase_s":      round(m.avg_phrase_length_s, 1),
            "register_breaks":   m.register_breaks,
            "dynamic_range_db":  round(m.dynamic_range_db, 1),
            "ref_melody_pct":    round(m.ref_melody_accuracy, 1) if m.ref_melody_accuracy >= 0 else None,
            "voice_type":        extra.get("voice_type", "unknown"),
            "intonation_level":  _score_to_level(m.intonation_score)[0],
            "breath_level":      _score_to_level(m.breath_support_score)[0],
            "resonance_level":   _score_to_level(m.resonance_score)[0],
        }
        with open(os.path.join(session_dir, "snapshot.json"), "w") as f:
            json.dump(snapshot, f, indent=2)

        # Save metadata
        meta = {
            "date":              time.strftime("%Y-%m-%d %H:%M"),
            "label":             label,
            "voice_type":        extra.get("voice_type", "unknown"),
            "reference_song":    reference_title,
            "duration_s":        round(len(y) / sr, 1),
            "practice_streak":   account.practice_streak,
            "session_number":    account.session_count,
            "tip_count":         len(tips),
            "areas_improved":    account.improved_areas,
            "username":          self.username,
            "display_name":      account.display_name,
        }
        with open(os.path.join(session_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return session_dir

    # ── list ────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """Return all saved sessions, newest first."""
        sessions = []
        if not os.path.exists(self.base_dir):
            return sessions
        for name in sorted(os.listdir(self.base_dir), reverse=True):
            path = os.path.join(self.base_dir, name)
            if not os.path.isdir(path):
                continue
            meta_path = os.path.join(path, "meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            audio_path = os.path.join(path, "audio.wav")
            size_mb = round(os.path.getsize(audio_path) / 1e6, 1) if os.path.exists(audio_path) else 0
            sessions.append({
                "folder":    name,
                "path":      path,
                "date":      meta.get("date", name),
                "label":     meta.get("label", ""),
                "voice_type": meta.get("voice_type", ""),
                "song":      meta.get("reference_song", ""),
                "duration_s": meta.get("duration_s", 0),
                "tip_count": meta.get("tip_count", 0),
                "streak":    meta.get("practice_streak", 0),
                "size_mb":   size_mb,
                "has_audio": os.path.exists(audio_path),
            })
        return sessions

    def print_session_list(self):
        sessions = self.list_sessions()
        if not sessions:
            console.print("[dim]No saved sessions yet.[/dim]")
            return sessions

        t = Table(title=f"Saved sessions for {self.username}",
                  box=box.ROUNDED, header_style="bold cyan", border_style="bright_black")
        t.add_column("#",        justify="right", min_width=3,  style="bold cyan")
        t.add_column("Date",     min_width=16,    style="white")
        t.add_column("Label",    min_width=14,    style="yellow")
        t.add_column("Voice",    min_width=10,    style="dim")
        t.add_column("Song",     min_width=20,    style="dim")
        t.add_column("Length",   justify="right", min_width=8,  style="dim")
        t.add_column("Tips",     justify="right", min_width=6,  style="cyan")
        t.add_column("Streak",   justify="right", min_width=8,  style="yellow")

        for i, s in enumerate(sessions, 1):
            dur = f"{s['duration_s']:.0f}s"
            streak = f"{s['streak']}d" if s["streak"] else ""
            t.add_row(str(i), s["date"], s["label"], s["voice_type"],
                      s["song"][:20] if s["song"] else "", dur,
                      str(s["tip_count"]), streak)

        console.print(t)
        return sessions

    # ── view a past session ──────────────────────────────────

    def view_session(self, session_path: str):
        """Re-display the tips from a saved session."""
        tips_json = os.path.join(session_path, "tips.json")
        meta_json = os.path.join(session_path, "meta.json")
        snap_json = os.path.join(session_path, "snapshot.json")

        meta = {}
        if os.path.exists(meta_json):
            with open(meta_json) as f:
                meta = json.load(f)

        snap = {}
        if os.path.exists(snap_json):
            with open(snap_json) as f:
                snap = json.load(f)

        # Header
        song_line  = f"  Song: [cyan]{meta['reference_song']}[/cyan]\n" if meta.get("reference_song") else ""
        label_line = f"  Label: [yellow]{meta['label']}[/yellow]\n"    if meta.get("label") else ""
        streak_line = f"  Streak at the time: [yellow]{meta['streak']} days[/yellow]\n" if meta.get("streak", 0) >= 2 else ""

        console.print(Panel(
            f"[bold white]{meta.get('date', 'Unknown date')}[/bold white]\n"
            f"  Voice: [cyan]{meta.get('voice_type', '?').title()}[/cyan]\n"
            f"{label_line}{song_line}{streak_line}",
            title="[bold cyan]Looking back at this session[/bold cyan]",
            border_style="cyan",
        ))

        # Quick snapshot
        if snap:
            console.print()
            t = Table(title="What your voice was doing that day",
                      box=box.SIMPLE_HEAD, header_style="bold cyan", border_style="bright_black")
            t.add_column("Area",  style="bold white", min_width=28)
            t.add_column("Measurement", style="white", min_width=30)
            t.add_column("Level",       justify="right", min_width=14)

            def srow(area, val, level=None):
                lev_str = ""
                if level:
                    for thr, lbl, col, stars in LEVEL_LABELS:
                        if lbl == level:
                            lev_str = f"[{col}]{lbl}[/{col}]  {stars}"
                            break
                t.add_row(area, str(val), lev_str)

            srow("Pitch accuracy",   "",                 snap.get("intonation_level"))
            srow("Breath support",   f"{snap.get('avg_phrase_s', 0):.1f}s per phrase",
                 snap.get("breath_level"))
            srow("Tone quality",     "",                 snap.get("resonance_level"))
            srow("Vibrato",          f"{'Present' if snap.get('vibrato_present') else 'Not detected'}  "
                                     + (f"{snap.get('vibrato_rate_hz', 0):.1f} Hz" if snap.get("vibrato_present") else ""))
            srow("Register breaks",  str(snap.get("register_breaks", 0)))
            srow("Dynamic range",    f"{snap.get('dynamic_range_db', 0):.0f} dB")
            if snap.get("ref_melody_pct") is not None:
                srow("Song match",   f"{snap['ref_melody_pct']:.0f}% of notes matched")
            console.print(t)

        # Tips
        if not os.path.exists(tips_json):
            console.print("[dim]No tips file found for this session.[/dim]")
            return

        with open(tips_json) as f:
            tips_data = json.load(f)

        console.print()
        console.print(Panel(
            f"[bold white]{len(tips_data)} tips from this session[/bold white]",
            border_style="magenta",
        ))

        severity_styles = {
            "critical": ("bold red",    "✖"),
            "warning":  ("bold yellow", "⚠"),
            "info":     ("bold cyan",   "ℹ"),
            "praise":   ("bold green",  "★"),
        }
        sev_label = {
            "critical": "Fix this first",
            "warning":  "Worth working on",
            "info":     "Good to know",
            "praise":   "Well done",
        }

        for i, tip in enumerate(tips_data, 1):
            color, icon = severity_styles.get(tip["severity"], ("white", "●"))
            label = sev_label.get(tip["severity"], "")
            console.print(Panel(
                f"[{color}]{icon}  {tip['headline']}[/{color}]\n\n"
                f"[white]{tip['detail']}[/white]\n\n"
                f"[dim cyan]How to practise this:[/dim cyan]\n[italic]{tip['exercise']}[/italic]",
                title=f"[bold white]{i}.  {tip['category']}[/bold white]"
                      + (f"  [dim]({label})[/dim]" if label else ""),
                border_style=color.replace("bold ", ""),
                padding=(0, 2),
            ))

    # ── play audio ───────────────────────────────────────────

    def play_audio(self, session_path: str):
        """Play back the saved audio through the default audio output."""
        audio_path = os.path.join(session_path, "audio.wav")
        if not os.path.exists(audio_path):
            console.print("[red]No audio file found in this session.[/red]")
            return
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            dur   = len(y) / sr
            console.print(f"\n[cyan]Playing back recording ({dur:.1f}s)...[/cyan]  "
                          "[dim]Press Ctrl+C to stop.[/dim]\n")
            sd.play(y, sr)
            sd.wait()
            console.print("[green]Done.[/green]")
        except KeyboardInterrupt:
            sd.stop()
            console.print("\n[dim]Stopped.[/dim]")
        except Exception as e:
            console.print(f"[red]Could not play audio: {e}[/red]")

    # ── delete a session ─────────────────────────────────────

    def delete_session(self, session_path: str):
        if not os.path.exists(session_path):
            console.print("[red]Session not found.[/red]")
            return
        confirm = input(f"  Delete this session? This cannot be undone. [y/N]: ").strip().lower()
        if confirm == "y":
            shutil.rmtree(session_path)
            console.print("[green]Session deleted.[/green]")
        else:
            console.print("[dim]Not deleted.[/dim]")

    # ── interactive browser ──────────────────────────────────

    def browse(self):
        """
        Interactive session browser.
        Shows the session list and lets the user pick one to view, play, or delete.
        """
        while True:
            console.print()
            sessions = self.print_session_list()
            if not sessions:
                return

            console.print(
                "\n  Type a session number to open it, or press Enter to go back.\n"
            )
            raw = input("  > ").strip()
            if not raw:
                return
            if not raw.isdigit() or not (1 <= int(raw) <= len(sessions)):
                console.print("[yellow]Please type a valid number from the list.[/yellow]")
                continue

            session = sessions[int(raw) - 1]
            console.print()

            while True:
                console.print(Panel(
                    f"[bold white]{session['date']}[/bold white]"
                    + (f"  —  {session['label']}" if session["label"] else ""),
                    title="[bold cyan]What do you want to do?[/bold cyan]",
                    border_style="cyan",
                ))
                console.print("  [cyan]1[/cyan]  Read the tips from this session")
                console.print("  [cyan]2[/cyan]  Play back the recording")
                console.print("  [cyan]3[/cyan]  Delete this session")
                console.print("  [cyan]4[/cyan]  Back to the session list")

                action = input("\n  Type 1, 2, 3, or 4: ").strip()
                if action == "1":
                    self.view_session(session["path"])
                elif action == "2":
                    self.play_audio(session["path"])
                elif action == "3":
                    self.delete_session(session["path"])
                    break
                elif action == "4":
                    break
                else:
                    console.print("[yellow]Just type 1, 2, 3, or 4.[/yellow]")

    # ── helpers ─────────────────────────────────────────────

    def _ask_label(self) -> str:
        console.print(
            "\n[bold white]Give this session a short label so you can find it later.[/bold white]\n"
            "[dim]Examples: warmup, chorus, verse 2, audition run, first try\n"
            "Press Enter to skip.[/dim]"
        )
        raw = input("  Label: ").strip()
        safe = "".join(c if c.isalnum() or c in " _-" else "" for c in raw)[:30].strip()
        return safe.replace(" ", "_")

    @staticmethod
    def _save_wav(y: np.ndarray, sr: int, path: str):
        import wave, struct
        y_int = np.clip(y, -1.0, 1.0)
        y_16  = (y_int * 32767).astype(np.int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(y_16.tobytes())

    @staticmethod
    def _save_tips_txt(tips: list, path: str):
        lines = []
        for i, tip in enumerate(tips, 1):
            lines.append(f"\n{'='*60}")
            lines.append(f"Tip {i}: {tip.category}")
            lines.append(f"{tip.headline}")
            lines.append(f"{'='*60}")
            lines.append(tip.detail)
            lines.append(f"\nHow to practise this:")
            lines.append(tip.exercise)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def _do_countdown_record(recorder: AudioRecorder, duration: float) -> np.ndarray:
    console.print(f"\n[bold yellow]Get ready to sing![/bold yellow]  Starting in 3 seconds...\n")
    for i in range(3, 0, -1):
        console.print(f"  [bold]{i}...[/bold]")
        time.sleep(1)
    return recorder.record(duration)


def main():
    parser = argparse.ArgumentParser(
        description="VocalIQ — Advanced Vocal Coach & Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vocal_analyzer.py                              # Record and analyse (asks about reference song)
  python vocal_analyzer.py -d 45                        # Record 45 seconds
  python vocal_analyzer.py -f my_voice.wav              # Analyse an existing file
  python vocal_analyzer.py --song "Shallow Lady Gaga"   # Search YouTube and auto-download reference
  python vocal_analyzer.py --song-url https://youtu.be/bo_efYhYU2A  # Use a direct YouTube URL
  python vocal_analyzer.py -f voice.wav -r my_song.mp3  # Use a local reference file
  python vocal_analyzer.py --list-songs                 # Show cached reference songs
  python vocal_analyzer.py --clear-songs                # Delete cached songs
  python vocal_analyzer.py --list-devices               # Show audio input devices
  python vocal_analyzer.py --no-plot                    # Skip the charts
  python vocal_analyzer.py --my-sessions                # Browse and replay your saved sessions
  python vocal_analyzer.py --no-save                    # Skip the save prompt after analysis

Requires yt-dlp for online song search and download:
  pip install yt-dlp
        """
    )
    parser.add_argument("-d", "--duration", type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("-f", "--file", type=str, default=None,
                        help="Path to WAV/MP3/FLAC file to analyze instead of recording")
    parser.add_argument("-r", "--reference", type=str, default=None,
                        help="Path to a local reference audio file (WAV/MP3/FLAC).")
    parser.add_argument("-s", "--song", type=str, default=None,
                        help="Search for and download a reference song by name, e.g. --song \"Shape of You Ed Sheeran\". "
                             "Requires: pip install yt-dlp")
    parser.add_argument("--song-url", type=str, default=None,
                        help="Direct YouTube URL to use as reference track.")
    parser.add_argument("--list-songs", action="store_true",
                        help="List previously downloaded reference songs in the cache.")
    parser.add_argument("--clear-songs", action="store_true",
                        help="Delete all cached reference songs.")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (default: 44100)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip visualization chart")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save metrics to a JSON file")
    parser.add_argument("--no-retry", action="store_true",
                        help="Skip the re-record prompt and analyse anyway even if quality is poor")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max re-record attempts before forcing analysis (default: 3)")
    parser.add_argument("--no-account", action="store_true",
                        help="Skip account login and run as a guest (tips won't be personalised)")
    parser.add_argument("--new-account", action="store_true",
                        help="Force creation of a new account even if accounts exist")
    parser.add_argument("--my-sessions", action="store_true",
                        help="Browse your saved sessions — replay tips and listen back to recordings")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip the save prompt after analysis and do not save this session")
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]VocalIQ[/bold cyan]  [white]Your personal singing coach[/white]\n"
        "[dim]Record yourself singing, get honest feedback, and track your improvement over time.[/dim]",
        border_style="cyan",
    ))

    recorder = AudioRecorder(sr=args.sr)

    if args.list_devices:
        recorder.list_devices()
        return

    # ── Song cache commands ───────────────────────────────────
    if args.list_songs:
        cached = SongFetcher.list_cached()
        if not cached:
            console.print("[dim]No songs cached yet. Use --song to download one.[/dim]")
        else:
            t = Table(title="Cached Reference Songs", box=box.ROUNDED,
                      header_style="bold cyan", border_style="bright_black")
            t.add_column("#",       justify="right", min_width=3)
            t.add_column("File",    min_width=50,    style="white")
            t.add_column("Size",    justify="right", min_width=8,  style="dim")
            t.add_column("Date",    justify="right", min_width=12, style="dim")
            for i, s in enumerate(cached, 1):
                t.add_row(str(i), s["filename"], f"{s['size_mb']} MB", s["date"])
            console.print(t)
            console.print(f"\n[dim]Cache location: {SONGS_CACHE_DIR}[/dim]")
        return

    if args.clear_songs:
        confirm = input("Delete all cached songs? This cannot be undone. [y/N]: ").strip().lower()
        if confirm == "y":
            SongFetcher.clear_cache()
        return

    # ── Resolve reference audio path ─────────────────────────
    # Priority: --reference (local file) > --song-url > --song (search)
    fetcher        = SongFetcher()
    reference_path = args.reference  # may be None

    if not reference_path and args.song_url:
        reference_path = fetcher.get(args.song_url)

    if not reference_path and args.song:
        reference_path = fetcher.get(args.song)

    # If no reference provided at all, offer an interactive prompt
    if not reference_path and not args.no_account:
        console.print()
        console.print(Panel(
            "[bold white]Were you singing along to a specific song?[/bold white]\n"
            "[dim]If yes, we can compare your vocal to the original and show you exactly where you matched it.[/dim]",
            border_style="bright_black",
        ))
        console.print("  [cyan]1[/cyan]  Yes, search for it by name")
        console.print("  [cyan]2[/cyan]  Yes, I have a YouTube link")
        console.print("  [cyan]3[/cyan]  No, just check my voice in general")
        song_choice = input("\n  Type 1, 2, or 3: ").strip()

        if song_choice == "1":
            query = input("\n  What song?  (name and artist, e.g. \"Shallow Lady Gaga\"): ").strip()
            if query:
                reference_path = fetcher.search_and_pick(query)
        elif song_choice == "2":
            url = input("\n  Paste the YouTube link: ").strip()
            if url:
                reference_path = fetcher.get(url)

        if reference_path:
            console.print(f"\n[green]Got it.[/green]  We will compare your singing to that song.\n")
        else:
            console.print("[dim]No song selected. We will just analyse your voice technique on its own.[/dim]\n")

    # ── Account flow ──────────────────────────────────────────
    account = None
    if not args.no_account:
        mgr = AccountManager()
        questionnaire = OnboardingQuestionnaire()
        account = account_login_flow(mgr, questionnaire, force_new=args.new_account)

    # ── Session browser ───────────────────────────────────────
    if args.my_sessions:
        if not account:
            console.print("[yellow]Log in to an account to browse saved sessions.[/yellow]")
            sys.exit(0)
        archive = SessionArchive(account.username)
        archive.browse()
        sys.exit(0)

    vocaliq  = VocalIQ(sr=args.sr)
    is_live  = args.file is None
    file_sr  = args.sr

    # ── Load or record ────────────────────────────────────────
    if args.file:
        console.print(f"[cyan]Loading:[/cyan] {args.file}")
        y, file_sr = librosa.load(args.file, sr=None, mono=False)
        if y.ndim == 1:
            console.print(f"[green]✓ Loaded {len(y)/file_sr:.1f}s  ({file_sr} Hz, mono)[/green]\n")
        else:
            console.print(f"[green]✓ Loaded {y.shape[1]/file_sr:.1f}s  ({file_sr} Hz, {y.shape[0]}-ch)[/green]\n")
            y = y.mean(axis=0)
    else:
        y = _do_countdown_record(recorder, args.duration)

    if np.abs(y).max() < 1e-6:
        console.print("[red]Error: Silent audio — check your microphone or file.[/red]")
        sys.exit(1)

    # ── Retry loop (live recording only) ─────────────────────
    attempt = 0
    while True:
        try:
            m, tips, extra = vocaliq.analyze(
                y, sr_in=file_sr,
                reference_path=reference_path,
                is_live=is_live,
                account=account,
            )
            break   # success

        except NeedsRetryError as e:
            quality = e.quality
            attempt += 1

            if args.no_retry or not is_live or attempt >= args.max_retries:
                console.print(
                    f"\n[yellow]Proceeding with analysis despite quality issues "
                    f"(attempt {attempt}/{args.max_retries}).[/yellow]\n"
                )
                m, tips, extra = vocaliq.analyze(
                    y, sr_in=file_sr,
                    reference_path=reference_path,
                    is_live=False,
                    account=account,
                )
                break

            # Ask user what to do
            console.print()
            console.print(Panel(
                "[bold red]The recording is too noisy to give you reliable feedback.[/bold red]\n\n"
                + "\n".join(f"  [red]Problem:[/red] {i}" for i in quality["issues"])
                + "\n\n[bold white]What do you want to do?[/bold white]\n"
                  "  [bold cyan]1[/bold cyan]  Try again  (sing again)\n"
                  "  [bold yellow]2[/bold yellow]  Analyse anyway  (results may be less accurate)\n"
                  "  [bold red]3[/bold red]  Quit",
                title="[bold red]Recording problem[/bold red]",
                border_style="red",
            ))

            for hint in quality.get("fix_hints", []):
                console.print(f"  [cyan]Quick fix:[/cyan] {hint}")

            console.print()
            choice = input("Type 1, 2, or 3: ").strip()

            if choice == "3":
                console.print("[dim]Exiting.[/dim]")
                sys.exit(0)
            elif choice == "2":
                console.print("[yellow]Proceeding with current recording...[/yellow]\n")
                m, tips, extra = vocaliq.analyze(
                    y, sr_in=file_sr,
                    reference_path=reference_path,
                    is_live=False,
                    account=account,
                )
                break
            else:
                # Re-record
                console.print(
                    f"\n[bold yellow]Re-recording (attempt {attempt + 1}/{args.max_retries})...[/bold yellow]"
                )
                y = _do_countdown_record(recorder, args.duration)
                file_sr = args.sr

        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # ── Save session to account history ──────────────────────
    if account:
        deep = extra.get("deep", {})
        issue_names = [c["root"] for c in deep.get("root_causes", []) if c["confidence"] >= 0.6]
        strength_names = extra.get("deep", {}).get("voice_profile", {})
        strengths = [t.category for t in tips if t.severity == "praise"]
        area_scores = {
            "Pitch":      m.intonation_score,
            "Breath":     m.breath_support_score,
            "Vibrato":    m.vibrato_score,
            "Resonance":  m.resonance_score,
            "Dynamics":   m.dynamic_control_score,
            "Stability":  m.pitch_stability_score,
            "Passaggio":  m.smoothness_through_break,
        }
        mgr.update_session(account, area_scores, issue_names, strengths)
        console.print(f"[dim]Session saved to {account.display_name}'s account.[/dim]")

    display_results(m, tips, vocaliq.pitch_analyzer, extra, account=account)

    if args.save_json:
        data = {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in m.__dict__.items()
                if not isinstance(v, list) or all(isinstance(x, (int, float)) for x in v)}
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]Metrics saved to {args.save_json}[/green]")

    if not args.no_plot:
        plot_results(m, extra, args.sr, extra.get("belt", {}))

    # ── Save session prompt ───────────────────────────────────
    if account and not args.no_save:
        console.print()
        console.print(Panel(
            "[bold white]Do you want to save this session?[/bold white]\n"
            "[dim]Saving keeps your recording and all your tips so you can look back at them later "
            "and hear how much you have improved.[/dim]",
            border_style="bright_black",
        ))
        console.print("  [cyan]1[/cyan]  Yes, save it")
        console.print("  [cyan]2[/cyan]  No thanks")
        save_choice = input("\n  Type 1 or 2: ").strip()

        if save_choice == "1":
            ref_title = ""
            if reference_path:
                ref_title = os.path.splitext(os.path.basename(reference_path))[0]
                # strip the hash suffix added by the fetcher
                import re as _re
                ref_title = _re.sub(r'_[a-f0-9]{12}$', '', ref_title).replace("_", " ").strip()

            archive   = SessionArchive(account.username)
            saved_dir = archive.save(
                y          = extra["y"],
                sr         = args.sr,
                tips       = tips,
                m          = m,
                extra      = extra,
                account    = account,
                reference_title = ref_title,
            )
            console.print()
            console.print(Panel(
                f"[bold green]Saved![/bold green]\n"
                f"[dim]You can come back to this any time with:[/dim]\n"
                f"  python vocal_analyzer.py --my-sessions",
                border_style="green",
            ))


if __name__ == "__main__":
    main()