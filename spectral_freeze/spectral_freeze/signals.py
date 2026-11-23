"""Test-signal synthesis and optional live recording.

Two signal families are produced for verification:

* ``make_musical_signal`` — deterministic homemade musical tones, including
  inharmonic / non-integer-period partials and phase discontinuities, which
  are the hardest freeze cases.
* ``make_ambient_signal`` — a short synthetic "environment" recording
  (filtered noise bed plus moving tone events), used when no microphone is
  available. With an audio device, ``record_ambient`` captures a real one.
"""

from __future__ import annotations

import numpy as np


def _ramp(x: np.ndarray, sr: int, fade_sec: float = 0.01) -> np.ndarray:
    n = int(fade_sec * sr)
    if n <= 0 or len(x) < 2 * n:
        return x
    env = np.ones(len(x))
    env[:n] = np.linspace(0.0, 1.0, n)
    env[-n:] = np.linspace(1.0, 0.0, n)
    return x * env


def make_musical_signal(
    sr: int = 48000,
    duration_sec: float = 3.0,
    channels: int = 2,
    seed: int = 7,
) -> np.ndarray:
    """A chord with harmonic + inharmonic partials and a phase jump.

    The signal deliberately contains:

    * non-integer-period partials (frequencies not dividing the sample rate),
    * an inharmonic partial at 2.31x the fundamental,
    * a 180-degree phase discontinuity at 40% of the duration,
    * a near-silent passage,
    * a small DC offset that the engine must reject.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sr)) / sr
    f0 = 220.0  # A3, deliberately not an integer divisor of 48000
    sig = np.zeros_like(t)
    partials = [(1.0, 0.55), (2.0, 0.30), (3.01, 0.18), (2.31, 0.16), (4.77, 0.10)]
    for mult, amp in partials:
        sig += amp * np.sin(2 * np.pi * f0 * mult * t + rng.uniform(0, 2 * np.pi))
    # Amplitude envelope with a near-silent gap.
    env = 0.75 + 0.25 * np.sin(2 * np.pi * 0.7 * t)
    gap = (t > 1.8) & (t < 2.05)
    env[gap] = 0.0008
    sig *= env
    # Phase discontinuity at 40%.
    jump = int(0.4 * len(t))
    sig[jump:] = -sig[jump:]
    sig += 0.002  # small DC offset
    sig = _ramp(sig, sr)
    if channels == 2:
        # Slightly decorrelated stereo image: a delayed, filtered copy on R.
        delay = int(0.0004 * sr)
        right = np.roll(sig, delay) * 0.95
        right[:delay] = 0.0
        stereo = np.stack([sig, right], axis=1)
        return stereo
    return sig[:, None]


def make_ambient_signal(
    sr: int = 48000, duration_sec: float = 3.0, channels: int = 2, seed: int = 11
) -> np.ndarray:
    """Synthetic environment bed: shaped noise + wandering tone events."""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.arange(n) / sr
    # Brown-ish noise via cumulative sum, then gentle low-pass via moving avg.
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown = brown / (np.max(np.abs(brown)) + 1e-12)
    kernel = np.ones(64) / 64.0
    bed = np.convolve(brown, kernel, mode="same") * 0.25
    # Two moving tone "events".
    events = np.zeros(n)
    for f_start, f_end, t0, dur, amp in [
        (660.0, 540.0, 0.4, 1.1, 0.22),
        (330.0, 392.0, 1.5, 1.0, 0.18),
    ]:
        idx = (t >= t0) & (t < t0 + dur)
        freq = np.linspace(f_start, f_end, np.sum(idx))
        phase = 2 * np.pi * np.cumsum(freq) / sr
        ev = np.zeros(n)
        ev[idx] = amp * np.sin(phase)
        events += ev
    mono = _ramp(bed + events, sr)
    if channels == 2:
        shifted = np.roll(mono, int(0.0009 * sr))
        return np.stack([mono, shifted * 0.9], axis=1)
    return mono[:, None]


def record_ambient(
    duration_sec: float = 3.0, sr: int = 48000, channels: int = 1
) -> np.ndarray:
    """Capture a real ambient recording via PortAudio (requires an input)."""
    import sounddevice as sd

    data = sd.rec(
        int(duration_sec * sr), samplerate=sr, channels=channels, dtype="float64"
    )
    sd.wait()
    return data
