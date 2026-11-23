"""Pure DSP helpers for spectral freeze: phase statistics and fades."""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Wrap phase to the half-open interval ``[-pi, pi)``."""
    return (phi + np.pi) % TWO_PI - np.pi


def principal_instantaneous_frequency(
    phase_new: np.ndarray, phase_old: np.ndarray, hop: int, fft_size: int
) -> np.ndarray:
    """Instantaneous frequency in rad/sample from two consecutive frames.

    The expected phase increment of bin ``k`` over one hop is
    ``2*pi*k*hop/fft_size``; deviations of the wrapped delta from that
    expectation give the true instantaneous frequency (phase vocoder).
    """
    bin_k = np.arange(phase_new.shape[0])[:, None]
    expected = TWO_PI * bin_k * hop / fft_size
    delta = wrap_phase(phase_new - phase_old - expected)
    return (expected + delta) / hop


def weighted_circular_mean(
    angles: np.ndarray, weights: np.ndarray, axis: int = 0
) -> np.ndarray:
    """Magnitude-weighted circular mean of angles along an axis.

    ``angles`` and ``weights`` share the same shape. Zero total weight yields
    a mean of 0.
    """
    weights = np.maximum(weights, 0.0)
    sin_sum = np.sum(weights * np.sin(angles), axis=axis)
    cos_sum = np.sum(weights * np.cos(angles), axis=axis)
    mean = np.arctan2(sin_sum, cos_sum)
    dead = (sin_sum**2 + cos_sum**2) <= 1e-20
    return np.where(dead, 0.0, mean)


def magnitude_weighted_mean(
    values: np.ndarray, weights: np.ndarray, axis: int = 0
) -> np.ndarray:
    total = np.sum(weights, axis=axis)
    safe = np.maximum(total, 1e-12)
    return np.sum(weights * values, axis=axis) / safe


def interchannel_coherence(
    spectra: np.ndarray, window: int
) -> np.ndarray:
    """Mean cross-channel coherence per bin over the last ``window`` frames.

    Parameters
    ----------
    spectra:
        Complex spectra, shape ``(frames, bins, channels)`` with exactly two
        channels.

    Returns
    -------
    np.ndarray
        Coherence in ``[0, 1]``, shape ``(bins,)``. Computed as the magnitude
        of the normalized cross-spectrum averaged over frames.
    """
    if spectra.ndim != 3 or spectra.shape[2] != 2:
        raise ValueError("coherence requires shape (frames, bins, 2)")
    recent = spectra[-window:, :, :]
    cross = recent[:, :, 0] * np.conj(recent[:, :, 1])
    power = np.sqrt(
        np.abs(recent[:, :, 0]) ** 2 * np.abs(recent[:, :, 1]) ** 2 + 1e-20
    )
    normalized = cross / power
    coh = np.abs(np.mean(normalized, axis=0))
    return np.clip(coh, 0.0, 1.0)


def equal_power_crossfade(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power (constant energy) fade pair for gain ``g`` in [0, 1]."""
    g = np.clip(g, 0.0, 1.0)
    return np.cos(0.5 * np.pi * g), np.sin(0.5 * np.pi * g)


def rms_db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x**2) + 1e-20))
    return 20.0 * np.log10(rms + 1e-12)
