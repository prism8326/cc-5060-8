"""STFT analysis/synthesis with overlap-add buffer management.

A single hop-size block of time-domain samples is consumed/produced per call.
Analysis and synthesis use the same periodic Hann window; weighted
overlap-add (WOLA) with explicit window-energy normalization gives perfect
reconstruction on the live path, and the frozen path shares this identical
reconstruction so no level mismatch appears at a transition.

Linear (non-circular) buffers are used deliberately: the oldest sample sits
at index 0, and both analysis and synthesis shift by one hop per call, which
makes the input/output alignment provably exact.
"""

from __future__ import annotations

import numpy as np


def hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (``n`` points), matching scipy hann(..., sym=False)."""
    if n == 1:
        return np.ones(1, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


class STFT:
    """Streaming short-time Fourier transform with overlap-add output.

    Parameters
    ----------
    block_size:
        Number of time-domain samples processed per hop.
    fft_size:
        FFT length (must be an integer multiple of ``block_size``).
    channels:
        Number of audio channels.
    """

    def __init__(self, block_size: int = 256, fft_size: int = 2048, channels: int = 2):
        if fft_size % block_size != 0:
            raise ValueError("fft_size must be an integer multiple of block_size")
        self.block_size = int(block_size)
        self.fft_size = int(fft_size)
        self.channels = int(channels)
        self.window = hann_window(fft_size)
        self._w2 = self.window**2
        # Analysis buffer: index 0 is the oldest sample.
        self._in_buf = np.zeros((fft_size, channels), dtype=np.float64)
        # Synthesis overlap-add buffer and its window-energy normalizer.
        self._out_acc = np.zeros((fft_size, channels), dtype=np.float64)
        self._norm_acc = np.zeros((fft_size, channels), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def push(self, hop_samples: np.ndarray) -> None:
        """Advance the analysis buffer by one hop without transforming it.

        The frozen path uses this to keep the window primed while no live
        spectrum is needed; a later exit can then resume analysis at a hop
        boundary with no gap in the input history.
        """
        if hop_samples.shape != (self.block_size, self.channels):
            raise ValueError(
                f"expected shape ({self.block_size}, {self.channels}), "
                f"got {hop_samples.shape}"
            )
        self._in_buf = np.roll(self._in_buf, -self.block_size, axis=0)
        self._in_buf[self.fft_size - self.block_size :, :] = hop_samples

    def analysis(self, hop_samples: np.ndarray) -> np.ndarray:
        """Push one hop of input and return the rFFT of the windowed frame."""
        self.push(hop_samples)
        windowed = self._in_buf * self.window[:, None]
        return np.fft.rfft(windowed, n=self.fft_size, axis=0)

    def current_frame(self) -> np.ndarray:
        """Return the (un-windowed) time-domain samples currently in the buffer."""
        return self._in_buf.copy()

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def synthesis(self, spectrum: np.ndarray) -> np.ndarray:
        """Overlap-add one complex spectrum and emit one hop of output.

        The frame is added at the *tail* of the accumulator; the hop emitted
        from the head has therefore accumulated every window that covers it.
        """
        if spectrum.shape != (self.fft_size // 2 + 1, self.channels):
            raise ValueError(
                f"expected spectrum shape "
                f"({self.fft_size // 2 + 1}, {self.channels}), got {spectrum.shape}"
            )
        frame = np.fft.irfft(spectrum, n=self.fft_size, axis=0)
        windowed = frame * self.window[:, None]
        self._out_acc += windowed
        self._norm_acc += self._w2[:, None]
        out = self._out_acc[: self.block_size].copy() / np.maximum(
            self._norm_acc[: self.block_size], 1e-12
        )
        self._out_acc = np.roll(self._out_acc, -self.block_size, axis=0)
        self._out_acc[self.fft_size - self.block_size :, :] = 0.0
        self._norm_acc = np.roll(self._norm_acc, -self.block_size, axis=0)
        self._norm_acc[self.fft_size - self.block_size :, :] = 0.0
        return out

    def reset(self) -> None:
        self._in_buf.fill(0.0)
        self._out_acc.fill(0.0)
        self._norm_acc.fill(0.0)
