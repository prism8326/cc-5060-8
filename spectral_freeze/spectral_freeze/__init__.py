"""Real-time spectral freeze effect processor.

The package implements an STFT/phase-vocoder based freeze: a short spectral
state is captured, its per-bin frequencies are estimated from multiple
analysis windows, and the frozen spectrum is *phase-advanced* hop by hop
instead of looping raw waveform samples or pasting identical complex frames.
"""

from .engine import FreezeEngine, FreezeParams, FreezeState
from .stft import STFT

__all__ = ["FreezeEngine", "FreezeParams", "FreezeState", "STFT"]
__version__ = "1.0.0"
