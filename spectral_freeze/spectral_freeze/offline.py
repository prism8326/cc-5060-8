"""Offline file processing with the same engine used for real-time audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .engine import FreezeEngine, FreezeParams, FreezeState


def read_audio(path: str | Path, expected_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Read an audio file and return float64 ``(samples, channels)`` at native SR."""
    data, sr = sf.read(str(path), always_2d=True, dtype="float64")
    if expected_sr is not None and sr != expected_sr:
        raise ValueError(f"sample rate mismatch: file={sr}, expected={expected_sr}")
    return data, sr


def write_audio(path: str | Path, data: np.ndarray, sr: int) -> float:
    """Write 16-bit PCM, normalizing gain instead of hard-clipping.

    Returns the applied linear gain (1.0 when no scaling was needed).
    """
    peak = np.max(np.abs(data)) if data.size else 0.0
    gain = 0.999 / peak if peak > 1.0 else 1.0
    sf.write(str(path), (data * gain).astype(np.float32), sr, subtype="PCM_16")
    return gain


def process_offline(
    input_path: str | Path,
    output_path: str | Path,
    freeze_at_sec: float,
    release_at_sec: float | None = None,
    params: FreezeParams | None = None,
    stop_input_at_sec: float | None = None,
) -> dict:
    """Run the freeze engine over a file.

    The freeze is armed at ``freeze_at_sec``; if ``release_at_sec`` is given
    the engine exits the freeze there. ``stop_input_at_sec`` replaces all later
    input with zeros, demonstrating that the frozen tone keeps sounding with no
    input energy — the output energy then comes solely from the captured
    spectral state.
    """
    data, sr = read_audio(input_path)
    params = params or FreezeParams(sample_rate=sr, channels=data.shape[1])
    if params.channels != data.shape[1]:
        raise ValueError("channel count mismatch between params and input file")
    engine = FreezeEngine(params)

    hop = params.block_size
    n_hops = data.shape[0] // hop
    out = np.zeros_like(data[: n_hops * hop])
    events = {"freeze_sample": int(freeze_at_sec * sr), "release_sample": None}
    freeze_pending = True
    release_pending = release_at_sec is not None

    for i in range(n_hops):
        start = i * hop
        block = data[start : start + hop].copy()
        if stop_input_at_sec is not None and start >= int(stop_input_at_sec * sr):
            block.fill(0.0)
        t = start / sr
        # The request is deferred until the analysis window is primed, so the
        # captured state reflects real signal rather than startup zeros.
        if (
            freeze_pending
            and t >= freeze_at_sec
            and engine.state == FreezeState.LIVE
            and engine._hist_count >= engine._warmup_hops
        ):
            engine.freeze()
            freeze_pending = False
        if (
            release_pending
            and t >= release_at_sec
            and engine.state in (FreezeState.ENTERING, FreezeState.FROZEN)
        ):
            engine.unfreeze()
            events["release_sample"] = start
            release_pending = False
        y, _ = engine.process_block(block)
        out[start : start + hop] = y

    write_audio(output_path, out, sr)
    return {
        "output_path": str(output_path),
        "sample_rate": sr,
        "hops": n_hops,
        "events": events,
        "snapshot": engine.snapshot(),
        "reports": engine.reports,
    }
