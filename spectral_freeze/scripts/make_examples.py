"""Regenerate the audible example WAVs and trace with the current engine.

Every file is produced by the same ``process_offline`` / ``FreezeEngine`` path
the GUI and real-time code use, so the examples cannot drift from the code.
Run from the repository root::

    python -m scripts.make_examples
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from spectral_freeze.engine import FreezeEngine, FreezeParams, FreezeState
from spectral_freeze.offline import process_offline, read_audio
from spectral_freeze.signals import make_ambient_signal, make_musical_signal

SR = 48000
EX = Path(__file__).resolve().parent.parent / "examples"


def _ensure_inputs() -> None:
    EX.mkdir(exist_ok=True)
    if not (EX / "musical_input.wav").exists():
        sf.write(EX / "musical_input.wav", make_musical_signal(SR, 3.0, 2).astype(np.float32), SR)
    if not (EX / "ambient_input.wav").exists():
        sf.write(EX / "ambient_input.wav", make_ambient_signal(SR, 3.5, 2).astype(np.float32), SR)


def _trace(out_path: Path, freeze_at: float, release_at=None, stop_at=None,
           capture="auto") -> dict:
    params = FreezeParams(sample_rate=SR, channels=2)
    result = process_offline(
        EX / "musical_input.wav", out_path, freeze_at_sec=freeze_at,
        release_at_sec=release_at, params=params, stop_input_at_sec=stop_at,
    )
    rep = result["snapshot"]["capture_report"]
    return {
        "events": result["events"],
        "capture": {
            "mode": rep.capture_mode,
            "phase_jump_bins": rep.n_phase_jump_bins,
            "coherent_bins": rep.n_coherent_bins,
            "top_frequencies_hz": rep.top_frequencies_hz,
            "dc_magnitude": float(rep.dc_magnitude),
            "near_silence": bool(rep.near_silence),
        },
        "frame_reports": [
            {"state": r.state.value, "live_rms_db": r.live_rms_db,
             "frozen_rms_db": r.frozen_rms_db, "output_rms_db": r.output_rms_db,
             "progress": r.freeze_progress, "note": r.note}
            for r in result["reports"]
        ],
    }


def _freeze_at_phase_jump() -> dict:
    """Freeze while the 180-degree discontinuity is inside the analysis window."""
    data, _ = read_audio(EX / "musical_input.wav")
    params = FreezeParams(sample_rate=SR, channels=2)
    eng = FreezeEngine(params)
    hop = params.block_size
    out = np.zeros_like(data)
    trace = []
    for i in range(data.shape[0] // hop):
        block = data[i * hop : (i + 1) * hop].copy()
        if i == int(1.23 * SR) // hop and eng.state == FreezeState.LIVE:
            eng.freeze("auto")
        y, r = eng.process_block(block)
        out[i * hop : (i + 1) * hop] = y
        trace.append({"state": r.state.value, "live_rms_db": r.live_rms_db,
                      "frozen_rms_db": r.frozen_rms_db,
                      "output_rms_db": r.output_rms_db,
                      "progress": r.freeze_progress, "note": r.note})
    sf.write(EX / "musical_freeze_phasejump.wav", out.astype(np.float32), SR, subtype="PCM_16")
    rep = eng._last_capture_report
    return {
        "events": {"freeze_sample": int(1.23 * SR), "release_sample": None},
        "capture": {"mode": rep.capture_mode, "phase_jump_bins": rep.n_phase_jump_bins,
                    "coherent_bins": rep.n_coherent_bins,
                    "top_frequencies_hz": rep.top_frequencies_hz,
                    "dc_magnitude": float(rep.dc_magnitude),
                    "near_silence": bool(rep.near_silence)},
        "frame_reports": trace,
    }


def main() -> None:
    _ensure_inputs()
    t = _trace(EX / "musical_freeze.wav", 0.9, 2.6)
    print("musical_freeze", t["capture"]["mode"], "jumps", t["capture"]["phase_jump_bins"])

    t = _trace(EX / "musical_freeze_indefinite.wav", 0.9, None, 1.3)
    print("musical_freeze_indefinite", t["capture"]["mode"])

    t = _trace(EX / "musical_freeze_nearsilence.wav", 1.9, 2.6)
    print("musical_freeze_nearsilence near_silence", t["capture"]["near_silence"])

    pj = _freeze_at_phase_jump()
    (EX / "phasejump_trace.json").write_text(json.dumps(pj, indent=2))
    print("musical_freeze_phasejump", pj["capture"]["mode"], "jumps",
          pj["capture"]["phase_jump_bins"])

    result = process_offline(EX / "ambient_input.wav", EX / "ambient_freeze.wav",
                             freeze_at_sec=0.9, release_at_sec=2.8,
                             params=FreezeParams(sample_rate=SR, channels=2))
    print("ambient_freeze", result["snapshot"]["capture_report"].capture_mode)


if __name__ == "__main__":
    main()
