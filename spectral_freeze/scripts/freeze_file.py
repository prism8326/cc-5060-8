"""Command-line offline freeze processor.

Example
-------
python -m scripts.freeze_file --input voice.wav --freeze-at 1.2 --release-at 3.0 \
        --stop-input-at 1.5 --output frozen.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_freeze.engine import FreezeParams
from spectral_freeze.offline import process_offline, read_audio


def main():
    ap = argparse.ArgumentParser(description="Offline spectral freeze")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="frozen_output.wav")
    ap.add_argument("--freeze-at", type=float, required=True, help="seconds")
    ap.add_argument("--release-at", type=float, default=None)
    ap.add_argument("--stop-input-at", type=float, default=None,
                    help="zero the input after this time; the freeze keeps sounding")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--fft-size", type=int, default=2048)
    ap.add_argument("--entry-fade", type=float, default=0.05)
    ap.add_argument("--exit-fade", type=float, default=0.10)
    ap.add_argument("--capture-windows", type=int, default=4)
    ap.add_argument("--coherence-threshold", type=float, default=0.6)
    ap.add_argument("--drift-hz", type=float, default=0.0)
    ap.add_argument("--report", default=None, help="optional JSON trace output")
    args = ap.parse_args()

    _, sr = read_audio(args.input)
    data, _ = read_audio(args.input)
    params = FreezeParams(
        sample_rate=sr,
        block_size=args.block_size,
        fft_size=args.fft_size,
        channels=data.shape[1],
        entry_fade_sec=args.entry_fade,
        exit_fade_sec=args.exit_fade,
        capture_windows=args.capture_windows,
        coherence_threshold=args.coherence_threshold,
        drift_compensation=args.drift_hz,
    )
    result = process_offline(
        args.input,
        args.output,
        freeze_at_sec=args.freeze_at,
        release_at_sec=args.release_at,
        params=params,
        stop_input_at_sec=args.stop_input_at,
    )
    rep = result["snapshot"]["capture_report"]
    print(f"wrote {result['output_path']} ({sr} Hz, {data.shape[1]} ch)")
    print(f"capture={rep.capture_mode} phase_jump_bins={rep.n_phase_jump_bins} "
          f"coherent_bins={rep.n_coherent_bins}")
    print(f"top frequencies (Hz): {rep.top_frequencies_hz}")
    if args.report:
        trace = {
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
                {
                    "state": r.state.value,
                    "live_rms_db": r.live_rms_db,
                    "frozen_rms_db": r.frozen_rms_db,
                    "output_rms_db": r.output_rms_db,
                    "progress": r.freeze_progress,
                    "note": r.note,
                }
                for r in result["reports"]
            ],
        }
        Path(args.report).write_text(json.dumps(trace, indent=2))
        print(f"trace report: {args.report}")


if __name__ == "__main__":
    main()
