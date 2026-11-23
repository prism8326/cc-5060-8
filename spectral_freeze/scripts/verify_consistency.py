"""Real-time / offline consistency and audio-quality verification.

Two execution paths are compared on identical input:

1. offline: direct ``FreezeEngine.process_block`` loop (same as file mode);
2. realtime: the PortAudio callback wrapper :class:`RealtimeFreeze`, driven
   block-by-block with an emulated device (works with no sound card).

Both must produce bit-identical output. The script also quantifies:

* frozen-tone frequency drift (should be ~0: frequencies are estimated once
  and then phase-advanced),
* frozen amplitude fluctuation,
* entry/exit switching transients,
* post-input-stop behavior proving the output energy comes from the captured
  spectral state rather than the (now silent) input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectral_freeze.engine import FreezeEngine, FreezeParams, FreezeState
from spectral_freeze.realtime import RealtimeFreeze
from spectral_freeze.signals import make_ambient_signal, make_musical_signal

SR = 48000
HOP = 256


def run_offline(signal: np.ndarray, params: FreezeParams, freeze_hop: int,
                stop_hop: int | None, release_hop: int | None, f_target_hz: float):
    eng = FreezeEngine(params)
    out = np.zeros_like(signal)
    freq_track = []
    bin_k = int(round(f_target_hz * params.fft_size / SR))
    for i in range(signal.shape[0] // HOP):
        block = signal[i * HOP : (i + 1) * HOP].copy()
        if i == freeze_hop:
            eng.freeze()
        if release_hop is not None and i == release_hop:
            eng.unfreeze()
        if stop_hop is not None and i >= stop_hop:
            block.fill(0.0)
        out[i * HOP : (i + 1) * HOP], _ = eng.process_block(block)
        if eng.state in (FreezeState.FROZEN, FreezeState.EXITING):
            freq_track.append(eng.snapshot()["frozen_frequency_hz"][bin_k].mean())
    return out, eng, np.array(freq_track)


def run_realtime_emulation(signal, params: FreezeParams, freeze_hop, stop_hop, release_hop):
    rt = RealtimeFreeze(params)
    out = np.zeros_like(signal)
    for i in range(signal.shape[0] // HOP):
        block = signal[i * HOP : (i + 1) * HOP].copy()
        if i == freeze_hop:
            rt.freeze()
        if release_hop is not None and i == release_hop:
            rt.unfreeze()
        if stop_hop is not None and i >= stop_hop:
            block.fill(0.0)
        out_block = np.zeros((HOP, params.channels))
        rt._callback(block, out_block, HOP, None, type("S", (), {"input_overflow": False})())
        out[i * HOP : (i + 1) * HOP] = out_block
    return out, rt.engine


def block_rms(x: np.ndarray, hop: int):
    n = len(x) // hop
    return np.array([np.sqrt(np.mean(x[i * hop : (i + 1) * hop] ** 2)) for i in range(n)])


def analyze(name: str, signal: np.ndarray, f_target: float):
    params = FreezeParams(sample_rate=SR, block_size=HOP, fft_size=2048,
                          channels=signal.shape[1])
    n_hops = signal.shape[0] // HOP
    freeze_hop = int(0.9 * SR) // HOP
    stop_hop = int(1.3 * SR) // HOP
    release_hop = int(2.6 * SR) // HOP

    # Build the exact post-stop input stream for energy-source accounting.
    input_stream = signal.copy()
    input_stream[stop_hop * HOP :] = 0.0

    off_out, off_eng, freq_track = run_offline(
        signal, params, freeze_hop, stop_hop, release_hop, f_target
    )
    rt_out, _ = run_realtime_emulation(
        signal, FreezeParams(sample_rate=SR, block_size=HOP, fft_size=2048,
                             channels=signal.shape[1]),
        freeze_hop, stop_hop, release_hop,
    )

    consistency = float(np.max(np.abs(off_out - rt_out)))

    # Frozen region well after the input has been zeroed.
    tail = off_out[int(1.8 * SR) : int(2.4 * SR)]
    rms = block_rms(tail.mean(axis=1), HOP)
    amp_fluct = float(np.std(rms) / (np.mean(rms) + 1e-12))

    # Frequency drift measured across successive frozen snapshots.
    drift = float(np.ptp(freq_track)) if freq_track.size else float("nan")
    median_freq = float(np.median(freq_track)) if freq_track.size else float("nan")

    diff = np.abs(np.diff(off_out.mean(axis=1)))
    entry_center = (freeze_hop + 4) * HOP
    exit_center = (release_hop + 12) * HOP
    baseline = np.percentile(np.abs(np.diff(signal[: int(0.5 * SR)].mean(axis=1))), 99)
    entry_transient = float(diff[entry_center - 4 * HOP : entry_center + 4 * HOP].max())
    exit_transient = float(diff[exit_center - 4 * HOP : exit_center + 4 * HOP].max())

    # Energy source trace: the *engine input* is zero here but output stays on.
    input_tail_rms = float(np.sqrt(np.mean(input_stream[int(1.8 * SR) : int(2.4 * SR)] ** 2)))
    output_tail_rms = float(np.sqrt(np.mean(tail**2)))

    return {
        "signal": name,
        "offline_realtime_max_abs_diff": consistency,
        "capture_mode": off_eng._last_capture_report.capture_mode,
        "phase_jump_bins": off_eng._last_capture_report.n_phase_jump_bins,
        "median_frozen_frequency_hz": round(median_freq, 3),
        "frequency_drift_hz_peak_to_peak": round(drift, 9),
        "amplitude_fluctuation_cv": round(amp_fluct, 4),
        "entry_max_sample_step": round(entry_transient, 5),
        "exit_max_sample_step": round(exit_transient, 5),
        "baseline_p99_sample_step": round(float(baseline), 5),
        "engine_input_tail_rms": input_tail_rms,
        "output_tail_rms": round(output_tail_rms, 5),
        "frozen_only_energy_audible": output_tail_rms > 0.01 and input_tail_rms < 1e-12,
    }


def main():
    results = []
    musical = make_musical_signal(SR, 3.5, 2)
    ambient = make_ambient_signal(SR, 3.5, 2)
    results.append(analyze("homemade_musical", musical, f_target=220.0))
    results.append(analyze("synthetic_ambient", ambient, f_target=600.0))

    out_dir = Path("examples")
    out_dir.mkdir(exist_ok=True)
    import soundfile as sf
    sf.write(out_dir / "musical_input.wav", musical.astype(np.float32), SR)
    sf.write(out_dir / "ambient_input.wav", ambient.astype(np.float32), SR)

    for r in results:
        print(json.dumps(r, indent=2))
        assert r["offline_realtime_max_abs_diff"] < 1e-12
        assert r["frequency_drift_hz_peak_to_peak"] < 1e-9
        assert r["amplitude_fluctuation_cv"] < 0.3
        assert r["entry_max_sample_step"] < 10 * r["baseline_p99_sample_step"] + 1e-3
        assert r["exit_max_sample_step"] < 10 * r["baseline_p99_sample_step"] + 1e-3
        assert r["frozen_only_energy_audible"]
    Path(out_dir / "verification.json").write_text(json.dumps(results, indent=2))
    print("all consistency / quality checks passed")


if __name__ == "__main__":
    main()
