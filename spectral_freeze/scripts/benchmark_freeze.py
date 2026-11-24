"""Sustained-synthesis cost breakdown and reference cross-check.

The frozen state is the expensive steady state: it must keep sounding for as
long as the input is silent. This script measures that path exactly as the
GUI / real-time / offline callers drive it (``FreezeEngine.process_block``),
under two loads:

* a long sustained note whose input has been zeroed (pure frozen output), and
* a parameter drag (coherence threshold and gain change every hop), which is
  what a real user does while shaping the freeze.

For each load it records per-hop CPU time and the frozen frequency track (the
phase-advance drift, which is zero by construction because the estimation is
done once). It then runs the *non-reusing* reference path and compares the two
spectra per hop: the complex phase difference and the number of bins the
reusing fast path failed to advance ("missed updates"), which must both be
zero. Nothing here is reported unless the numbers were actually measured.

Output: ``examples/freeze_benchmark.json``.

Optional baseline
-----------------
Pass ``--baseline-src DIR`` pointing at an older checkout of the package to
also measure the *original* implementation on the same machine in a fresh
subprocess, and record the honest original-vs-optimized ratio. The subprocess
imports ``spectral_freeze`` from ``DIR`` so the two code versions never mix::

    git worktree add /tmp/baseline HEAD
    python -m scripts.benchmark_freeze --baseline-src /tmp/baseline/spectral_freeze
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from spectral_freeze.engine import FreezeEngine, FreezeParams, FreezeState
from spectral_freeze.signals import make_musical_signal

SR = 48000
HOP = 256


def _params(reuse: bool) -> FreezeParams:
    return FreezeParams(
        sample_rate=SR, block_size=HOP, fft_size=2048, channels=2, reuse_cache=reuse
    )


def _prime(eng: FreezeEngine, signal: np.ndarray, seconds: float = 1.0) -> None:
    for i in range(int(seconds * SR) // HOP):
        eng.process_block(signal[i * HOP : (i + 1) * HOP])
    eng.freeze()
    while eng.state != FreezeState.FROZEN:
        eng.process_block(np.zeros((HOP, 2)))


def measure_sustain(reuse: bool, seconds: float = 16.0, repeats: int = 7) -> dict:
    # Interleave nothing here; the caller alternates fast/reference so slow
    # system periods hit both paths equally. Best-case per-hop time is robust
    # to scheduler noise, so it is the headline number.
    sig = make_musical_signal(SR, 2.0, 2)
    eng = FreezeEngine(_params(reuse))
    _prime(eng, sig)
    f0_bin = int(round(220.0 * eng.p.fft_size / SR))
    blk = np.zeros((HOP, 2))
    n_hops = int(seconds * SR) // HOP
    per_hop = []
    freq_track = []
    spec_hashes = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for k in range(n_hops):
            eng.process_block(blk)
            if k % 64 == 0:
                freq_track.append(eng._frozen_freq[f0_bin].mean())
                spec_hashes.append(hash(eng._frozen_spec.tobytes()))
        per_hop.append((time.perf_counter() - t0) / n_hops * 1e6)
    freq_hz = np.array(freq_track) * SR / (2 * np.pi)
    return {
        "path": "reuse" if reuse else "reference",
        "us_per_hop_best": round(float(np.min(per_hop)), 2),
        "us_per_hop_median": round(float(np.median(per_hop)), 2),
        "realtime_budget_us_per_hop": round(HOP / SR * 1e6, 1),
        "frozen_frequency_drift_hz_ptp": round(float(np.ptp(freq_hz)), 9),
        "total_frozen_spectrum_changes": len(set(spec_hashes)) - 1,
        "hops": n_hops * repeats,
    }


def measure_drag(reuse: bool, seconds: float = 8.0, repeats: int = 5) -> dict:
    sig = make_musical_signal(SR, 2.0, 2)
    p = _params(reuse)
    eng = FreezeEngine(p)
    _prime(eng, sig)
    blk = np.zeros((HOP, 2))
    n_hops = int(seconds * SR) // HOP
    per_hop = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(n_hops):
            # A user dragging the coherence slider and the output gain.
            p.coherence_threshold = 0.3 + 0.3 * (0.5 + 0.5 * np.sin(i * 0.02))
            p.gain = 0.5 + 0.5 * (0.5 + 0.5 * np.sin(i * 0.01))
            eng.process_block(blk)
        per_hop.append((time.perf_counter() - t0) / n_hops * 1e6)
    return {
        "path": "reuse" if reuse else "reference",
        "us_per_hop_best": round(float(np.min(per_hop)), 2),
        "us_per_hop_median": round(float(np.median(per_hop)), 2),
        "hops": n_hops * repeats,
    }


def cross_check(seconds: float = 4.0) -> dict:
    """Run fast and reference paths in lock-step; compare spectra per hop."""
    sig = make_musical_signal(SR, 2.0, 2)
    fast = FreezeEngine(_params(True))
    ref = FreezeEngine(_params(False))
    _prime(fast, sig)
    _prime(ref, sig)
    blk = np.zeros((HOP, 2))
    n_hops = int(seconds * SR) // HOP
    max_phase_diff = 0.0
    max_complex_diff = 0.0
    missed_updates = 0
    prev_ref = ref._frozen_spec.copy()
    prev_fast = fast._frozen_spec.copy()
    for _ in range(n_hops):
        fast.process_block(blk)
        ref.process_block(blk)
        d = ref._frozen_spec - fast._frozen_spec
        max_complex_diff = max(max_complex_diff, float(np.max(np.abs(d))))
        mag = np.abs(ref._frozen_spec)
        # Phase difference only where there is energy to define a phase.
        live = mag > mag.max() * 1e-4
        if np.any(live):
            max_phase_diff = max(
                max_phase_diff,
                float(
                    np.max(
                        np.abs(
                            np.angle(
                                ref._frozen_spec[live] * np.conj(fast._frozen_spec[live])
                            )
                        )
                    )
                ),
            )
        # Missed update: reference moved a bin this hop but fast did not.
        ref_moved = np.abs(ref._frozen_spec - prev_ref) > 1e-12
        fast_moved = np.abs(fast._frozen_spec - prev_fast) > 1e-12
        missed_updates += int(np.count_nonzero(ref_moved & ~fast_moved))
        prev_ref = ref._frozen_spec.copy()
        prev_fast = fast._frozen_spec.copy()
    return {
        "hops": n_hops,
        "max_bin_phase_difference_rad": max_phase_diff,
        "max_bin_complex_difference": max_complex_diff,
        "missed_phase_updates": missed_updates,
    }


def cost_breakdown(seconds: float = 6.0) -> dict:
    """Attribute frozen-hop time to synthesis, phase advance and reporting."""
    sig = make_musical_signal(SR, 2.0, 2)
    eng = FreezeEngine(_params(True))
    _prime(eng, sig)
    blk = np.zeros((HOP, 2))
    n = int(seconds * SR) // HOP
    # Whole hop.
    t0 = time.perf_counter()
    for _ in range(n):
        eng.process_block(blk)
    whole = (time.perf_counter() - t0) / n * 1e6
    # Component timings, averaged over the same count.
    spec = eng._frozen_spec.copy()
    t0 = time.perf_counter()
    for _ in range(n):
        eng.stft.push(blk)
    push = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        eng.stft.synthesis(spec)
    synth = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        eng._advance_frozen()
    advance = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        eng._build_report(blk, spec, blk, 1.0, "")
    report = (time.perf_counter() - t0) / n * 1e6
    return {
        "whole_hop_us": round(whole, 2),
        "analysis_push_us": round(push, 2),
        "synthesis_us": round(synth, 2),
        "phase_advance_us": round(advance, 2),
        "report_us": round(report, 2),
        "remainder_us": round(whole - (push + synth + advance + report), 2),
    }


def _measure_original(src: Path) -> dict:
    """Measure an older package checkout in a fresh subprocess.

    Runs the same sustain and drag loops with ``spectral_freeze`` imported
    from ``src``'s parent, so the original implementation is timed without any
    of the optimized code being importable.
    """
    if not (src / "engine.py").exists():
        raise FileNotFoundError(f"no spectral_freeze/engine.py under {src}")
    code = r"""
import json, sys, time
import numpy as np
import spectral_freeze
from spectral_freeze.engine import FreezeEngine, FreezeParams, FreezeState
from spectral_freeze.signals import make_musical_signal
sys.stderr.write('baseline package: ' + spectral_freeze.__file__ + '\n')
SR, HOP = 48000, 256
sig = make_musical_signal(SR, 2.0, 2)
def prime(e):
    for i in range(int(1.0 * SR) // HOP):
        e.process_block(sig[i*HOP:(i+1)*HOP])
    e.freeze()
    while e.state != FreezeState.FROZEN:
        e.process_block(np.zeros((HOP, 2)))
def sustain():
    p = FreezeParams(sample_rate=SR, block_size=HOP, fft_size=2048, channels=2)
    e = FreezeEngine(p); prime(e)
    blk = np.zeros((HOP, 2)); best = 1e9
    n = int(16.0 * SR) // HOP
    for _ in range(7):
        t0 = time.perf_counter()
        for _ in range(n): e.process_block(blk)
        best = min(best, (time.perf_counter()-t0)/n*1e6)
    return best
def drag():
    p = FreezeParams(sample_rate=SR, block_size=HOP, fft_size=2048, channels=2)
    e = FreezeEngine(p); prime(e)
    blk = np.zeros((HOP, 2)); best = 1e9; n = int(8.0*SR)//HOP
    for _ in range(5):
        t0 = time.perf_counter()
        for i in range(n):
            p.coherence_threshold = 0.3 + 0.3*(0.5+0.5*np.sin(i*0.02))
            p.gain = 0.5 + 0.5*(0.5+0.5*np.sin(i*0.01))
            e.process_block(blk)
        best = min(best, (time.perf_counter()-t0)/n*1e6)
    return best
print(json.dumps({"sustain_us_per_hop_min": sustain(), "drag_us_per_hop_min": drag()}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src.parent)
    # Run from a neutral directory: ``python -c`` puts the current directory
    # first on sys.path, which would otherwise import this checkout instead.
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(src.parent),
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Sustained freeze benchmark")
    ap.add_argument("--baseline-src", default=None,
                    help="path to an older spectral_freeze package to measure "
                         "the original implementation on the same machine")
    args = ap.parse_args()

    fast_sustain = measure_sustain(True)
    ref_sustain = measure_sustain(False)
    fast_drag = measure_drag(True)
    ref_drag = measure_drag(False)
    check = cross_check()
    breakdown = cost_breakdown()

    # Best-case time is the clean signal; report both ratios.
    fast_best = fast_sustain["us_per_hop_best"]
    ref_best = ref_sustain["us_per_hop_best"]
    speedup = ref_best / fast_best
    result = {
        "sample_rate": SR,
        "block_size": HOP,
        "sustain_fast": fast_sustain,
        "sustain_reference": ref_sustain,
        "sustain_speedup_reference_over_fast": round(speedup, 3),
        "drag_fast": fast_drag,
        "drag_reference": ref_drag,
        "drag_speedup_reference_over_fast": round(
            ref_drag["us_per_hop_best"] / fast_drag["us_per_hop_best"], 3
        ),
        "fast_vs_reference": check,
        "frozen_hop_cost_breakdown_fast": breakdown,
    }

    if args.baseline_src:
        orig = _measure_original(Path(args.baseline_src))
        result["original_baseline"] = orig
        result["sustain_speedup_original_over_optimized"] = round(
            orig["sustain_us_per_hop_min"] / fast_best, 3
        )
        result["drag_speedup_original_over_optimized"] = round(
            orig["drag_us_per_hop_min"] / fast_drag["us_per_hop_best"], 3
        )

    out_dir = Path(__file__).resolve().parent.parent / "examples"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "freeze_benchmark.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # The optimization must not change behavior: zero missed updates, zero
    # phase difference beyond float rounding, and zero frequency drift.
    assert check["missed_phase_updates"] == 0, "fast path missed a phase update"
    assert check["max_bin_complex_difference"] < 1e-12, "fast/reference spectra diverge"
    assert fast_sustain["frozen_frequency_drift_hz_ptp"] < 1e-9
    assert ref_sustain["frozen_frequency_drift_hz_ptp"] < 1e-9
    assert fast_sustain["total_frozen_spectrum_changes"] > 0, "frozen spectrum never advanced"
    print("benchmark checks passed")


if __name__ == "__main__":
    main()
