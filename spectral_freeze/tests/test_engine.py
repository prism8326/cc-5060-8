"""Numerical tests for the spectral freeze engine."""

import numpy as np
import pytest

from spectral_freeze.dsp import (
    equal_power_crossfade,
    interchannel_coherence,
    principal_instantaneous_frequency,
    wrap_phase,
)
from spectral_freeze.engine import FreezeEngine, FreezeParams, FreezeState
from spectral_freeze.signals import make_ambient_signal, make_musical_signal
from spectral_freeze.stft import STFT


SR = 48000


def feed(engine, signal, n_hops=None):
    hop = engine.p.block_size
    n_hops = n_hops or signal.shape[0] // hop
    outputs, reports = [], []
    for i in range(n_hops):
        block = signal[i * hop : (i + 1) * hop]
        y, r = engine.process_block(block)
        outputs.append(y)
        reports.append(r)
    return np.concatenate(outputs), reports


def make_engine(**kw):
    p = FreezeParams(sample_rate=SR, **kw)
    return FreezeEngine(p)


def test_stft_reconstructs_passthrough():
    sr, fft, hop, ch = SR, 2048, 256, 2
    stft = STFT(hop, fft, ch)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((hop * 60, ch)) * 0.3
    out = []
    # Warm-up fills the overlap buffers.
    for i in range(sig.shape[0] // hop):
        spec = stft.analysis(sig[i * hop : (i + 1) * hop])
        out.append(stft.synthesis(spec))
    out = np.concatenate(out)
    # Streaming WOLA latency measured empirically: over a steady region,
    # out[i0 : i0+N] == in[lag : lag+N] with a fixed lag of 3 hops
    # (frame fill + one-hop emission delay).
    lag = 3 * hop
    n = hop * 25
    a = sig[lag : lag + n]
    b = out[hop * 10 : hop * 10 + n]
    err = np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-12)
    assert err < 1e-6


def test_wrap_phase():
    x = np.array([[3 * np.pi, -3 * np.pi, 0.1]])
    w = wrap_phase(x)
    assert np.all(np.abs(w) <= np.pi)
    assert np.allclose(np.exp(1j * w), np.exp(1j * x))


def test_instantaneous_frequency_recovers_tone():
    fft, hop, sr = 2048, 256, SR
    f = 1000.0
    n = fft + hop * 4
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * f * t)
    stft = STFT(hop, fft, 1)
    phases = []
    for i in range(n // hop):
        spec = stft.analysis(tone[i * hop : (i + 1) * hop, None])
        phases.append(np.angle(spec))
    ifreq = principal_instantaneous_frequency(phases[-1], phases[-2], hop, fft)
    bin_k = int(round(f * fft / sr))
    est_hz = ifreq[bin_k, 0] * sr / (2 * np.pi)
    assert abs(est_hz - f) < 2.0  # within 2 Hz at 1 kHz


def test_freeze_continues_after_input_stops():
    """Output energy after input goes silent must come from the frozen state."""
    sig = make_musical_signal(SR, 3.0, 1)
    eng = make_engine(channels=1)
    hop = eng.p.block_size
    n_hops = sig.shape[0] // hop
    freeze_hop = int(1.0 * SR) // hop
    stop_hop = int(1.4 * SR) // hop
    outputs = []
    for i in range(n_hops):
        block = sig[i * hop : (i + 1) * hop].copy()
        if i == freeze_hop:
            eng.freeze()
        if i >= stop_hop:
            block.fill(0.0)
        y, _ = eng.process_block(block)
        outputs.append(y)
    out = np.concatenate(outputs)
    # Long after input silence, frozen output remains clearly audible.
    tail = out[int(2.4 * SR) : int(2.9 * SR)]
    rms = np.sqrt(np.mean(tail**2))
    assert rms > 0.01
    assert eng.state == FreezeState.FROZEN


def test_frozen_frequency_stays_on_track():
    """Estimated frozen frequencies must not drift over a long sustain."""
    fft, hop, sr = 2048, 256, SR
    dur = 4.0
    t = np.arange(int(dur * sr)) / sr
    f = 440.0
    sig = (0.5 * np.sin(2 * np.pi * f * t))[:, None]
    eng = make_engine(fft_size=fft, block_size=hop, channels=1)
    n_hops = sig.shape[0] // hop
    eng.freeze  # noop reference
    # Feed some history then freeze.
    for i in range(int(1.0 * sr) // hop):
        eng.process_block(sig[i * hop : (i + 1) * hop])
    eng.freeze()
    freqs = []
    for i in range(int(1.0 * sr) // hop, n_hops):
        eng.process_block(sig[i * hop : (i + 1) * hop])
        snap = eng.snapshot()
        bin_k = int(round(f * fft / sr))
        freqs.append(snap["frozen_frequency_hz"][bin_k, 0])
    freqs = np.array(freqs)
    # After the entry fade, IF estimates are held fixed: zero drift by design.
    assert np.ptp(freqs[4:]) < 1e-9
    assert abs(np.median(freqs) - f) < 3.0


def test_no_periodic_pulses_from_frame_copying():
    """Phase advancement must avoid buzz: output stays tonal, not hop-rate pulsed."""
    sig = make_musical_signal(SR, 2.5, 1)
    eng = make_engine(channels=1)
    out, _ = feed_then_freeze(eng, sig, freeze_sec=0.8)
    tail = out[int(1.8 * SR) : int(2.3 * SR), 0]
    spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
    freqs = np.fft.rfftfreq(len(tail), 1 / SR)
    peak_f = freqs[np.argmax(spectrum)]
    # Tonal content (partials of 220 Hz) dominates rather than a hop-rate buzz.
    assert 180 < peak_f < 1200
    # Frame-copy looping would amplitude-modulate the output at the hop rate
    # (SR/block_size = 187.5 Hz). Measure that envelope component explicitly.
    hop_rate = SR / eng.p.block_size
    envelope = np.abs(tail)
    env_spec = np.abs(np.fft.rfft(envelope - envelope.mean()))
    env_freqs = np.fft.rfftfreq(len(envelope), 1 / SR)
    hop_bin = np.argmin(np.abs(env_freqs - hop_rate))
    band = env_spec[max(0, hop_bin - 2) : hop_bin + 3]
    tonal_bin = np.argmin(np.abs(env_freqs - 220.0))
    assert band.max() < 0.5 * (env_spec[tonal_bin] + 1e-12)


def feed_then_freeze(engine, signal, freeze_sec=1.0, release_sec=None):
    hop = engine.p.block_size
    n_hops = signal.shape[0] // hop
    outs, reports = [], []
    frozen = False
    released = False
    for i in range(n_hops):
        t = i * hop / SR
        if not frozen and t >= freeze_sec and engine.state == FreezeState.LIVE:
            engine.freeze()
            frozen = True
        if (
            release_sec
            and not released
            and t >= release_sec
            and engine.state in (FreezeState.ENTERING, FreezeState.FROZEN)
        ):
            engine.unfreeze()
            released = True
        y, r = engine.process_block(signal[i * hop : (i + 1) * hop])
        outs.append(y)
        reports.append(r)
    return np.concatenate(outs), reports


def test_entry_and_exit_are_click_free():
    sig = make_musical_signal(SR, 5.0, 1)
    eng = make_engine(channels=1, entry_fade_sec=0.08, exit_fade_sec=0.12)
    out, reports = feed_then_freeze(eng, sig, freeze_sec=1.0, release_sec=2.5)
    # Sample-to-sample discontinuity around transitions must stay bounded.
    diff = np.abs(np.diff(out[:, 0]))
    live_diff = np.percentile(np.abs(np.diff(sig[: int(0.5 * SR), 0])), 99)
    assert np.max(diff) < 10 * live_diff + 1e-3
    states = [r.state for r in reports]
    assert FreezeState.ENTERING in states
    assert FreezeState.FROZEN in states
    assert FreezeState.EXITING in states
    assert states[-1] == FreezeState.LIVE


def test_phase_jump_triggers_multi_window_capture():
    sig = make_musical_signal(SR, 3.0, 1)  # jump at 40% = 1.2 s
    eng = make_engine(channels=1, capture_windows=4)
    # Freeze while the discontinuity is still inside the analysis window.
    hop = eng.p.block_size
    for i in range(int(1.25 * SR) // hop):
        eng.process_block(sig[i * hop : (i + 1) * hop])
    eng.freeze("auto")
    report = eng._last_capture_report
    assert report.capture_mode == "multi"
    assert report.n_phase_jump_bins > 0


def test_steady_tone_uses_single_window_capture():
    sig = make_musical_signal(SR, 3.0, 1)
    eng = make_engine(channels=1, capture_windows=4)
    hop = eng.p.block_size
    for i in range(int(0.9 * SR) // hop):
        eng.process_block(sig[i * hop : (i + 1) * hop])
    eng.freeze("auto")
    assert eng._last_capture_report.capture_mode == "single"
    assert eng._last_capture_report.n_phase_jump_bins == 0


def test_near_silence_capture_is_suppressed():
    eng = make_engine(channels=1, near_silence_db=-60.0)
    hop = eng.p.block_size
    rng = np.random.default_rng(3)
    silence = rng.standard_normal((hop * 10, 1)) * 1e-6
    for i in range(10):
        eng.process_block(silence[i * hop : (i + 1) * hop])
    eng.freeze()
    assert np.allclose(eng._frozen_mag, 0.0, atol=1e-12)
    out = []
    for i in range(20):
        y, _ = eng.process_block(np.zeros((hop, 1)))
        out.append(y)
    out = np.concatenate(out)
    # WOLA window energy floor bounds the residual of a suppressed capture.
    assert np.max(np.abs(out)) < 1e-5
    assert np.sqrt(np.mean(out**2)) < 1e-6


def test_dc_bin_does_not_accumulate():
    sig = make_musical_signal(SR, 2.0, 1)  # includes 0.002 DC offset
    eng = make_engine(channels=1)
    out, _ = feed_then_freeze(eng, sig, freeze_sec=0.8)
    tail = out[int(1.5 * SR) :, 0]
    assert abs(np.mean(tail)) < 0.01


def test_stereo_coherence_collapses_coherent_bins():
    # Fully coherent stereo signal.
    t = np.arange(int(2.0 * SR)) / SR
    mono = 0.5 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([mono, mono], axis=1)
    eng = make_engine(channels=2, coherence_threshold=0.6)
    hop = eng.p.block_size
    for i in range(int(1.0 * SR) // hop):
        eng.process_block(stereo[i * hop : (i + 1) * hop])
    eng.freeze()
    snap = eng.snapshot()
    # Coherence is meaningful only at energetic bins (the tone and its leakage).
    mag = snap["frozen_magnitude"].mean(axis=1)
    active = mag > mag.max() * 1e-3
    assert np.mean(snap["coherence"][active] > 0.6) > 0.8
    # After several frozen hops, coherent bins are identical across channels.
    for _ in range(10):
        eng.process_block(stereo[:hop])
    spec = eng._current_frozen_spectrum()
    bin_k = int(round(330 * eng.p.fft_size / SR))
    assert np.isclose(spec[bin_k, 0], spec[bin_k, 1], rtol=1e-9)


def test_coherence_helper():
    rng = np.random.default_rng(0)
    fft = 256
    base = np.exp(1j * rng.uniform(0, 2 * np.pi, (20, fft // 2 + 1)))
    coherent = np.stack([base, base], axis=2)
    coh = interchannel_coherence(coherent, 8)
    assert np.all(coh > 0.99)
    noise = np.exp(1j * rng.uniform(0, 2 * np.pi, (20, fft // 2 + 1, 2)))
    coh2 = interchannel_coherence(noise, 8)
    assert np.mean(coh2) < 0.4


def test_equal_power_crossfade():
    g = np.linspace(0, 1, 11)
    a, b = equal_power_crossfade(g)
    assert np.allclose(a**2 + b**2, 1.0)


def test_ambient_signal_freeze_is_stable():
    sig = make_ambient_signal(SR, 3.0, 2)
    eng = make_engine(channels=2)
    out, reports = feed_then_freeze(eng, sig, freeze_sec=1.0)
    # Frozen tail amplitude should be stable frame-to-frame (low fluctuation).
    rms_per_block = []
    hop = eng.p.block_size
    tail = out[int(2.0 * SR) :]
    for i in range(0, len(tail) - hop, hop):
        rms_per_block.append(np.sqrt(np.mean(tail[i : i + hop] ** 2)))
    rms_per_block = np.array(rms_per_block)
    fluct = np.std(rms_per_block) / (np.mean(rms_per_block) + 1e-12)
    assert fluct < 0.25


def test_reports_trace_energy_source():
    sig = make_musical_signal(SR, 2.5, 1)
    eng = make_engine(channels=1)
    _, reports = feed_then_freeze(eng, sig, freeze_sec=1.0)
    frozen_reports = [r for r in reports if r.state == FreezeState.FROZEN]
    assert frozen_reports and all(r.frozen_rms_db > -40 for r in frozen_reports)
    assert all(isinstance(r.top_frequencies_hz, list) for r in frozen_reports)


# --------------------------------------------------------------------- #
# Fast frozen path (phasor reuse) vs the non-reusing reference
# --------------------------------------------------------------------- #
def _frozen_run(reuse: bool, seconds: float = 1.5, drift_at=None, drift_hz=6.0):
    sig = make_musical_signal(SR, seconds + 0.8, 2)
    p = FreezeParams(sample_rate=SR, channels=2, reuse_cache=reuse)
    eng = FreezeEngine(p)
    hop = p.block_size
    n_hops = sig.shape[0] // hop
    out = np.zeros_like(sig)
    for i in range(n_hops):
        block = sig[i * hop : (i + 1) * hop].copy()
        t = i * hop / SR
        if t >= 0.6 and eng.state == FreezeState.LIVE and eng._hist_count >= eng._warmup_hops:
            eng.freeze()
        if drift_at is not None and t >= drift_at:
            p.drift_compensation = drift_hz
        out[i * hop : (i + 1) * hop] = eng.process_block(block)[0]
    return out, eng


def test_fast_path_matches_reference_path():
    """Reusing static bins must reproduce the full per-bin recomputation."""
    a, ea = _frozen_run(reuse=True)
    b, eb = _frozen_run(reuse=False)
    assert np.max(np.abs(a - b)) < 1e-12
    # Every bin the reference advanced agrees to float rounding; none were
    # left frozen by the fast path.
    diff = np.abs(ea._frozen_spec - eb._frozen_spec)
    assert np.max(diff) < 1e-12
    assert np.count_nonzero(diff > 1e-12) == 0


def test_static_bins_really_static():
    """Bins with zero estimated frequency keep a constant complex value."""
    _, eng = _frozen_run(reuse=True)
    static = ~eng._active_mask
    assert np.any(static)
    before = eng._frozen_spec.copy()
    eng.process_block(np.zeros((eng.p.block_size, 2)))
    changed = np.abs(eng._frozen_spec - before) > 0
    assert not np.any(changed & static)


def test_drift_change_mid_freeze_matches_reference():
    """Enabling a pitch nudge while frozen must advance every bin."""
    a, _ = _frozen_run(reuse=True, drift_at=0.9)
    b, _ = _frozen_run(reuse=False, drift_at=0.9)
    assert np.max(np.abs(a - b)) < 1e-12


def test_sample_rate_change_mid_freeze_matches_reference():
    """Changing sample rate + drift while frozen stays consistent."""
    sig = make_musical_signal(SR, 2.0, 1)

    def run(reuse):
        p = FreezeParams(sample_rate=SR, channels=1, reuse_cache=reuse)
        eng = FreezeEngine(p)
        hop = p.block_size
        out = np.zeros_like(sig)
        for i in range(sig.shape[0] // hop):
            t = i * hop / SR
            if t >= 0.6 and eng.state == FreezeState.LIVE and eng._hist_count >= eng._warmup_hops:
                eng.freeze()
            if t >= 1.0:
                p.sample_rate = 96000
                p.drift_compensation = 5.0
            out[i * hop : (i + 1) * hop] = eng.process_block(sig[i * hop : (i + 1) * hop])[0]
        return out

    a, b = run(True), run(False)
    assert np.all(np.isfinite(a))
    assert np.max(np.abs(a - b)) < 1e-12


def test_frozen_analysis_skipped_but_exit_recovers_live():
    """Analysis is skipped while frozen yet the exit resumes live cleanly."""
    sig = make_musical_signal(SR, 3.0, 1)
    eng = make_engine(channels=1)
    hop = eng.p.block_size
    out, reports = feed_then_freeze(eng, sig, freeze_sec=0.8, release_sec=2.0)
    states = [r.state for r in reports]
    assert FreezeState.EXITING in states and states[-1] == FreezeState.LIVE
    # After the exit completes, live passthrough tracks the input again.
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out[int(2.4 * SR) :])) > 0.01

