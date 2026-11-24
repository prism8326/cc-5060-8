"""Streaming spectral-freeze engine.

State machine
-------------
LIVE -> ENTERING -> FROZEN -> EXITING -> LIVE

* LIVE: normal passthrough (analysis and synthesis run identically so the
  reconstruction path is shared and verified).
* ENTERING: equal-power crossfade from live spectrum to the captured frozen
  spectrum over ``entry_fade_sec``.
* FROZEN: the captured magnitude distribution is held; phases are advanced
  every hop by per-bin instantaneous frequencies estimated over several
  analysis windows. Raw waveform looping and complex-frame copying are never
  used.
* EXITING: crossfade back to the live input spectrum.

Every emitted hop carries a :class:`FrameReport` so the energy source,
frequency trajectories and post-stop behavior are fully traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .dsp import (
    equal_power_crossfade,
    interchannel_coherence,
    magnitude_weighted_mean,
    principal_instantaneous_frequency,
    weighted_circular_mean,
    wrap_phase,
)
from .stft import STFT, hann_window


class FreezeState(Enum):
    LIVE = "live"
    ENTERING = "entering"
    FROZEN = "frozen"
    EXITING = "exiting"


@dataclass
class FreezeParams:
    sample_rate: int = 48000
    block_size: int = 256          # hop size in samples
    fft_size: int = 2048
    channels: int = 2
    capture_windows: int = 4       # multi-window frequency estimation depth
    coherence_window: int = 8      # frames used for inter-channel coherence
    coherence_threshold: float = 0.6
    entry_fade_sec: float = 0.05
    exit_fade_sec: float = 0.10
    near_silence_db: float = -60.0
    drift_compensation: float = 0.0  # user pitch nudge in Hz, applied in freeze
    gain: float = 1.0
    # Reuse the per-bin phasor cache in the frozen state. Set to False for a
    # reference run that recomputes every bin every hop (benchmark/validation).
    reuse_cache: bool = True


@dataclass
class FrameReport:
    """Per-hop trace record (one row per processed hop)."""

    state: FreezeState
    live_rms_db: float
    frozen_rms_db: float
    output_rms_db: float
    freeze_progress: float          # 0..1 crossfade position
    capture_mode: str = ""          # "single" | "multi" | ""
    n_phase_jump_bins: int = 0
    n_coherent_bins: int = 0
    top_frequencies_hz: list = field(default_factory=list)
    dc_magnitude: float = 0.0
    near_silence: bool = False
    note: str = ""


class FreezeEngine:
    def __init__(self, params: FreezeParams):
        self.p = params
        self.stft = STFT(params.block_size, params.fft_size, params.channels)
        self.bins = params.fft_size // 2 + 1
        self._state = FreezeState.LIVE
        # Ring of recent complex spectra (oldest .. newest), depth enough for
        # multi-window IF estimation and coherence analysis.
        depth = max(params.capture_windows + 1, params.coherence_window) + 1
        self._history = np.zeros((depth, self.bins, params.channels), dtype=complex)
        self._hist_count = 0
        # Frozen snapshot.
        self._frozen_mag = np.zeros((self.bins, params.channels))
        self._frozen_phase = np.zeros((self.bins, params.channels))
        self._frozen_freq = np.zeros((self.bins, params.channels))  # rad/sample
        self._frozen_coherence = np.zeros(self.bins)
        self._frozen_hops = 0
        self._frozen_rms_db = -240.0
        # Per-bin complex value for the frozen state. Bins whose estimated
        # frequency is exactly zero and with no global drift nudge keep their
        # captured phase forever, so ``mag * exp(1j*phase)`` is constant and is
        # computed once on capture instead of every hop. Only bins that
        # actually move in phase are re-phased each hop.
        self._frozen_spec = np.zeros((self.bins, params.channels), dtype=complex)
        self._active_mask: np.ndarray | None = None
        self._active_empty = True
        self._drift_cached: float | None = None
        self._all_mask = np.ones((self.bins, params.channels), dtype=bool)
        # Coherence mask cache: rebuilt only when the threshold actually moves,
        # so dragging the coherence slider does not recompute it every hop.
        self._coh_mask: np.ndarray | None = None
        self._coh_threshold_cached: float | None = None
        # Last frozen spectrum emitted, for the exit crossfade origin and the
        # report RMS (the exact complex frame synthesis consumed).
        self._last_frozen_spec = np.zeros((self.bins, params.channels), dtype=complex)
        # Transition bookkeeping.
        self._fade_steps_total = 1
        self._fade_step = 0
        self._start_spectrum: np.ndarray | None = None  # blend origin
        self._stop_mode: str | None = None
        self.reports: list[FrameReport] = []
        # Minimum hops before a capture is valid: one full FFT window must
        # have streamed through, otherwise the analysis frame contains
        # leading zeros and a freeze would capture silence or a partial tone.
        self._warmup_hops = params.fft_size // params.block_size
        # Precomputed DC drift floor.
        self._bin_hz = np.fft.rfftfreq(params.fft_size, d=1.0 / params.sample_rate)
        # Hann window tail over the newest hop, for O(hop) report RMS.
        self._hann_tail = hann_window(params.fft_size)[-params.block_size :, None]

    @staticmethod
    def _spec_rms(spec: np.ndarray) -> float:
        """Frame RMS of an rFFT spectrum via Parseval (no inverse FFT).

        Matches ``sqrt(mean(irfft(spec)**2))`` averaged over samples and
        channels. irfft treats DC and Nyquist as real, so only their real part
        contributes.
        """
        n = 2 * (spec.shape[0] - 1)
        p = spec.real**2 + spec.imag**2
        power = 2.0 * p[1:-1].sum() + (spec[0].real**2).sum() + (spec[-1].real**2).sum()
        return float(np.sqrt(power / (n * n * spec.shape[1])))

    # ------------------------------------------------------------------ #
    # Public control API
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> FreezeState:
        return self._state

    def freeze(self, mode: str = "auto") -> None:
        """Capture the current spectral state and enter the freeze.

        Parameters
        ----------
        mode:
            "auto" picks single vs multi-window capture from phase-jump
            statistics; "single" forces last-frame copy; "multi" forces the
            multi-window estimator.
        """
        if self._state in (FreezeState.ENTERING, FreezeState.FROZEN):
            return
        if self._hist_count < self._warmup_hops:
            raise RuntimeError(
                f"analysis buffer not primed: need {self._warmup_hops} hops, "
                f"have {self._hist_count}"
            )
        self._stop_mode = mode
        self._capture(mode)
        self._start_spectrum = self._history[-1].copy()
        self._frozen_hops = 0
        self._fade_step = 0
        self._fade_steps_total = max(
            1, int(round(self.p.entry_fade_sec * self.p.sample_rate / self.p.block_size))
        )
        self._state = FreezeState.ENTERING

    def unfreeze(self) -> None:
        """Begin the exit transition back to live audio."""
        if self._state not in (FreezeState.ENTERING, FreezeState.FROZEN):
            return
        self._start_spectrum = self._current_frozen_spectrum()
        self._fade_step = 0
        self._fade_steps_total = max(
            1, int(round(self.p.exit_fade_sec * self.p.sample_rate / self.p.block_size))
        )
        self._state = FreezeState.EXITING
        # The analysis history may be stale (frozen hops skipped analysis).
        # Force a fresh warm-up before any re-freeze so a capture never builds
        # on stale spectra; exit already recomputes live analysis next hop.
        self._history.fill(0.0)
        self._hist_count = 0

    # ------------------------------------------------------------------ #
    # Capture and estimation
    # ------------------------------------------------------------------ #
    def _capture(self, mode: str) -> FrameReport:
        p = self.p
        n = min(self._hist_count, self._history.shape[0])
        hist = self._history[-n:]
        latest = hist[-1]
        mag_latest = np.abs(latest)

        # Phase-jump detection at dominant spectral peaks. Phases of the last
        # (w+1) frames are unwrapped and a line is fitted to the older w
        # frames; the newest frame is then extrapolated. Steady tones follow
        # the line to within ~0.05 rad, while a discontinuity leaves a large
        # extrapolation residual (0.15..pi rad). Per-pair instantaneous
        # frequencies cannot be used because they absorb the jump in one hop
        # and their hop-to-hop sum telescopes to the observed phase change.
        w = min(p.capture_windows, n - 1)
        if n >= w + 1 and w >= 3:
            inst = principal_instantaneous_frequency(
                np.angle(latest), np.angle(hist[-2]), p.block_size, p.fft_size
            )
            phase_stack = np.unwrap(
                np.angle(np.stack([hist[-1 - w + k] for k in range(w + 1)])), axis=0
            )  # shape (w+1, bins, channels)
            # Vectorized least-squares slope over the first w frames; np.polyfit
            # only accepts up to 2-D y.
            x = np.arange(w, dtype=np.float64)
            xc = x - x.mean()
            denom = np.sum(xc**2)
            slope = np.sum(xc[:, None, None] * phase_stack[:-1], axis=0) / denom
            predicted = phase_stack[-2] + slope
            residual = np.abs(wrap_phase(phase_stack[-1] - predicted)).mean(axis=1)
            mag_mean = mag_latest.mean(axis=1)
            peaks = np.zeros(self.bins, dtype=bool)
            eligible = mag_mean >= mag_mean.max() * 0.2
            for k in np.where(eligible)[0]:
                lo, hi = max(1, k - 2), min(self.bins - 1, k + 2)
                if mag_mean[k] == mag_mean[lo : hi + 1].max():
                    peaks[k] = True
            # Steady tones stay below ~0.05 rad, smooth chirps below ~0.3 rad,
            # a real discontinuity leaves 0.5 rad..pi residuals at the peaks.
            n_jumps = int(np.count_nonzero(peaks & (residual > 0.4)))
        elif n == 2:
            inst = principal_instantaneous_frequency(
                np.angle(latest), np.angle(hist[-2]), p.block_size, p.fft_size
            )
            n_jumps = 0
        else:
            inst = np.zeros_like(mag_latest)
            n_jumps = 0

        near_silence = self._is_near_silent(latest)

        if mode == "auto":
            capture_mode = "single" if n_jumps == 0 or n < p.capture_windows else "multi"
        else:
            capture_mode = mode

        if capture_mode == "multi" and n >= 3:
            w = min(p.capture_windows, n - 1)
            window_mags = np.abs(hist[-w - 1 :])
            # Magnitude distribution: weighted mean over the capture windows
            # (smooths single-frame amplitude spikes).
            weights = window_mags**2
            self._frozen_mag = magnitude_weighted_mean(window_mags, weights, axis=0)
            # Per-bin instantaneous frequencies between every consecutive
            # window pair, combined with magnitude-weighted circular means.
            ifs = []
            for k in range(1, w + 1):
                ifs.append(
                    principal_instantaneous_frequency(
                        np.angle(hist[-w - 1 + k]),
                        np.angle(hist[-w - 1 + k - 1]),
                        p.block_size,
                        p.fft_size,
                    )
                )
            ifs = np.stack(ifs, axis=0)
            w_if = window_mags[1:]
            self._frozen_freq = magnitude_weighted_mean(ifs, w_if, axis=0)
            # Starting phase: magnitude-weighted circular mean so a jump on
            # the final frame does not dominate.
            self._frozen_phase = weighted_circular_mean(
                np.angle(hist[-w - 1 :]), weights, axis=0
            )
        else:
            # Last-instant capture: exact final frame, single-window IF.
            self._frozen_mag = mag_latest.copy()
            self._frozen_phase = np.angle(latest).copy()
            self._frozen_freq = inst.copy()
            capture_mode = "single"

        # DC / near-silence protection.
        dc = float(self._frozen_mag[0].max())
        if near_silence:
            self._frozen_mag *= 0.0
            self._frozen_freq[:] = 0.0
        # Force the DC bin to stay at zero phase increment; clamp its
        # magnitude to avoid DC wander during long freezes.
        self._frozen_freq[0, :] = 0.0
        self._frozen_mag[0, :] = min(self._frozen_mag[0].max(), dc)
        # Limit extreme IF estimates to +/- Nyquist and kill IFs of bins with
        # negligible energy to stop numerical divergence in long freezes.
        nyq = np.pi
        floor = max(self._frozen_mag.max() * 1e-4, 1e-12)
        weak = self._frozen_mag < floor
        self._frozen_freq = np.clip(self._frozen_freq, -nyq, nyq)
        self._frozen_freq[weak] = 0.0

        # Stereo coherence decides mono/stereo continuation per bin.
        if p.channels == 2:
            self._frozen_coherence = interchannel_coherence(hist, p.coherence_window)
        else:
            self._frozen_coherence = np.ones(self.bins)

        # Bins with a zero estimated frequency and no drift nudge never advance:
        # their captured value is static and is reused. A global drift nudge
        # moves every bin, so it disables the static fast path.
        self._refresh_active_mask()
        self._coh_mask = None
        # Seed the complex frozen spectrum once from magnitude and phase.
        self._frozen_spec = self._frozen_mag * np.exp(1j * self._frozen_phase)
        # Frozen RMS is phase-invariant, so it is fixed by the captured
        # magnitude and computed once instead of every hop in the report.
        self._frozen_rms_db = 20.0 * np.log10(self._spec_rms(self._frozen_spec) + 1e-12)

        self._last_capture_report = FrameReport(
            state=FreezeState.ENTERING,
            live_rms_db=0.0,
            frozen_rms_db=0.0,
            output_rms_db=0.0,
            freeze_progress=0.0,
            capture_mode=capture_mode,
            n_phase_jump_bins=n_jumps,
            n_coherent_bins=int(np.sum(self._frozen_coherence > p.coherence_threshold)),
            top_frequencies_hz=self._top_frequencies(),
            dc_magnitude=dc,
            near_silence=near_silence,
            note="near-silence capture suppressed" if near_silence else "",
        )
        return self._last_capture_report

    def _is_near_silent(self, spectrum: np.ndarray) -> bool:
        # Evaluate only the most recent hop of the analysis frame: during
        # stream start-up the FFT window still contains leading zeros, which
        # would otherwise make every early capture look silent.
        frame = np.fft.irfft(spectrum, n=self.p.fft_size, axis=0)
        recent = frame[self.p.fft_size - self.p.block_size :]
        rms = float(np.sqrt(np.mean(recent**2) + 1e-20))
        db = 20.0 * np.log10(rms + 1e-12)
        return db < self.p.near_silence_db

    def _top_frequencies(self, k: int = 5) -> list:
        energy = self._frozen_mag.mean(axis=1)
        energy[:1] = 0.0
        idx = np.argsort(energy)[::-1][:k]
        idx = idx[energy[idx] > energy.max() * 1e-3]
        return [round(float(self._bin_hz[i]), 2) for i in idx]

    # ------------------------------------------------------------------ #
    # Frozen synthesis
    # ------------------------------------------------------------------ #
    def _current_frozen_spectrum(self) -> np.ndarray:
        """Current frozen spectrum rebuilt from the per-bin cache.

        Uncollapsed (per-channel independent), matching what the frozen state
        stored before coherence collapse is applied at synthesis. Independent of
        the fast path, so snapshots and exit crossfades never depend on it.
        """
        return self._frozen_spec.copy()

    def _collapse_coherent(self, spec: np.ndarray) -> np.ndarray:
        if self.p.channels == 2:
            t = self.p.coherence_threshold
            if self._coh_mask is None or self._coh_threshold_cached != t:
                self._coh_mask = self._frozen_coherence > t
                self._coh_threshold_cached = t
            coherent = self._coh_mask
            if np.any(coherent):
                spec[coherent, :] = spec[coherent].mean(axis=1, keepdims=True)
        return spec

    def _refresh_active_mask(self) -> float:
        """Rebuild the set of bins whose phase must advance.

        A bin is active when its estimated frequency is non-zero, or when a
        global drift nudge is in effect (which moves every bin). The mask and
        the nudge depend only on the captured frequencies, the drift value and
        the sample rate, so they are rebuilt only when one of those changes --
        including a sample-rate change made while frozen. Returns the per-hop
        phase nudge in radians.
        """
        drift = self.p.drift_compensation
        sr = self.p.sample_rate
        if self._drift_cached != (drift, sr) or self._active_mask is None:
            nudge = 2.0 * np.pi * drift / sr
            if nudge == 0.0:
                self._active_mask = self._frozen_freq != 0.0
            else:
                self._active_mask = self._all_mask
            self._active_empty = not self._active_mask.any()
            self._drift_cached = (drift, sr)
        return 2.0 * np.pi * drift / sr

    def _advance_frozen(self) -> np.ndarray:
        """Advance phases by the *estimated* frequencies (never repeat frame).

        With ``reuse_cache`` only bins that actually move are advanced and
        re-phased; static bins keep a constant complex value from capture. With
        ``reuse_cache=False`` every bin is recomputed each hop -- the original
        non-reusing reference, used to cross-check phase and detect any bin the
        static fast path wrongly froze. Nothing is ever copied or looped as a
        raw waveform.
        """
        p = self.p
        if p.reuse_cache:
            nudge = self._refresh_active_mask()
            act = self._active_mask
            if not self._active_empty:
                self._frozen_phase[act] = wrap_phase(
                    self._frozen_phase[act]
                    + self._frozen_freq[act] * p.block_size
                    + nudge
                )
                self._frozen_spec[act] = self._frozen_mag[act] * np.exp(
                    1j * self._frozen_phase[act]
                )
        else:
            nudge = 2.0 * np.pi * p.drift_compensation / p.sample_rate
            self._frozen_phase = wrap_phase(
                self._frozen_phase + self._frozen_freq * p.block_size + nudge
            )
            self._frozen_spec = self._frozen_mag * np.exp(1j * self._frozen_phase)
        spec = self._collapse_coherent(self._frozen_spec.copy())
        self._frozen_hops += 1
        self._last_frozen_spec = spec
        return spec

    # ------------------------------------------------------------------ #
    # Main streaming entry point
    # ------------------------------------------------------------------ #
    def process_block(self, hop_samples: np.ndarray) -> tuple[np.ndarray, FrameReport]:
        """Process one hop and return ``(output_samples, report)``."""
        p = self.p
        hop_samples = np.asarray(hop_samples, dtype=np.float64)
        # While fully frozen the live analysis FFT is unnecessary: only the
        # window buffer must advance so an exit resumes at a hop boundary with
        # no gap in the input history. The frozen spectrum needs no input at
        # all. On every other state the analysis FFT runs as before.
        if self._state == FreezeState.FROZEN:
            spectrum = None
            self.stft.push(hop_samples)
        else:
            spectrum = self.stft.analysis(hop_samples)
            self._push_history(spectrum)

        note = ""
        if self._state == FreezeState.LIVE:
            out_spec = spectrum.copy()
            progress = 0.0

        elif self._state == FreezeState.ENTERING:
            g = (self._fade_step + 1) / self._fade_steps_total
            gl, gf = equal_power_crossfade(np.array([g]))
            frozen = self._advance_frozen()
            out_spec = float(gl[0]) * self._start_spectrum + float(gf[0]) * frozen
            progress = g
            if self._fade_step + 1 >= self._fade_steps_total:
                self._state = FreezeState.FROZEN
                note = "freeze reached steady state"
            self._fade_step += 1

        elif self._state == FreezeState.FROZEN:
            out_spec = self._advance_frozen()
            progress = 1.0

        else:  # EXITING
            g = (self._fade_step + 1) / self._fade_steps_total
            gf, gl = equal_power_crossfade(np.array([g]))  # g=1 -> all live
            frozen = self._current_frozen_spectrum()
            out_spec = float(gf[0]) * frozen + float(gl[0]) * spectrum
            progress = 1.0 - g
            if self._fade_step + 1 >= self._fade_steps_total:
                self._state = FreezeState.LIVE
                note = "exit complete"
            self._fade_step += 1

        out = self.stft.synthesis(out_spec * p.gain)
        # Safety clip guard against numerical blow-up rather than hiding it:
        peak = np.max(np.abs(out))
        if peak > 10.0:
            note = (note + " | warning: output level divergence").strip(" |")

        report = self._build_report(hop_samples, out_spec, out, progress, note)
        self.reports.append(report)
        return out, report

    def _push_history(self, spectrum: np.ndarray) -> None:
        self._history = np.roll(self._history, -1, axis=0)
        self._history[-1] = spectrum
        self._hist_count = min(self._hist_count + 1, self._history.shape[0])

    def _build_report(
        self,
        hop_samples: np.ndarray,
        out_spec: np.ndarray,
        out_samples: np.ndarray,
        progress: float,
        note: str,
    ) -> FrameReport:
        # Live RMS of this input hop. The analysis frame's most recent block is
        # ``hop * window_tail``; the Hann tail is constant, so this equals the
        # former irfft-then-last-block measurement to rounding (no extra FFT).
        live_db = 20 * np.log10(
            np.sqrt(np.mean((hop_samples * self._hann_tail) ** 2)) + 1e-12
        )
        # Frozen RMS by Parseval from the emitted frozen spectrum (no second
        # inverse FFT); phase-invariant, so cached at capture.
        if self._state != FreezeState.LIVE:
            fro_db = self._frozen_rms_db
        else:
            fro_db = live_db
        out_db = 20 * np.log10(np.sqrt(np.mean(out_samples**2)) + 1e-12)
        base = getattr(self, "_last_capture_report", None)
        return FrameReport(
            state=self._state,
            live_rms_db=float(live_db),
            frozen_rms_db=float(fro_db),
            output_rms_db=float(out_db),
            freeze_progress=float(progress),
            capture_mode=base.capture_mode if base else "",
            n_phase_jump_bins=base.n_phase_jump_bins if base else 0,
            n_coherent_bins=base.n_coherent_bins if base else 0,
            top_frequencies_hz=base.top_frequencies_hz if base else [],
            dc_magnitude=float(np.abs(out_spec[0]).max()),
            near_silence=bool(live_db < self.p.near_silence_db),
            note=note,
        )

    # ------------------------------------------------------------------ #
    # Snapshot for the GUI / offline analysis
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        return {
            "state": self._state.value,
            "frozen_magnitude": self._frozen_mag.copy(),
            "frozen_phase": self._frozen_phase.copy(),
            "frozen_frequency_hz": self._frozen_freq * self.p.sample_rate / (2 * np.pi),
            "coherence": self._frozen_coherence.copy(),
            "bin_frequencies_hz": self._bin_hz.copy(),
            "frozen_hops": self._frozen_hops,
            "capture_report": getattr(self, "_last_capture_report", None),
        }
