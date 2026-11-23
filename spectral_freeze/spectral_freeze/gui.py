"""Spectral freeze editing window (PySide6).

Launch:

    python -m spectral_freeze.gui

The window combines:

* a source area (synthesized musical/ambient signal or a loaded WAV/FLAC),
* a clickable waveform with freeze / release points,
* real-time transition and freeze parameters,
* a result area showing the frozen magnitude spectrum, estimated frequency
    trajectories and inter-channel coherence, plus a continuously synthesized
    tone that the engine keeps producing after the input stops.

Everything displayed comes from actual :class:`FreezeEngine` computations.
Processing errors are caught and surfaced in the status line of the window.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .engine import FreezeEngine, FreezeParams, FreezeState
from .signals import make_ambient_signal, make_musical_signal

SR = 48000
HOP = 256
FFT = 2048
MAX_F_HZ = 4000.0


# ---------------------------------------------------------------------- #
# Canvas widgets
# ---------------------------------------------------------------------- #
class WaveformView(QtWidgets.QWidget):
    """Clickable waveform; a click arms the freeze at that time position."""

    positionClicked = QtCore.Signal(float)  # time in seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self._signal = None
        self._play_pos = 0.0
        self._freeze_pos: float | None = None
        self._release_pos: float | None = None
        self._duration = 1.0

    def set_signal(self, signal: np.ndarray, duration: float):
        self._signal = signal
        self._duration = max(duration, 1e-6)
        self._freeze_pos = None
        self._release_pos = None
        self.update()

    def set_markers(self, freeze: float | None, release: float | None):
        self._freeze_pos = freeze
        self._release_pos = release
        self.update()

    def set_play_position(self, t: float):
        self._play_pos = t
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        x = event.position().x()
        t = max(0.0, min(self._duration, x / max(self.width(), 1) * self._duration))
        self.positionClicked.emit(t)

    def paintEvent(self, event: QtGui.QPaintEvent):  # noqa: ARG002
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#101418"))
        if self._signal is None:
            p.setPen(QtGui.QColor("#8899aa"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, self.tr("No source loaded"))
            return
        mono = self._signal.mean(axis=1)
        w = self.width()
        mid = self.height() / 2
        scale = self.height() * 0.42 / (np.max(np.abs(mono)) + 1e-9)
        pen = QtGui.QPen(QtGui.QColor("#4fc3f7"))
        pen.setWidth(1)
        p.setPen(pen)
        n = len(mono)
        prev = None
        for x in range(0, w, 2):
            i0 = int(x / w * n)
            i1 = max(i0 + 1, int((x + 2) / w * n))
            y_hi = mid - np.max(mono[i0:i1]) * scale
            y_lo = mid - np.min(mono[i0:i1]) * scale
            if prev is not None:
                p.drawLine(prev[0], prev[1], x, int(y_hi))
                p.drawLine(prev[0], prev[2], x, int(y_lo))
            prev = (x, int(y_hi), int(y_lo))
        self._draw_marker(p, self._freeze_pos, QtGui.QColor("#ffb300"), "freeze")
        self._draw_marker(p, self._release_pos, QtGui.QColor("#81c784"), "release")
        # Playhead.
        px = int(self._play_pos / self._duration * w)
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
        p.drawLine(px, 0, px, self.height())

    def _draw_marker(self, p, t, color, label):
        if t is None:
            return
        x = int(t / self._duration * self.width())
        p.setPen(QtGui.QPen(color, 2))
        p.drawLine(x, 0, x, self.height())
        p.fillRect(x + 2, 2, 58, 16, color)
        p.setPen(QtGui.QColor("#101418"))
        p.drawText(x + 5, 15, label)


class SpectrumView(QtWidgets.QWidget):
    """Frozen magnitude spectrum + estimated frequency trajectories."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._snapshot: dict | None = None
        self._mode = "magnitude"  # or "frequency"

    def set_mode(self, mode: str):
        self._mode = mode
        self.update()

    def update_snapshot(self, snapshot: dict):
        self._snapshot = snapshot
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):  # noqa: ARG002
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#0d1117"))
        if not self._snapshot:
            p.setPen(QtGui.QColor("#8899aa"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       self.tr("Click Freeze to capture a spectral state"))
            return
        snap = self._snapshot
        f = snap["bin_frequencies_hz"]
        keep = f <= MAX_F_HZ
        x = f[keep]
        w, h = self.width(), self.height()
        if self._mode == "magnitude":
            mag = np.mean(snap["frozen_magnitude"][keep], axis=1)
            mag_db = 20 * np.log10(mag + 1e-10)
            lo, hi = -90.0, max(0.0, float(mag_db.max()) + 6.0)
            pen = QtGui.QPen(QtGui.QColor("#ffb74d"))
            p.setPen(pen)
            prev = None
            for i in range(len(x)):
                xx = int(x[i] / MAX_F_HZ * w)
                yy = int(h - (mag_db[i] - lo) / (hi - lo) * h)
                if prev is not None:
                    p.drawLine(prev[0], prev[1], xx, yy)
                prev = (xx, yy)
            # Coherent-bin overlay.
            coh = snap["coherence"][keep]
            p.setPen(QtGui.QPen(QtGui.QColor("#64b5f6"), 4))
            for i in np.where(coh > 0.6)[0][::3]:
                xx = int(x[i] / MAX_F_HZ * w)
                p.drawPoint(xx, h - 6)
        else:
            freq = np.mean(snap["frozen_frequency_hz"][keep], axis=1)
            mag = np.mean(snap["frozen_magnitude"][keep], axis=1)
            active = mag > mag.max() * 1e-3
            p.setPen(QtGui.QPen(QtGui.QColor("#aed581")))
            for i in np.where(active)[0]:
                xx = int(x[i] / MAX_F_HZ * w)
                yy = int(h - min(freq[i], MAX_F_HZ) / MAX_F_HZ * h)
                p.drawEllipse(QtCore.QPoint(xx, yy), 2, 2)
        p.setPen(QtGui.QColor("#90a4ae"))
        p.drawText(8, 16, self._mode.upper())


# ---------------------------------------------------------------------- #
# Main window
# ---------------------------------------------------------------------- #
class FreezeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.tr("Spectral Freeze Editor"))
        self.resize(1180, 760)

        self.params = FreezeParams(sample_rate=SR, block_size=HOP, fft_size=FFT, channels=2)
        self.engine = FreezeEngine(self.params)
        self.source: np.ndarray | None = None
        self.cursor = 0
        self.playing = False
        self.frozen_armed_at: float | None = None
        self.release_at: float | None = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

        self._build_ui()
        self._load_builtin("musical")

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Source bar.
        src = QtWidgets.QHBoxLayout()
        src.addWidget(QtWidgets.QLabel(self.tr("Source:")))
        self.cmb_source = QtWidgets.QComboBox()
        self.cmb_source.addItem(self.tr("Homemade musical tones"), "musical")
        self.cmb_source.addItem(self.tr("Synthetic ambient recording"), "ambient")
        self.cmb_source.currentIndexChanged.connect(
            lambda: self._load_builtin(self.cmb_source.currentData())
        )
        src.addWidget(self.cmb_source)
        btn_open = QtWidgets.QPushButton(self.tr("Load audio file..."))
        btn_open.clicked.connect(self._open_file)
        src.addWidget(btn_open)
        self.btn_play = QtWidgets.QPushButton(self.tr("Play"))
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_play)
        src.addWidget(self.btn_play)
        self.btn_loop = QtWidgets.QCheckBox(self.tr("Loop source"))
        self.btn_loop.setChecked(True)
        src.addWidget(self.btn_loop)
        self.chk_stop_input = QtWidgets.QCheckBox(
            self.tr("Zero input after freeze point (trace frozen-only energy)")
        )
        src.addWidget(self.chk_stop_input)
        src.addStretch(1)
        root.addLayout(src)

        # Waveform.
        self.wave = WaveformView()
        self.wave.positionClicked.connect(self._wave_clicked)
        root.addWidget(self.wave, 2)

        # Spectrum result area.
        spec_bar = QtWidgets.QHBoxLayout()
        spec_bar.addWidget(QtWidgets.QLabel(self.tr("Result — frozen spectral state:")))
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItem(self.tr("Magnitude + coherence"), "magnitude")
        self.cmb_view.addItem(self.tr("Estimated frequency trajectories"), "frequency")
        self.cmb_view.currentIndexChanged.connect(
            lambda: self.spectrum.set_mode(self.cmb_view.currentData())
        )
        spec_bar.addWidget(self.cmb_view)
        spec_bar.addStretch(1)
        root.addLayout(spec_bar)
        self.spectrum = SpectrumView()
        root.addWidget(self.spectrum, 3)

        # Controls + status.
        bottom = QtWidgets.QHBoxLayout()
        bottom.addLayout(self._build_controls(), 3)
        bottom.addWidget(self._build_status(), 2)
        root.addLayout(bottom)

        self.status = self.statusBar()
        self.status.showMessage(self.tr("Ready. Click the waveform to place a freeze point."))

    def _build_controls(self):
        box = QtWidgets.QGridLayout()
        row = 0

        def slider(label, lo, hi, val, step, slot):
            nonlocal row
            box.addWidget(QtWidgets.QLabel(label), row, 0)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(int(lo / step), int(hi / step))
            s.setValue(int(val / step))
            s.setTickPosition(QtWidgets.QSlider.TicksBelow)
            lab = QtWidgets.QLabel(f"{val:g}")
            s.valueChanged.connect(lambda v: lab.setText(f"{v * step:g}"))
            s.valueChanged.connect(lambda v: slot(v * step))
            box.addWidget(s, row, 1)
            box.addWidget(lab, row, 2)
            row += 1
            return s

        slider(self.tr("Entry transition (s)"), 0.0, 0.5, 0.05, 0.005,
               lambda v: setattr(self.params, "entry_fade_sec", v))
        slider(self.tr("Exit transition (s)"), 0.0, 0.5, 0.10, 0.005,
               lambda v: setattr(self.params, "exit_fade_sec", v))
        slider(self.tr("Coherence threshold"), 0.0, 1.0, 0.6, 0.05,
               lambda v: setattr(self.params, "coherence_threshold", v))
        slider(self.tr("Pitch drift nudge (Hz)"), -50.0, 50.0, 0.0, 1.0,
               lambda v: setattr(self.params, "drift_compensation", v))
        slider(self.tr("Output gain"), 0.0, 2.0, 1.0, 0.05,
               lambda v: setattr(self.params, "gain", v))
        box.addWidget(QtWidgets.QLabel(self.tr("Capture mode")), row, 0)
        self.cmb_capture = QtWidgets.QComboBox()
        self.cmb_capture.addItems(["auto", "multi", "single"])
        box.addWidget(self.cmb_capture, row, 1, 1, 2)
        row += 1

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_freeze = QtWidgets.QPushButton(self.tr("Freeze now"))
        self.btn_freeze.setStyleSheet("background:#ffb300;color:#101418;font-weight:bold;")
        self.btn_freeze.clicked.connect(lambda: self._do_freeze(self.cmb_capture.currentText()))
        btn_row.addWidget(self.btn_freeze)
        self.btn_release = QtWidgets.QPushButton(self.tr("Release freeze"))
        self.btn_release.clicked.connect(self._do_release)
        btn_row.addWidget(self.btn_release)
        self.btn_reset = QtWidgets.QPushButton(self.tr("Reset engine"))
        self.btn_reset.clicked.connect(self._reset_engine)
        btn_row.addWidget(self.btn_reset)
        box.addLayout(btn_row, row, 0, 1, 3)
        return box

    def _build_status(self):
        group = QtWidgets.QGroupBox(self.tr("Processing trace (live engine values)"))
        lay = QtWidgets.QFormLayout(group)
        self.lbl_state = QtWidgets.QLabel("-")
        self.lbl_mode = QtWidgets.QLabel("-")
        self.lbl_jumps = QtWidgets.QLabel("-")
        self.lbl_coh = QtWidgets.QLabel("-")
        self.lbl_freqs = QtWidgets.QLabel("-")
        self.lbl_rms = QtWidgets.QLabel("-")
        self.lbl_dc = QtWidgets.QLabel("-")
        for lab, w in [
            (self.tr("State"), self.lbl_state),
            (self.tr("Capture mode"), self.lbl_mode),
            (self.tr("Phase-jump bins"), self.lbl_jumps),
            (self.tr("Coherent bins"), self.lbl_coh),
            (self.tr("Top frequencies (Hz)"), self.lbl_freqs),
            (self.tr("RMS live / frozen / out (dB)"), self.lbl_rms),
            (self.tr("DC magnitude"), self.lbl_dc),
        ]:
            lay.addRow(lab, w)
        return group

    # ------------------------------------------------------------------ #
    # Source handling
    # ------------------------------------------------------------------ #
    def _load_builtin(self, kind: str):
        try:
            if kind == "ambient":
                self.source = make_ambient_signal(SR, 4.0, 2)
            else:
                self.source = make_musical_signal(SR, 4.0, 2)
            self._after_source_changed()
            self.status.showMessage(self.tr("Built-in source synthesized (4 s, 48 kHz stereo)."))
        except Exception:
            self._show_exception()

    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("Open audio"), "", self.tr("Audio (*.wav *.flac *.ogg)")
        )
        if not path:
            return
        try:
            import soundfile as sf

            data, sr = sf.read(path, always_2d=True, dtype="float64")
            if sr != SR:
                raise ValueError(f"only {SR} Hz files are supported by this demo (got {sr} Hz)")
            if data.shape[1] != 2:
                if data.shape[1] == 1:
                    data = np.repeat(data, 2, axis=1)
                else:
                    raise ValueError("expected mono or stereo input")
            self.source = data
            self._after_source_changed()
            self.status.showMessage(self.tr("Loaded %s") % Path(path).name)
        except Exception:
            self._show_exception()

    def _after_source_changed(self):
        self._reset_engine()
        self.cursor = 0
        dur = self.source.shape[0] / SR
        self.wave.set_signal(self.source, dur)

    def _reset_engine(self):
        self.engine = FreezeEngine(self.params)
        self.frozen_armed_at = None
        self.release_at = None
        self.spectrum._snapshot = None
        self.spectrum.update()
        self.wave.set_markers(None, None)

    # ------------------------------------------------------------------ #
    # Freeze control
    # ------------------------------------------------------------------ #
    def _wave_clicked(self, t: float):
        if self.frozen_armed_at is None:
            self.frozen_armed_at = t
            self.status.showMessage(self.tr("Freeze armed at %.3f s. Press Play to run.") % t)
        else:
            self.release_at = t
            self.status.showMessage(
                self.tr("Freeze @ %.3f s, release @ %.3f s.") % (self.frozen_armed_at, t)
            )
        self.wave.set_markers(self.frozen_armed_at, self.release_at)

    def _do_freeze(self, mode: str):
        try:
            self.engine.freeze(mode)
            self.status.showMessage(self.tr("Freeze captured (%s).") % mode)
        except Exception:
            self._show_exception()

    def _do_release(self):
        try:
            self.engine.unfreeze()
            self.status.showMessage(self.tr("Release transition started."))
        except Exception:
            self._show_exception()

    def _toggle_play(self, on: bool):
        self.playing = on
        self.btn_play.setText(self.tr("Stop") if on else self.tr("Play"))
        if on:
            self.timer.start()
        else:
            self.timer.stop()

    # ------------------------------------------------------------------ #
    # Processing loop — the same engine path as offline / real-time code
    # ------------------------------------------------------------------ #
    def _tick(self):
        try:
            hop = self.params.block_size
            if self.source is None:
                return
            if self.cursor + hop > len(self.source):
                if self.btn_loop.isChecked():
                    self.cursor = 0
                    self._reset_engine()
                else:
                    self.btn_play.setChecked(False)
                    return
            block = self.source[self.cursor : self.cursor + hop].copy()
            t = self.cursor / SR
            if (
                self.frozen_armed_at is not None
                and t >= self.frozen_armed_at
                and self.engine.state == FreezeState.LIVE
                and self.engine._hist_count >= self.engine._warmup_hops
            ):
                self.engine.freeze(self.cmb_capture.currentText())
            if (
                self.release_at is not None
                and t >= self.release_at
                and self.engine.state in (FreezeState.ENTERING, FreezeState.FROZEN)
            ):
                self.engine.unfreeze()
            if (
                self.chk_stop_input.isChecked()
                and self.frozen_armed_at is not None
                and self.engine.state in (FreezeState.ENTERING, FreezeState.FROZEN)
                and t >= self.frozen_armed_at + self.params.entry_fade_sec + 0.05
            ):
                block.fill(0.0)
            _, report = self.engine.process_block(block)
            self.cursor += hop
            self.wave.set_play_position(self.cursor / SR)
            self._refresh_status(report)
            self.spectrum.update_snapshot(self.engine.snapshot())
        except Exception:
            self.btn_play.setChecked(False)
            self._show_exception()

    def _refresh_status(self, report):
        self.lbl_state.setText(report.state.value)
        self.lbl_mode.setText(report.capture_mode or "-")
        self.lbl_jumps.setText(str(report.n_phase_jump_bins))
        self.lbl_coh.setText(str(report.n_coherent_bins))
        self.lbl_freqs.setText(", ".join(map(str, report.top_frequencies_hz)) or "-")
        self.lbl_rms.setText(
            f"{report.live_rms_db:.1f} / {report.frozen_rms_db:.1f} / {report.output_rms_db:.1f}"
        )
        self.lbl_dc.setText(f"{report.dc_magnitude:.2e}")
        if report.near_silence:
            self.status.showMessage(self.tr("Near-silence detected: capture suppressed to avoid a click."))
        elif report.note:
            self.status.showMessage(report.note)

    def _show_exception(self):
        msg = traceback.format_exc(limit=1).strip().splitlines()[-1]
        self.statusBar().showMessage(self.tr("Processing error: %s") % msg)
        QtWidgets.QMessageBox.warning(self, self.tr("Spectral freeze error"), msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FreezeWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
