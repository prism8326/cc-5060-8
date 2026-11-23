"""Real-time audio streaming around :class:`FreezeEngine`.

The PortAudio callback runs the exact same per-hop ``process_block`` path as
the offline processor, which guarantees real-time / offline consistency.
Input and output devices are configurable; when no hardware is available the
class still validates configuration and can be driven manually in tests.
"""

from __future__ import annotations

import numpy as np

from .engine import FreezeEngine, FreezeParams


class RealtimeFreeze:
    def __init__(self, params: FreezeParams, input_device=None, output_device=None):
        self.params = params
        self.engine = FreezeEngine(params)
        self.input_device = input_device
        self.output_device = output_device
        self._stream = None
        self.xruns = 0

    def _callback(self, indata, outdata, frames, time_info, status):  # noqa: ARG002
        if status:
            # Input underflow/overflow: surface it instead of hiding it.
            self.xruns += 1
        if frames != self.params.block_size:
            # PortAudio may deliver a partial final block; pad to hop length.
            block = np.zeros((self.params.block_size, self.params.channels))
            block[:frames] = indata
        else:
            block = np.array(indata, copy=True)
        y, _ = self.engine.process_block(block)
        outdata[:] = y[:frames]

    def start(self) -> None:
        import sounddevice as sd

        if not sd.query_devices():
            raise RuntimeError(
                "no audio devices available in this environment; use the "
                "offline processor or connect an audio device"
            )
        self._stream = sd.Stream(
            samplerate=self.params.sample_rate,
            blocksize=self.params.block_size,
            channels=self.params.channels,
            dtype="float64",
            callback=self._callback,
            device=(self.input_device, self.output_device),
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def freeze(self, mode: str = "auto") -> None:
        self.engine.freeze(mode)

    def unfreeze(self) -> None:
        self.engine.unfreeze()

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active
