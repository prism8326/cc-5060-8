"""Live real-time freeze via audio input/output devices.

Requires a working PortAudio device. In this minimal container there is no
sound card; run on a normal machine (or any host with ALSA/PulseAudio) after
sourcing env.sh.

Examples
--------
python -m scripts.live_freeze                       # default devices, GUI-style params
python -m scripts.live_freeze --list-devices
python -m scripts.live_freeze --input 1 --output 2 --entry-fade 0.08
"""

from __future__ import annotations

import argparse
import time

from spectral_freeze.engine import FreezeParams
from spectral_freeze.realtime import RealtimeFreeze


def main():
    ap = argparse.ArgumentParser(description="Real-time spectral freeze")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input", type=int, default=None)
    ap.add_argument("--output", type=int, default=None)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--fft-size", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--entry-fade", type=float, default=0.05)
    ap.add_argument("--exit-fade", type=float, default=0.10)
    ap.add_argument("--auto-freeze-sec", type=float, default=None,
                    help="automatically freeze after N seconds (smoke/demo use)")
    args = ap.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return

    params = FreezeParams(
        sample_rate=args.sample_rate,
        block_size=args.block_size,
        fft_size=args.fft_size,
        channels=args.channels,
        entry_fade_sec=args.entry_fade,
        exit_fade_sec=args.exit_fade,
    )
    rt = RealtimeFreeze(params, input_device=args.input, output_device=args.output)
    rt.start()
    print("Streaming. Press Enter to freeze, Enter again to release, Ctrl-C to quit.")
    try:
        if args.auto_freeze_sec is not None:
            time.sleep(args.auto_freeze_sec)
            rt.freeze()
            print("FREEZE ON")
        while True:
            input()
            if rt.engine.state.value in ("entering", "frozen"):
                rt.unfreeze()
                print("FREEZE OFF")
            else:
                rt.freeze()
                print("FREEZE ON")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rt.stop()
        print("stopped; xruns:", rt.xruns)


if __name__ == "__main__":
    main()
