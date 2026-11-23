#!/usr/bin/env bash
# User-space environment setup for minimal Debian containers (no root needed).
# Installs Python packages into ~/.local and pulls the PortAudio/ALSA/Qt
# shared libraries that have no wheel into a local prefix.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYS_DIR="${SYS_LIB_DIR:-$HOME/.local/spectral-freeze-libs}"
DEB_TMP="$(mktemp -d)"
trap 'rm -rf "$DEB_TMP"' EXIT

python3 -m pip --version >/dev/null 2>&1 || {
  echo "Installing pip (user site)..."
  curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$DEB_TMP/get-pip.py"
  python3 "$DEB_TMP/get-pip.py" --user --break-system-packages
}

python3 -m pip install --user --break-system-packages -r "$HERE/requirements.txt"

# System shared libraries (only needed when the distro does not provide them).
need_libs=()
python3 - <<'PY' || need_libs+=("audio")
import ctypes; ctypes.CDLL("libportaudio.so.2")
PY
python3 - <<'PY' || need_libs+=("qt")
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets
app = QtWidgets.QApplication([])
PY

if [[ ${#need_libs[@]} -eq 0 ]]; then
  echo "All system libraries already present."
  exit 0
fi

echo "Fetching local system libraries into $SYS_DIR ..."
mkdir -p "$SYS_DIR" "$DEB_TMP/debs"
cd "$DEB_TMP/debs"
BASE=http://ftp.debian.org/debian/pool/main
fetch() { curl -sSL -O "$1"; }

if [[ " ${need_libs[*]} " == *"audio"* ]]; then
  fetch "$BASE/p/portaudio19/libportaudio2_19.6.0-1.1_arm64.deb"
  fetch "$BASE/a/alsa-lib/libasound2_1.2.8-1+b1_arm64.deb"
  fetch "$BASE/a/alsa-lib/libasound2-data_1.2.8-1_all.deb"
  fetch "$BASE/j/jackd2/libjack-jackd2-0_1.9.21~dfsg-3_arm64.deb"
fi
if [[ " ${need_libs[*]} " == *"qt"* ]]; then
  fetch "$BASE/libg/libglvnd/libegl1_1.6.0-1_arm64.deb"
  fetch "$BASE/libg/libglvnd/libgles2_1.6.0-1_arm64.deb"
  fetch "$BASE/libg/libglvnd/libgl1_1.6.0-1_arm64.deb"
  fetch "$BASE/libg/libglvnd/libglx0_1.6.0-1_arm64.deb"
  fetch "$BASE/libg/libglvnd/libopengl0_1.6.0-1_arm64.deb"
  fetch "$BASE/libg/libglvnd/libglvnd0_1.6.0-1_arm64.deb"
  fetch "$BASE/libx/libxkbcommon/libxkbcommon0_1.5.0-1_arm64.deb"
  fetch "$BASE/libx/libxkbcommon/libxkbcommon-x11-0_1.5.0-1_arm64.deb"
  fetch "$BASE/d/dbus/libdbus-1-3_1.14.10-1~deb12u1_arm64.deb"
fi

for d in *.deb; do dpkg-deb -x "$d" "$SYS_DIR"; done

cat > "$HERE/env.sh" <<EOF
# Source this before launching the GUI / real-time audio:
#   source env.sh
export LD_LIBRARY_PATH="$SYS_DIR/usr/lib/aarch64-linux-gnu:$SYS_DIR/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export ALSA_CONFIG_PATH="$SYS_DIR/usr/share/alsa/alsa.conf"
EOF
echo "Wrote $HERE/env.sh — source it before launching the application."
