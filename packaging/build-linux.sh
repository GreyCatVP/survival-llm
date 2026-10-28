#!/bin/bash
# Build on Linux for the host architecture. Ollama is installed separately.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
[ "$(uname -s)" = Linux ] || { echo 'Build the Linux package on Linux.' >&2; exit 2; }
case "$(uname -m)" in
  x86_64) ARCH=x86_64 ;;
  aarch64|arm64) ARCH=aarch64 ;;
  *) echo 'Supported Linux architectures: x86_64 and aarch64.' >&2; exit 2 ;;
esac
CACHE="$HERE/cache"
DIST="$REPO/dist"
STAGE="$DIST/linux-$ARCH"
APP="$STAGE/Survival LLM"
PBS_TAG=20261021
PBS_PY=3.12.15
FILE="cpython-$PBS_PY+$PBS_TAG-$ARCH-unknown-linux-gnu-install_only_stripped.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/$FILE"
mkdir -p "$CACHE" "$DIST"
if [ ! -f "$CACHE/$FILE" ]; then
  curl -L --fail -sS -o "$CACHE/$FILE.partial" "$URL"
  mv "$CACHE/$FILE.partial" "$CACHE/$FILE"
fi
rm -rf "$STAGE"
mkdir -p "$APP"
tar -xzf "$CACHE/$FILE" -C "$APP"
# Qt and its web engine travel with the app; Linux desktop libraries remain system dependencies.
"$APP/python/bin/python3" -m pip install -q --no-compile --disable-pip-version-check \
  'pywebview[qt]==6.2.1' 'pypdf==6.19.0' 'truststore==0.10.4'
mkdir -p "$APP/app/library" "$APP/app/app/static"
cp "$REPO/core.py" "$REPO/memory.py" "$APP/app/"
cp "$REPO/library/search.py" "$REPO/library/passages.py" "$REPO/library/myfiles.py" "$APP/app/library/"
cp "$REPO/app/server.py" "$APP/app/app/"
cp "$REPO/app/static/"* "$APP/app/app/static/"
cp "$REPO/LICENSE" "$REPO/README.md" "$REPO/SOURCES.md" "$APP/"
cp "$HERE/VERSION" "$APP/app/version.txt"
cp "$HERE/linux-launcher.sh" "$APP/survival-llm"
cp "$HERE/linux-desktop.py" "$APP/install-desktop.py"
chmod +x "$APP/survival-llm"
# Zip dereferences Python's symlinks: both graphical extractors and our updater get runnable files.
ARCHIVE="$DIST/Survival-LLM-linux-$ARCH.zip"
rm -f "$ARCHIVE"
(cd "$STAGE" && zip -qr "$ARCHIVE" 'Survival LLM')
ls -lh "$ARCHIVE"
