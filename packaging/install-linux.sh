#!/bin/bash
# Install Survival LLM on Linux from this copy of the code: a Python environment for the app window and PDF
# reading, a survival-llm command and an entry in the desktop's app menu. The app then walks you through the AI
# engine (Ollama), the model and the library, as on Mac and Windows. Run it again after a git pull; it's quick.
# Uninstall: packaging/install-linux.sh --uninstall (keeps your models, library and files).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="$HOME/.survival-llm"  # the app's settings folder on Linux (APP_DIR in app/server.py)
VENV="$APP_DIR/venv"
BIN="$HOME/.local/bin/survival-llm"
DESKTOP="${XDG_DATA_HOME:-$HOME/.local/share}/applications/survival-llm.desktop"
PYWEBVIEW="pywebview==6.2.1"  # the versions packaging/build.sh puts in the Mac and Windows apps
PYPDF="pypdf==6.19.0"

if [ "${1:-}" = "--uninstall" ]; then
  rm -rf "$VENV" "$BIN" "$DESKTOP"
  echo "Removed the survival-llm command and app menu entry. Your models (~/.ollama/models), library"
  echo "(~/Survival LLM Library), files (~/Survival LLM Files) and settings ($APP_DIR) are still there."
  exit 0
fi

command -v python3 >/dev/null || { echo "Survival LLM needs Python 3. Install it, then run this again." >&2; exit 1; }

# The app window is the system's WebKitGTK web view, reached through PyGObject; the venv sees the system's copy.
if ! python3 -c "import gi; gi.require_version('WebKit2', '4.1')" 2>/dev/null; then
  if command -v pacman >/dev/null; then hint="sudo pacman -S python-gobject webkit2gtk-4.1"
  elif command -v apt-get >/dev/null; then hint="sudo apt install python3-gi gir1.2-webkit2-4.1"
  elif command -v dnf >/dev/null; then hint="sudo dnf install python3-gobject webkit2gtk4.1"
  else hint="install PyGObject and WebKitGTK 4.1"; fi
  echo "Note: without WebKitGTK the app opens in your web browser instead of its own window."
  echo "For its own window: $hint, then run this again."
fi

mkdir -p "$APP_DIR" "$(dirname "$BIN")" "$(dirname "$DESKTOP")"
[ -x "$VENV/bin/python3" ] || python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python3" -m pip install -q --disable-pip-version-check "$PYWEBVIEW" "$PYPDF"

cat > "$BIN" <<EOF
#!/bin/sh
exec "$VENV/bin/python3" "$REPO/app/server.py" "\$@"
EOF
chmod +x "$BIN"

cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Survival LLM
Comment=Offline AI assistant and survival library
Exec="$BIN"
Icon=help-browser
Terminal=false
Categories=Utility;
Keywords=emergency;first aid;offline;AI;
EOF

echo "Installed. Open Survival LLM from your app menu, or run: survival-llm"
command -v ollama >/dev/null || echo "The app will show you how to install the AI engine (Ollama) the first time it opens."
