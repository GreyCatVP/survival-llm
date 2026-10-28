#!/bin/sh
# A relocatable launcher; no system Python or virtualenv needed.
set -eu
APP=$(CDPATH= cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
# Select the Qt renderer bundled by build-linux.sh (source runs may still choose GTK).
export PYWEBVIEW_GUI="${PYWEBVIEW_GUI:-qt}"
if [ "${1:-}" = --install-desktop ]; then
    exec "$APP/python/bin/python3" -I "$APP/install-desktop.py"
fi
if [ ! -t 1 ]; then
    LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/survival-llm"
    mkdir -p "$LOG_DIR"
    exec >> "$LOG_DIR/app.log" 2>&1
fi
exec "$APP/python/bin/python3" -s "$APP/app/app/server.py" "$@"
