#!/usr/bin/env bash
# One command to launch the Notes Deskewer.
# Creates a virtualenv on first run, installs deps, then starts the local app.
set -e
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

if [ ! -d .venv ]; then
  echo "→ Creating virtual environment ($PYTHON)…"
  "$PYTHON" -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi

# Keep dependencies in sync (fast no-op when already satisfied).
./.venv/bin/pip install --quiet -r requirements.txt

# Headless Chromium renders the Borders chapter header exactly like the design.
# No-op once installed; if it can't be fetched the server still runs (a Pillow
# fallback draws the header instead).
./.venv/bin/playwright install chromium >/dev/null 2>&1 || true

exec ./.venv/bin/python server.py
