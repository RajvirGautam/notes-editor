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
  echo "→ Installing dependencies…"
  ./.venv/bin/pip install --quiet -r requirements.txt
fi

exec ./.venv/bin/python server.py
