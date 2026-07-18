#!/usr/bin/env bash
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and rerun."
  exit 2
fi

if ! brew list --versions python@3.12 >/dev/null 2>&1; then
  brew install python@3.12
fi

PYTHON_312="$(brew --prefix python@3.12)/bin/python3.12"

if [[ ! -x "$PYTHON_312" ]]; then
  echo "Python 3.12 was not found at: $PYTHON_312"
  exit 3
fi

rm -rf .venv
"$PYTHON_312" -m venv .venv
source .venv/bin/activate

python scripts/check_python.py
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
bash scripts/run_quality_checks.sh

echo "PASS: macOS Python 3.12 environment is ready."
