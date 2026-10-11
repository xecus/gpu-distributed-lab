#!/usr/bin/env bash
set -euo pipefail
# Run on an already-approved DLAMI; preserves system EFA installation.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
mkdir -p runs
python -m pip freeze > runs/pip-freeze.txt
python scripts/preflight.py --require-gpus 8
