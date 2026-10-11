#!/usr/bin/env bash
set -euo pipefail
# Run on an already-approved DLAMI; preserves system EFA installation.
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null 2>&1 \
  || { echo "$PY not found; run: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "$PY is older than 3.10; set PYTHON=/path/to/python3.10+" >&2; exit 1; }
# Triton JIT-compiles a C helper on first use (deepspeed import) and needs Python.h.
"$PY" -c 'import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' \
  || { echo "Python.h missing; run: sudo apt-get install -y ${PY}-dev" >&2; exit 1; }
"$PY" -m venv .venv || { echo "venv failed; try: sudo apt-get install -y ${PY}-venv" >&2; exit 1; }
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
mkdir -p runs
python -m pip freeze > runs/pip-freeze.txt
python scripts/preflight.py --require-gpus 8
