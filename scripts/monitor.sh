#!/usr/bin/env bash
set -euo pipefail
# Sample nvidia-smi once per second into a CSV. Run in a separate shell; stop with Ctrl-C.
out=${1:-runs/gpu.csv}
mkdir -p "$(dirname "$out")"
exec nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
 --format=csv --loop=1 > "$out"
