#!/usr/bin/env bash
set -euo pipefail
# Entry point for Part A lessons: bash scripts/part_a.sh A01..A14 [extra lab.train args].
# Multi-GPU lessons go through launch.sh; set NPROC / NNODES / NODE_RANK as needed.
lesson=${1:?A01..A14 required};shift
case "$lesson" in
 A01) exec python -m lab.cuda_basics "$@" ;;
 A02) NPROC=1 bash scripts/launch.sh lab.train --mode single --out runs/A02 "$@" ;;
 A03) exec python scripts/batch_sweep.py "$@" ;;
 A04) bash scripts/launch.sh lab.collectives --out runs/A04/collectives.csv "$@" ;;
 A05) bash scripts/launch.sh lab.ring --out runs/A05/ring.csv "$@" ;;
 A06) bash scripts/launch.sh lab.train --mode manual --out runs/A06 "$@" ;;
 A07) bash scripts/launch.sh lab.train --mode ddp --sampler --out runs/A07 "$@" ;;
 A08) bash scripts/launch.sh lab.train --mode zero1 --out runs/A08 "$@" ;;
 A09) bash scripts/launch.sh lab.train --mode zero2 --out runs/A09 "$@" ;;
 A10) bash scripts/launch.sh lab.train --mode zero3 --out runs/A10 "$@" ;;
 A11) bash scripts/launch.sh lab.train --mode fsdp --out runs/A11 "$@" ;;
 A12) bash scripts/launch.sh lab.train --mode tp --out runs/A12 "$@" ;;
 A13) : "${NNODES:?set 1..5}";bash scripts/launch.sh lab.train --mode ddp --sampler --out runs/A13 "$@" ;;
 A14) bash scripts/launch.sh lab.train --mode ddp --sampler --profile --out runs/A14 "$@" ;;
 *) echo 'Unknown lesson' >&2;exit 2 ;;
esac
