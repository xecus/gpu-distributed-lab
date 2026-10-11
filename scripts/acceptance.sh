#!/usr/bin/env bash
set -euo pipefail
# Short GPU acceptance run: every Part A mode for a few steps, then resume from step_4
# to check checkpoint/restore, then the TP forward/gradient/update parity check.
# Start clean: previous acceptance outputs, the DeepSpeed/torch JIT extension cache (FusedAdam) and
# the Triton kernel cache. Set KEEP_CACHE=1 to keep the build caches and skip the FusedAdam rebuild.
rm -rf runs/acceptance
if [[ "${KEEP_CACHE:-0}" != 1 ]];then
 for d in "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}" "${TRITON_CACHE_DIR:-$HOME/.triton/cache}";do
  [[ -n "$d" && "$d" != / ]] && { echo "removing $d";rm -rf "$d"; }
 done
fi
python scripts/preflight.py --require-gpus 8
for lesson in A02 A04 A05 A06 A07 A08 A09 A10 A11 A12;do
 if [[ "$lesson" == A04 ]];then bash scripts/part_a.sh "$lesson" --iters 5 --max-power 20 --out runs/acceptance/A04/collectives.csv
 elif [[ "$lesson" == A05 ]];then bash scripts/part_a.sh "$lesson" --iters 5 --elements 1024 --out runs/acceptance/A05/ring.csv
 else bash scripts/part_a.sh "$lesson" --steps 4 --warmup 1 --save-every 2 --dim 64 --layers 2 --seq 32 --out "runs/acceptance/$lesson";fi
done
for item in 'A07 ddp' 'A08 zero1' 'A09 zero2' 'A10 zero3' 'A11 fsdp' 'A12 tp';do
 read -r lesson mode <<< "$item"
 extra=();if [[ "$mode" == ddp ]];then extra+=(--sampler);fi
 bash scripts/launch.sh lab.train --mode "$mode" --steps 6 --warmup 1 --save-every 2 --dim 64 --layers 2 --seq 32 \
 --out "runs/acceptance/$lesson" --resume "runs/acceptance/$lesson/step_4" "${extra[@]}"
done

NPROC=2 bash scripts/launch.sh lab.tp_check
