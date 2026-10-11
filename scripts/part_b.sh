#!/usr/bin/env bash
set -euo pipefail
# Entry point for Part B lessons: bash scripts/part_b.sh B01..B05 [extra args].
# B01-B03 train Qwen3 via lab.train; B04/B05 run upstream GR00T in its own uv environment.
# Every run is wrapped by run_record.py so OOM and timeouts are recorded as results.
lesson=${1:?B01..B05 required};shift
if [[ "$lesson" == B04 || "$lesson" == B05 ]];then
 out="runs/$lesson";args=("$@")
 for ((i=0;i<${#args[@]};i++));do
  if [[ "${args[$i]}" == --out ]];then out="${args[$((i+1))]}";fi
 done
 if [[ "$lesson" == B04 ]];then
  if [[ "${NNODES:-1}" != 1 ]];then echo 'B04 is single node; set NNODES=1' >&2;exit 2;fi
  total=8
 else
  : "${NNODES:?set 2..5}"
  if ((NNODES<2 || NNODES>5));then echo 'B05 requires 2..5 nodes' >&2;exit 2;fi
  total=$((8*NNODES))
 fi
 price_args=();if [[ -n "${NODE_HOURLY_USD:-}" ]];then price_args+=(--hourly-usd "$NODE_HOURLY_USD");fi
 exec python scripts/run_record.py --id "$lesson" --out "$out" --nodes "${NNODES:-1}" --gpus "$total" \
  --timeout "${EXPERIMENT_TIMEOUT:-3600}" "${price_args[@]}" -- \
  python scripts/groot_run.py --gpus "$total" --out "$out" "$@"
fi
: "${TOKENS:?prepared train.bin required}"
: "${MODEL_REVISION:?HF commit SHA required}"
: "${NODE_HOURLY_USD:?reviewed node hourly USD required for cost recording}"
MODE=${MODE:-zero3}
case "$lesson" in
 B01) model=Qwen/Qwen3-8B;MODE=${B01_MODE:-ddp} ;;
 B02) model=Qwen/Qwen3-8B ;;
 B03) model=Qwen/Qwen3-32B ;;
 *) echo 'Unknown lesson' >&2;exit 2 ;;
esac
out="runs/$lesson/${MODE}_${NNODES:-1}node"
args=("$@")
for ((i=0;i<${#args[@]};i++));do
 if [[ "${args[$i]}" == --out ]];then out="${args[$((i+1))]}";fi
done
exec python scripts/run_record.py --id "$lesson" --out "$out" --nodes "${NNODES:-1}" \
 --gpus "$(( ${NPROC:-8} * ${NNODES:-1} ))" --hourly-usd "${NODE_HOURLY_USD:-0}" --timeout "${EXPERIMENT_TIMEOUT:-3600}" -- \
 bash scripts/launch.sh lab.train --mode "$MODE" --model "$model" --revision "$MODEL_REVISION" --tokens "$TOKENS" \
 --seq 256 --batch 1 --accum "${ACCUM:-2}" --steps 100 --warmup 10 --save-every 50 --sampler \
 --activation-checkpointing --out "$out" "$@"
