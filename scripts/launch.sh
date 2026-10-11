#!/usr/bin/env bash
set -euo pipefail
# One command per node. No SSH fanout, no cloud provisioning.
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
if ((NNODES<1 || NNODES>5 || NPROC<1 || NPROC>8));then echo "Supported: 1..5 nodes, 1..8 processes/node" >&2;exit 2;fi
# DeepSpeed checks the CUDA toolkit (nvcc) at import time and JIT-builds FusedAdam on first use.
# Prefer the runfile-installed 12.8 toolkit; torch is cu128 and DeepSpeed requires the same CUDA major.
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then export CUDA_HOME=/usr/local/cuda-12.8
  elif ! command -v nvcc >/dev/null 2>&1; then
    [[ -x /usr/local/cuda/bin/nvcc ]] && export CUDA_HOME=/usr/local/cuda \
      || { echo "CUDA 12.8 toolkit (nvcc) not found; see README (runfile --toolkit) or set CUDA_HOME" >&2; exit 2; }
  fi
fi
[[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH"
# nvcc 12.8 rejects host GCC newer than 14 (FusedAdam JIT build). Use gcc-14 for nvcc, torch and Triton.
if [[ -z "${CC:-}" ]] && (( $(gcc -dumpversion | cut -d. -f1) > 14 )); then
  command -v g++-14 >/dev/null 2>&1 \
    || { echo "system gcc is newer than 14; run: sudo apt-get install -y gcc-14 g++-14" >&2; exit 2; }
  export CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS="-ccbin g++-14 ${NVCC_PREPEND_FLAGS:-}"
fi
if [[ "$NNODES" == 1 ]]; then
  exec python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --module "$@"
else
  : "${MASTER_ADDR:?private IPv4 of node 0 required}"
  : "${RUN_ID:?same unique run ID on every node required}"
  exec python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:${MASTER_PORT:-29500}" \
    --rdzv_id="$RUN_ID" --max_restarts=0 --module "$@"
fi
