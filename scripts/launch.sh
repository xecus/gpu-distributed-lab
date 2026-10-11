#!/usr/bin/env bash
set -euo pipefail
# One command per node. No SSH fanout, no cloud provisioning.
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
if ((NNODES<1 || NNODES>5 || NPROC<1 || NPROC>8));then echo "Supported: 1..5 nodes, 1..8 processes/node" >&2;exit 2;fi
if [[ "$NNODES" == 1 ]]; then
  exec python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --module "$@"
else
  : "${MASTER_ADDR:?private IPv4 of node 0 required}"
  : "${RUN_ID:?same unique run ID on every node required}"
  exec python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:${MASTER_PORT:-29500}" \
    --rdzv_id="$RUN_ID" --max_restarts=0 --module "$@"
fi
