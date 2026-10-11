#!/usr/bin/env bash
# =============================================================================
# scripts/launch.sh — torchrun で分散学習を起動する共通ランチャー
#
# 目的:
#   1 ノードあたり 1 回このスクリプトを実行すると、そのノードの GPU 数だけ
#   Python プロセスを起動する。SSH で他ノードに配ったり、クラウドのリソースを
#   作ったりはしない (複数ノードでは、各ノードで人間がこのコマンドを実行する)。
#
# 使い方:
#   # 1 ノード 8 GPU
#   bash scripts/launch.sh lab.train --mode ddp --out runs/A07
#
#   # 2 ノード (各ノードで NODE_RANK だけ変えて実行する)
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.11 RUN_ID=exp1 bash scripts/launch.sh lab.train --mode ddp
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.11 RUN_ID=exp1 bash scripts/launch.sh lab.train --mode ddp
#
#   第 1 引数は「python -m で実行するモジュール名」、残りはそのモジュールへの引数。
#
# 読み取る環境変数:
#   NPROC       : 1 ノードあたりのプロセス数 = GPU 数 (既定 8, 1〜8)
#   NNODES      : ノード数 (既定 1, 1〜5)
#   NODE_RANK   : このノードの番号 (既定 0, ノード 0 がランデブーの受付役)
#   MASTER_ADDR : (複数ノード時に必須) ノード 0 のプライベート IPv4 アドレス
#   MASTER_PORT : (複数ノード時) ランデブー用ポート (既定 29500)
#   RUN_ID      : (複数ノード時に必須) 全ノードで同じ、実行ごとに一意な ID
#   CUDA_HOME   : CUDA ツールキットの場所 (未設定なら自動検出)
#   CC          : C コンパイラ (未設定かつ gcc が 15 以上なら gcc-14 を使う)
#   NVCC_PREPEND_FLAGS : nvcc に追加で渡すフラグ (既存の値は保持される)
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# プロセス数・ノード数の設定と範囲チェック
#   この教材は p5 系 (8 GPU/ノード) を最大 5 ノードまで使う前提なので、それ以外は拒否する。
# -----------------------------------------------------------------------------
NPROC=${NPROC:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
if ((NNODES<1 || NNODES>5 || NPROC<1 || NPROC>8)); then
  echo "Supported: 1..5 nodes, 1..8 processes/node" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# CUDA_HOME の自動設定
#   DeepSpeed は import 時に CUDA ツールキット (nvcc) を確認し、初回使用時に
#   FusedAdam を JIT ビルドする。そのため nvcc が見つからないと学習が始まらない。
#   torch は cu128 (CUDA 12.8 版) なので、DeepSpeed のビルドにも同じ CUDA メジャー版 (12) が必要。
#   runfile でインストールした /usr/local/cuda-12.8 を最優先で使う。
#
#   判定の順番:
#     1. CUDA_HOME が設定済みなら何もしない (利用者の指定を尊重)
#     2. /usr/local/cuda-12.8/bin/nvcc があればそれを使う
#     3. PATH に nvcc があればそのまま使う (CUDA_HOME は設定しない)
#     4. /usr/local/cuda/bin/nvcc があればそれを使う
#     5. どれもなければエラー終了
# -----------------------------------------------------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.8
  elif ! command -v nvcc >/dev/null 2>&1; then
    [[ -x /usr/local/cuda/bin/nvcc ]] && export CUDA_HOME=/usr/local/cuda \
      || { echo "CUDA 12.8 toolkit (nvcc) not found; see README (runfile --toolkit) or set CUDA_HOME" >&2; exit 2; }
  fi
fi
# CUDA_HOME が決まっていれば、その nvcc が PATH の先頭で見つかるようにする。
[[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH"

# -----------------------------------------------------------------------------
# ホストコンパイラ (gcc) の設定
#   nvcc 12.8 は gcc 15 以上をホストコンパイラとして受け付けない
#   (FusedAdam の JIT ビルドが失敗する)。新しい OS の既定 gcc が 15 以上の場合は、
#   nvcc・torch の C++ 拡張・Triton のすべてで gcc-14 を使うように切り替える。
#   CC が設定済みなら利用者の指定を尊重して何もしない。
# -----------------------------------------------------------------------------
if [[ -z "${CC:-}" ]] && (( $(gcc -dumpversion | cut -d. -f1) > 14 )); then
  command -v g++-14 >/dev/null 2>&1 \
    || { echo "system gcc is newer than 14; run: sudo apt-get install -y gcc-14 g++-14" >&2; exit 2; }
  # CC/CXX は torch 拡張と Triton 用、-ccbin は nvcc にホストコンパイラを指定するため。
  export CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS="-ccbin g++-14 ${NVCC_PREPEND_FLAGS:-}"
fi

# -----------------------------------------------------------------------------
# torchrun (python -m torch.distributed.run) で起動する
#   exec でこのシェルを置き換えるので、Ctrl-C などのシグナルが torchrun に直接届く。
# -----------------------------------------------------------------------------
if [[ "$NNODES" == 1 ]]; then
  # 1 ノード: --standalone モード。
  #   ローカルに一時的なランデブーを自動で用意してくれるので、
  #   MASTER_ADDR やポートを考えずに済む。
  exec python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" --module "$@"
else
  # 複数ノード: c10d ランデブー。
  #   全ノードが MASTER_ADDR:MASTER_PORT (ノード 0) に集まり、同じ RUN_ID を持つ
  #   プロセス同士でグループを作る。RUN_ID が違うと別の実行とみなされて合流しない。
  #   --max_restarts=0 : 失敗時に自動再起動しない (エラーを隠さず、すぐ気付けるように)。
  : "${MASTER_ADDR:?private IPv4 of node 0 required}"
  : "${RUN_ID:?same unique run ID on every node required}"
  exec python -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:${MASTER_PORT:-29500}" \
    --rdzv_id="$RUN_ID" --max_restarts=0 --module "$@"
fi
