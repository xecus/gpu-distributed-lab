#!/usr/bin/env bash
# =============================================================================
# scripts/part_a.sh — Part A (分散学習の基礎) のレッスン実行用エントリポイント
#
# 使い方:
#   bash scripts/part_a.sh <A01..A14> [lab.train などへの追加引数]
#
#   例: bash scripts/part_a.sh A07 --steps 100
#       NPROC=4 bash scripts/part_a.sh A08
#       NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0のprivate IP> RUN_ID=<RUN_ID> \
#         bash scripts/part_a.sh A13
#
#   追加引数は各コマンドの末尾に付くので、--out などを後から指定すると既定値を上書きできる。
#
# 読み取る環境変数:
#   複数 GPU のレッスンは launch.sh 経由で起動するので、NPROC / NNODES / NODE_RANK など
#   launch.sh の環境変数がそのまま効く。A13 では NNODES の指定が必須。
# =============================================================================
set -euo pipefail

# 第 1 引数 = レッスン番号 (必須)。残りの引数は "$@" としてそのまま下のコマンドに渡す。
lesson=${1:?A01..A14 required}
shift

case "$lesson" in
  # A01: GPU 情報の記録と FP32/BF16 行列積の計測 (CUDA event)。1 プロセスで直接実行。
  A01) exec python -m lab.cuda_basics "$@" ;;

  # A02: 1 GPU での学習 (以降の分散学習と比べる基準)。NPROC=1 に固定。
  A02) NPROC=1 bash scripts/launch.sh lab.train --mode single --out runs/A02 "$@" ;;

  # A03: GPU 数 1/2/4/8 で strong/weak scaling を測る (batch_sweep.py)。
  A03) exec python scripts/batch_sweep.py "$@" ;;

  # A04: 集団通信 (all-reduce, all-gather など) の帯域ベンチマーク。
  A04) bash scripts/launch.sh lab.collectives --out runs/A04/collectives.csv "$@" ;;

  # A05: リング all-reduce を自分で実装して動きを確かめる。
  A05) bash scripts/launch.sh lab.ring --out runs/A05/ring.csv "$@" ;;

  # A06: 勾配を手で all-reduce するデータ並列 (DDP の中身を理解する)。
  A06) bash scripts/launch.sh lab.train --mode manual --out runs/A06 "$@" ;;

  # A07: PyTorch DDP。DistributedSampler で各 rank に別々のデータを配る。
  A07) bash scripts/launch.sh lab.train --mode ddp --sampler --out runs/A07 "$@" ;;

  # A08: DeepSpeed ZeRO-1 (オプティマイザ状態を GPU 間で分割)。
  A08) bash scripts/launch.sh lab.train --mode zero1 --out runs/A08 "$@" ;;

  # A09: DeepSpeed ZeRO-2 (さらに勾配も分割)。
  A09) bash scripts/launch.sh lab.train --mode zero2 --out runs/A09 "$@" ;;

  # A10: DeepSpeed ZeRO-3 (さらにパラメータも分割)。
  A10) bash scripts/launch.sh lab.train --mode zero3 --out runs/A10 "$@" ;;

  # A11: PyTorch FSDP (ZeRO-3 相当を PyTorch 標準機能で)。
  A11) bash scripts/launch.sh lab.train --mode fsdp --out runs/A11 "$@" ;;

  # A12: テンソル並列 (1 つの層の行列を GPU 間で分割)。
  A12) bash scripts/launch.sh lab.train --mode tp --out runs/A12 "$@" ;;

  # A13: マルチノード DDP (8〜40 GPU)。ノード数 (1〜5) を必ず明示させるため、
  #      NNODES 未設定ならエラーにする。
  A13)
    : "${NNODES:?set 1..5}"
    bash scripts/launch.sh lab.train --mode ddp --sampler --out runs/A13 "$@"
    ;;

  # A14: DDP + プロファイラ。MFU・スケーリング効率と、計算/通信の重なりを
  #      タイムラインで確認する。
  A14) bash scripts/launch.sh lab.train --mode ddp --sampler --profile --out runs/A14 "$@" ;;

  # それ以外のレッスン番号はエラー。
  *)
    echo 'Unknown lesson' >&2
    exit 2
    ;;
esac
