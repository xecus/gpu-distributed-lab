#!/usr/bin/env bash
# =============================================================================
# scripts/part_a.sh — Part A (分散学習の基礎) のレッスン実行用エントリポイント
#
# 使い方:
#   bash scripts/part_a.sh <A01..A14> [lab.train などへの追加引数]
#
#   例: bash scripts/part_a.sh A07 --steps 100
#       NPROC=4 bash scripts/part_a.sh A08
#       NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.11 RUN_ID=exp1 bash scripts/part_a.sh A13
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
  # A01: CUDA の基礎 (デバイス情報・テンソル演算・同期など)。1 プロセスで直接実行。
  A01) exec python -m lab.cuda_basics "$@" ;;

  # A02: 1 GPU での学習 (以降の分散学習と比べる基準)。NPROC=1 に固定。
  A02) NPROC=1 bash scripts/launch.sh lab.train --mode single --out runs/A02 "$@" ;;

  # A03: バッチサイズを変えながらスループットとメモリ使用量を測る。
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

  # A13: 複数ノードでの DDP。NNODES を明示的に指定しないとエラーにする
  #      (うっかり 1 ノードで実行して「複数ノードの結果」と誤解しないように)。
  A13)
    : "${NNODES:?set 1..5}"
    bash scripts/launch.sh lab.train --mode ddp --sampler --out runs/A13 "$@"
    ;;

  # A14: DDP + プロファイラ。計算と通信の重なり (オーバーラップ) を観察する。
  A14) bash scripts/launch.sh lab.train --mode ddp --sampler --profile --out runs/A14 "$@" ;;

  # それ以外のレッスン番号はエラー。
  *)
    echo 'Unknown lesson' >&2
    exit 2
    ;;
esac
