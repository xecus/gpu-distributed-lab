#!/usr/bin/env bash
# =============================================================================
# scripts/part_b.sh — Part B (実モデルでの大規模学習) のレッスン実行用エントリポイント
#
# 使い方:
#   bash scripts/part_b.sh <B01..B05> [追加引数]
#
#   例: TOKENS=data/train.bin MODEL_REVISION=<HF のコミット SHA> NODE_HOURLY_USD=98.32 \
#         bash scripts/part_b.sh B02
#       NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.11 RUN_ID=exp1 ... bash scripts/part_b.sh B05
#
# レッスン構成:
#   B01-B03 : Qwen3 を lab.train で学習する (8B を DDP / ZeRO-3、32B を ZeRO-3 など)
#   B04/B05 : NVIDIA GR00T の上流コードを専用 uv 環境で実行する (B04 は 1 ノード、B05 は複数ノード)
#
#   すべての実行を run_record.py で包むので、OOM やタイムアウトで失敗しても
#   「失敗した」という結果 (所要時間・概算コストなど) が記録として残る。
#
# 読み取る環境変数:
#   NNODES             : ノード数 (B04 は 1 のみ、B05 は 2〜5 が必須)
#   NPROC              : 1 ノードあたりの GPU 数 (既定 8、GPU 総数の記録に使う)
#   NODE_HOURLY_USD    : 1 ノード 1 時間あたりの料金 (コスト記録用。B01-B03 では必須)
#   EXPERIMENT_TIMEOUT : 実験のタイムアウト秒数 (既定 3600)
#   TOKENS             : (B01-B03 必須) 事前に作ったトークン列 train.bin のパス
#   MODEL_REVISION     : (B01-B03 必須) Hugging Face モデルのコミット SHA (再現性のため固定)
#   MODE               : 並列化モード (既定 zero3)。B01 では B01_MODE (既定 ddp) が優先される
#   ACCUM              : 勾配累積ステップ数 (既定 2)
#   その他、launch.sh が読む環境変数 (NODE_RANK, MASTER_ADDR, RUN_ID など)
# =============================================================================
set -euo pipefail

# 第 1 引数 = レッスン番号 (必須)。残りの引数は "$@" としてそのまま渡す。
lesson=${1:?B01..B05 required}
shift

# =============================================================================
# B04 / B05: GR00T (上流コード) のファインチューニング
# =============================================================================
if [[ "$lesson" == B04 || "$lesson" == B05 ]]; then
  # 出力先: 既定は runs/B04 など。追加引数に --out があればその値を使う
  # (run_record.py にも同じ出力先を教える必要があるため、ここで取り出しておく)。
  out="runs/$lesson"
  args=("$@")
  for ((i=0;i<${#args[@]};i++)); do
    if [[ "${args[$i]}" == --out ]]; then
      out="${args[$((i+1))]}"
    fi
  done

  # GPU 総数を決める。
  if [[ "$lesson" == B04 ]]; then
    # B04 は 1 ノード (8 GPU) 専用。
    if [[ "${NNODES:-1}" != 1 ]]; then
      echo 'B04 is single node; set NNODES=1' >&2
      exit 2
    fi
    total=8
  else
    # B05 は 2〜5 ノード。NNODES の指定が必須。
    : "${NNODES:?set 2..5}"
    if ((NNODES<2 || NNODES>5)); then
      echo 'B05 requires 2..5 nodes' >&2
      exit 2
    fi
    total=$((8*NNODES))
  fi

  # 料金が分かっていればコスト記録用に渡す (B04/B05 では任意)。
  price_args=()
  if [[ -n "${NODE_HOURLY_USD:-}" ]]; then
    price_args+=(--hourly-usd "$NODE_HOURLY_USD")
  fi

  # run_record.py が "--" 以降のコマンド (groot_run.py) を実行し、結果を記録する。
  exec python scripts/run_record.py --id "$lesson" --out "$out" --nodes "${NNODES:-1}" --gpus "$total" \
    --timeout "${EXPERIMENT_TIMEOUT:-3600}" "${price_args[@]}" -- \
    python scripts/groot_run.py --gpus "$total" --out "$out" "$@"
fi

# =============================================================================
# B01-B03: Qwen3 の学習 (lab.train)
# =============================================================================

# 必須の環境変数。どれかが未設定ならメッセージを出して即終了する。
#   TOKENS          : 学習データ (事前にトークン化した train.bin)
#   MODEL_REVISION  : モデルの版を固定し、後から同じ条件で再現できるようにする
#   NODE_HOURLY_USD : 料金を確認したうえで指定させ、コストを必ず記録する
: "${TOKENS:?prepared train.bin required}"
: "${MODEL_REVISION:?HF commit SHA required}"
: "${NODE_HOURLY_USD:?reviewed node hourly USD required for cost recording}"

# 並列化モードの既定は ZeRO-3 (大きなモデルでもメモリに収まりやすい)。
MODE=${MODE:-zero3}

case "$lesson" in
  # B01: 8B モデルを DDP で学習 (既定)。各 GPU にモデル全体を載せるので、
  #      メモリ不足 (OOM) になるかどうかを観察するレッスン。
  B01)
    model=Qwen/Qwen3-8B
    MODE=${B01_MODE:-ddp}
    ;;

  # B02: 8B モデルを ZeRO-3 (既定) で学習。B01 との比較でメモリ削減効果を見る。
  B02) model=Qwen/Qwen3-8B ;;

  # B03: 32B モデルを ZeRO-3 (既定) で学習。より大きなモデルを複数 GPU に分割する。
  B03) model=Qwen/Qwen3-32B ;;

  # それ以外のレッスン番号はエラー。
  *)
    echo 'Unknown lesson' >&2
    exit 2
    ;;
esac

# 出力先: 既定は runs/B02/zero3_1node のように、モードとノード数を名前に含める。
# 追加引数に --out があればその値を使う。
out="runs/$lesson/${MODE}_${NNODES:-1}node"
args=("$@")
for ((i=0;i<${#args[@]};i++)); do
  if [[ "${args[$i]}" == --out ]]; then
    out="${args[$((i+1))]}"
  fi
done

# run_record.py で包んで launch.sh → lab.train を実行する。
#   --seq 256 --batch 1     : 系列長・バッチを小さくしてメモリを節約
#   --accum                 : 勾配累積で実効バッチサイズを稼ぐ
#   --activation-checkpointing : 活性値を保存せず逆伝播時に再計算してメモリを節約
#   --save-every 50         : 途中でチェックポイントを保存 (保存/復元の時間も計測できる)
exec python scripts/run_record.py --id "$lesson" --out "$out" --nodes "${NNODES:-1}" \
  --gpus "$(( ${NPROC:-8} * ${NNODES:-1} ))" --hourly-usd "${NODE_HOURLY_USD:-0}" --timeout "${EXPERIMENT_TIMEOUT:-3600}" -- \
  bash scripts/launch.sh lab.train --mode "$MODE" --model "$model" --revision "$MODEL_REVISION" --tokens "$TOKENS" \
  --seq 256 --batch 1 --accum "${ACCUM:-2}" --steps 100 --warmup 10 --save-every 50 --sampler \
  --activation-checkpointing --out "$out" "$@"
