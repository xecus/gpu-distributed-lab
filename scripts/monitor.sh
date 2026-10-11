#!/usr/bin/env bash
# =============================================================================
# scripts/monitor.sh — GPU の使用状況を 1 秒ごとに CSV へ記録する
#
# 目的:
#   学習中の GPU 使用率・メモリ使用量・消費電力を記録し、
#   GPU 使用率が極端に低い時間帯 (データ待ち等) がないか、メモリに余裕があるかを
#   後から確認する。NCCL の通信カーネルも「使用中」と数えられるので、
#   通信待ちの判定はプロファイラのタイムラインで行う。
#
# 使い方 (学習とは別のシェルで実行し、Ctrl-C で止める):
#   bash scripts/monitor.sh                    # runs/gpu.csv に記録
#   bash scripts/monitor.sh runs/A07/gpu.csv   # 出力先を指定
#
# 引数:
#   $1 : 出力 CSV のパス (省略時 runs/gpu.csv)
# =============================================================================
set -euo pipefail

out=${1:-runs/gpu.csv}
mkdir -p "$(dirname "$out")"

# nvidia-smi を --loop=1 で 1 秒ごとにサンプリングし続ける。
# 記録する列: 時刻, GPU 番号, UUID, GPU 使用率, 使用メモリ, 総メモリ, 消費電力
exec nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv --loop=1 > "$out"
