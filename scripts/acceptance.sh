#!/usr/bin/env bash
# =============================================================================
# scripts/acceptance.sh — GPU ノードの受け入れテスト (短時間で全モードを一通り動かす)
#
# 目的:
#   新しく借りた 8 GPU ノードで、教材の全モードが「とりあえず最後まで動く」ことを
#   数分で確認する。本番の長い実験を始める前の動作確認として使う。
#
#   フェーズ 1: 環境チェック (preflight.py で GPU 数と pip の依存整合性を確認)
#   フェーズ 2: Part A の各レッスンを数ステップだけ実行
#   フェーズ 3: step_4 のチェックポイントから再開し、保存/復元が壊れていないか確認
#   フェーズ 4: テンソル並列 (TP) の forward / 勾配 / 更新が単一 GPU と一致するか確認
#
# 使い方:
#   bash scripts/acceptance.sh
#   KEEP_CACHE=1 bash scripts/acceptance.sh   # ビルドキャッシュを残して再実行を速くする
#
# 読み取る環境変数:
#   KEEP_CACHE           : 1 ならビルドキャッシュを消さない (FusedAdam の再ビルドを省略)
#   TORCH_EXTENSIONS_DIR : torch / DeepSpeed の JIT 拡張キャッシュ (既定 ~/.cache/torch_extensions)
#   TRITON_CACHE_DIR     : Triton カーネルキャッシュ (既定 ~/.triton/cache)
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 準備: クリーンな状態から始める
#   - 前回の受け入れテストの出力 (runs/acceptance) を削除する。
#     古いチェックポイントが残っていると、再開テストが前回の結果を拾ってしまう。
#   - DeepSpeed / torch の JIT 拡張キャッシュ (FusedAdam) と Triton のキャッシュも削除する。
#     CUDA / gcc / torch のバージョンを変えた後に古いビルド成果物が残っていると、
#     原因の分かりにくいエラーになるため。ここで「ゼロからビルドできる」ことも確認する。
# -----------------------------------------------------------------------------
rm -rf runs/acceptance
if [[ "${KEEP_CACHE:-0}" != 1 ]]; then
  for d in "${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}" "${TRITON_CACHE_DIR:-$HOME/.triton/cache}"; do
    # 空文字やルート (/) を誤って rm -rf しないための安全確認。
    [[ -n "$d" && "$d" != / ]] && {
      echo "removing $d"
      rm -rf "$d"
    }
  done
fi

# -----------------------------------------------------------------------------
# フェーズ 1: 環境チェック
#   GPU 数 (8 枚以上) と pip の依存整合性を確認し、nvidia-smi / topo / fi_info / df の
#   結果を runs/preflight.json に記録する。
# -----------------------------------------------------------------------------
python scripts/preflight.py --require-gpus 8

# -----------------------------------------------------------------------------
# フェーズ 2: Part A の各レッスンを短く実行する
#   A04 (集団通信のベンチマーク) と A05 (リング all-reduce) は学習ではないので、
#   反復回数とデータサイズを小さくして実行する。
#   それ以外の学習レッスンは 4 ステップ・小さなモデルで、2 ステップごとに保存する
#   (→ step_2 と step_4 のチェックポイントができる)。
# -----------------------------------------------------------------------------
for lesson in A02 A04 A05 A06 A07 A08 A09 A10 A11 A12; do
  if [[ "$lesson" == A04 ]]; then
    bash scripts/part_a.sh "$lesson" --iters 5 --max-power 20 --out runs/acceptance/A04/collectives.csv
  elif [[ "$lesson" == A05 ]]; then
    bash scripts/part_a.sh "$lesson" --iters 5 --elements 1024 --out runs/acceptance/A05/ring.csv
  else
    bash scripts/part_a.sh "$lesson" --steps 4 --warmup 1 --save-every 2 --dim 64 --layers 2 --seq 32 \
      --out "runs/acceptance/$lesson"
  fi
done

# -----------------------------------------------------------------------------
# フェーズ 3: チェックポイントからの再開テスト
#   フェーズ 2 で保存した step_4 から 6 ステップ目まで学習を続ける。
#   DDP / ZeRO-1/2/3 / FSDP / TP はそれぞれ保存形式 (シャードの切り方) が違うので、
#   全モードで「保存 → 復元」が通ることを確認する。
#   DDP は A07 と同じく --sampler を付け、DistributedSampler 経路でも再開できることを確認する。
# -----------------------------------------------------------------------------
for item in 'A07 ddp' 'A08 zero1' 'A09 zero2' 'A10 zero3' 'A11 fsdp' 'A12 tp'; do
  # 'A07 ddp' のような文字列を、レッスン名とモード名に分解する。
  read -r lesson mode <<< "$item"
  extra=()
  if [[ "$mode" == ddp ]]; then
    extra+=(--sampler)
  fi
  bash scripts/launch.sh lab.train --mode "$mode" --steps 6 --warmup 1 --save-every 2 --dim 64 --layers 2 --seq 32 \
    --out "runs/acceptance/$lesson" --resume "runs/acceptance/$lesson/step_4" "${extra[@]}"
done

# -----------------------------------------------------------------------------
# フェーズ 4: テンソル並列の数値チェック
#   2 GPU で TP したモデルの forward 出力・勾配・パラメータ更新が、
#   分割しない場合と一致するかを確認する (分割の仕方のバグを検出する)。
# -----------------------------------------------------------------------------
NPROC=2 bash scripts/launch.sh lab.tp_check
