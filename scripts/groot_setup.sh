#!/usr/bin/env bash
# =============================================================================
# scripts/groot_setup.sh — NVIDIA Isaac GR00T (Part B04/B05 用) のセットアップ
#
# 目的:
#   GR00T の上流リポジトリを、固定したコミットで third_party/ に取得し、
#   専用の uv 環境を作る。教材本体の .venv とは依存関係が衝突するので、環境は分ける。
#
# 使い方:
#   GROOT_COMMIT=<configs/revisions.lock.json に書かれたコミット> bash scripts/groot_setup.sh
#
# 前提:
#   EC2 インスタンスは作成しない。承認済みで起動中のホスト上で実行すること。
#   ffmpeg 4〜7, git-lfs, uv, CUDA 12.8 が事前にインストールされていること (docs/aws.md)。
#
# 読み取る環境変数:
#   GROOT_COMMIT (必須) : チェックアウトする GR00T のコミット SHA
# =============================================================================
set -euo pipefail

# 再現性のため、コミットは必ず固定値 (lock ファイルの値) を指定させる。未設定なら即エラー。
: "${GROOT_COMMIT:?from configs/revisions.lock.json}"

# -----------------------------------------------------------------------------
# リポジトリの取得 (初回のみ clone)
#   GIT_LFS_SKIP_SMUDGE=1 : clone 時に大きな LFS ファイルをダウンロードしない
#   --no-checkout         : 作業ツリーは後で指定コミットに checkout する
# -----------------------------------------------------------------------------
mkdir -p third_party
if [[ ! -d third_party/Isaac-GR00T/.git ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T third_party/Isaac-GR00T
fi

# -----------------------------------------------------------------------------
# 指定コミットに固定し、サブモジュールと LFS ファイルを取得する
# -----------------------------------------------------------------------------
cd third_party/Isaac-GR00T
git fetch origin "$GROOT_COMMIT"
git checkout --detach "$GROOT_COMMIT"
git submodule update --init --recursive
git lfs install
git lfs pull

# -----------------------------------------------------------------------------
# GR00T 専用の Python 3.12 環境を作る
#   --frozen : uv.lock どおりの版を入れる (lock を書き換えない = 再現性を保つ)
#   ※ ffmpeg 4〜7, git-lfs, uv, CUDA 12.8 が先にインストールされている必要がある (docs/aws.md)。
# -----------------------------------------------------------------------------
uv sync --frozen --python 3.12

# 動作確認: ファインチューニング用 CLI のヘルプを記録しておく。
# 上流の引数が変わっていないかを、後で groot_run.py と突き合わせるため。
uv run python gr00t/experiment/launch_finetune.py --help > ../../runs/groot-cli-help.txt
