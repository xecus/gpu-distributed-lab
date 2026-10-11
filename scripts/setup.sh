#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — GPU ノード上に教材用の Python 環境 (.venv) を作る
#
# 目的:
#   前提ソフトウェアがそろっているかを順番にチェックし、問題があれば
#   「何をインストールすればよいか」を表示して止まる。そろっていれば .venv を作り、
#   PyTorch・DeepSpeed などをインストールし、最後に GPU 8 枚が使えるか確認する。
#
# 使い方 (リポジトリのルートで):
#   bash scripts/setup.sh
#   PYTHON=python3.11 bash scripts/setup.sh   # 別の Python を使う場合
#
# 前提:
#   承認済みで起動中の DLAMI (Deep Learning AMI) 上で実行する。
#   システムにインストール済みの EFA ドライバ / ライブラリには手を付けない。
#
# 読み取る環境変数:
#   PYTHON    : 使用する Python 実行ファイル (既定 python3.12、3.10 以上が必要)
#   CUDA_HOME : CUDA ツールキットの場所 (既定 /usr/local/cuda-12.8)
# =============================================================================
set -euo pipefail

PY="${PYTHON:-python3.12}"

# -----------------------------------------------------------------------------
# チェック 1: Python 本体が存在するか
# -----------------------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 \
  || { echo "$PY not found; run: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev" >&2; exit 1; }

# -----------------------------------------------------------------------------
# チェック 2: Python のバージョンが 3.10 以上か (依存ライブラリの要件)
# -----------------------------------------------------------------------------
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "$PY is older than 3.10; set PYTHON=/path/to/python3.10+" >&2; exit 1; }

# -----------------------------------------------------------------------------
# チェック 3: Python の開発用ヘッダ (Python.h) があるか
#   Triton は初回使用時 (deepspeed の import 時) に C のヘルパーを JIT コンパイルするため、
#   Python.h が必要。python3.12-dev のようなパッケージに含まれている。
# -----------------------------------------------------------------------------
"$PY" -c 'import os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' \
  || { echo "Python.h missing; run: sudo apt-get install -y ${PY}-dev" >&2; exit 1; }

# -----------------------------------------------------------------------------
# チェック 4: CUDA 12.x のツールキット (nvcc) があるか
#   DeepSpeed は import 時に nvcc を確認する (FusedAdam の JIT ビルドに使う)。
#   CUDA_HOME (既定 /usr/local/cuda-12.8) → PATH 上の nvcc の順に探し、
#   バージョン表示に "release 12." が含まれるかで判定する。
#   (実行時の CUDA_HOME の自動設定は launch.sh が行う)
# -----------------------------------------------------------------------------
NVCC="${CUDA_HOME:-/usr/local/cuda-12.8}/bin/nvcc"
[ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] && "$NVCC" --version | grep 'release 12\.' >/dev/null \
  || { echo "CUDA 12.x toolkit (nvcc) not found; install the CUDA 12.8 runfile with --toolkit (see README) or set CUDA_HOME" >&2; exit 1; }

# -----------------------------------------------------------------------------
# チェック 5: gcc のバージョン
#   nvcc 12.8 がホストコンパイラとして対応するのは gcc 14 まで。
#   システムの gcc が 15 以上なら、FusedAdam の JIT ビルド用に gcc-14 / g++-14 が必要
#   (実際の切り替えは launch.sh が行う)。
# -----------------------------------------------------------------------------
if [ "$(gcc -dumpversion | cut -d. -f1)" -gt 14 ]; then
  command -v g++-14 >/dev/null 2>&1 \
    || { echo "system gcc is newer than 14; run: sudo apt-get install -y gcc-14 g++-14" >&2; exit 1; }
fi

# -----------------------------------------------------------------------------
# 仮想環境 (.venv) を作って有効化する
#   venv の作成に失敗する場合は、多くは python3.12-venv パッケージが未インストール。
# -----------------------------------------------------------------------------
"$PY" -m venv .venv || { echo "venv failed; try: sudo apt-get install -y ${PY}-venv" >&2; exit 1; }
source .venv/bin/activate

# -----------------------------------------------------------------------------
# パッケージのインストール
#   - PyTorch は CUDA 12.8 版 (cu128) を専用のインデックスから入れる。
#   - DS_BUILD_OPS=0 : DeepSpeed の C++/CUDA 拡張をインストール時にビルドしない
#                      (必要になった時点で JIT ビルドされる)。
#   - 教材パッケージ (lab) 自体は編集可能モード (-e) で入れる。依存は requirements.txt で
#     入れ済みなので --no-deps。
# -----------------------------------------------------------------------------
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# 再現性のため、実際にインストールされたパッケージの版をすべて記録しておく。
mkdir -p runs
python -m pip freeze > runs/pip-freeze.txt

# 最終確認: GPU 数 (8 枚以上) と pip の依存整合性を確認し、
# nvidia-smi / topo / fi_info / df の結果を runs/preflight.json に記録する。
python scripts/preflight.py --require-gpus 8
