#!/usr/bin/env bash
set -euo pipefail
# No EC2 creation. Requires existing approved host; separate uv environment.
: "${GROOT_COMMIT:?from configs/revisions.lock.json}"
mkdir -p third_party
if [[ ! -d third_party/Isaac-GR00T/.git ]];then
 GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T third_party/Isaac-GR00T
fi
cd third_party/Isaac-GR00T
git fetch origin "$GROOT_COMMIT"
git checkout --detach "$GROOT_COMMIT"
git submodule update --init --recursive
git lfs install
git lfs pull
# ffmpeg 4..7, git-lfs, uv and CUDA 12.8 must be installed first (docs/aws.md).
uv sync --frozen --python 3.12
uv run python gr00t/experiment/launch_finetune.py --help > ../../runs/groot-cli-help.txt
