"""configs/revisions.lock.json に固定されたコミット SHA でモデルをダウンロードするスクリプト。

目的:
    Hugging Face のモデルは "main" が随時更新されるため、SHA で固定 (pin) して
    取得することで、全員・全ノードが完全に同じ重みで実験できるようにする。
    lock ファイルは resolve_revisions.py で作成する。

入力:
    --lock    revision lock ファイル
    --models  ダウンロードするモデル ID (複数可)

出力:
    各モデルについて「モデル ID とローカルのスナップショットパス」を表示する。
"""
import argparse, json
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--lock', default='configs/revisions.lock.json')
p.add_argument('--models', nargs='+', default=['Qwen/Qwen3-8B'])
a = p.parse_args()
lock = json.load(open(a.lock))

for model in a.models:
    # lock に無いモデルは KeyError になる (意図的: 未固定のモデルは使わせない)
    print(model, snapshot_download(model, revision=lock[model]))
