"""モデル / データセット / upstream リポジトリの正確なコミット SHA を記録するスクリプト。

目的:
    Hugging Face の "main" や GitHub の main ブランチは日々更新されるため、
    セットアップ前にその時点の SHA を取得して configs/revisions.lock.json に固定 (pin) する。
    以降は download_models.py / groot_run.py が lock を直接読み、download_corpus.py
    (--revision) や groot_setup.sh (GROOT_COMMIT)、lab.train (--revision) には
    lock の値を渡して使う。こうすることで、
    全ノード・全受講者が同じ重みとコードで実験でき、結果を再現できる。

    読み取り専用のメタデータ API を叩くだけで、何もダウンロードしない。

入力:
    --out  lock ファイルの出力先

出力:
    {名前: SHA} の JSON。
"""
import argparse, json, urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', default='configs/revisions.lock.json')
a = p.parse_args()

items = {}

# Hugging Face のモデル: /api/models/<id> の 'sha' が最新コミット
for model in ['Qwen/Qwen3-8B',
              'Qwen/Qwen3-32B',
              'nvidia/GR00T-N1.7-3B',
              'nvidia/Cosmos-Reason2-2B']:
    with urllib.request.urlopen(
        'https://huggingface.co/api/models/' + model,
        timeout=30,
    ) as r:
        items[model] = json.load(r)['sha']

# Hugging Face のデータセット (WikiText)
with urllib.request.urlopen(
    'https://huggingface.co/api/datasets/Salesforce/wikitext',
    timeout=30,
) as r:
    items['Salesforce/wikitext'] = json.load(r)['sha']

# GitHub の Isaac-GR00T main ブランチの最新コミット (GitHub API は User-Agent 必須)
req = urllib.request.Request(
    'https://api.github.com/repos/NVIDIA/Isaac-GR00T/commits/main',
    headers={'User-Agent': 'gpu-lab'},
)
with urllib.request.urlopen(req, timeout=30) as r:
    items['Isaac-GR00T'] = json.load(r)['sha']

Path(a.out).write_text(json.dumps(items, indent=2))
print(json.dumps(items, indent=2))
