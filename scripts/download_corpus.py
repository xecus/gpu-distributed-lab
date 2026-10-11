"""full fine-tuning のシステム実験用に、小さな公開コーパスを取得するスクリプト。

目的:
    学習スループットやメモリを測るためのデータとして、WikiText-2 (raw) を
    JSONL 形式で保存する。私的なデータは一切不要。

入力:
    --revision  データセットの revision (既定: main。再現性のため
                revisions.lock.json の 'Salesforce/wikitext' の SHA を渡す)
    --out       出力先 JSONL (既定: data/wiki.jsonl)

出力:
    1 行 1 レコード {"text": ...} の JSONL。空行は除外する。
"""
import argparse, json
from pathlib import Path
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument('--revision', default='main')
p.add_argument('--out', default='data/wiki.jsonl')
a = p.parse_args()

# revision を固定すると、いつ実行しても同じデータで実験できる
d = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train', revision=a.revision)

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w') as f:
    for row in d:
        # WikiText には空行が多いのでスキップする
        if row['text'].strip():
            f.write(json.dumps({'text': row['text']}, ensure_ascii=False) + '\n')
print(out)
