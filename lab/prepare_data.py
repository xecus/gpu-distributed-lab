"""Part B 用データ準備: JSONL ({"text": str}) をトークン化し uint32 の memmap ファイルに書き出す。

B01 以降の実モデル (Qwen3 など) 学習で使います。
- 各文書を train / val に決定的に振り分ける (文書テキストの SHA-256 で約 1/10 を val に)。
  乱数を使わないので、何度実行しても・どのマシンで実行しても同じ分割になる。
- 文書ごとに末尾へ EOS トークンを付け、train.bin / val.bin に uint32 で連結して書く。
- 再現性のため、tokenizer 名・revision・トークン数・入力ファイルのハッシュを metadata.json に残す。
"""
import argparse, json, hashlib
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--model', default='Qwen/Qwen3-8B')
p.add_argument('--revision', default='main')
p.add_argument('--out', default='data/qwen')
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# 出力ファイル (train.bin / val.bin) と、それぞれに書いたトークン数
handles = {k: (out / f'{k}.bin').open('wb') for k in ['train', 'val']}
counts = {k: 0 for k in handles}

for line in Path(a.input).open():
    text = json.loads(line)['text']

    # テキストのハッシュ値の先頭8桁 (16進) を整数にし、10で割った余りが0なら val にする
    split = 'val' if int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'

    # 特殊トークンは付けず、文書の区切りとして EOS だけを末尾に足す
    ids = tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]

    # uint32 のバイナリとして追記 (学習側は np.memmap で読む)
    np.asarray(ids, dtype=np.uint32).tofile(handles[split])
    counts[split] += len(ids)

for f in handles.values():
    f.close()

# 再現性のためのメタデータ
(out / 'metadata.json').write_text(json.dumps(
    {
        'tokenizer': a.model,
        'revision': a.revision,
        'counts': counts,
        'input_sha256': hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
    },
    indent=2,
))
print(counts)
