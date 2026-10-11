"""JSONL {text: str} -> deterministic episode/document split and uint32 memmaps."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',default='Qwen/Qwen3-8B')
p.add_argument('--revision',default='main');p.add_argument('--out',default='data/qwen');a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
handles={k:(out/f'{k}.bin').open('wb') for k in ['train','val']};counts={k:0 for k in handles}
for line in Path(a.input).open():
    text=json.loads(line)['text']
    split='val' if int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%10==0 else 'train'
    ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
    np.asarray(ids,dtype=np.uint32).tofile(handles[split]);counts[split]+=len(ids)
for f in handles.values():f.close()
(out/'metadata.json').write_text(json.dumps({'tokenizer':a.model,'revision':a.revision,'counts':counts,
    'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest()},indent=2))
print(counts)
