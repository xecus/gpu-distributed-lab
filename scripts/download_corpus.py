"""Small public corpus for full fine-tuning systems experiments; no private data needed."""
import argparse,json
from pathlib import Path
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--revision',default='main');p.add_argument('--out',default='data/wiki.jsonl');a=p.parse_args()
d=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train',revision=a.revision)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
 for row in d:
  if row['text'].strip():f.write(json.dumps({'text':row['text']},ensure_ascii=False)+'\n')
print(out)
