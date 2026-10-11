"""Read-only metadata lookup; record exact model/upstream commits before setup."""
import argparse,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='configs/revisions.lock.json');a=p.parse_args()
items={}
for model in ['Qwen/Qwen3-8B','Qwen/Qwen3-32B','nvidia/GR00T-N1.7-3B','nvidia/Cosmos-Reason2-2B']:
 with urllib.request.urlopen('https://huggingface.co/api/models/'+model,timeout=30) as r:items[model]=json.load(r)['sha']
with urllib.request.urlopen('https://huggingface.co/api/datasets/Salesforce/wikitext',timeout=30) as r:items['Salesforce/wikitext']=json.load(r)['sha']
req=urllib.request.Request('https://api.github.com/repos/NVIDIA/Isaac-GR00T/commits/main',headers={'User-Agent':'gpu-lab'})
with urllib.request.urlopen(req,timeout=30) as r:items['Isaac-GR00T']=json.load(r)['sha']
Path(a.out).write_text(json.dumps(items,indent=2));print(json.dumps(items,indent=2))
