"""Download model snapshots at the commit SHA pinned in configs/revisions.lock.json."""
import argparse,json
from huggingface_hub import snapshot_download
p=argparse.ArgumentParser();p.add_argument('--lock',default='configs/revisions.lock.json')
p.add_argument('--models',nargs='+',default=['Qwen/Qwen3-8B']);a=p.parse_args();lock=json.load(open(a.lock))
for model in a.models:
 print(model,snapshot_download(model,revision=lock[model]))
