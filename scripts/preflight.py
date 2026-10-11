"""Host checks before experiments: GPU count, CUDA/torch versions, topology, EFA provider, pip check.

Writes runs/preflight.json and exits non-zero if fewer GPUs than --require-gpus or pip conflicts.
"""
import argparse,json,os,platform,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--require-gpus',type=int,default=8);a=p.parse_args()
import torch
checks={};checks['python']=sys.version;checks['torch']=torch.__version__;checks['cuda']=torch.version.cuda
checks['host']=platform.node();checks['gpu_count']=torch.cuda.device_count()
for command in [['nvidia-smi','-L'],['nvidia-smi','topo','-m'],['fi_info','-p','efa'],['df','-h'],['python','-m','pip','check']]:
 try:
  r=subprocess.run(command,text=True,capture_output=True);checks[' '.join(command)]={'code':r.returncode,'output':r.stdout+r.stderr}
 except FileNotFoundError:checks[' '.join(command)]={'code':127,'output':'not installed / PATH missing'}
Path('runs').mkdir(exist_ok=True);Path('runs/preflight.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
if checks['gpu_count']<a.require_gpus:sys.exit('GPU count too small')
if checks['python -m pip check']['code']!=0:sys.exit('dependency conflicts')
