"""Run a local command with timeout and record ok/OOM/error as data; no provisioning."""
import argparse,csv,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--out',required=True)
p.add_argument('--timeout',type=int,default=3600);p.add_argument('--nodes',type=int,default=1)
p.add_argument('--gpus',type=int,default=8);p.add_argument('--hourly-usd',type=float)
p.add_argument('cmd',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.cmd[1:] if a.cmd[:1]==['--'] else a.cmd
if not cmd:raise SystemExit('command required after --')
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);node=os.environ.get('NODE_RANK','0');t=time.time()
with (out/f'command_node{node}.log').open('w') as f:
 proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=proc.wait(timeout=a.timeout);timed=False
 except subprocess.TimeoutExpired:
  import signal
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  code=124;timed=True
elapsed=time.time()-t;text=(out/f'command_node{node}.log').read_text(errors='replace')
extra=out/f'train_node{node}.log'
if extra.exists():text+='\n'+extra.read_text(errors='replace')
status='timeout' if timed else 'ok' if code==0 else 'oom' if any(x in text.lower() for x in ['cuda out of memory','cuda error: out of memory','outofmemoryerror']) else 'error'
row={'id':a.id,'node_rank':node,'nodes':a.nodes,'gpus':a.gpus,'status':status,'returncode':code,
 'wall_seconds':elapsed,'estimated_usd':None if a.hourly_usd is None else elapsed/3600*a.nodes*a.hourly_usd,'command':json.dumps(cmd)}
with (out/f'status_node{node}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
print(json.dumps(row,indent=2));raise SystemExit(code)
