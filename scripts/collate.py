"""Collect every run under runs/ into one CSV with the same columns as results/experiment_results.csv.

Picks up training metrics, OOM/timeout status rows, A01 matmul, A04/A05 communication benchmarks
and GR00T metrics. Only rank0 metrics are averaged; peak memory is the max over all rank files.
"""
import argparse,csv,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--runs',default='runs');p.add_argument('--out',default='results/measured_results.csv');a=p.parse_args()
fields=next(csv.reader(open('results/experiment_results.csv')));rows=[]
for env in Path(a.runs).rglob('environment_rank0.json'):
 path=env.parent;meta=json.loads(env.read_text());args=meta['args'];m=[];peaks=[]
 for f in path.glob('metrics_rank*_from*.csv'):
  data=list(csv.DictReader(f.open()));peaks += [float(r['allocated_gib']) for r in data]
  if 'rank0_' in f.name:m+=data
 m=list({r['step']:r for r in m}.values())
 ids=re.findall(r'[AB]\d\d',str(path));ident=ids[-1] if ids else 'unknown'
 node_status=path/'status_node0.csv';status=list(csv.DictReader(node_status.open()))[0] if node_status.exists() else {}
 row=dict(id=ident,variant=str(path),model=args['model'],nodes=status.get('nodes',''),gpus=meta['world'],
  micro_batch=args['batch'],accum=args['accum'],global_batch=m[-1]['global_batch'] if m else '',seq=args['seq'],
  dtype='FP32 state / BF16 compute' if not args.get('cpu') else 'FP32 CPU',mode=args['mode'],status=status.get('status','ok' if m else 'no_metrics'),
  tokens_s=statistics.mean(float(r['tokens_s']) for r in m) if m else '',median_step_s=statistics.median(float(r['step_s']) for r in m) if m else '',
  peak_vram_gib=max(peaks) if peaks else '',wall_s=status.get('wall_seconds',''),estimated_usd=status.get('estimated_usd',''),
  notes='communication/MFU/scaling/billing: attach separate analyses; collect all rank files for cluster peak')
 rows.append(row)
# Failed runs may have no environment file (e.g. OOM during model loading).
known={r['variant'] for r in rows}
for f in Path(a.runs).rglob('status_node0.csv'):
 if str(f.parent) in known:continue
 r=list(csv.DictReader(f.open()))[0];rows.append(dict(id=r['id'],variant=str(f.parent),nodes=r['nodes'],gpus=r['gpus'],status=r['status'],wall_s=r['wall_seconds'],estimated_usd=r['estimated_usd'],notes='see command log; no complete training metrics'))

for f in Path(a.runs).rglob('matmul.csv'):
 for data in csv.DictReader(f.open()):
  rows.append(dict(id='A01',variant=str(f.parent)+'/'+data['dtype'],gpus=1,dtype=data['dtype'],status='ok',
   median_step_s=float(data['matmul_ms'])/1000,peak_vram_gib=data['peak_allocated_gib'],notes='GEMM TFLOPs='+data['TFLOPs']))
for f in Path(a.runs).rglob('*.csv'):
 if f.name in ['groot_metrics.csv','matmul.csv'] or f.name.startswith(('status_','metrics_')):continue
 with f.open() as file:
  reader=csv.DictReader(file)
  if not reader.fieldnames or 'bus_GBs' not in reader.fieldnames:continue
  for data in reader:
   ident='A05' if 'implementation' in data else 'A04'
   rows.append(dict(id=ident,variant=str(f)+'/'+data.get('collective',data.get('implementation','')),
     gpus=data['world'],status='ok',nccl_bus_GBs=data['bus_GBs'],communication_ms=data['latency_ms'],
     notes='bytes='+data['bytes']+'; collective microbenchmark, not training step'))
for f in Path(a.runs).rglob('groot_metrics.csv'):
 data=list(csv.DictReader(f.open()));ids=re.findall(r'B0[45]',str(f));ident=ids[-1] if ids else 'B04'
 wall=f.parent/'wall_node0.json';values=json.loads(wall.read_text()) if wall.exists() else {}
 rows.append(dict(id=ident,variant=str(f.parent),model='nvidia/GR00T-N1.7-3B',gpus=values.get('gpus',''),
  status=values.get('status','ok' if data else 'no_metrics'),wall_s=values.get('elapsed_s',''),
  notes='VLA samples/s and action MAE/MSE in groot_metrics.csv/open_loop_metrics.json; tokens_s/MFU intentionally blank'))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(f'{len(rows)} run variants -> {out}')
