"""Summarize one run's rank0 metrics: median/p95 step time, tokens/s, memory and estimated cost.

Cost assumes 8 GPUs per node and is a wall-clock estimate; reconcile with aws/cost.py afterwards.
"""
import argparse,csv,statistics,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--node-hourly-usd',type=float,required=True)
p.add_argument('--nodes',type=int,default=1);p.add_argument('--wall-seconds',type=float,required=True)
p.add_argument('--extras-usd',type=float,default=0);a=p.parse_args()
rows=[]
for f in Path(a.run).glob('metrics_rank0_from*.csv'):rows+=list(csv.DictReader(f.open()))
if not rows:raise SystemExit('no rank0 CSV')
# Resume segments may overlap: latest input wins by step.
rows=list({r['step']:r for r in rows}.values());seconds=[float(r['step_s']) for r in rows]
rate=statistics.mean(float(r['tokens_s']) for r in rows);cost=a.wall_seconds/3600*a.node_hourly_usd*a.nodes+a.extras_usd
report={'gpu_hours':a.wall_seconds/3600*8*a.nodes,'measured_updates':len(rows),'median_step_s':statistics.median(seconds),
 'p95_step_s':sorted(seconds)[min(len(seconds)-1,int(.95*len(seconds)))],
 'mean_tokens_s':rate,'max_rank0_allocated_gib':max(float(r['allocated_gib']) for r in rows),
 'wall_estimated_usd':cost,'steady_usd_per_million_tokens':a.node_hourly_usd*a.nodes/3600/rate*1e6,
 'note':'Estimate; wall includes setup/checkpoint/idle. Reconcile billed amount with Cost Explorer. Memory is rank0 only.'}
Path(a.run,'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
