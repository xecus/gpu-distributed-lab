"""Convert upstream Hugging Face trainer_state logs to a stable CSV; no invented timing."""
import argparse,json,csv,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();root=Path(a.run)
rows={}
for path in sorted(root.rglob('trainer_state.json'),key=lambda x:x.stat().st_mtime):
 for entry in json.loads(path.read_text()).get('log_history',[]):
  step=entry.get('step');
  if step is not None:
   row=rows.setdefault(step,{'step':step});row.update({k:entry[k] for k in ['loss','learning_rate','epoch','grad_norm','train_runtime','train_samples_per_second'] if k in entry})
keys=['step','loss','learning_rate','epoch','grad_norm','train_runtime','train_samples_per_second']
with (root/'groot_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows[k] for k in sorted(rows))
text='\n'.join(p.read_text(errors='replace') for p in root.glob('*eval*.log'))
metrics={}
for metric in ['MSE','MAE']:
 matches=re.findall(r'Average '+metric+r'[^\n:]*[:=]\s*([0-9.eE+-]+)',text)
 if matches:metrics[metric]=float(matches[-1])
(root/'open_loop_metrics.json').write_text(json.dumps(metrics,indent=2))
if not rows:raise SystemExit('No trainer_state log_history found; inspect pinned upstream logger. CSV header only.')
