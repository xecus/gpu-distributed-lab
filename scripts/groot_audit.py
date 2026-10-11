"""Static check of the pinned GR00T source for torchrun compatibility before any multi-node run.

Passing only means the code reads WORLD_SIZE/LOCAL_RANK, initializes a process group and does not
spawn its own workers. It is not runtime verification.
"""
import argparse,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',default='third_party/Isaac-GR00T');p.add_argument('--out',default='runs/groot_audit.json');a=p.parse_args()
r=Path(a.repo);entry=(r/'gr00t/experiment/launch_finetune.py').read_text();experiment=(r/'gr00t/experiment/experiment.py').read_text()
checks={'world_size_read':'WORLD_SIZE' in experiment,'local_rank_read':'LOCAL_RANK' in experiment,
 'process_group_init':'init_process_group' in experiment,'world_size_validation':'_assert_num_gpus_matches_world_size' in experiment,
 'no_nested_spawn':not bool(re.search(r'mp\.spawn|multiprocessing\.spawn|subprocess\.Popen',entry)),
 'trainer':'Gr00tTrainer' in experiment}
report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),
 'checks':checks,'static_gate_passed':all(checks.values()),
 'runtime_verified':False,'next':'2 nodes x 8 GPUs, 2 updates + checkpoint + resume + sample-count check before 5 nodes'}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
raise SystemExit(0 if all(checks.values()) else 2)
