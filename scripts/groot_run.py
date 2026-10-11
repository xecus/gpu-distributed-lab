"""Run a commit-pinned upstream GR00T finetune in its own uv env, not torchrun twice."""
import argparse,subprocess,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',default='third_party/Isaac-GR00T')
p.add_argument('--dataset',default='demo_data/cube_to_bowl_5');p.add_argument('--modality',default='examples/SO100/so100_config.py')
p.add_argument('--out',default='runs/B05');p.add_argument('--gpus',type=int,choices=[1,8,16,24,32,40],default=1)
p.add_argument('--steps',type=int,default=100);p.add_argument('--batch',type=int,default=32)
p.add_argument('--resume',action='store_true');p.add_argument('--allow-multinode-smoke','--multinode-verified',dest='multinode_verified',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
repo=Path(a.repo).resolve();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
entry=str(Path(__file__).resolve().with_name('groot_entry.py'))
cmd=['uv','run','python',entry,'--base-model-path','nvidia/GR00T-N1.7-3B',
 '--dataset-path',a.dataset,'--embodiment-tag','NEW_EMBODIMENT','--modality-config-path',a.modality,
 '--num-gpus',str(a.gpus),'--output-dir',str(out),'--max-steps',str(a.steps),
 '--save-steps',str(min(50,a.steps)),'--save-total-limit','3','--global-batch-size',str(a.batch),'--dataloader-num-workers','4']
if a.resume:cmd+=['--resume-from-checkpoint']
nnodes=int(os.environ.get('NNODES','1'));node_rank=int(os.environ.get('NODE_RANK','0'))
if nnodes>1 and (not a.multinode_verified or a.gpus!=nnodes*8):
 raise SystemExit('B05 requires --allow-multinode-smoke after audit and --gpus = NNODES*8')
if a.batch%a.gpus:raise SystemExit('--batch must be divisible by total GPUs')
if a.gpus>1:
 launch=['uv','run','torchrun','--nproc_per_node',str(8 if nnodes>1 else a.gpus)]
 if nnodes==1:launch+=['--standalone']
 else:
  master=os.environ['MASTER_ADDR'];run_id=os.environ['RUN_ID']
  launch+=['--nnodes',str(nnodes),'--node_rank',str(node_rank),'--rdzv_backend','c10d',
           '--rdzv_endpoint',master+':'+os.environ.get('MASTER_PORT','29500'),'--rdzv_id',run_id,'--max_restarts','0']
 cmd=launch+cmd[3:]
if a.dry_run:print(json.dumps(cmd,indent=2));raise SystemExit(0)
from huggingface_hub import snapshot_download
lock=json.loads((Path(__file__).resolve().parents[1]/'configs/revisions.lock.json').read_text())
base=snapshot_download('nvidia/GR00T-N1.7-3B',revision=lock['nvidia/GR00T-N1.7-3B'])
cosmos=snapshot_download('nvidia/Cosmos-Reason2-2B',revision=lock['nvidia/Cosmos-Reason2-2B'])
cmd[cmd.index('--base-model-path')+1]=base
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
helptext=subprocess.check_output(['uv','run','python',entry,'--help'],cwd=repo,text=True)
for flag in ['--base-model-path','--dataset-path','--num-gpus','--max-steps']+(['--resume-from-checkpoint'] if a.resume else []):
 if flag not in helptext:raise SystemExit(f'Pinned upstream does not expose {flag}; consult its config before running')
(out/f'launch_node{node_rank}.json').write_text(json.dumps({'commit':commit,'cmd':cmd},indent=2))
env=os.environ.copy();env['COSMOS_LOCAL_PATH']=cosmos;env['WANDB_MODE']='disabled';env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(min(8,a.gpus))))
t=time.time()
with (out/f'train_node{node_rank}.log').open('w') as f:r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
(out/f'wall_node{node_rank}.json').write_text(json.dumps({'elapsed_s':time.time()-t,'returncode':r.returncode,'gpus':a.gpus,'status':'ok' if r.returncode==0 else 'error','runtime_multinode_verified':False}))
raise SystemExit(r.returncode)
