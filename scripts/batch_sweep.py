"""A03: run 1/2/4/8-GPU sweeps for strong scaling (fixed global batch) and weak scaling (fixed per-GPU batch)."""
import argparse,os,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=40);p.add_argument('--global-batch',type=int,default=16)
a=p.parse_args()
for gpus in [1,2,4,8]:
 for kind in ['strong','weak']:
  accum=a.global_batch//gpus if kind=='strong' else 2
  if accum<1 or (kind=='strong' and a.global_batch%gpus):raise ValueError('invalid global batch')
  out=f'runs/A03/{kind}_{gpus}gpu';cmd=[sys.executable,'scripts/run_record.py','--id','A03','--out',out,'--gpus',str(gpus),'--',
   'bash','scripts/launch.sh','lab.train','--mode','single' if gpus==1 else 'ddp','--batch','1','--accum',str(accum),
   '--steps',str(a.steps),'--out',out,'--sampler']
  subprocess.run(cmd,env={**os.environ,'NPROC':str(gpus)},check=False)
