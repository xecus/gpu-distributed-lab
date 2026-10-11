import os,sys,tempfile,subprocess,unittest,json
from pathlib import Path
import torch
from lab.model import Transformer
ROOT=Path(__file__).resolve().parents[1]
class TrainingTests(unittest.TestCase):
 def run_train(self,out,mode='single',steps=4,resume=None):
  cmd=[sys.executable]
  if mode in ['manual','ddp']:cmd+=['-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=29501','--nproc_per_node=2','--module']
  else:cmd+=['-m']
  cmd+=['lab.train','--cpu','--mode',mode,'--dim','32','--layers','1','--heads','4','--seq','16','--batch','2','--steps',str(steps),'--warmup','0','--save-every','2','--out',str(out)]
  if resume:cmd+=['--resume',str(resume)]
  subprocess.run(cmd,cwd=ROOT,check=True,env={**os.environ,'OMP_NUM_THREADS':'1'},capture_output=True,timeout=120)
 def load(self,path):return torch.load(path,map_location='cpu',weights_only=False)
 def test_causal_and_gradients(self):
  torch.manual_seed(1);m=Transformer(dim=32,layers=1,heads=4,seq=16)
  x=torch.randint(0,256,(2,16));y=x.clone();y[:,8:]=(y[:,8:]+1)%256
  torch.testing.assert_close(m(x)[:,:8],m(y)[:,:8])
  m(x).square().mean().backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
 def test_resume_exact_and_loss(self):
  with tempfile.TemporaryDirectory() as tmp:
   full=Path(tmp)/'full';split=Path(tmp)/'split'
   self.run_train(full,steps=4);self.run_train(split,steps=2);self.run_train(split,steps=4,resume=split/'step_2')
   a=self.load(full/'step_4/rank0.pt');b=self.load(split/'step_4/rank0.pt')
   for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
 @unittest.skipIf(os.environ.get('LAB_TEST_NO_SOCKETS')=='1','container blocks Gloo TCP sockets; run on AWS host')
 def test_manual_matches_ddp(self):
  with tempfile.TemporaryDirectory() as tmp:
   manual=Path(tmp)/'manual';ddp=Path(tmp)/'ddp'
   self.run_train(manual,mode='manual');self.run_train(ddp,mode='ddp')
   a=self.load(manual/'step_4/rank0.pt');b=self.load(ddp/'step_4/rank0.pt')
   for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=1e-5,atol=1e-6)
 def test_average_gradient_equals_global_batch(self):
  import copy
  torch.manual_seed(9);m=Transformer(dim=32,layers=1,heads=4,seq=16);n=copy.deepcopy(m)
  x=torch.randint(0,256,(4,16));target=torch.randint(0,256,(4,16))
  loss=torch.nn.functional.cross_entropy(m(x).reshape(-1,256),target.reshape(-1));loss.backward()
  partial=[]
  for shard in [slice(0,2),slice(2,4)]:
   n.zero_grad();torch.nn.functional.cross_entropy(n(x[shard]).reshape(-1,256),target[shard].reshape(-1)).backward()
   partial.append([p.grad.clone() for p in n.parameters()])
  for i,p in enumerate(m.parameters()):torch.testing.assert_close(p.grad,(partial[0][i]+partial[1][i])/2,rtol=1e-4,atol=1e-7)
 def test_aws_guard(self):
  r=subprocess.run(['bash','aws/launch_after_approval.sh'],cwd=ROOT,env={k:v for k,v in os.environ.items() if k!='AWS_PAID_RESOURCE_APPROVAL'},capture_output=True)
  self.assertEqual(r.returncode,2);self.assertIn(b'Refusing',r.stderr)
 def test_plan_never_launches(self):
  with tempfile.TemporaryDirectory() as tmp:
   dest=Path(tmp)/'plan.json'
   subprocess.run([sys.executable,'aws/plan.py','--ami','ami-example','--subnet','subnet-example','--sg','sg-example','--profile','example','--placement','example','--run-id','test','--out',str(dest)],cwd=ROOT,check=True,capture_output=True)
   r=json.loads(dest.read_text());self.assertEqual(len(r['NetworkInterfaces']),32);self.assertEqual(r['MaxCount'],1)
   subprocess.run([sys.executable,'aws/plan.py','--ami','ami-example','--subnet','subnet-example','--sg','sg-example','--profile','example','--placement','example','--run-id','test','--instance-type','p6-b200.48xlarge','--out',str(dest)],cwd=ROOT,check=True,capture_output=True)
   r=json.loads(dest.read_text());self.assertEqual(r['InstanceType'],'p6-b200.48xlarge');self.assertEqual(len(r['NetworkInterfaces']),8)
if __name__=='__main__':unittest.main()
