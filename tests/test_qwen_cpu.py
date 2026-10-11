"""CPU integration only: tiny random Qwen3 architecture, not 8B/32B weights."""
import unittest,tempfile,os,subprocess,sys
from pathlib import Path
import numpy as np
from transformers import Qwen3Config,Qwen3ForCausalLM
ROOT=Path(__file__).resolve().parents[1]
class QwenCPUIntegration(unittest.TestCase):
 def test_qwen_training_path_and_csv(self):
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);model=tmp/'model';data=tmp/'tokens.bin';out=tmp/'run'
   m=Qwen3ForCausalLM(Qwen3Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
    num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,use_sliding_window=False))
   m.save_pretrained(model);(np.arange(4096,dtype=np.uint32)%256).tofile(data)
   cmd=[sys.executable,'-m','lab.train','--cpu','--model',str(model),'--tokens',str(data),'--seq','16',
    '--batch','2','--steps','4','--warmup','0','--save-every','2','--out',str(out),'--sampler']
   subprocess.run(cmd,cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1'},capture_output=True,timeout=60,check=True)
   import csv,json
   with (out/'metrics_rank0_from0.csv').open() as file:rows=list(csv.DictReader(file))
   self.assertEqual(len(rows),4)
   self.assertTrue(all(float(r['loss'])>0 for r in rows))
   meta=json.loads((out/'step_4/meta_rank0.json').read_text());self.assertEqual(len(meta['data_sha256']),64)
if __name__=='__main__':unittest.main()
