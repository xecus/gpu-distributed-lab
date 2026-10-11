"""Single/DDP rank checkpoint -> HF adapter/model. For ZeRO use zero_to_fp32 first."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--revision',default='main')
p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--lora',action='store_true')
p.add_argument('--zero',action='store_true');a=p.parse_args()
m=AutoModelForCausalLM.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16)
if a.lora:
 from peft import LoraConfig,get_peft_model
 m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
if a.zero:
 from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
 state=get_fp32_state_dict_from_zero_checkpoint(a.checkpoint,tag='ds')
else:state=torch.load(Path(a.checkpoint)/'rank0.pt',map_location='cpu',weights_only=False)['model']
m.load_state_dict(state,strict=True);m.save_pretrained(a.out);AutoTokenizer.from_pretrained(a.model,revision=a.revision).save_pretrained(a.out)
