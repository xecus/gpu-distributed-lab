"""Part B 用エクスポート: 単一GPU / DDP の rank チェックポイントを Hugging Face 形式に変換する。

- 通常 (単一GPU / DDP): rank0.pt の 'model' に保存された state_dict を読み込む
  (DDP では全 rank が同じ重みを持つので rank 0 の分だけで十分)。
- ZeRO (--zero): FP32 のマスター重みが optimizer 状態として rank ごとに分割保存されている
  (ZeRO-3 ではモデル重み自体も分割) ので、DeepSpeed の zero_to_fp32 で
  FP32 の完全な state_dict に組み立て直してから読み込む。
- --lora: 学習時と同じ LoRA 構成でモデルを包んでから読み込み、adapter として保存する。
FSDP のチェックポイントは fsdp_export.py を使う。
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--revision', default='main')
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
p.add_argument('--lora', action='store_true')
p.add_argument('--zero', action='store_true')
a = p.parse_args()

# ベースモデルを読み込む (重みは後でチェックポイントの値で上書きする)
m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)

if a.lora:
    # state_dict のキーを学習時と一致させるため、学習時と同じ LoRA 設定で包む
    from peft import LoraConfig, get_peft_model
    m = get_peft_model(
        m,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            task_type='CAUSAL_LM',
            target_modules=[
                'q_proj',
                'k_proj',
                'v_proj',
                'o_proj',
                'gate_proj',
                'up_proj',
                'down_proj',
            ],
        ),
    )

if a.zero:
    # ZeRO: 分割保存された FP32 マスター重みを FP32 の1つの state_dict に統合する
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    state = get_fp32_state_dict_from_zero_checkpoint(a.checkpoint, tag='ds')
else:
    # 単一GPU / DDP: rank 0 のチェックポイントだけを読む
    state = torch.load(
        Path(a.checkpoint) / 'rank0.pt',
        map_location='cpu',
        weights_only=False,
    )['model']

# strict=True でキーの過不足があればエラーにする (変換ミスを見逃さないため)
m.load_state_dict(state, strict=True)
m.save_pretrained(a.out)
# tokenizer も同じディレクトリに保存して、そのまま from_pretrained できるようにする
AutoTokenizer.from_pretrained(a.model, revision=a.revision).save_pretrained(a.out)
