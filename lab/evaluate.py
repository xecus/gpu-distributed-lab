"""Part B 用評価: Hugging Face モデル (または保存した LoRA adapter) を検証用トークンで評価する。

prepare_data.py が作った val.bin (uint32 memmap) から長さ seq+1 の窓を先頭から順に切り出し、
次トークン予測の cross entropy loss を平均して、loss と perplexity を JSON に保存します。
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--revision', default='main')
p.add_argument('--adapter')
p.add_argument('--tokens', required=True)
p.add_argument('--seq', type=int, default=256)
p.add_argument('--windows', type=int, default=20)
p.add_argument('--out', default='runs/eval.json')
a = p.parse_args()

# ---- モデルの読み込み (BF16, 1GPU) ----
m = AutoModelForCausalLM.from_pretrained(
    a.model,
    revision=a.revision,
    torch_dtype=torch.bfloat16,
).cuda()
if a.adapter:
    # LoRA adapter を指定した場合は、ベースモデルに重ねて読み込む
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, a.adapter)

m.eval()
data = np.memmap(a.tokens, dtype=np.uint32, mode='r')
losses = []

# ---- 評価ループ (勾配不要なので inference_mode) ----
with torch.inference_mode():
    # 窓数は「指定数」と「データに収まる数」の小さい方
    for i in range(min(a.windows, (len(data) - 1) // a.seq)):
        # seq+1 トークンを切り出す: 入力は先頭 seq 個、正解は1つずらした seq 個
        ids = torch.tensor(
            data[i * a.seq:i * a.seq + a.seq + 1].astype('int64'),
            device='cuda',
        )[None]
        logits = m(input_ids=ids[:, :-1]).logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
        )
        losses.append(loss.item())

if not losses:
    raise ValueError('not enough validation tokens')

# ---- 結果の保存 ----
value = sum(losses) / len(losses)
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
# perplexity = exp(loss)。loss が大きすぎるとオーバーフローするので 80 で頭打ちにする
out.write_text(json.dumps(
    {
        'loss': value,
        'perplexity': math.exp(min(value, 80)),
        'windows': len(losses),
        'seq': a.seq,
    },
    indent=2,
))
