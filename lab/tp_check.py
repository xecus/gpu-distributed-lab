"""A12: Attention + MLP の Tensor Parallel が正しいかを、分割しないモデルと比べて確認する。

同じ初期重みから「分割しない参照モデル (ref)」と「apply_tp で TP 化したモデル (tp)」を作り、
次の3点が一致することを確かめます。
    1. forward の loss
    2. backward 後の全パラメータの勾配
    3. SGD で1ステップ更新した後の全パラメータ

起動: NPROC=2 bash scripts/launch.sh lab.tp_check
"""
import os, copy

import torch
import torch.distributed as dist
from lab.model import Transformer, apply_tp

# ---- 分散環境の初期化 (TP は GPU + NCCL 前提) ----
local = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local)
dist.init_process_group('nccl')
world = dist.get_world_size()

# 全 rank で同じ seed を使い、同じ重み・同じ入力を作る
torch.manual_seed(42)
ref = Transformer(dim=64, layers=2, heads=8, seq=16).cuda()
# deepcopy で同じ重みを複製してから TP 化する (ref 自体は分割しない)
tp = apply_tp(copy.deepcopy(ref), world)

x = torch.randint(0, 256, (2, 16), device='cuda')
y = torch.randint(0, 256, (2, 16), device='cuda')
loss = lambda m: torch.nn.functional.cross_entropy(m(x).reshape(-1, 256), y.reshape(-1))

# ---- 1. forward の loss を比較 ----
a = loss(ref)
b = loss(tp)
torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

# ---- 2. 勾配を比較 ----
a.backward()
b.backward()
for (n, p), (name, q) in zip(ref.named_parameters(), tp.named_parameters()):
    assert n == name
    # TP 化したパラメータの勾配は DTensor (各 rank が一部だけ持つ) なので、
    # full_tensor() で全 rank 分を集めて元の形に戻してから比較する
    grad = q.grad.full_tensor() if hasattr(q.grad, 'full_tensor') else q.grad
    torch.testing.assert_close(p.grad, grad, rtol=1e-3, atol=1e-5)

# ---- 3. SGD 1ステップ後のパラメータを比較 ----
# tp 側は DTensor と通常の Tensor が混在するので foreach (まとめて更新) を無効にする
torch.optim.SGD(ref.parameters(), lr=.01).step()
torch.optim.SGD(tp.parameters(), lr=.01, foreach=False).step()  # DTensor/Tensor 混在
for p, q in zip(ref.parameters(), tp.parameters()):
    full = q.full_tensor() if hasattr(q, 'full_tensor') else q
    torch.testing.assert_close(p, full, rtol=1e-4, atol=1e-5)

if dist.get_rank() == 0:
    print('TP attention+MLP parity PASS')
dist.destroy_process_group()
