"""A01: GPU の情報を記録し、FP32 と BF16 の正方行列積を CUDA event で計測する。

出力:
    device.json  GPU 名、SM 数、HBM 容量、Compute Capability
    matmul.csv   dtype ごとの matmul 時間、TFLOPs、ピークメモリ

TF32 を無効にしているので、FP32 は本当に FP32 で計算されます
(有効だと Tensor Core が内部で TF32 に丸めて計算し、FP32 の性能を正しく測れない)。
TFLOPs は n×n 行列積1回あたり 2*n^3 FLOPs (掛け算と足し算) として計算します。
"""
import argparse, csv, time, json
from pathlib import Path

import torch

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--size', type=int, default=4096)
p.add_argument('--iters', type=int, default=30)
p.add_argument('--out', default='runs/A01')
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# ---- GPU の基本情報を記録 ----
prop = torch.cuda.get_device_properties(0)
(out / 'device.json').write_text(json.dumps(
    {
        'name': prop.name,
        'SMs': prop.multi_processor_count,
        'HBM_gib': prop.total_memory / 2**30,
        'compute_capability': [prop.major, prop.minor],
    },
    indent=2,
))

# ---- FP32 と BF16 で matmul の速度を比較 ----
rows = []
for dtype in [torch.float32, torch.bfloat16]:
    # TF32 を無効化して、FP32 を本来の精度・速度で測る
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(a.size, a.size, device='cuda', dtype=dtype)
    y = torch.randn_like(x)

    # warmup: cuBLAS のアルゴリズム選択やクロック上昇を計測から除く
    for _ in range(5):
        z = x @ y

    # CUDA event は GPU ストリーム上の実行時間そのものを測る。
    # ホスト側の time.time() ではカーネルの「起動」時間しか測れない (GPU は非同期に動くため)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(a.iters):
        z = x @ y
    end.record()
    torch.cuda.synchronize()  # end event まで GPU の処理が終わるのを待つ
    ms = start.elapsed_time(end) / a.iters  # 1回あたりのミリ秒

    rows.append({
        'dtype': str(dtype),
        'size': a.size,
        'matmul_ms': ms,
        'TFLOPs': 2 * a.size**3 / (ms / 1000) / 1e12,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
    })
    del x, y, z

with (out / 'matmul.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
print(rows)
