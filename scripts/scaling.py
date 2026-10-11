"""複数の summary.json から strong/weak scaling 効率と概算 MFU を計算するスクリプト。

    A03 の sweep や A14 / B02、マルチノード run の比較に使う。

目的:
    GPU 数の異なる run の summary.json (summary.py の出力) を並べ、
    最初の run を基準 (baseline) としてスケーリング効率を求める。

計算式:
    ratio      = GPU 数 / 基準 GPU 数
    speedup    = 基準の median step 時間 / この run の median step 時間
    strong 効率 = speedup / ratio
                 (global batch 固定なので、GPU が N 倍なら step 時間が 1/N になるのが理想)
    weak 効率   = (tokens/s / 基準 tokens/s) / ratio
                 (GPU あたりの batch 固定なので、tokens/s が N 倍になるのが理想)
    MFU        = 6 * params * tokens/s / (GPU 数 * GPU 1 枚の BF16 ピーク FLOPS)
                 学習 1 トークンあたり forward 2*params + backward 4*params ≒ 6*params FLOPs
                 という標準的な近似を使う。

    ピーク値は実際に使った GPU の dense (sparsity なし) BF16 値を渡すこと。
    H100 / H200 / B200 で値が異なる。

入力:
    --runs / --gpus  run ディレクトリと、それぞれの GPU 数 (同じ順・同じ個数)
    --kind           strong または weak
    --params / --peak-bf16-tflops-per-gpu  両方指定したときのみ MFU を計算

出力:
    runs/scaling.csv (既定)
"""
import argparse, json, csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--gpus', nargs='+', type=int, required=True)
p.add_argument('--kind', choices=['strong', 'weak'], required=True)
p.add_argument('--params', type=int)
p.add_argument(
    '--peak-bf16-tflops-per-gpu',
    type=float,
    help='non-sparse peak from the actual GPU specification',
)
p.add_argument('--out', default='runs/scaling.csv')
a = p.parse_args()

if len(a.runs) != len(a.gpus):
    raise ValueError('run/GPU count mismatch')

# (GPU 数, summary) のリストを作る
items = []
for path, gpu in zip(a.runs, a.gpus):
    s = json.loads(Path(path, 'summary.json').read_text())
    items.append((gpu, s))

# 先頭の run を基準にする (--runs/--gpus の最初の要素。
# A03 なら 1 GPU、マルチノード比較なら 8 GPU など)
basegpu, bases = items[0]
rows = []

for gpu, s in items:
    ratio = gpu / basegpu
    speed = bases['median_step_s'] / s['median_step_s']
    efficiency = (
        speed / ratio if a.kind == 'strong'
        else s['mean_tokens_s'] / bases['mean_tokens_s'] / ratio
    )

    # MFU: 実効 FLOPS (6 * params * tokens/s) ÷ 理論ピーク (GPU 数 * TFLOPS * 1e12)
    mfu = None
    if a.params and a.peak_bf16_tflops_per_gpu:
        mfu = 6 * a.params * s['mean_tokens_s'] / (gpu * a.peak_bf16_tflops_per_gpu * 1e12)

    rows.append({
        'gpus': gpu,
        'kind': a.kind,
        'step_speedup': speed,
        'scaling_efficiency': efficiency,
        'approx_MFU': mfu,
        'tokens_s': s['mean_tokens_s'],
        'usd_per_million_tokens': s['steady_usd_per_million_tokens'],
    })

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
print(rows)
