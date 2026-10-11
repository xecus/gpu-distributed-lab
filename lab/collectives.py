"""A04: 集団通信ベンチマーク (AllReduce / AllGather / ReduceScatter)。

GPU では NCCL、--cpu 時は Gloo を使います (Gloo では ReduceScatter を省略)。

メッセージサイズを変えながら、3種類の集団通信の latency と帯域を測ります。
各集団通信は、まず数値的に正しい結果になるかを確認してから計測します。
bus_GBs には NCCL 標準の bus bandwidth 係数を掛けているので、
GPU 数 (world size) が違う結果同士でも比較できます。

起動: torchrun 経由で実行する (scripts/part_a.sh A04)。
"""
import argparse, csv, os, time
from pathlib import Path

import torch
import torch.distributed as dist

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--out', default='runs/A04/collectives.csv')
p.add_argument('--iters', type=int, default=50)
p.add_argument('--cpu', action='store_true')
p.add_argument('--max-power', type=int, default=28)
a = p.parse_args()

# ---- 分散環境の初期化 ----
# torchrun が設定する LOCAL_RANK でノード内の GPU を選ぶ。CPU 時は Gloo、GPU 時は NCCL
local = int(os.environ.get('LOCAL_RANK', 0))
if not a.cpu:
    torch.cuda.set_device(local)
dev = 'cpu' if a.cpu else f'cuda:{local}'
dist.init_process_group('gloo' if a.cpu else 'nccl')
rank = dist.get_rank()
world = dist.get_world_size()


def sync():
    # GPU 上の通信は非同期なので、時間計測の前後で完了を待つ
    if not a.cpu:
        torch.cuda.synchronize()


rows = []

# メッセージサイズは 2^10 .. 2^max_power バイト (4倍刻み)。
# fp32 は 4 バイトなので要素数は bytes/4。さらに world の倍数に切り捨てる
for power in range(10, a.max_power + 1, 2):
    n = max(world, (2**power // 4) // world * world)

    for name in ['all_reduce', 'all_gather', 'reduce_scatter']:
        # rank r は値 r+1 で埋めたテンソルを持つ
        # (合計系の期待値は world*(world+1)/2、all_gather では chunk r が r+1)
        x = torch.full((n,), float(rank + 1), device=dev)

        if name == 'all_reduce':
            # 全 rank の合計を全員が受け取る
            fn = lambda: dist.all_reduce(x)
            fn()
            torch.testing.assert_close(x, torch.full_like(x, world * (world + 1) / 2))
            size = n * 4
            # Ring AllReduce は ReduceScatter + AllGather なので、
            # 各 rank が 2*(world-1)/world 倍を送る
            factor = 2 * (world - 1) / world

        elif name == 'all_gather':
            # 各 rank の x を連結した n*world 要素のテンソルを全員が受け取る
            y = torch.empty(n * world, device=dev)
            fn = lambda: dist.all_gather_into_tensor(y, x)
            fn()
            # r 番目の chunk は rank r の値 (r+1) になっているはず
            for r, chunk in enumerate(y.chunk(world)):
                torch.testing.assert_close(chunk, torch.full_like(chunk, r + 1))
            size = n * 4 * world  # 出力全体のサイズで数える
            factor = (world - 1) / world

        else:
            # Gloo は reduce_scatter_tensor に対応していない場合がある (NCCL は対応)
            if a.cpu:
                continue
            # 合計を計算し、その 1/world ずつを各 rank が受け取る
            y = torch.empty(n // world, device=dev)
            fn = lambda: dist.reduce_scatter_tensor(y, x)
            fn()
            torch.testing.assert_close(y, torch.full_like(y, world * (world + 1) / 2))
            size = n * 4  # 入力全体のサイズで数える
            factor = (world - 1) / world

        x.zero_()

        # warmup: NCCL コミュニケータの準備や遅延確保 (lazy allocation) を計測から除く
        for _ in range(5):
            fn()

        sync()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(a.iters):
            fn()

        # 一番遅い rank の時間を報告する。集団通信は全 rank が終わって初めて完了するため
        sync()
        dt = torch.tensor([(time.perf_counter() - t) / a.iters], device=dev)
        dist.all_reduce(dt, op=dist.ReduceOp.MAX)
        secs = dt.item()

        # algorithm_GBs = size / 時間、bus_GBs = それに bus bandwidth 係数を掛けたもの
        rows.append({
            'collective': name,
            'bytes': size,
            'world': world,
            'latency_ms': secs * 1000,
            'algorithm_GBs': size / secs / 1e9,
            'bus_GBs': size / secs / 1e9 * factor,
        })

# 結果の書き出しは rank 0 だけが行う
if rank == 0:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    print(rows)

dist.destroy_process_group()
