"""A05: P2P 通信だけで自作する Ring AllReduce (ReduceScatter + AllGather)。

教育用の実装で、本番向けの最適化はしていません。
dist.batch_isend_irecv による隣接 rank との送受信だけで AllReduce を組み立て、
まず NCCL (または Gloo) の dist.all_reduce と結果が一致することを確認してから、
両者の latency / 帯域を計測して CSV に保存します。

起動例: torchrun --nproc_per_node=8 -m lab.ring  (CPU で試す場合は --cpu で Gloo を使用)
"""
import os, argparse, csv, time
from pathlib import Path

import torch
import torch.distributed as dist


def ring_allreduce(x):
    """in-place の Ring AllReduce (合計) = ReduceScatter フェーズ + AllGather フェーズ。

    x を world 個の chunk に分割する。
    ReduceScatter: world-1 ステップの間、各 rank は1つの chunk を右隣へ送り、
        左隣から受け取った chunk を自分の chunk に足し込む。
        終了後、rank r は chunk (r+1)%world の「全 rank 分の合計」を持つ。
    AllGather: さらに world-1 ステップで、合計済みの chunk をリング上で回して全員に配る。
    各 rank が送るデータ量は全体の 2*(world-1)/world 倍。
    bus_GBs でこの係数を掛けているのはそのため。
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    if x.numel() % world:
        raise ValueError('tensor size must divide world size')

    # chunk() はコピーではなく x の view を返すので、chunk を書き換えると x も書き換わる (in-place)
    chunks = list(x.chunk(world))
    left = (rank - 1) % world  # 受信相手 (左隣)
    right = (rank + 1) % world  # 送信相手 (右隣)

    def exchange(send_index, recv_index, add):
        # 受信用バッファは別に確保する。
        # send と recv を対にして nonblocking で同時に発行することで、
        # 全員が先に send して待ち合う「リングのデッドロック」を防ぐ
        recv = torch.empty_like(chunks[recv_index])
        send = chunks[send_index].contiguous()
        works = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send, right),
            dist.P2POp(dist.irecv, recv, left),
        ])
        for work in works:
            work.wait()

        if add:
            # ReduceScatter: 受け取った部分和を自分の chunk に足し込む
            chunks[recv_index].add_(recv)
        else:
            # AllGather: 完成済みの chunk で上書きする
            chunks[recv_index].copy_(recv)

    # ReduceScatter: ステップ i で rank は chunk (rank-i) を送り、chunk (rank-i-1) を受け取って加算。
    # 例 world=4, rank=0: 送信 0,3,2 / 受信 3,2,1 -> 最後に受信した chunk 1 = (0+1)%4 が完成する
    for i in range(world - 1):
        exchange((rank - i) % world, (rank - i - 1) % world, True)  # ReduceScatter

    # AllGather: 完成済みの chunk (rank+1) から送り始め、左隣の完成済み chunk を順に受け取って上書き
    for i in range(world - 1):
        exchange((rank - i + 1) % world, (rank - i) % world, False)  # AllGather

    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--elements', type=int, default=1048576)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', default='runs/A05/ring.csv')
    a = p.parse_args()

    # torchrun が設定する LOCAL_RANK でノード内の GPU を選ぶ。CPU 時は Gloo、GPU 時は NCCL
    local = int(os.environ.get('LOCAL_RANK', 0))
    if not a.cpu:
        torch.cuda.set_device(local)
    dev = 'cpu' if a.cpu else f'cuda:{local}'
    dist.init_process_group('gloo' if a.cpu else 'nccl')

    rank = dist.get_rank()
    world = dist.get_world_size()
    n = a.elements // world * world  # world で割り切れるように要素数を切り捨てる

    # NCCL では P2P 通信の前に集団通信 (barrier) でコミュニケータを初期化しておく
    dist.barrier()
    # rank ごとに異なる seed で入力を作る (全 rank 同じ値だと加算の検証にならない)
    g = torch.Generator().manual_seed(rank + 123)
    original = torch.randn(n, generator=g).to(dev)

    # 計測の前に、ライブラリの all_reduce と結果が一致するかを検証する
    expected = original.clone()
    dist.all_reduce(expected)
    actual = ring_allreduce(original.clone())
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    rows = []

    def sync():
        # GPU の処理は非同期なので、時間を測る前後で完了を待つ
        if not a.cpu:
            torch.cuda.synchronize()

    for name, fn in [('ring', ring_allreduce), ('nccl_or_gloo', dist.all_reduce)]:
        samples = []
        for i in range(a.iters + 5):
            x = original.clone()
            dist.barrier()
            sync()
            t = time.perf_counter()
            fn(x)
            sync()

            # 一番遅い rank の時間を採用する (全 rank が終わって初めて集団通信は完了)
            dt = torch.tensor([time.perf_counter() - t], device=dev)
            dist.all_reduce(dt, op=dist.ReduceOp.MAX)

            # 最初の5回は warmup として捨てる
            if i >= 5:
                samples.append(dt.item())

        secs = sum(samples) / len(samples)
        size = n * 4  # fp32 = 4 bytes
        # algorithm_GBs: データサイズ / 時間
        # bus_GBs: 各 rank が実際に送る量 (2*(world-1)/world 倍) で補正した帯域
        rows.append({
            'implementation': name,
            'world': world,
            'bytes': size,
            'latency_ms': secs * 1000,
            'algorithm_GBs': size / secs / 1e9,
            'bus_GBs': size / secs / 1e9 * 2 * (world - 1) / world,
        })

    # CSV の書き出しは rank 0 だけが行う
    if rank == 0:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
