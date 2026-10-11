"""PyTorch profiler のトレースから、CUDA / NCCL カーネル区間の和集合 (union) を求めるスクリプト。

目的:
    Chrome trace 形式 (traceEvents) のトレースを読み、
    - GPU カーネルが動いていた合計時間
    - NCCL (通信) カーネルが動いていた合計時間
    を求める。

なぜ区間の「和集合」を取るのか:
    複数の CUDA stream 上でカーネルは同時に (重なって) 実行される。
    単純に dur を合計すると重なり部分を二重に数えてしまうため、
    区間を開始時刻でソートしてマージし、実際に「何かが動いていた時間」を求める。

注意:
    得られるのは「通信カーネルの実行時間」であって、計算が通信待ちで止まった
    時間 (exposed stall) ではない。NCCL は計算とオーバーラップするので、
    この値を step 時間から引いてはいけない。露出した待ち時間はタイムラインで確認する。

入力:
    trace  トレース JSON ファイル
    --out  出力 CSV (既定: runs/trace_stats.csv)
"""
import argparse, json, csv
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('trace')
p.add_argument('--out', default='runs/trace_stats.csv')
a = p.parse_args()

data = json.loads(Path(a.trace).read_text())


def union_us(intervals):
    """(開始, 終了) 区間のリストを受け取り、重なりをマージした合計長 [us] を返す。"""
    total = 0
    start = end = None  # 現在マージ中の区間
    for l, r in sorted(intervals):
        if start is None:
            # 最初の区間
            start, end = l, r
        elif l <= end:
            # 重なっている (または接している) -> 現在の区間を延長
            end = max(end, r)
        else:
            # 離れている -> 現在の区間を確定して合計し、新しい区間を開始
            total += end - start
            start, end = l, r
    # 最後にマージ中だった区間を加える
    return total + (0 if start is None else end - start)


# ph=='X' は「開始時刻 ts と長さ dur を持つ完了イベント」。cat に kernel を含むものが GPU カーネル。
kernels = [
    e for e in data['traceEvents']
    if e.get('ph') == 'X' and 'kernel' in str(e.get('cat', '')).lower()
]
# そのうち名前に nccl を含むものが通信カーネル
comm = [e for e in kernels if 'nccl' in e.get('name', '').lower()]

# イベントを (開始, 終了) の区間 [us] に変換
ints = lambda events: [(e['ts'], e['ts'] + e['dur']) for e in events if 'dur' in e]

row = {
    'gpu_kernel_union_ms': union_us(ints(kernels)) / 1000,  # us -> ms
    'nccl_kernel_union_ms': union_us(ints(comm)) / 1000,
    'nccl_kernel_count': len(comm),
    'gpu_kernel_count': len(kernels),
    'source': a.trace,
    'note': 'NCCL overlaps compute; do not subtract this from step time. Inspect timeline for exposed stalls.',
}

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=row)
    w.writeheader()
    w.writerow(row)
print(row)
