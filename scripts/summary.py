"""1 つの run の rank0 メトリクスを要約するスクリプト。

目的:
    lab.train が出力した metrics_rank0_from*.csv から、
    median / p95 step 時間、平均 tokens/s、rank0 のピークメモリ、推定コストを求め、
    <run>/summary.json に保存する。scaling.py はこの summary.json を入力にする。

なぜ rank0 だけか:
    DDP などでは全 rank が同期して同じ step を進むので、step 時間や tokens/s は
    rank0 を代表値とすれば十分。メモリも rank0 のみ (全 rank の最大は collate.py で見る)。

コストについて:
    1 ノード 8 GPU を前提とした wall-clock ベースの概算。
    実際の請求額は後で aws/cost.py (Cost Explorer) と突き合わせること。

入力:
    run                 run ディレクトリ
    --node-hourly-usd   1 ノードの時間単価 [USD]
    --nodes             ノード数
    --wall-seconds      run 全体の wall 時間 [s] (セットアップ等も含む)
    --extras-usd        ストレージ等の追加コスト [USD]
"""
import argparse, csv, statistics, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run')
p.add_argument('--node-hourly-usd', type=float, required=True)
p.add_argument('--nodes', type=int, default=1)
p.add_argument('--wall-seconds', type=float, required=True)
p.add_argument('--extras-usd', type=float, default=0)
a = p.parse_args()

# resume すると from<step> ごとに別ファイルになるので全部読む
rows = []
for f in Path(a.run).glob('metrics_rank0_from*.csv'):
    rows += list(csv.DictReader(f.open()))
if not rows:
    raise SystemExit('no rank0 CSV')

# resume の区間は重複し得る: step をキーにして後から読んだ行を優先する
rows = list({r['step']: r for r in rows}.values())
seconds = [float(r['step_s']) for r in rows]

rate = statistics.mean(float(r['tokens_s']) for r in rows)
# wall 時間ベースの総コスト = 時間 [h] x 単価 x ノード数 + 追加費用
cost = a.wall_seconds / 3600 * a.node_hourly_usd * a.nodes + a.extras_usd

report = {
    # 1 ノード 8 GPU を前提とした GPU 時間
    'gpu_hours': a.wall_seconds / 3600 * 8 * a.nodes,
    'measured_updates': len(rows),
    'median_step_s': statistics.median(seconds),
    # p95: ソートして 95% 位置の要素 (範囲外にならないよう末尾で打ち止め)
    'p95_step_s': sorted(seconds)[min(len(seconds) - 1, int(.95 * len(seconds)))],
    'mean_tokens_s': rate,
    'max_rank0_allocated_gib': max(float(r['allocated_gib']) for r in rows),
    'wall_estimated_usd': cost,
    # 定常状態のコスト効率: 1 秒あたりの費用 ÷ tokens/s x 100 万トークン
    'steady_usd_per_million_tokens': a.node_hourly_usd * a.nodes / 3600 / rate * 1e6,
    'note': 'Estimate; wall includes setup/checkpoint/idle. Reconcile billed amount with Cost Explorer. Memory is rank0 only.',
}

Path(a.run, 'summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
