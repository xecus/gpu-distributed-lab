"""runs/ 以下の全 run を集め、1 つの CSV にまとめるスクリプト。

目的:
    各レッスンで出力された結果を、results/experiment_results.csv と同じ列構成の
    CSV (既定: results/measured_results.csv) に集約し、横並びで比較できるようにする。

収集対象:
    - 学習メトリクス (environment_rank0.json + metrics_rank*_from*.csv)
    - environment_rank0.json が無い run の status 行 (status_node0.csv)。
      OOM / timeout の run のほか、GR00T (B04/B05) の run もここに入る
    - A01 の matmul ベンチマーク (matmul.csv)
    - A04 / A05 の通信ベンチマーク (bus_GBs 列を持つ CSV)
    - B04 / B05 の GR00T メトリクス (groot_metrics.csv)

集計方針:
    - tokens/s は rank0 の平均、step 時間は rank0 の中央値を使う。
      lab.train が step_s を全 rank の最大値に all_reduce 済みで、tokens_s も
      global batch から計算しているため、rank0 の値がそのまま全体の値になる
      (全 rank を混ぜると同じ step を重複して数えてしまう)。
    - peak memory は全 rank ファイルの最大値を取る (rank ごとに偏りがあり得るため)。
"""
import argparse, csv, json, re, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--runs', default='runs')
p.add_argument('--out', default='results/measured_results.csv')
a = p.parse_args()

# 出力 CSV の列は、手書きの結果表 (experiment_results.csv) のヘッダに合わせる
fields = next(csv.reader(open('results/experiment_results.csv')))
rows = []

# ---------------------------------------------------------------------------
# 1. 学習 run: environment_rank0.json がある run ディレクトリを走査
# ---------------------------------------------------------------------------
for env in Path(a.runs).rglob('environment_rank0.json'):
    path = env.parent
    meta = json.loads(env.read_text())
    args = meta['args']
    m = []      # rank0 のメトリクス行
    peaks = []  # 全 rank の allocated メモリ

    for f in path.glob('metrics_rank*_from*.csv'):
        data = list(csv.DictReader(f.open()))
        # peak memory は全 rank から集める
        peaks += [float(r['allocated_gib']) for r in data]
        # スループット等は rank0 のみ
        if 'rank0_' in f.name:
            m += data

    # resume で区間が重複する場合があるため、step をキーに重複排除 (後勝ち)。
    # 「後」は glob の列挙順で、resume の新しい区間が勝つとは限らない。
    m = list({r['step']: r for r in m}.values())

    # パスに含まれる実験 ID (A01, B05 など) のうち最後のものを採用
    ids = re.findall(r'[AB]\d\d', str(path))
    ident = ids[-1] if ids else 'unknown'

    # run_record.py が書いた status (ノード数・wall 時間・推定コストなど) があれば読む
    node_status = path / 'status_node0.csv'
    status = list(csv.DictReader(node_status.open()))[0] if node_status.exists() else {}

    row = dict(
        id=ident,
        variant=str(path),
        model=args['model'],
        nodes=status.get('nodes', ''),
        gpus=meta['world'],
        micro_batch=args['batch'],
        accum=args['accum'],
        global_batch=m[-1]['global_batch'] if m else '',
        seq=args['seq'],
        dtype='FP32 state / BF16 compute' if not args.get('cpu') else 'FP32 CPU',
        mode=args['mode'],
        status=status.get('status', 'ok' if m else 'no_metrics'),
        tokens_s=statistics.mean(float(r['tokens_s']) for r in m) if m else '',
        median_step_s=statistics.median(float(r['step_s']) for r in m) if m else '',
        peak_vram_gib=max(peaks) if peaks else '',
        wall_s=status.get('wall_seconds', ''),
        estimated_usd=status.get('estimated_usd', ''),
        notes='communication/MFU/scaling/billing: attach separate analyses; collect all rank files for cluster peak')
    rows.append(row)

# ---------------------------------------------------------------------------
# 2. environment_rank0.json が無い run (例: モデル読み込み中の OOM、GR00T の run) も
#    status_node0.csv だけから 1 行として記録する。失敗も重要な実験結果。
#    status は ok の場合もある (GR00T の run は 5. でも別の行になる)。
# ---------------------------------------------------------------------------
known = {r['variant'] for r in rows}
for f in Path(a.runs).rglob('status_node0.csv'):
    # 既に 1. で拾った run はスキップ
    if str(f.parent) in known:
        continue
    r = list(csv.DictReader(f.open()))[0]
    rows.append(dict(
        id=r['id'],
        variant=str(f.parent),
        nodes=r['nodes'],
        gpus=r['gpus'],
        status=r['status'],
        wall_s=r['wall_seconds'],
        estimated_usd=r['estimated_usd'],
        notes='see command log; no complete training metrics'))

# ---------------------------------------------------------------------------
# 3. A01: 単一 GPU の matmul (GEMM) ベンチマーク。dtype ごとに 1 行。
# ---------------------------------------------------------------------------
for f in Path(a.runs).rglob('matmul.csv'):
    for data in csv.DictReader(f.open()):
        rows.append(dict(
            id='A01',
            variant=str(f.parent) + '/' + data['dtype'],
            gpus=1,
            dtype=data['dtype'],
            status='ok',
            median_step_s=float(data['matmul_ms']) / 1000,  # ms -> s
            peak_vram_gib=data['peak_allocated_gib'],
            notes='GEMM TFLOPs=' + data['TFLOPs']))

# ---------------------------------------------------------------------------
# 4. A04/A05: 通信 (collective) マイクロベンチマーク。
#    ファイル名ではなく「bus_GBs 列があるか」で判定する。
# ---------------------------------------------------------------------------
for f in Path(a.runs).rglob('*.csv'):
    # 他の種類の CSV は除外
    if f.name in ['groot_metrics.csv', 'matmul.csv'] or f.name.startswith(('status_', 'metrics_')):
        continue
    with f.open() as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or 'bus_GBs' not in reader.fieldnames:
            continue
        for data in reader:
            # implementation 列があるのは A05 (実装比較)、無ければ A04 (collective 比較)
            ident = 'A05' if 'implementation' in data else 'A04'
            rows.append(dict(
                id=ident,
                variant=str(f) + '/' + data.get('collective', data.get('implementation', '')),
                gpus=data['world'],
                status='ok',
                nccl_bus_GBs=data['bus_GBs'],
                communication_ms=data['latency_ms'],
                notes='bytes=' + data['bytes'] + '; collective microbenchmark, not training step'))

# ---------------------------------------------------------------------------
# 5. B04/B05: GR00T (VLA モデル) の run。
#    VLA は「トークン」単位ではないので tokens_s/MFU は意図的に空欄にする。
# ---------------------------------------------------------------------------
for f in Path(a.runs).rglob('groot_metrics.csv'):
    data = list(csv.DictReader(f.open()))
    ids = re.findall(r'B0[45]', str(f))
    ident = ids[-1] if ids else 'B04'

    # groot_run.py が書いた wall 時間・GPU 数・終了状態
    wall = f.parent / 'wall_node0.json'
    values = json.loads(wall.read_text()) if wall.exists() else {}

    rows.append(dict(
        id=ident,
        variant=str(f.parent),
        model='nvidia/GR00T-N1.7-3B',
        gpus=values.get('gpus', ''),
        status=values.get('status', 'ok' if data else 'no_metrics'),
        wall_s=values.get('elapsed_s', ''),
        notes='VLA samples/s and action MAE/MSE in groot_metrics.csv/open_loop_metrics.json; tokens_s/MFU intentionally blank'))

# ---------------------------------------------------------------------------
# 6. CSV 書き出し (各行に無い列は DictWriter が空欄で埋める)
# ---------------------------------------------------------------------------
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f'{len(rows)} run variants -> {out}')
