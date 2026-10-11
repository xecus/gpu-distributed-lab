"""upstream (Hugging Face Trainer) の trainer_state ログを安定した CSV に変換するスクリプト。

目的:
    GR00T の学習ログ (trainer_state.json の log_history) から step ごとの
    loss / 学習率 / samples/s などを抜き出し、groot_metrics.csv にまとめる。
    ログに無いタイミング値を推測で作ることはしない。

入力:
    run  GR00T run の出力ディレクトリ (例: runs/B05)

出力:
    <run>/groot_metrics.csv       step ごとの学習メトリクス
    <run>/open_loop_metrics.json  *eval*.log から抽出した open-loop 評価の MSE/MAE
"""
import argparse, json, csv, re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run')
a = p.parse_args()
root = Path(a.run)

# step -> メトリクス辞書
rows = {}

# checkpoint ごとに trainer_state.json があるので、更新時刻の古い順に読み、
# 同じ step は新しいファイルの値で上書きする (resume 後の値を優先)
for path in sorted(root.rglob('trainer_state.json'), key=lambda x: x.stat().st_mtime):
    for entry in json.loads(path.read_text()).get('log_history', []):
        step = entry.get('step')
        if step is not None:
            row = rows.setdefault(step, {'step': step})
            # 必要なキーだけ取り出す (ログの種類によって含まれるキーが異なる)
            row.update({
                k: entry[k]
                for k in ['loss', 'learning_rate', 'epoch', 'grad_norm', 'train_runtime',
                          'train_samples_per_second']
                if k in entry
            })

# CSV の列は固定 (upstream のログ形式が変わっても列構成を安定させる)
keys = ['step', 'loss', 'learning_rate', 'epoch', 'grad_norm', 'train_runtime',
        'train_samples_per_second']
with (root / 'groot_metrics.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows[k] for k in sorted(rows))

# open-loop 評価ログから "Average MSE: 0.123" のような行を探し、最後の値を採用する
text = '\n'.join(p.read_text(errors='replace') for p in root.glob('*eval*.log'))
metrics = {}
for metric in ['MSE', 'MAE']:
    matches = re.findall(r'Average ' + metric + r'[^\n:]*[:=]\s*([0-9.eE+-]+)', text)
    if matches:
        metrics[metric] = float(matches[-1])
(root / 'open_loop_metrics.json').write_text(json.dumps(metrics, indent=2))

# ログが 1 件も無ければ、ヘッダだけの CSV を残した上でエラー終了する
if not rows:
    raise SystemExit(
        'No trainer_state log_history found; inspect pinned upstream logger. CSV header only.'
    )
