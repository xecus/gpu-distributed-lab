"""コマンドを timeout 付きで実行し、ok / OOM / timeout / error を「データ」として記録する。

目的:
    分散学習の実験では OOM やハングも重要な結果。失敗してもログと状態を必ず残し、
    collate.py で他の run と同じ表に並べられるようにする。
    インスタンスの起動 (provisioning) は行わない。

使い方:
    python scripts/run_record.py --id A03 --out runs/A03/x --gpus 8 -- <実行するコマンド...>

出力 (--out 以下, {node} は環境変数 NODE_RANK):
    command_node{node}.log  コマンドの標準出力・標準エラー
    status_node{node}.csv   状態・終了コード・wall 時間・推定コスト
    終了コードは子プロセスのもの (timeout 時は 124)。
"""
import argparse, csv, json, os, subprocess, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--id', required=True)
p.add_argument('--out', required=True)
p.add_argument('--timeout', type=int, default=3600)
p.add_argument('--nodes', type=int, default=1)
p.add_argument('--gpus', type=int, default=8)
p.add_argument('--hourly-usd', type=float)
# '--' 以降の残り全部を実行コマンドとして受け取る
p.add_argument('cmd', nargs=argparse.REMAINDER)
a = p.parse_args()

# REMAINDER には先頭の '--' も含まれるので取り除く
cmd = a.cmd[1:] if a.cmd[:1] == ['--'] else a.cmd
if not cmd:
    raise SystemExit('command required after --')

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
node = os.environ.get('NODE_RANK', '0')
t = time.time()

# ---------------------------------------------------------------------------
# コマンド実行。start_new_session=True で新しいプロセスグループにしておくと、
# timeout 時に torchrun とその子 (各 rank) をまとめて kill できる。
# ---------------------------------------------------------------------------
with (out / f'command_node{node}.log').open('w') as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=a.timeout)
        timed = False
    except subprocess.TimeoutExpired:
        import signal
        # まず SIGTERM で穏やかに終了を促し、15 秒待っても終わらなければ SIGKILL
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        # 124 は coreutils の timeout コマンドと同じ慣習
        code = 124
        timed = True

# ---------------------------------------------------------------------------
# 状態の判定: ログ (と、あれば学習ログ) から OOM のメッセージを探す
# ---------------------------------------------------------------------------
elapsed = time.time() - t
text = (out / f'command_node{node}.log').read_text(errors='replace')
extra = out / f'train_node{node}.log'
if extra.exists():
    text += '\n' + extra.read_text(errors='replace')

# 優先順位: timeout > ok (終了コード 0) > OOM (ログにメッセージあり) > その他 error
status = (
    'timeout' if timed
    else 'ok' if code == 0
    else 'oom' if any(x in text.lower() for x in ['cuda out of memory',
                                                  'cuda error: out of memory',
                                                  'outofmemoryerror'])
    else 'error'
)

# 推定コスト = wall 時間 [h] x ノード数 x 1 ノードの時間単価 (単価未指定なら None)
row = {
    'id': a.id,
    'node_rank': node,
    'nodes': a.nodes,
    'gpus': a.gpus,
    'status': status,
    'returncode': code,
    'wall_seconds': elapsed,
    'estimated_usd': None if a.hourly_usd is None else elapsed / 3600 * a.nodes * a.hourly_usd,
    'command': json.dumps(cmd),
}
with (out / f'status_node{node}.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=row)
    w.writeheader()
    w.writerow(row)

print(json.dumps(row, indent=2))
# 呼び出し元 (シェル) が失敗を検知できるよう、子プロセスの終了コードを引き継ぐ
raise SystemExit(code)
