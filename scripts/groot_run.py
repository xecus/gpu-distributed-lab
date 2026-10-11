"""コミット固定した upstream GR00T の finetune を、専用の uv 環境で実行するスクリプト (B04/B05)。

目的:
    NVIDIA Isaac-GR00T (VLA モデル) の公式 finetune を、このラボの計測形式に合わせて起動する。

なぜ専用の uv 環境で動かすのか:
    GR00T は torch / transformers などの依存バージョンを独自に固定しており、
    ラボ本体の venv と混ぜると衝突する。そこで upstream リポジトリ付属の
    uv 環境 (`uv run`) の中で実行し、依存を完全に分離する。
    複数 GPU のときは、このスクリプトが uv 環境内の torchrun を直接 1 回だけ起動する
    (launch.sh 経由で torchrun を二重に重ねない)。1 GPU では torchrun を使わない。

入力 (主なもの):
    --repo / --dataset / --modality  GR00T リポジトリ・データセット・modality 設定
    --gpus     合計 GPU 数 (1/8/16/24/32/40 のいずれか、既定 1。
               マルチノード時は NNODES*8 と一致必須)
    --steps / --batch                学習ステップ数と global batch
    --resume   checkpoint から再開
    --allow-multinode-smoke (--multinode-verified)
               NNODES>1 で必須の明示フラグ。groot_audit.py の合格を人が確認してから付ける
               (スクリプトは audit 結果を検査しない)
    --dry-run  実行せずにコマンドだけ表示
    環境変数 NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / RUN_ID (マルチノード時)

出力 (--out 以下):
    launch_node{rank}.json  使用した upstream コミットと実行コマンド
    train_node{rank}.log    学習ログ
    wall_node{rank}.json    経過時間・終了コード・状態
"""
import argparse, subprocess, json, time, os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default='third_party/Isaac-GR00T')
p.add_argument('--dataset', default='demo_data/cube_to_bowl_5')
p.add_argument('--modality', default='examples/SO100/so100_config.py')
p.add_argument('--out', default='runs/B05')
p.add_argument('--gpus', type=int, choices=[1, 8, 16, 24, 32, 40], default=1)
p.add_argument('--steps', type=int, default=100)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--resume', action='store_true')
p.add_argument(
    '--allow-multinode-smoke',
    '--multinode-verified',
    dest='multinode_verified',
    action='store_true',
)
p.add_argument('--dry-run', action='store_true')
a = p.parse_args()

repo = Path(a.repo).resolve()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)

# 公式 launch_finetune.py を包むアダプタ (Cosmos のローカルパス差し込み等)
entry = str(Path(__file__).resolve().with_name('groot_entry.py'))

# ---------------------------------------------------------------------------
# 単一 GPU 用の基本コマンド。先頭 3 要素 ['uv','run','python'] は
# 複数 GPU のとき torchrun に置き換える (下の cmd[3:] を参照)。
# --base-model-path は後で revision 固定のローカルパスに差し替える。
# ---------------------------------------------------------------------------
cmd = [
    'uv', 'run', 'python', entry,
    '--base-model-path', 'nvidia/GR00T-N1.7-3B',
    '--dataset-path', a.dataset,
    '--embodiment-tag', 'NEW_EMBODIMENT',
    '--modality-config-path', a.modality,
    '--num-gpus', str(a.gpus),
    '--output-dir', str(out),
    '--max-steps', str(a.steps),
    '--save-steps', str(min(50, a.steps)),
    '--save-total-limit', '3',
    '--global-batch-size', str(a.batch),
    '--dataloader-num-workers', '4',
]
if a.resume:
    cmd += ['--resume-from-checkpoint']

# ---------------------------------------------------------------------------
# 安全確認: マルチノードは audit 合格後に明示フラグを付けた場合のみ許可し、
# GPU 数は「ノード数 x 8」と一致している必要がある。
# ---------------------------------------------------------------------------
nnodes = int(os.environ.get('NNODES', '1'))
node_rank = int(os.environ.get('NODE_RANK', '0'))
if nnodes > 1 and (not a.multinode_verified or a.gpus != nnodes * 8):
    raise SystemExit('B05 requires --allow-multinode-smoke after audit and --gpus = NNODES*8')

# global batch は全 GPU に均等に割り振れなければならない
if a.batch % a.gpus:
    raise SystemExit('--batch must be divisible by total GPUs')

# ---------------------------------------------------------------------------
# 複数 GPU: torchrun で起動する (uv 環境内の torchrun を使う)
# ---------------------------------------------------------------------------
if a.gpus > 1:
    # 1 ノードあたりのプロセス数: マルチノードなら 8、単一ノードなら指定 GPU 数
    launch = ['uv', 'run', 'torchrun', '--nproc_per_node', str(8 if nnodes > 1 else a.gpus)]
    if nnodes == 1:
        # 単一ノード: rendezvous サーバ不要の standalone モード
        launch += ['--standalone']
    else:
        # マルチノード: c10d rendezvous で全ノードが MASTER_ADDR に集合する。
        # max_restarts=0 で、失敗時に勝手に再起動させず結果をそのまま記録する。
        master = os.environ['MASTER_ADDR']
        run_id = os.environ['RUN_ID']
        launch += [
            '--nnodes', str(nnodes),
            '--node_rank', str(node_rank),
            '--rdzv_backend', 'c10d',
            '--rdzv_endpoint', master + ':' + os.environ.get('MASTER_PORT', '29500'),
            '--rdzv_id', run_id,
            '--max_restarts', '0',
        ]
    # ['uv','run','python'] を torchrun 起動部分に置き換え、entry 以降はそのまま使う
    cmd = launch + cmd[3:]

if a.dry_run:
    print(json.dumps(cmd, indent=2))
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# revision 固定でモデルを取得する。lock ファイルの SHA を使うことで、
# 全ノードが同じ重みを使い、後から同じ条件で再現できる。
# (huggingface_hub は dry-run では不要なので、ここで遅延 import する)
# ---------------------------------------------------------------------------
from huggingface_hub import snapshot_download
lock = json.loads((Path(__file__).resolve().parents[1] / 'configs/revisions.lock.json').read_text())
base = snapshot_download('nvidia/GR00T-N1.7-3B', revision=lock['nvidia/GR00T-N1.7-3B'])
cosmos = snapshot_download('nvidia/Cosmos-Reason2-2B', revision=lock['nvidia/Cosmos-Reason2-2B'])

# --base-model-path の値を、ダウンロード済みのローカルパスに差し替える
cmd[cmd.index('--base-model-path') + 1] = base

# 実際に使った upstream のコミットを記録する
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()

# upstream の CLI が想定どおりのフラグを持っているか --help の出力で確認する。
# (コミットによってフラグ名が変わることがあるため、黙って誤動作させない)
helptext = subprocess.check_output(['uv', 'run', 'python', entry, '--help'], cwd=repo, text=True)
for flag in (['--base-model-path', '--dataset-path', '--num-gpus', '--max-steps']
             + (['--resume-from-checkpoint'] if a.resume else [])):
    if flag not in helptext:
        raise SystemExit(
            f'Pinned upstream does not expose {flag}; consult its config before running'
        )

(out / f'launch_node{node_rank}.json').write_text(
    json.dumps({'commit': commit, 'cmd': cmd}, indent=2)
)

# ---------------------------------------------------------------------------
# 実行環境: Cosmos はローカルパスから読み (groot_entry.py が参照)、
# W&B は無効化、1 ノード内で使う GPU を最大 8 枚に限定する。
# ---------------------------------------------------------------------------
env = os.environ.copy()
env['COSMOS_LOCAL_PATH'] = cosmos
env['WANDB_MODE'] = 'disabled'
env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(min(8, a.gpus))))

# 学習を実行し、wall 時間と終了状態を記録する
t = time.time()
with (out / f'train_node{node_rank}.log').open('w') as f:
    r = subprocess.run(cmd, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT)
(out / f'wall_node{node_rank}.json').write_text(json.dumps({
    'elapsed_s': time.time() - t,
    'returncode': r.returncode,
    'gpus': a.gpus,
    'status': 'ok' if r.returncode == 0 else 'error',
    'runtime_multinode_verified': False,
}))
raise SystemExit(r.returncode)
