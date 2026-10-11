"""A03: 1/2/4/8 GPU のスケーリング・スイープを一括実行するスクリプト。

目的:
    GPU 数を 1 -> 2 -> 4 -> 8 と増やしながら同じ学習を走らせ、
    strong scaling と weak scaling の両方を測定する (レッスン A03)。

    - strong scaling: global batch を固定し、GPU を増やすほど 1 GPU あたりの
      仕事量 (accum) を減らす。理想的には GPU 数に比例して step 時間が短くなる。
    - weak scaling: 1 GPU あたりの batch (accum=2) を固定し、GPU を増やすほど
      global batch が大きくなる。理想的には step 時間が一定のまま tokens/s が伸びる。

入力:
    --steps         各 run の学習ステップ数
    --global-batch  strong scaling で固定する global batch

出力:
    runs/A03/{strong|weak}_{N}gpu/ 以下に、run_record.py が記録する
    status_node*.csv / command_node*.log と、lab.train のメトリクス CSV。
"""
import argparse, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, default=40)
p.add_argument('--global-batch', type=int, default=16)
a = p.parse_args()

for gpus in [1, 2, 4, 8]:
    for kind in ['strong', 'weak']:
        # strong: micro batch=1 なので、global batch / GPU 数 = 1 GPU あたりの勾配累積回数
        # weak  : 1 GPU あたり accum=2 で固定 -> GPU を増やすと global batch が増える
        accum = a.global_batch // gpus if kind == 'strong' else 2

        # strong scaling では global batch が GPU 数で割り切れないと
        # 「同じ global batch」の比較にならないためエラーにする
        if accum < 1 or (kind == 'strong' and a.global_batch % gpus):
            raise ValueError('invalid global batch')

        out = f'runs/A03/{kind}_{gpus}gpu'

        # run_record.py で包むことで、OOM/timeout も「結果データ」として記録される。
        # '--' 以降が実際に実行される学習コマンド (launch.sh 経由で torchrun 起動)。
        cmd = [
            sys.executable, 'scripts/run_record.py',
            '--id', 'A03', '--out', out, '--gpus', str(gpus), '--',
            'bash', 'scripts/launch.sh', 'lab.train',
            '--mode', 'single' if gpus == 1 else 'ddp',
            '--batch', '1',
            '--accum', str(accum),
            '--steps', str(a.steps),
            '--out', out,
            '--sampler',
        ]

        # NPROC 環境変数で launch.sh に起動プロセス数 (=GPU 数) を伝える。
        # check=False: 1 つの構成が失敗してもスイープ全体は続行する。
        subprocess.run(cmd, env={**os.environ, 'NPROC': str(gpus)}, check=False)
