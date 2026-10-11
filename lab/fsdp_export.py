"""全 rank 分の FSDP チェックポイントを Hugging Face 形式のディレクトリに書き出す。

B03 など実モデル (Hugging Face) を FSDP で学習したチェックポイント用
(toy モデルの A11 チェックポイントは lab.train の --export-hf が拒否する)。
FSDP では重みが全 rank に分割 (shard) されているので、torchrun で全 rank を起動して実行する。
学習時と同じ rank 数で起動すること (各 rank が自分の meta_rank{rank}.json とシャードを読むため)。
このスクリプト自体は変換処理を持たず、チェックポイントに保存された学習時の引数を読み出して
lab.train を「--mode fsdp --resume ... --export-hf ...」付きで呼び直すだけの薄いラッパー。
(学習と同じモデル構成・同じ FSDP の包み方で復元しないと state_dict が一致しないため)
"""
import argparse, json, sys, runpy, os
from pathlib import Path

# ---- コマンドライン引数 ----
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()

# 自分の rank のメタデータから、学習時のコマンドライン引数 (args) を取り出す
rank = int(os.environ.get('RANK', 0))
meta = json.loads((Path(a.checkpoint) / f'meta_rank{rank}.json').read_text())
old = meta['args']

# lab.train に渡す引数を組み立てる
cmd = [
    'lab.train',
    '--mode', 'fsdp',
    '--resume', a.checkpoint,
    '--export-hf', a.out,
    '--out', str(Path(a.out).parent / 'export-state'),
]

# 値を持つ引数: 学習時に指定されていたものだけを引き継ぐ (例: seq -> --seq 256)
for key in [
    'model', 'revision', 'tokens', 'batch', 'accum', 'seq',
    'dim', 'layers', 'heads', 'seed', 'lr',
]:
    if old.get(key) is not None:
        cmd += ['--' + key.replace('_', '-'), str(old[key])]

# フラグ引数 (store_true): True だったものだけを付ける
for key in ['sampler', 'activation_checkpointing', 'lora']:
    if old.get(key):
        cmd += ['--' + key.replace('_', '-')]

# sys.argv を差し替えて、lab.train を `python -m lab.train ...` と同じように実行する
sys.argv = cmd
runpy.run_module('lab.train', run_name='__main__')
