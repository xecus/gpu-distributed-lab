"""実験前のホスト検査 (preflight) スクリプト。

目的:
    高価な GPU インスタンスで実験を始める前に、環境が整っているかをまとめて確認する。
    - GPU 枚数、Python / torch / CUDA のバージョン
    - GPU 間トポロジ (nvidia-smi topo -m: NVLink/PCIe の接続関係)
    - EFA プロバイダ (fi_info -p efa: AWS のノード間高速ネットワーク)
    - ディスク空き容量、pip の依存関係の整合性

入力:
    --require-gpus  必要な GPU 枚数 (既定 8)

出力:
    runs/preflight.json に全結果を保存。GPU 数が --require-gpus 未満、
    または PATH 上の python で実行した pip check が非ゼロ (未インストールの 127 を含む)
    なら非ゼロで終了する。pip check は sys.executable ではなく PATH 上の python で
    実行されるので、別の環境を検査している可能性がある点に注意。
"""
import argparse, json, os, platform, subprocess, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--require-gpus', type=int, default=8)
a = p.parse_args()

# torch の import は重いので、引数解析 (--help など) の後で行う
import torch

# ---------------------------------------------------------------------------
# Python 側から取れる情報
# ---------------------------------------------------------------------------
checks = {}
checks['python'] = sys.version
checks['torch'] = torch.__version__
checks['cuda'] = torch.version.cuda
checks['host'] = platform.node()
checks['gpu_count'] = torch.cuda.device_count()

# ---------------------------------------------------------------------------
# 外部コマンドの結果 (終了コードと出力) をそのまま記録する。
# コマンドが無い環境 (例: EFA 非搭載) でも止めずに 127 として記録する。
# ---------------------------------------------------------------------------
for command in [['nvidia-smi', '-L'],
                ['nvidia-smi', 'topo', '-m'],
                ['fi_info', '-p', 'efa'],
                ['df', '-h'],
                ['python', '-m', 'pip', 'check']]:
    try:
        r = subprocess.run(command, text=True, capture_output=True)
        checks[' '.join(command)] = {'code': r.returncode, 'output': r.stdout + r.stderr}
    except FileNotFoundError:
        checks[' '.join(command)] = {'code': 127, 'output': 'not installed / PATH missing'}

Path('runs').mkdir(exist_ok=True)
Path('runs/preflight.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))

# 致命的な問題だけ終了コードで知らせる (他の項目は JSON を見て人が判断する)
if checks['gpu_count'] < a.require_gpus:
    sys.exit('GPU count too small')
if checks['python -m pip check']['code'] != 0:
    sys.exit('dependency conflicts')
