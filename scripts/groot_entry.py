"""GR00T 公式 finetune を起動するための、バージョン限定の小さなアダプタ。

目的:
    upstream のコードを書き換えずに、以下の 2 点だけを差し込んでから
    公式の launch_finetune.py を実行する。
      - Cosmos (VLM バックボーン) を、revision 固定でダウンロード済みのローカルパスから読む
      - 環境変数 GROOT_PROFILE=1 のとき公式プロファイラを有効にする
    あわせて logging_steps=1 にし、groot_csv.py が全 step のログを拾えるようにする。

使い方:
    groot_run.py から、GR00T リポジトリを cwd として uv 環境内で実行される。
    コマンドライン引数はそのまま launch_finetune.py に渡る。
"""
import os, runpy, sys
from pathlib import Path

# cwd はコミット固定された公式 GR00T リポジトリ。そこを import パスの先頭に追加する。
sys.path.insert(0, str(Path.cwd()))
import gr00t.experiment.experiment as experiment

# 元の run 関数を保持しておき、ラッパーから呼び出す (モンキーパッチ)
original = experiment.run


def pinned_run(config):
    # groot_run.py が設定したローカルの Cosmos スナップショットを使う
    if os.environ.get('COSMOS_LOCAL_PATH'):
        config.model.model_name = os.environ['COSMOS_LOCAL_PATH']

    # 毎 step ログを出す (メトリクス CSV の粒度を揃えるため)
    config.training.logging_steps = 1

    if os.environ.get('GROOT_PROFILE') == '1':
        config.training.enable_profiling = True
    return original(config)


experiment.run = pinned_run

# 公式の起動スクリプトを __main__ として実行 (python launch_finetune.py ... と同等)
runpy.run_path(str(Path.cwd() / 'gr00t/experiment/launch_finetune.py'), run_name='__main__')
