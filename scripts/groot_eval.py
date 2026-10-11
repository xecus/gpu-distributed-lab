# upstream の uv 環境で実行する。upstream の評価 CLI を呼ぶ前に、独自の modality 設定を登録する。
#
# 目的:
#   GR00T の open-loop 評価 (gr00t/eval/open_loop_eval.py) は、データセットの
#   modality (どの関節・カメラを使うか) が事前に登録されている必要がある。
#   そこで --modality で指定した Python 設定ファイルを先に import して登録し、
#   残りの引数をそのまま公式評価スクリプトへ渡す。
#
# 入力:
#   --repo      GR00T リポジトリのパス
#   --modality  modality 設定ファイル (例: examples/SO100/so100_config.py)。
#               相対パスは --repo ではなく cwd 基準で解決される
#   それ以外の引数は open_loop_eval.py にそのまま渡る。
import argparse, runpy, sys, importlib.util
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--modality', required=True)

# parse_known_args: 自分が知らない引数は rest に残して評価スクリプトへ転送する
a, rest = p.parse_known_args()
repo = Path(a.repo).resolve()
sys.path.insert(0, str(repo))

# modality 設定ファイルをモジュールとして読み込む (import 時に登録処理が走る)
config = Path(a.modality).resolve()
spec = importlib.util.spec_from_file_location('custom_modality', config)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# 公式スクリプトから見て、自分が直接起動されたように sys.argv を差し替える
sys.argv = ['open_loop_eval.py'] + rest
runpy.run_path(str(repo / 'gr00t/eval/open_loop_eval.py'), run_name='__main__')
