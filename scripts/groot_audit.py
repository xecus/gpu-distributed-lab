"""固定 (pin) した GR00T ソースが torchrun に対応しているかを静的に検査するスクリプト。

目的:
    マルチノード run を始める前に、upstream の GR00T コードが
    torchrun から起動される前提で書かれているかをソースの文字列検索で確認する。

合格の意味:
    experiment.py が WORLD_SIZE / LOCAL_RANK を読み、init_process_group を呼び、
    GPU 数と world size の検証関数と Gr00tTrainer を含み、launch_finetune.py が
    自前でワーカープロセスを spawn しない (= torchrun と二重起動にならない)、
    という文字列レベルの条件を満たすことだけ。
    実行時の動作検証ではない (runtime_verified は常に False)。

入力:
    --repo  GR00T リポジトリのパス
    --out   レポート JSON の出力先

出力:
    レポート JSON。全チェック合格なら終了コード 0、そうでなければ 2。
"""
import argparse, json, re, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--repo', default='third_party/Isaac-GR00T')
p.add_argument('--out', default='runs/groot_audit.json')
a = p.parse_args()

r = Path(a.repo)
entry = (r / 'gr00t/experiment/launch_finetune.py').read_text()   # 起動スクリプト
experiment = (r / 'gr00t/experiment/experiment.py').read_text()    # 学習本体

# 各チェックは「ソースに特定の文字列が含まれるか」だけを見る
checks = {
    # torchrun が設定する環境変数を読んでいるか
    'world_size_read': 'WORLD_SIZE' in experiment,
    'local_rank_read': 'LOCAL_RANK' in experiment,
    # torch.distributed の初期化をしているか
    'process_group_init': 'init_process_group' in experiment,
    # 指定 GPU 数と world size の不一致を検出する仕組みがあるか
    'world_size_validation': '_assert_num_gpus_matches_world_size' in experiment,
    # 起動スクリプト内で独自に子プロセスを spawn していないか (torchrun と二重になる)
    'no_nested_spawn': not bool(
        re.search(r'mp\.spawn|multiprocessing\.spawn|subprocess\.Popen', entry)
    ),
    # 想定している Trainer クラスが使われているか
    'trainer': 'Gr00tTrainer' in experiment,
}

report = {
    # どのコミットを検査したかを記録 (pin した版と一致するか後で確認できる)
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=r, text=True).strip(),
    'checks': checks,
    'static_gate_passed': all(checks.values()),
    'runtime_verified': False,
    'next': '2 nodes x 8 GPUs, 2 updates + checkpoint + resume + sample-count check before 5 nodes',
}

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

# シェルスクリプトから合否を判定できるよう、終了コードで結果を返す
raise SystemExit(0 if all(checks.values()) else 2)
