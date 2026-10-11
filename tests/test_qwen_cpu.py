"""Qwen3 学習経路の CPU 統合テスト。

本物の 8B / 32B の重みは使わず、Qwen3 と同じアーキテクチャの「ごく小さなランダム初期化モデル」を
その場で作って lab.train に通す。Part B (B01-B03) と同じ HF モデル読み込み・記録の経路
(Hugging Face モデルの読み込み → トークン列 train.bin の読み込み → 学習 → CSV / メタ情報の保存。
並列化は single モード)
が壊れていないことを、GPU なしで数十秒で確認するのが目的。
"""
import unittest,tempfile,os,subprocess,sys
from pathlib import Path

import numpy as np
from transformers import Qwen3Config,Qwen3ForCausalLM

# リポジトリのルート (tests/ の 1 つ上)。サブプロセスの cwd に使う。
ROOT=Path(__file__).resolve().parents[1]


class QwenCPUIntegration(unittest.TestCase):

    def test_qwen_training_path_and_csv(self):
        """小さな Qwen3 を 4 ステップ学習し、メトリクス CSV とメタ情報が
        正しく出ることを確認する。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            model=tmp/'model'
            data=tmp/'tokens.bin'
            out=tmp/'run'

            # Qwen3 と同じ構造 (GQA: KV ヘッド数 < クエリヘッド数) を保ったまま、
            # 隠れ次元 32・2 層まで縮めたモデルを作り、HF 形式で保存する。
            m=Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=256,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    max_position_embeddings=128,
                    use_sliding_window=False,
                )
            )
            m.save_pretrained(model)

            # 学習データ: 0..255 を繰り返す 4096 トークンを uint32 の生バイナリで保存する。
            # (本番の prepared train.bin と同じ形式)
            (np.arange(4096,dtype=np.uint32)%256).tofile(data)

            # lab.train を CPU・1 プロセスで実行する。--save-every 2 なので step_2, step_4 を保存。
            cmd=[
                sys.executable,'-m','lab.train','--cpu',
                '--model',str(model),'--tokens',str(data),'--seq','16',
                '--batch','2','--steps','4','--warmup','0','--save-every','2',
                '--out',str(out),'--sampler',
            ]
            subprocess.run(
                cmd,
                cwd=ROOT,
                env={**os.environ,'OMP_NUM_THREADS':'1'},
                capture_output=True,
                timeout=60,
                check=True,
            )

            import csv,json

            # メトリクス CSV: 1 ステップ 1 行で 4 行あり、loss がすべて正の値であること。
            # (loss が 0 や NaN なら、ラベルのずれや数値不安定を疑う)
            with (out/'metrics_rank0_from0.csv').open() as file:
                rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),4)
            self.assertTrue(all(float(r['loss'])>0 for r in rows))

            # チェックポイントのメタ情報に、学習データの SHA-256 (16 進 64 文字) が
            # 記録されていること。
            # 再開時に「同じデータで続きを学習しているか」を検証するために使われる。
            meta=json.loads((out/'step_4/meta_rank0.json').read_text())
            self.assertEqual(len(meta['data_sha256']),64)


if __name__=='__main__':
    unittest.main()
