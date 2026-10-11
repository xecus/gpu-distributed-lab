"""分散学習ラボのパッケージ: モデル定義・学習ループ・通信実験をまとめたもの。

主なモジュール:
    model.py        自作の小さな因果Transformerと Tensor Parallel 化 (A02, A12)
    train.py        単一GPU / DDP / ZeRO / FSDP / TP を切り替えられる学習ループ
                    (A02, A03, A06〜A14, B01〜B03)
    cuda_basics.py  GPU情報の記録と FP32 / BF16 matmul の計測 (A01)
    collectives.py  NCCL集団通信のベンチマーク (A04)
    ring.py         P2P通信で自作した Ring AllReduce (A05)
    tp_check.py     TPモデルと非分割モデルの一致確認 (A12)
    prepare_data.py JSONL → 決定的な train/val 分割と uint32 トークン列 (B01〜B03)
    evaluate.py     検証用トークンでの loss / perplexity 評価 (Part B)
    export.py       単一GPU / DDP / ZeRO のチェックポイント → Hugging Face 形式 (Part B)
    fsdp_export.py  実モデルの FSDP 全 rank チェックポイント → Hugging Face 形式 (B03)
"""
