# 検証状況：2026-10-10

## 実行済みの検証

- Pythonの全ファイルのcompileall、全shell scriptおよび教材のbash command blockの構文確認。
- 全19教材の6項目、JSON、比較CSVの19行/column数、Markdown相対リンク確認。
- PyTorch 2.7.1 CPUで、因果マスク、全パラメータgradientの生成、rank平均勾配とglobal batch勾配の数値一致。
- 単一プロセス学習の中断なし/チェックポイント再開がbit一致。データSHA256をmetadataへ保存。
- paid-resource approvalがないとAWS起動scriptが拒否することと、plan.pyが1nodeのJSONしか作らず、インスタンス種別ごとのEFA数（p5:32、p6-b200:8）になること。
- 小さいrandom初期化Qwen3 architectureのCPU読込→token memmap→loss/backward/AdamW→checkpoint→CSVの統合確認。
- 固定GR00T commitの公式sourceを読み、WORLD_SIZE/LOCAL_RANK、NCCL process group、num_gpus/world consistency、resume bool、nested spawn不在を静的確認。

テスト結果：7 tests、6 passed、1 skipped。結果logは`results/cpu_validation.log`。
CPUのQwen testは8B/32Bの公開weightの学習ではなく、実モデル用コード経路を同architectureのtiny modelで検証したもの。

## 未実行のこと

- AWS EC2/EBS/FSx/NAT等の有料リソース起動。
- H100/H200/B200上でのCUDA BF16、NCCL/OFI/EFA、DeepSpeed ZeRO、FSDP、TPの実行。
- Qwen3-8B/32B公開weightを使うFull Fine-tuning。
- GR00T weightの取得/学習/Action評価、および16〜40GPU runtime verification。

ローカル環境はGloo TCP socketを許可しておらず、manual vs DDPの2processテストは明示的にskip。
最初の試行でhostname rendezvous待ち、その後localhost static rendezvousではGlooのsocketがOperation not permittedとなった。
分散実行の成功と誤って報告せず、AWSホストで再実行するテストを残した。

## GPUノード上での受け入れ順序

1. preflight：GPU8、CUDA版、pip check、EFA provider、OFI plugin path。
2. `scripts/acceptance.sh`：toy single/manual/DDP/ZeRO/FSDP/TP、collectives/ringの数値確認、保存再開、TP forward/gradient/更新のparity。
3. 2nodeでA04/A13の短い試験、EFA log確認、checkpoint回収・再開。
4. Qwen8B、Qwen32Bを各2〜10updateで確認してから、100updateの正式比較。
5. GR00T 8GPUのfit/eval、static audit後に16GPUで2→4step resume、data shard/global batch確認。
6. 追加承認の後に40GPU。

`requirements.txt`は固定候補ベースライン。GPU/DLAMI上で受け入れ完了したstackとは言わない。
速度・メモリ・コストの期待結果は理論的な仮説であり、GPUの実測値は空欄。
