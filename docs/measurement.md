# 計測・CSV・チェックポイント

## 計測契約

同じモデル・dtype・sequence・global batch・学習率・更新数・checkpoint設定で比較する。
Strong scalingはglobal batch固定、Weak scalingはmicro batchとaccum固定。
最初は100 total updates、最初10 updatesをwarmupとしてCSVから除外。
このwarmupはGPU/optimizerの立ち上がりを除くためであり、LRスケジューラのwarmupではない。LRは一定。
性能計測用runでは`--save-every 0`で中間保存を外し、最終保存のwall費用は別に計測する。
精度評価用にはチェックポイントを残し、同一held-out windowsで比較する。

学習CSVは `metrics_rank{r}_from{resume_step}.csv`。

| column | 定義 |
|---|---|
| step | optimizer更新数、accum後の1step |
| global_batch | DPならb×a×world、純TPならb×a |
| loss | 同じサイズのDP rank平均のmicro loss平均 |
| step_s | batch生成・転送・forward/backward/optimizerを含む同期wall時間のrank最大値 |
| tokens_s | global_batch×sequence/step_s |
| allocated_gib | 当stepのPyTorch peak allocation、rank別 |
| reserved_gib | allocator予約ピーク、rank別 |

barrierとstats reduction、CSV書込、checkpointはstep_sに入らない。
このため最適化されたproductionスループットより慎重な、同期付きの学習実験値となる。
プロファイル用の余分な更新はmetricsのstep数に含めず、最後の保存後に実施する。
profilerを有効にしたrunと通常runの速度を同一条件の値として混ぜない。
allocatedとreservedはnvidia-smi memory.usedと異なる。NCCL/GPU runtime等はPyTorch外にメモリを使う。

```bash
bash scripts/monitor.sh runs/A07/gpu.csv &
MONITOR_PID=$!
NPROC=8 bash scripts/part_a.sh A07 --steps 100 --profile
kill "$MONITOR_PID"
python scripts/summary.py runs/A07 --node-hourly-usd REVIEWED_PRICE --nodes 1 --wall-seconds ACTUAL_SECONDS
python scripts/trace_stats.py runs/A07/trace_rank0.json --out runs/A07/communication.csv
```

`summary.py`はrank0時間（全rank最大を記録）とrank0メモリを出す。
クラスタpeak VRAMは**全rankのallocated_gibの最大**を取る。rank0だけでノード全体を代表させない。
`trace_stats.py`はNCCL CUDA kernel時間の区間和集合を出す。通信と演算が重なるため、step timeから単純減算できない。
traceをChrome/Perfettoに読込み、NCCL stream、GEMM、idle gap、H2D、CPU queueを調べる。
NCCL kernelに見えないplugin通信や待機もある。通信の露出時間はタイムラインで判定する。

## MFUとスケーリング

LLMの簡易演算見積 `F≈6P×tokens/s`、`MFU≈F/(GPU数×BF16 dense peak FLOPs/s)`。
使ったGPUのdense・非sparseピーク値を確認して入力し、FP8やsparseピークを使わない。
self-attentionのsequence依存項、再計算、optimizer等を省いた近似。長いsequence、VLA、TP複製部分にはそのまま使えない。
VLAは画像encoderやaction flowの演算があるためLLM式のMFUを報告せずsamples/sとaction metricsを優先。
TPのtokens/sは複製サンプルをworld倍にしない。

```bash
python scripts/scaling.py --runs runs/scale8 runs/scale16 runs/scale40 --gpus 8 16 40 \
 --kind strong --params ACTUAL_PARAMETER_COUNT --peak-bf16-tflops-per-gpu VERIFIED_DENSE_PEAK
```

小さいtoyモデルで8GPUが1GPUより遅くても異常とは限らない。通信latencyと小GEMMが支配する。
再現性は同じ条件を3回実行しmedian/p95/範囲を報告する。初期探索の失敗runを除去せずstatusへ残す。

## 結果一覧

`results/experiment_results.csv`は19実験の条件＋結果の空表。未実行はstatus=not_run。
`run_record.py`はログとstatus_nodeN.csvを保存、OOM/timeout/errorを別に扱う。
GPU上で未測定の値は空欄のまま。実測CSVを生成してから比較表へ取り込む。

```bash
python scripts/run_record.py --id A10 --out runs/A10-guarded --gpus 8 --nodes 1 \
 --hourly-usd REVIEWED_PRICE --timeout 600 -- \
 bash scripts/part_a.sh A10 --steps 10 --warmup 2 --out runs/A10-guarded
```

OOMの場合はGPUのモデル読込、forward、backward、Adam初期化のどの段階で起きたかログから判定。
host RAM OOM/killedはGPU OOMと断定せずstatus=error、dmesg/OS memoryも確認。
タイムアウトで一方のnodeだけが終了した場合は残りのnodeのプロセスも終了する。

## チェックポイント：保存・再開

`step_N`に各rankのmetadataとstateを保存。データSHA256の一致を復元時に確認し、同じpathのデータ差し替えを拒否する。
標準は直近2世代を保持（`--keep-checkpoints 2`）。全rank保存完了後に古いstep directoryだけを除去する。
保存中は旧2世代＋新世代が一時的に共存するため、32BではローカルNVMe/共有filesystemへの保存か容量増加・保存頻度をレビューする。single/manual/DDPはnative state＋optimizer＋RNG。
TPはDTensor shard、FSDPはLOCAL_STATE_DICTとローカルoptimizerで、**同じmode/world/layout/PyTorch版**にだけ復元する。
DeepSpeedはengine.save_checkpoint/load_checkpoint、全rankで呼び出す。
ZeRO-3は保存したengineへそのまま再読込せず、新プロセスでmodel/engineを再構築してloadする。

```bash
NPROC=8 bash scripts/part_a.sh A07 --steps 50 --save-every 25 --out runs/resume-demo
NPROC=8 bash scripts/part_a.sh A07 --steps 100 --save-every 25 --out runs/resume-demo \
 --resume runs/resume-demo/step_50
```

モデル/seed/batch/accum/sequence/tokens/revision/Sampler設定は一致させる。
`--steps`は再開後の追加数ではなく合計更新数。AMPの非決定性でGPU再開がbit一致する保証はない。
CPU単一GPU相当の検証では中断なしと中断再開の重みがbit一致することをチェックする。

保存中のディレクトリを復元対象にしない。all-rank barrier完了とshard個数を確認する。
本教材は同じtopologyの教育用checkpoint。worldサイズ変更のelastic reshardは実装していない。
FSDP/TP shardはtorch.load時にprocess groupが必要。pickleは自身の信頼できるcheckpointだけを読込む。

multi-nodeでlocal diskへ保存する場合、全nodeの`step_N`を同じdirへ統合し、同じrank shardを元のnodeへ戻す。
ZeRO metadataを含むすべてのファイルが揃ったことを確かめる。共有FSなら全node同じパスを使う。
checkpoint容量はoptimizer state込みでは重みだけの何倍も大きい。32B exportではCPU RAMとdiskにも余裕が必要。

## Qwenの評価用export

```bash
# ZeRO checkpointをCPUでまとめてHF形式へ。32BはFP32 stateで128GB以上＋モデル構築分のRAMが必要。
python -m lab.export --model Qwen/Qwen3-8B --revision "$MODEL_REVISION" \
 --checkpoint runs/B01/zero2_1node/step_100 --zero --out runs/qwen8-export
python -m lab.evaluate --model runs/qwen8-export --tokens data/qwen/val.bin --out runs/qwen8-eval.json
```

DDP/nativeは`--zero`なし。LoRAを追加実験する場合は`--lora`でexportし`evaluate --adapter PATH`。
評価コードは1GPUでmodel weightsのみ読込み、固定token windowsのloss/perplexityを計測。
32BのBF16重みだけでも約64GBで、評価sequenceのactivationがさらに必要。短いsequenceで確認。
FSDPの評価は全rankでload→FULL_STATE_DICT offload/rank0_onlyでgather→HF saveが必要。
`lab.fsdp_export`がその手順を実装。shard directoryのrank0.ptだけを`lab.export`へ渡してはいけない。

## GR00Tの計測

`groot_run.py`は公式Trainerの学習ログ、node別wall、commit/commandを保存。
`groot_csv.py RUN_DIR`はtrainer_state.jsonのlog_historyをCSV化。
loggerがlog_historyにstep timeを出さない場合は空欄を保ち、wall/optimizer更新数から平均を別途出す。
per-stepの性能解析は公式`experiment.py`のenable_profiling設定を有効にした短い別runで行う。
Action MAE/MSEは非正規化単位と各次元の単位を明記する。違うembodimentの値を混ぜない。
Open-loopの低誤差はclosed-loopの成功率を保証しない。

FSDPのHF export例：
```bash
NPROC=8 bash scripts/launch.sh lab.fsdp_export --checkpoint runs/B03/fsdp-8/step_10 --out runs/qwen32-fsdp-export
```

結果集約：`python scripts/collate.py --runs runs --out results/measured_results.csv`。
学習、GEMM、collective、GR00TのCSVを読み、通信/MFU/スケーリング/請求は各解析結果を添付する。
異なるタスクの指標は空欄を保ち、VLAのsamples/sをLLMのtokens/sへ変換しない。
