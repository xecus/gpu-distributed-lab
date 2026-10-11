# 公開モデル・データを用意する

## Qwen用の小さい公開corpus

製造・物流の機密テキストがなくても、公開WikiTextでシステム実験を開始できる。
WikiTextはこの用途の代表的な小規模corpusで、製造・物流への適応性能を評価するデータではない。
元のtrain splitを入力に文単位hash splitでheld-outを作り、本文の重複は同じsplitへ入る。
正式な精度実験では元データの公式validation/testを保つ、document単位・時間・domainの分離へ拡張する。

```bash
source .venv/bin/activate
export DATA_REVISION=$(python -c "import json;print(json.load(open('configs/revisions.lock.json'))['Salesforce/wikitext'])")
python scripts/download_corpus.py --revision "$DATA_REVISION"
export MODEL_REVISION=$(python -c "import json;print(json.load(open('configs/revisions.lock.json'))['Qwen/Qwen3-8B'])")
python -m lab.prepare_data --input data/wiki.jsonl --model Qwen/Qwen3-8B \
 --revision "$MODEL_REVISION" --out data/qwen
cat data/qwen/metadata.json
sha256sum data/qwen/*.bin
python scripts/download_models.py --models Qwen/Qwen3-8B
```

8B/32Bはtokenizer語彙の一致を両revisionのtokenizer/configで確認して同じtoken memmapを使う。
一致しなければ32Bで再tokenizeする。研究用runではデータrevision、tokenizer revision、SHA256を記録。
JSONLのスキーマは`{"text":"..."}`。サンプル例は`data/example.jsonl.example`だが、2行だけでLLM fine-tuningを行うためのcorpusではない。
準備したtrain.binとval.binを各ノード同じ絶対パスへコピーする。

正式なQwen比較はB01〜B03。追加でLoRAを試す場合：
```bash
NPROC=1 bash scripts/launch.sh lab.train --mode single --model Qwen/Qwen3-8B \
 --revision "$MODEL_REVISION" --tokens data/qwen/train.bin --lora --batch 1 --accum 16 \
 --activation-checkpointing --out runs/extra-lora
```

LoRAは主実験のFull Fine-tuningと区別する。低い更新パラメータ数でメモリが減るため、直接同じ土俵のFull FTの比較にはしない。
この教材のQwenはdense Qwen3-8B/32Bに固定。Qwen3-VLやMoEモデルへ無条件に差し替えない。

## GR00Tの準備

Ubuntu22.04/24.04等でffmpeg4〜7、git-lfs、uv、Python3.12、CUDA12.8、対応driverを用意。
```bash
# すでに承認済みのホストでOS packagesを用意。
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs
python -m pip install uv
mkdir -p runs
export GROOT_COMMIT=$(python -c "import json;print(json.load(open('configs/revisions.lock.json'))['Isaac-GR00T'])")
bash scripts/groot_setup.sh
# 使用権の確認後、必要なgated modelアクセスをHFで申請してログイン。
huggingface-cli login
```

GR00TとCosmos-Reason2-2Bのmodel pageの利用条件とアクセス権を確認する。
`groot_run.py`はlockのrevisionで両snapshotを取得し、base-model-pathとCOSMOS_LOCAL_PATHへ渡す。
`groot_entry.py`は公式CLIを起動し、Cosmosのmodel_nameを固定snapshotへ設定、logging_steps=1と任意profile設定だけを適用する薄いadapter。
学習器・model forward・dataset実装は公式コードを使う。公式コードをコピーして別物のVLAを実装したふりはしない。

デモは公式git-lfsの`demo_data/cube_to_bowl_5`。modalityは`examples/SO100/so100_config.py`。
最初に5episode上のfit確認。その後、独立session/episode/taskをheld-outへ分離した自社データを使う。
自社データは公式data preparation guideに従いmeta/modality.json、stats、episode index、video/parquetを検証。
カメラ名、画像時刻、state/action次元、関節順、単位、horizon16、gripper表現を合わせる。
実機を動かす工程はこの教材には含まれない。

## Action評価を実行する

実験後のcheckpoint-100を例に、公式evaluatorへmodality登録した状態で渡す：
```bash
export LAB_ROOT=$PWD
export GROOT_REPO=$PWD/third_party/Isaac-GR00T
export GROOT_CHECKPOINT=$PWD/runs/B04/checkpoint-100
cd "$GROOT_REPO"
uv run python "$LAB_ROOT/scripts/groot_eval.py" --repo "$GROOT_REPO" \
 --modality "$GROOT_REPO/examples/SO100/so100_config.py" \
 --dataset-path "$GROOT_REPO/demo_data/cube_to_bowl_5" --embodiment-tag NEW_EMBODIMENT \
 --model-path "$GROOT_CHECKPOINT" --traj-ids 0 --execution-horizon 16 --steps 400 \
 --modality-keys single_arm gripper --save-plot-path "$LAB_ROOT/runs/B04/action-plots" \
 2>&1 | tee "$LAB_ROOT/runs/B04/open-loop-eval.log"
cd "$LAB_ROOT"
python scripts/groot_csv.py runs/B04
```

MAE=`mean(|a_pred−a_gt|)`、MSE=`mean((a_pred−a_gt)^2)`。
関節角/rad、EEF/m、rotation等、異なる単位を1つの平均へ潰すと解釈が難しい。dimension/group別の誤差も併記。
訓練trajectory0の結果はfitのsanity check。未学習episodeの誤差、実機タスク成功率、衝突/停止率とは区別する。

## B05のチェックリスト

`results/groot_static_audit.json`は公開時点の固定commitでsourceを確認した結果。runtime_multinode_verified=false。
16GPUの短いrunで以下をすべて検証してから40GPUへ進む。

- world=16、local=0..7、global rank=0..15。num_gpus=16。
- global batch32なら各GPU batch2、accum設定込みのeffective batchを確認。
- 全rankのデータshardが異なる。同じepisodeは境界/再sampling以外で意図なく全rankへ複製されない。
- NET/OFI/efaを全node logで確認。GPUの全rankが計算している。
- loss finite、checkpoint全shardが揃う、新しいプロセスから再開できる。
- 学習後のAction評価を同じhold-outで実行できる。

`--allow-multinode-smoke`は静的監査を確認して短い実験へ進むフラグ。実行検証済みを意味しない。
対応しないcommitではblockedと記録し、torchrunで無理に起動しない。
B05で使用する出力は全node共通の共有パスを推奨。ローカルvolumeのみならshard回収・配置・resumeを先に検証。
40GPUのStrong比較を16GPUと同じG80で行うならbatch80、accum1。8GPUのB04比較側もG80に合わせる。
24/32/40を全てG固定で比べるならG480等のworldで割り切れるbatchへ設定するが、初回には大きすぎるので1/2/5nodeから進む。
