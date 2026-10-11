# GPU Distributed Lab — AWS 分散学習ハンズオン

AWSの8GPUノード（H100 / H200 / B200）を使い、分散学習を**手を動かして**理解するための教材です。
自作の小さなTransformerでDDP・ZeRO・FSDP・Tensor Parallelの仕組みを一つずつ確かめ、最後にQwen3-8B/32Bの全パラメータFine-tuningとGR00T N1.7（VLA）のFine-tuningまで進みます。

- **Part A（14実験）**：CUDA基礎 → NCCL通信 → 自作Ring AllReduce → DDP → ZeRO-1/2/3 → FSDP → TP → マルチノード → プロファイリング
- **Part B（5実験）**：実モデル（Qwen3-8B/32B、GR00T N1.7）での学習・スケーリング・評価
- 最大構成：8GPUノード × 5台 = 40GPU

> [!WARNING]
> **GPUでの検証は途中です。** H100×8の単一ノードで受け入れテスト（`scripts/acceptance.sh`）が通った段階で、EFA・マルチノード・実モデルweightでの学習はまだ実行していません。
> 期待結果はすべて理論上の仮説で、`results/experiment_results.csv`の実測欄は空です。
> 検証済みの範囲と未検証の範囲は[docs/validation.md](docs/validation.md)にまとめています。

> [!CAUTION]
> **AWSの料金に注意してください。** p5/p6インスタンスは1台で1時間あたり数十ドルかかり、1GPUしか使わない実験でもノード全体が課金されます。
> このリポジトリのスクリプトが勝手にインスタンスを起動することはありません（詳細は[安全設計](#安全設計)）。
> それでも、実験が終わったら必ず停止・削除し、請求額を確認してください。

## 対応ハードウェア

| EC2インスタンス | GPU | EFA network cards |
|---|---|---|
| p5.48xlarge（既定） | H100 80GB × 8 | 32 |
| p5e.48xlarge | H200 141GB × 8 | 32 |
| p5en.48xlarge | H200 141GB × 8 | 16 |
| p6-b200.48xlarge | B200 180GB × 8 | 8 |

学習コードは通常のPyTorch・NCCLだけで書いているので、特定のGPUに依存しません。
GPUの種類で変わるのは、起動プランのEFA数、OOMになる境界、MFU計算に使う理論性能、ノード単価です。
B200を使う場合はCUDA 12.8以上が必要です。詳しくは[docs/aws.md](docs/aws.md)を参照してください。

## ロードマップ

| Phase | 実験 | 学ぶこと |
|---|---|---|
| 1 | A01〜A03 | CUDA、Transformer、Batch・Accumulation |
| 2 | A04〜A07 | NCCL、Ring、手動Data Parallel、DDP |
| 3 | A08〜A12 | ZeRO-1/2/3、FSDP、Attention/MLPのTP |
| 4 | A13〜A14 | EFA、最大5ノード、Profiler・MFU |
| 5 | B01〜B05 | 実際のLLM/VLAの学習・評価 |

### Part A：14実験

| ID | 実験 | GPU | 教材 |
|---|---|---|---|
| A01 | GPUアーキテクチャ・CUDA基礎 | 1 | [A01](docs/lessons/A01.md) |
| A02 | 自作Transformerの単一GPU学習 | 1 | [A02](docs/lessons/A02.md) |
| A03 | GPU数とBatch Size | 1〜8 | [A03](docs/lessons/A03.md) |
| A04 | NCCL通信ベンチマーク | 2〜8 | [A04](docs/lessons/A04.md) |
| A05 | 自作Ring AllReduce | 2〜8 | [A05](docs/lessons/A05.md) |
| A06 | 手動Data Parallel | 2〜8 | [A06](docs/lessons/A06.md) |
| A07 | DDP＋DistributedSampler | 2〜8 | [A07](docs/lessons/A07.md) |
| A08 | ZeRO-1 | 8 | [A08](docs/lessons/A08.md) |
| A09 | ZeRO-2 | 8 | [A09](docs/lessons/A09.md) |
| A10 | ZeRO-3 | 8 | [A10](docs/lessons/A10.md) |
| A11 | FSDP | 8 | [A11](docs/lessons/A11.md) |
| A12 | Attention・MLP Tensor Parallel | 2〜8 | [A12](docs/lessons/A12.md) |
| A13 | マルチノード | 8〜40 | [A13](docs/lessons/A13.md) |
| A14 | 性能解析 | 1〜40 | [A14](docs/lessons/A14.md) |

### Part B：5実験

| ID | モデル・比較 | GPU | 教材 |
|---|---|---|---|
| B01 | Qwen3-8B：DDP／ZeRO-1/2/3 全パラメータ学習 | 8 | [B01](docs/lessons/B01.md) |
| B02 | Qwen3-8B：Strong／Weak Scaling | 8／16 | [B02](docs/lessons/B02.md) |
| B03 | Qwen3-32B：ZeRO-3／FSDP | 8〜40 | [B03](docs/lessons/B03.md) |
| B04 | GR00T N1.7：単一ノード微調整・Action評価 | 8 | [B04](docs/lessons/B04.md) |
| B05 | GR00T N1.7：対応確認・マルチノード検証 | 16〜40 | [B05](docs/lessons/B05.md) |

各教材は「理論 → コード → 実行コマンド → 測定項目 → 期待結果 → 考察課題」の6節構成です。

## はじめかた

### 1. GPUなしで試す（ローカル）

PyTorch 2.7.1（CPU版で可）が入ったPython 3.10以上の環境で実行します。

```bash
git clone <this-repo> gpu-distributed-lab && cd gpu-distributed-lab
python -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 -r requirements.txt && pip install -e . --no-deps
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

因果マスク、勾配平均の数値一致、チェックポイント再開のbit一致、AWS起動ガードなどを確認します。
2プロセス通信のテストは、ローカルTCPソケットが使えない環境では`LAB_TEST_NO_SOCKETS=1`でskipできます。

### 2. GPUノードで実行する

まず[docs/aws.md](docs/aws.md)の手順でノードを用意します（IAM、EFA、セキュリティグループ、DLAMI）。
`setup.sh`はデフォルトで`python3.12`を使います（`PYTHON=python3.11 bash scripts/setup.sh`のように変更可）。
ノードに次のaptパッケージが入っていることを確認してください。

| パッケージ | 用途 |
|---|---|
| `python3.12` | venvのベースになるPython |
| `python3.12-venv` | `python3.12 -m venv`に必要 |
| `python3.12-dev` | `Python.h`。DeepSpeed import時にTritonがCヘルパーをJITコンパイルするため必須 |
| `build-essential` | Tritonのコンパイルに使う`gcc` |
| `gcc-14` `g++-14` | システムのgccが15以上の場合（Ubuntu 26.04など）。CUDA 12.8のnvccはGCC 14までしか対応しないため、DeepSpeed FusedAdamのJITビルドで使う |

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential gcc-14 g++-14
```

Ubuntu 24.04以降のDLAMIなら標準リポジトリで入ります。22.04ベースでは`python3.12`が標準にないため、承認済みの手順で追加してください。

#### CUDA 12.8 Toolkit

DeepSpeedはimport時に`nvcc`があるかを確認し、初回にFusedAdamをJITビルドします。ビルド時にはnvccとPyTorch（cu128）のCUDAメジャー版が一致している必要があるため、CUDA 13のnvccでは動きません。
DLAMIに`/usr/local/cuda-12.8`がない場合（デフォルトがCUDA 13のDLAMIなど）は、NVIDIAのrunfileでToolkitだけを入れます。

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
# Ubuntu 26.04などlibxml2.so.2がない（libxml2.so.16のみ）環境ではインストーラが起動しないため、
# インストール中だけ一時的にリンクを張る（ABIが異なるので終わったら必ず消す）
XML2=/usr/lib/x86_64-linux-gnu/libxml2.so.2
TMP_XML=
if [ ! -e "$XML2" ]; then sudo ln -s libxml2.so.16 "$XML2"; TMP_XML=1; fi
sudo sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --no-man-page --override
if [ -n "$TMP_XML" ]; then sudo rm "$XML2"; sudo ldconfig; fi
/usr/local/cuda-12.8/bin/nvcc --version   # release 12.8
nvidia-smi | head -4                      # ドライバの版が変わっていないこと
```

`--toolkit`のみを指定するとドライバには触れません（`--override`はgcc版チェックを飛ばすため）。

`launch.sh`は`CUDA_HOME`が未設定なら`/usr/local/cuda-12.8`を優先して使います。
またシステムのgccが15以上（Ubuntu 26.04など）なら、FusedAdamのJITビルドがnvcc 12.8の対応範囲（GCC 14まで）に収まるよう、次の環境変数を自動で設定します。

| 変数 | 効き先 |
|---|---|
| `CC=gcc-14` | Tritonのヘルパービルド、torch cpp_extension（nvccへ`-ccbin`として渡す） |
| `CXX=g++-14` | torch cpp_extensionの`.cpp`ファイルのコンパイル |
| `NVCC_PREPEND_FLAGS="-ccbin g++-14"` | nvcc本体が直接読む。torchの受け渡しに依存せずホストコンパイラを固定する |

`launch.sh`を通さずに`torchrun`や`python -m lab.train`を直接実行する場合は、同じ設定を手動で行ってください。

```bash
export CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS="-ccbin g++-14"
```

gcc 15のままビルドに失敗した場合は、残ったビルド結果を消してから再実行します（`rm -rf ~/.cache/torch_extensions/py312_cu128/fused_adam`）。

> **Ubuntu標準の`nvidia-cuda-toolkit`はaptで入れないでください。** Ubuntu版のNVIDIAユーザー空間ライブラリが入り、DLAMIのカーネルドライバと版がずれて`nvidia-smi`が`Driver/library version mismatch`になります。

そのあと:

```bash
bash scripts/setup.sh            # 前提条件の確認、venv作成、CUDA 12.8版PyTorch、preflight
source .venv/bin/activate
bash scripts/acceptance.sh       # 全モードの短い受け入れテスト（前回の出力とビルドキャッシュを消してから実行）
bash scripts/part_a.sh A01
bash scripts/part_a.sh A02 --steps 30 --warmup 5
NPROC=2 bash scripts/part_a.sh A04 --iters 10 --max-power 22
```

`acceptance.sh`は毎回`runs/acceptance/`、torch拡張のビルドキャッシュ（`~/.cache/torch_extensions`）、Tritonのキャッシュ（`~/.triton/cache`）を消してから始めるため、初回はFusedAdamのビルドに数分かかります。
キャッシュを残したい場合は`KEEP_CACHE=1 bash scripts/acceptance.sh`としてください（`runs/acceptance/`は常に消します）。

全19実験をまとめて自動実行する仕組みはあえて用意していません。
1つずつ短く試し、予算と結果を見ながら規模を広げてください。

## 読む順番

1. [検証状況](docs/validation.md)：検証済みの範囲と未検証の範囲を確認する
2. [AWS構築・EFA・運用](docs/aws.md)：インスタンス選択、ネットワーク、コスト、後片付け
3. [理論の共通部分](docs/theory.md)：データ並列・モデル並列とメモリ見積もり
4. [計測・チェックポイント](docs/measurement.md)：公平な比較条件
5. [公開モデルとデータ・VLA評価](docs/data_and_real_models.md)
6. 各実験の教材 `docs/lessons/A01.md`〜`B05.md`

## リポジトリ構成

```
gpu-distributed-lab/
├── README.md
├── LICENSE
├── pyproject.toml            パッケージ定義（lab/ を pip install -e . で使う）
├── requirements.txt          固定した依存パッケージ（torchは setup.sh が cu128 版を入れる）
│
├── lab/                      学習・通信の本体（Pythonパッケージ）
│   ├── model.py              Attention・MLPまで見える自作Transformer と TP適用（apply_tp）
│   ├── train.py              single/manual/DDP/ZeRO/FSDP/TP 共通の学習ループ・計測・保存・再開
│   ├── cuda_basics.py        A01：GPU情報とFP32/BF16 matmulの計測
│   ├── collectives.py        A04：AllReduce/AllGather/ReduceScatter のベンチマーク
│   ├── ring.py               A05：P2Pで書いたRing AllReduce（教育用）
│   ├── tp_check.py           A12：TPと非分割モデルのforward/勾配/更新の一致確認
│   ├── prepare_data.py       JSONL → 決定的な分割と uint32 token memmap
│   ├── evaluate.py           HFモデル・LoRA adapterのheld-out評価
│   ├── export.py             single/DDPのcheckpoint → HF形式
│   └── fsdp_export.py        FSDPの全rank checkpoint → HF形式（torchrunで実行）
│
├── scripts/                  実験の入口と補助ツール
│   ├── setup.sh              前提条件の確認、venv作成、依存インストール、preflight
│   ├── preflight.py          GPU数・CUDA/torch版・トポロジ・EFA・pip check の確認
│   ├── acceptance.sh         全モードの短い受け入れテスト（保存・再開、TP parity含む）
│   ├── launch.sh             torchrunの共通起動口（CUDA_HOME・gcc-14の設定、マルチノード対応）
│   ├── part_a.sh             Part A（A01〜A14）の入口
│   ├── part_b.sh             Part B（B01〜B05）の入口
│   ├── efa_env.sh            EFA/OFI用の環境変数（source して使う）
│   ├── monitor.sh            nvidia-smi を1秒ごとにCSVへ記録
│   ├── batch_sweep.py        A03：1/2/4/8 GPUのstrong/weak scaling sweep
│   ├── run_record.py         タイムアウト付きで実行し ok/OOM/error を記録
│   ├── summary.py            1 runのstep時間・tokens/s・メモリ・推定コストを集計
│   ├── scaling.py            複数runからscaling効率とMFUを計算
│   ├── trace_stats.py        profiler traceからNCCLカーネルの時間を集計
│   ├── collate.py            runs/ 以下を results/experiment_results.csv と同じ形式に集約
│   ├── download_models.py    固定commitのモデルを取得
│   ├── download_corpus.py    公開コーパスの取得
│   ├── resolve_revisions.py  モデル・上流リポジトリのcommitを調べて lock に記録
│   └── groot_*.py / groot_setup.sh   GR00T（B04/B05）の環境構築・実行・評価・静的監査
│
├── aws/                      AWS操作（承認ガード付き）
│   ├── plan.py               起動リクエストJSONを書き出すだけ（AWS APIは呼ばない）
│   ├── launch_after_approval.sh  明示的な承認の環境変数がないと起動を拒否、1回1台
│   ├── cleanup.sh            RUN_IDのインスタンス一覧（--terminate-confirmed で終了）
│   ├── cost.py               Cost Explorerで実費を照合（読み取りのみ）
│   └── *.json                IAM信頼関係・S3ポリシーの例
│
├── configs/
│   ├── zero1.json / zero2.json / zero3.json   DeepSpeed ZeRO設定
│   ├── experiments.json      各実験のGPU実行状況（status）
│   └── revisions.lock.json   モデル・データ・GR00Tのcommit SHA
│
├── docs/
│   ├── validation.md         検証済み・未検証の範囲
│   ├── aws.md                インスタンス選択、IAM、ネットワーク、EFA、課金、後片付け
│   ├── theory.md             データ並列・モデル並列とメモリ見積もり
│   ├── measurement.md        計測とcheckpointの公平な比較条件
│   ├── data_and_real_models.md   公開モデル・データ・VLA評価
│   ├── sources.md            参照元とライセンスのリンク
│   └── lessons/A01.md〜B05.md   19本の教材
│
├── data/example.jsonl.example   学習データ（JSONL）の形式例
├── results/                  比較表（実測欄は空）、CPU検証ログ、GR00T静的監査結果
└── tests/                    CPUで動く単体・統合テスト
```

実行すると次のものが作られます（いずれも`.gitignore`対象）。

| パス | 中身 |
|---|---|
| `.venv/` | `setup.sh`が作るPython環境 |
| `runs/` | 各実験の出力（metrics CSV、環境JSON、checkpoint、preflight結果）。受け入れテストは`runs/acceptance/` |
| `third_party/` | GR00Tの上流リポジトリ |
| `data/*.jsonl`、`data/**/*.bin` | ダウンロード・生成したデータとtoken memmap |
| `~/.cache/torch_extensions/` | DeepSpeed FusedAdamのJITビルド結果（リポジトリ外） |

## 安全設計

- `aws/plan.py`は起動リクエストのJSONを**書き出すだけ**で、AWS APIは呼びません。
- `aws/launch_after_approval.sh`は、`AWS_PAID_RESOURCE_APPROVAL=I_HAVE_EXPLICIT_USER_APPROVAL`を設定しないと起動を拒否します。1回の実行で起動するのは1台だけです。
- `aws/cleanup.sh`は既定では対象の一覧を表示するだけで、`--terminate-confirmed`を付けたときにだけ終了処理をします。
- `scripts/launch.sh`は、ログイン済みのノード上でtorchrunを起動するだけです。SSHで他ノードへ展開したり、クラウドのリソースを作成したりはしません。

## 再現性

- `configs/revisions.lock.json`に、モデル・データセット・GR00Tのcommit SHAを固定しています。
- 基礎実験とQwenはPyTorch 2.7.1を使います。GR00Tは公式リポジトリの`uv.lock`を使い、別の環境で動かします。
- `requirements.txt`は候補として固定した依存構成です。GPU上での受け入れ確認は途中です（[docs/validation.md](docs/validation.md)）。
- 各runについて、HF revision、データのSHA256、pip freeze、EFA/OFIのバージョンを記録します。

## スコープ外

- 32Bモデルをゼロから学習するPre-training
- 実機ロボットの制御
- GPU・EFAのない環境での、NCCL・ZeRO・FSDP・TPの性能測定

## ライセンス

コードとドキュメントは[MIT License](LICENSE)です。
Qwen3、GR00T N1.7、Cosmos、WikiTextなどのモデル・データセット、および依存パッケージには、それぞれのライセンスと利用条件が適用されます。
ダウンロードして使う前に、各配布元の条件を確認してください（リンクは[docs/sources.md](docs/sources.md)）。

## 貢献

GPUでの実測結果、不具合の報告、教材の改善を歓迎します。
実測値を共有していただく場合は、インスタンス種別、リージョン、ドライバ・CUDA・NCCL・EFAのバージョン、実行コマンドを添えてください。
