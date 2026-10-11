# GPU Distributed Lab — AWS 分散学習ハンズオン

AWSの8GPUノード（H100 / H200 / B200）を使い、分散学習を**手を動かして**理解するための教材です。
自作の小さなTransformerでDDP・ZeRO・FSDP・Tensor Parallelの仕組みを一つずつ確かめ、最後にQwen3-8B/32Bの全パラメータFine-tuningとGR00T N1.7（VLA）のFine-tuningまで進みます。

- **Part A（14実験）**：CUDA基礎 → NCCL通信 → 自作Ring AllReduce → DDP → ZeRO-1/2/3 → FSDP → TP → マルチノード → プロファイリング
- **Part B（5実験）**：実モデル（Qwen3-8B/32B、GR00T N1.7）での学習・スケーリング・評価
- 最大構成：8GPUノード × 5台 = 40GPU

> [!WARNING]
> **現状はCPU上の検証のみです。** GPU・EFA・実モデルweightでの学習はまだ実行していません。
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

DeepSpeedはimport時に`nvcc`の存在とCUDAメジャー版（PyTorchのcu128と同じ12）を確認し、初回にFusedAdamをJITビルドします。
DLAMIに`/usr/local/cuda-12.8`がない場合（デフォルトがCUDA 13のDLAMIなど）は、NVIDIAのrunfileでToolkitだけを入れます。

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --no-man-page --override
/usr/local/cuda-12.8/bin/nvcc --version   # release 12.8
```

`--toolkit`のみを指定するとドライバには触れません（`--override`はgcc版チェックを飛ばすため）。

Ubuntu 26.04では`libxml2.so.2`がなく（`libxml2.so.16`のみ）、インストーラが起動しません。
その場合はインストール中だけ一時的にリンクを張り、終わったら必ず消してください（ABIが異なるため常設しない）。

```bash
sudo ln -s libxml2.so.16 /usr/lib/x86_64-linux-gnu/libxml2.so.2
sudo sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --no-man-page --override
sudo rm /usr/lib/x86_64-linux-gnu/libxml2.so.2 && sudo ldconfig
```
`launch.sh`は`CUDA_HOME`が未設定なら`/usr/local/cuda-12.8`を優先して使い、`gcc-14`があれば`CC=gcc-14 CXX=g++-14`を設定します。

> **Ubuntu標準の`nvidia-cuda-toolkit`はaptで入れないでください。** Ubuntu版のNVIDIAユーザー空間ライブラリが入り、DLAMIのカーネルドライバと版がずれて`nvidia-smi`が`Driver/library version mismatch`になります。

そのあと:

```bash
bash scripts/setup.sh            # venv作成、CUDA 12.8版PyTorch、preflight
source .venv/bin/activate
bash scripts/acceptance.sh       # 全モードの短い受け入れテスト
bash scripts/part_a.sh A01
bash scripts/part_a.sh A02 --steps 30 --warmup 5
NPROC=2 bash scripts/part_a.sh A04 --iters 10 --max-power 22
```

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
lab/          学習・通信の本体
  model.py      Attention・MLPまで見える自作Transformer と TP適用
  train.py      single/manual/DDP/ZeRO/FSDP/TP 共通の学習ループ・計測・保存・復元
  ring.py       P2Pで書いたRing AllReduce（教育用）
  collectives.py  AllReduce/AllGather/ReduceScatter ベンチマーク
scripts/      実験ごとの入口（part_a.sh / part_b.sh）、起動、集計、GR00T連携
aws/          起動プラン生成、承認ガード付き起動、コスト照合、後片付け
configs/      DeepSpeed設定、モデル・データのcommit固定
docs/         理論・運用ドキュメントと19本の教材
results/      比較表（未実測）、CPU検証ログ、GR00T静的監査結果
tests/        CPUで動く単体・統合テスト
```

## 安全設計

- `aws/plan.py`は起動リクエストのJSONを**書き出すだけ**で、AWS APIは呼びません。
- `aws/launch_after_approval.sh`は、`AWS_PAID_RESOURCE_APPROVAL=I_HAVE_EXPLICIT_USER_APPROVAL`を設定しないと起動を拒否します。1回の実行で起動するのは1台だけです。
- `aws/cleanup.sh`は既定では対象の一覧を表示するだけで、`--terminate-confirmed`を付けたときにだけ終了処理をします。
- `scripts/launch.sh`は、ログイン済みのノード上でtorchrunを起動するだけです。SSHで他ノードへ展開したり、クラウドのリソースを作成したりはしません。

## 再現性

- `configs/revisions.lock.json`に、モデル・データセット・GR00Tのcommit SHAを固定しています。
- 基礎実験とQwenはPyTorch 2.7.1を使います。GR00Tは公式リポジトリの`uv.lock`を使い、別の環境で動かします。
- `requirements.txt`は候補として固定した依存構成です。GPU上での受け入れ確認はまだ済んでいません。
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
