# AWS構築、EFA、データ、課金、後片付け

## 0. 承認境界と構成

このリポジトリの開発・CPU検証では、AWSの有料リソースを起動していない。
既存の承認済みGPUノードがあればそれを使用する。追加のEC2、EBS、FSx、NAT、Capacity Reservation/Block等は、地域・台数・単価・上限時間・保存先を提示し承認後に作成する。
`plan.py`はJSONを書くだけ。`launch_after_approval.sh`は明示承認を表す環境変数なしでは拒否する。
その変数自体は承認の代わりにならない。予算・台数・時間について費用負担者の承認を得てから設定する。

どのノードも8GPU/ノードで、コード・手順は共通。`aws/plan.py --instance-type`で選ぶ。

| インスタンス | GPU | network cards（EFA数） | 備考 |
|---|---|---|---|
| p5.48xlarge（既定） | H100 80GB×8 | 32 | Hopper |
| p5e.48xlarge | H200 141GB×8 | 32 | Hopper、メモリ増 |
| p5en.48xlarge | H200 141GB×8 | 16 | Hopper、EFAv3 |
| p6-b200.48xlarge | B200 180GB×8 | 8 | Blackwell。CUDA 12.8以上必須 |

EFA数は承認前に`describe-instance-types`の`NetworkInfo.MaximumNetworkCards`で確認し、違えば`--network-cards`で上書き。
B200はCUDA 12.8+のPyTorch wheel（setup.shはcu128を使用）とBlackwell対応driver/DLAMIが必要。DeepSpeedのFusedAdamはJIT buildなので、PyTorch（cu128）と同じメジャー版のnvcc 12.8が必要（CUDA 13のnvccは不可）。
GPUメモリが違うのでOOMの境界、MFU用のpeak TFLOPS（`scaling.py --peak-bf16-tflops-per-gpu`）、ノード単価はインスタンスごとに変える。異なる種別の結果を同じ表で直接比較しない。

同一AZ、同一Subnet、cluster placement group、EFA対応のDLAMI。
1→2→5ノードの順に拡張。先にEC2 P系vCPU quota、AZの容量、subnet空きIP、placement制約をread-onlyで確認。
5台は40GPU、1GPU実験でも課金はノード全体。未使用7GPU分が無料になるわけではない。

## 1. IAMとアクセス

操作担当：EC2 Describe、必要な場合のRunInstances/TerminateInstances、iam:PassRole、タグ操作、SSM StartSession、Cost Explorer GetCostAndUsage。
EC2 instance profile：AmazonSSMManagedInstanceCore相当＋実験prefixだけのS3 GetObject/PutObject/ListBucket。
Role信頼関係の例は`aws/instance-trust.json`、S3限定policyの例は`aws/s3-instance-policy.example.json`。REVIEWED_BUCKETを実際の許可済みbucketへ置換し、IAM担当者が適用する。
S3の暗号化KMSキーを使う場合は必要なKMS権限も追加。AdministratorAccessを前提にしない。
root volumeは暗号化、IMDSv2 required。HF tokenは環境/credential storeへ置き、git/CSVへ書かない。

```bash
export AWS_REGION=ap-northeast-1
aws sts get-caller-identity
export INSTANCE_TYPE=p5.48xlarge   # p5e.48xlarge / p5en.48xlarge / p6-b200.48xlarge
aws ec2 describe-instance-types --region "$AWS_REGION" --instance-types "$INSTANCE_TYPE" \
  --query 'InstanceTypes[0].{GPU:GpuInfo,Network:NetworkInfo}'
aws ec2 describe-instance-type-offerings --region "$AWS_REGION" --location-type availability-zone \
  --filters Name=instance-type,Values=$INSTANCE_TYPE
aws service-quotas list-service-quotas --service-code ec2 --region "$AWS_REGION"
# AMIはリージョン依存。承認前に最新の対応DLAMIのIDとroot device、driver/CUDAを確認。
aws ec2 describe-images --owners amazon --region "$AWS_REGION" \
  --filters 'Name=name,Values=*Deep Learning*GPU*Ubuntu*' \
  --query 'Images[].{Id:ImageId,Name:Name,Date:CreationDate,Root:RootDeviceName}'
```

DLAMIはCUDA12.8とdriver互換性を満たすものを選ぶ。setup.shはデフォルトでpython3.12を使う（`PYTHON=...`で3.10以上の別のPythonを指定可）。
必要なaptパッケージとCUDA 12.8 Toolkitの入れ方はREADMEの「GPUノードで実行する」を参照（デフォルトがCUDA 13のDLAMIでは12.8のToolkitを別途入れる）。
Base DLAMIに必要なPythonがなければ承認済みホストでインストールする。GPU driverを不用意に置換しない。
特にUbuntu標準の`nvidia-cuda-toolkit`をaptで入れると、NVIDIAのユーザー空間ライブラリが入れ替わり`Driver/library version mismatch`になる。
unattended-upgradesがNVIDIA関連パッケージ（Fabric Managerなど）を更新すると、同様にドライバとの版がずれることがある。短期の実験ノードでは無効化するか、NVIDIA関連とカーネルを更新対象から外す。
既存CUDAが違う場合は選択したDLAMIの公式手順に従い、まず受け入れテストでwheelとOFI pluginの相性を確認。
AMI IDを固定値で教材に埋め込まない。

AWS CLI v2がなければ公式EC2/CLI手順でインストールして`aws --version`を確認する。

SSM例：
```bash
aws ssm start-session --target i-REVIEWED_INSTANCE_ID --region "$AWS_REGION"
```

ノードごとに同じcodeをコピーする。private subnetならSSM/S3/HFへのegressが必要。
既存NATやVPC endpointを使う。新しいNAT/FSxは自動作成しない。

## 2. ネットワーク

SGは同じSGからの全プロトコルのinbound/outboundを許可するEFA向け構成。
外からの全ポート公開は不要。SSMに必要なHTTPS egress、モデル取得のegressを組織の規則に合わせる。
NACL、MTU、ルートも確認。EFAは同一AZ内、torchrun rendezvousはprivate TCP 29500。
同じcluster placement groupに全ノードを置く。IAM/SG/placement groupが既存なら再利用。

以下は環境構築用のコマンド例であり、開発時には実行していない。
必要な有料構成を含む全体承認を得た後に実行する。

```bash
# 既存VPC/Subnetを指定。作成結果のIDsを確認して以降に入力する。
aws ec2 create-placement-group --region "$AWS_REGION" --group-name gpu-lab-pg --strategy cluster
aws ec2 create-security-group --region "$AWS_REGION" --vpc-id vpc-REVIEWED \
  --group-name gpu-lab-efa --description 'GPU laboratory intra-group EFA'
export SG_ID=sg-REVIEWED
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_ID}]"
aws ec2 authorize-security-group-egress --region "$AWS_REGION" --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_ID}]"
```

`aws/plan.py`はnetwork cards数（p5/p5e:32、p5en:16、p6-b200:8）へ各1つのEFA ENIを置く既存方式のプランを作る。
primaryはcard0/device0、残りはdevice1。全帯域を使うため各GPUに近いNICを露出させる。
Subnetに1台あたりその数のENI分のprivate IPが必要。新しいEFA-only方式への変更はAWSの現行docsと比較してレビューする。

```bash
python aws/plan.py --ami ami-REVIEWED --subnet subnet-REVIEWED --sg "$SG_ID" \
  --profile APPROVED_INSTANCE_PROFILE --placement gpu-lab-pg --run-id gpu-lab-20261010 \
  --instance-type "$INSTANCE_TYPE"
# ここまでプランのみ。各値・ストレージサイズ・予算をレビューし承認を得る。
# 承認後にだけ以下を設定・実行。台数分を勝手にループしない。
export AWS_PAID_RESOURCE_APPROVAL=I_HAVE_EXPLICIT_USER_APPROVAL
export REQUEST_JSON=aws/launch.local.json
export RUN_ID=gpu-lab-20261010
export NODE_LABEL=0
bash aws/launch_after_approval.sh
```

requestは1台限定。同じRunId、異なるNODE_LABELで追加台数を明示的に起動する。
root device名はAMIのRootDeviceNameへ合わせてJSONをレビューする。32Bのoptimizer込みcheckpointを2世代＋exportする容量を考慮し、toyのみならストレージを小さくする。defaultは2000GiB gp3、DeleteOnTermination。
起動直後から課金される。予算アラームは停止機構ではない。

## 3. Python/CUDA/NCCL/EFA

各ノードで（READMEのaptパッケージとCUDA 12.8 Toolkitを入れた後）：
```bash
cd gpu-distributed-lab
bash scripts/setup.sh
source .venv/bin/activate
source scripts/efa_env.sh
which fi_info
fi_info -p efa
nvidia-smi -L
nvidia-smi topo -m
python -m pip check
python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.nccl.version())'
find /opt/amazon /opt/aws-ofi-nccl -name '*nccl-net*' 2>/dev/null
bash scripts/acceptance.sh
```

`fi_info`成功だけでNCCLのEFA利用を断定しない。2ノードでA04を実行しログの`NET/OFI`、`efa`、GPUDirect情報を確認。
PyTorch wheelが含むNCCLとDLAMIのOFI pluginの組合せは実際の環境で検証。
LD_LIBRARY_PATHへ複数のNCCL本体を混ぜない。`NCCL_DEBUG=INFO`のログはnode別に保存する。

EFA/SocketのA/B比較：同条件で正常EFA実行の後、別runで`NCCL_NET=Socket`を指定。
Socket側では`unset FI_PROVIDER`。EFA側のrunでは`unset NCCL_NET`して再実行する。
`NCCL_NET=OFI`等はインストール済みpluginの実際のnetwork名を確認せず強制しない。

## 4. マルチノード起動

全ノードが同じ実験引数・RUN_ID・MASTER_ADDR・NNODESを使い、NODE_RANKだけを変更。
rank0のprivate IPはmetadata/EC2 Describeで確認。時計、データchecksum、commit、依存版を一致させる。

```bash
export MASTER_ADDR=172.31.REVIEWED.PRIVATE_IP
export MASTER_PORT=29500
export NNODES=2 NPROC=8 RUN_ID=A13-efa-test
# node0
export NODE_RANK=0
bash scripts/part_a.sh A13 --steps 20 --warmup 3 --batch 1 --accum 5 2>&1 | tee runs/A13-node0.log
# node1の別sessionで（同時に）
export NODE_RANK=1
bash scripts/part_a.sh A13 --steps 20 --warmup 3 --batch 1 --accum 5 2>&1 | tee runs/A13-node1.log
```

ここでglobal batch=16×1×5=80。1nodeの8GPUではaccum10、5nodeの40GPUではaccum2で同じG=80。
3/4nodeも全て同じGで比較するならG=480、accum=60/30/20/15/12で1..5nodeに対応。
`launch.sh`は既存ホスト上のプロセス起動だけ。SSH自動接続やEC2作成はしない。

## 5. データ・ストレージ・チェックポイント

最初は承認済みEBSまたはinstance-local NVMeへデータをコピー。各ノードへ同じtoken corpusを置く。
NVMeは停止/終了で失われうる。FSx for Lustreは高速共有保存に有効だが新規作成は別承認が必要。
S3はcheckpointアーカイブ先として使い、各stepのランダム小I/Oの直接保存先にはしない。

```bash
sha256sum data/qwen/train.bin data/qwen/val.bin
# 既存の許可済みS3 bucket/prefixを指定。保存先を間違えない。
export CHECKPOINT_URI=s3://APPROVED_BUCKET/gpu-lab/REVIEWED_RUN_ID
aws s3 sync runs/ "$CHECKPOINT_URI/node${NODE_RANK:-0}/runs/" --exclude '*trace*.json'
```

multi-nodeで各ローカルvolumeに分散保存する場合、全rankのshardを集めないと復元できない。
保存後に全ノードのstep directoryを1つのcanonical directoryへ統合する。同名ds metadataが競合しないかチェック。
最初の2node検証は共通パスの共有filesystemまたは、停止前の全rank回収と復元試験を必須とする。
詳しくはmeasurement.md。機密ロボットデータをpublic bucketへ置かない。

## 6. コスト計測

EC2単価は地域・On-Demand/Spot/契約で変動する。起動前にAWS Pricing/見積で確認し、ノード単価を入力する。
`NODE_HOURLY_USD`はGPU1枚単価ではなく、使用インスタンス種別のノード単価。
GPU hour=8×nodes×instance稼働h。請求はGPU利用率や学習ステップ数ではなくinstance稼働時間に基づく。
セットアップ、download、idle、checkpoint、エラー再試行も含める。Spot中断の再試行も加算。

```bash
export NODE_HOURLY_USD=REPLACE_WITH_REVIEWED_NUMERIC_PRICE
python scripts/summary.py runs/A02 --node-hourly-usd "$NODE_HOURLY_USD" --nodes 1 \
  --wall-seconds REPLACE_WITH_INSTANCE_RUNNING_SECONDS --extras-usd REPLACE_WITH_EBS_S3_EGRESS_ESTIMATE
# RunIdをcost allocation tagとしてBillingで事前activateする。endはexclusive。
python aws/cost.py --start 2026-10-10 --end 2026-10-11 --run-id "$RUN_ID"
```

tag活性化前の費用や共有NAT等はRunId集計に出ないことがある。請求反映は遅延する。
Cost Explorerの実績はestimated flagも保存し、後日CUR/請求と照合する。
`run_record.py`の費用はコマンドwall時間の概算であり、ノード稼働時間の総費用とは別。
ノードごとのCSVのcostを合計すると同じクラスタ費用を二重計上するので、rank0/主ノードの1行だけを採用する。

## 7. 終了・後片付け

1. 学習プロセスとmonitorを終了。
2. 全rank checkpoint、metric、ログ、freeze、revision、データchecksumを保存。
3. 復元可能性とS3一覧を確認。
4. 対象instance IDをレビューし停止/終了。
5. EBS、snapshot、FSx、NAT、Elastic IP、Capacity Reservation/Blockを別に確認する。

```bash
# 停止：instance-storeのデータ消失とEBSの継続課金を理解した上で実行。
aws ec2 stop-instances --region "$AWS_REGION" --instance-ids i-REVIEWED
# 終了：まずdry-run inventory。
bash aws/cleanup.sh
# 全shardバックアップ後に、表示されたIDが実験専用であることを確認して実行。
bash aws/cleanup.sh --terminate-confirmed
```

Terminate後もDeleteOnTermination=falseのvolumeやsnapshotsは課金が残る。
placement group/SGの削除は実験専用で依存がなくなったものだけ。共有SGや既存bucketを削除しない。
本リポジトリは既存データを一括削除するコマンドを含めない。
