r"""EC2 起動リクエスト (JSON) を生成するだけのスクリプト。RunInstances は絶対に呼ばない。

目的:
    8 GPU ノード 1 台分の `aws ec2 run-instances --cli-input-json` 用 JSON を作る。
    実際の起動は、人間が JSON をレビューし明示的に承認したあとで
    aws/launch_after_approval.sh が行う。「計画」と「実行」を分けることで、
    高額な GPU インスタンスを誤って起動する事故を防ぐ。

使い方の例:
    python aws/plan.py --ami ami-xxxx --subnet subnet-xxxx --sg sg-xxxx \
        --profile my-instance-profile --placement my-cluster-pg --run-id lab-20261001

    1 リクエスト = 1 ノード。複数ノードは同じ JSON・同じ RUN_ID のまま NODE_LABEL を変えて
    launch_after_approval.sh を 1 台ずつ明示的に実行する (docs/aws.md)。

生成される JSON のポイント:
    - MinCount = MaxCount = 1 (1 ノードずつ)
    - クラスタ配置グループ (Placement) でノード間を低遅延に配置
    - 全ネットワークカードに EFA インターフェースを割り当てる
    - IMDSv2 必須 (HttpTokens=required)、暗号化 gp3 2TB ルートボリューム
    - Project / RunId タグをインスタンスとボリュームに付与 (cleanup.sh と cost.py が使う)
"""
import argparse,json
from pathlib import Path

# インスタンスタイプごとの (GPU 名, ノードあたりのネットワークカード数)。
# EFA インターフェースはネットワークカード 1 枚につき 1 つ作るので、
# カード数がそのまま EFA 数になる。
# 値は次のコマンドで確認すること:
#   aws ec2 describe-instance-types --instance-types p5.48xlarge \
#     --query 'InstanceTypes[0].NetworkInfo.MaximumNetworkCards'
INSTANCES={
    'p5.48xlarge':('H100 80GB',32),
    'p5e.48xlarge':('H200 141GB',32),
    'p5en.48xlarge':('H200 141GB',16),
    'p6-b200.48xlarge':('B200 180GB',8),
}

# ---------------------------------------------------------------------------
# コマンドライン引数
#   --ami / --subnet / --sg / --profile / --placement : 事前に用意した AWS リソースの ID / 名前
#   --instance-type   : 上の INSTANCES のいずれか
#   --network-cards   : カード数を手動で上書きしたい場合 (describe-instance-types で確認後に)
#   --run-id          : この実験を識別するタグ値。後片付け (cleanup.sh) や費用集計 (cost.py) に使う
#   --out             : 出力 JSON のパス
# ---------------------------------------------------------------------------
p=argparse.ArgumentParser()
p.add_argument('--ami',required=True)
p.add_argument('--subnet',required=True)
p.add_argument('--sg',required=True)
p.add_argument('--profile',required=True)
p.add_argument('--placement',required=True)
p.add_argument('--instance-type',choices=sorted(INSTANCES),default='p5.48xlarge')
p.add_argument(
    '--network-cards',
    type=int,
    help='override after checking describe-instance-types MaximumNetworkCards',
)
p.add_argument('--run-id',required=True)
p.add_argument('--out',default='aws/launch.local.json')
a=p.parse_args()

# GPU 名と EFA 数を決める (--network-cards が指定されていればそちらを優先)。
gpu,cards=INSTANCES[a.instance_type]
cards=a.network_cards or cards

req={
    'ImageId':a.ami,
    'InstanceType':a.instance_type,
    # 起動台数は必ず 1 台。複数ノードは NODE_LABEL を変えて同じ JSON で 1 台ずつ起動する。
    'MinCount':1,
    'MaxCount':1,
    'IamInstanceProfile':{'Name':a.profile},
    # クラスタ配置グループに入れて、ノード間通信の遅延を小さくする。
    'Placement':{'GroupName':a.placement},
    # IMDSv2 (トークン必須) を強制する。セキュリティ上のベストプラクティス。
    'MetadataOptions':{'HttpTokens':'required'},
    # ルートボリューム: 2TB の gp3。チェックポイントやモデル重みを置くため大きめ。
    # 終了時に削除 & 暗号化を有効にする。
    'BlockDeviceMappings':[
        {
            'DeviceName':'/dev/sda1',
            'Ebs':{
                'VolumeSize':2000,
                'VolumeType':'gp3',
                'DeleteOnTermination':True,
                'Encrypted':True,
            },
        }
    ],
    # ネットワークカードごとに EFA インターフェースを 1 つ作る。
    # カード 0 だけ DeviceIndex=0 (プライマリ)、それ以外は DeviceIndex=1 にするのが EFA の作法。
    'NetworkInterfaces':[
        {
            'DeviceIndex':0 if card==0 else 1,
            'NetworkCardIndex':card,
            'SubnetId':a.subnet,
            'Groups':[a.sg],
            'InterfaceType':'efa',
            'DeleteOnTermination':True,
        }
        for card in range(cards)
    ],
    # インスタンスとボリュームの両方にタグを付ける。
    # cleanup.sh は Project=gpu-lab と RunId で削除対象を絞り込む。
    'TagSpecifications':[
        {
            'ResourceType':t,
            'Tags':[{'Key':'Project','Value':'gpu-lab'},{'Key':'RunId','Value':a.run_id}],
        }
        for t in ['instance','volume']
    ],
}

# JSON を書き出して終わり。ここでは AWS API を一切呼ばない。
Path(a.out).write_text(json.dumps(req,indent=2))
print(
    f'Plan only: {a.out}; 1 node {a.instance_type}, 8x {gpu}, {cards} EFA interfaces. '
    'Review before explicit approval.'
)
