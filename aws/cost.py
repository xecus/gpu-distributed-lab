"""AWS Cost Explorer から実際の請求額を取得して CSV に書き出す (読み取り専用)。

目的:
    実験で使った EC2 などの費用を、RunId タグ単位で日別・サービス別に集計し、
    runs/ 以下の実験記録 (run_record.py が見積もった概算コスト) と突き合わせる。
    このスクリプトは Cost Explorer を「読む」だけで、リソースの作成・削除は一切しない。

使い方の例:
    python aws/cost.py --start 2026-10-01 --end 2026-10-08 --run-id lab-20261001

必要な権限:
    IAM の ce:GetCostAndUsage。
    Cost Explorer API はグローバルサービスなので、リージョンは常に us-east-1 を指定する。

注意:
    タグで費用を絞り込むには、事前に請求コンソールで RunId を
    「コスト配分タグ」として有効化しておく必要がある (有効化前の費用はタグで拾えない)。
    また請求月が締まるまでの費用は見積もり値 (estimated 列が True) で、後で変わり得る。
"""
import argparse,subprocess,json,csv
from pathlib import Path

# ---------------------------------------------------------------------------
# コマンドライン引数
#   --start / --end : 集計期間 (YYYY-MM-DD)。Cost Explorer の仕様で end は「含まない」。
#   --run-id        : インスタンス作成時に付けた RunId タグの値。
#   --out           : 出力 CSV のパス。
# ---------------------------------------------------------------------------
p=argparse.ArgumentParser()
p.add_argument('--start',required=True)
p.add_argument('--end',required=True)
p.add_argument('--run-id',required=True)
p.add_argument('--out',default='runs/billing.csv')
a=p.parse_args()

# RunId タグが完全一致するリソースの費用だけに絞り込むフィルタ。
f={'Tags':{'Key':'RunId','Values':[a.run_id],'MatchOptions':['EQUALS']}}

# aws CLI で Cost Explorer を呼ぶ。
#   - 日別 (DAILY) の非ブレンドコスト (UnblendedCost = 実際に請求される単価ベースの金額)
#   - サービスごとにグループ化 (例: Amazon Elastic Compute Cloud - Compute、
#     EBS 等を含む EC2 - Other)
cmd=[
    'aws','ce','get-cost-and-usage','--region','us-east-1',
    '--time-period',f'Start={a.start},End={a.end}',
    '--granularity','DAILY','--metrics','UnblendedCost','--filter',json.dumps(f),
    '--group-by','Type=DIMENSION,Key=SERVICE','--output','json',
]

# CLI の JSON 出力をパースし、出力先ディレクトリを (無ければ) 作成する。
r=json.loads(subprocess.check_output(cmd,text=True))
out=Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)

# 1 行 = 「ある日 × あるサービス」の費用として CSV に書き出す。
with out.open('w',newline='') as file:
    w=csv.writer(file)
    w.writerow(['date','service','amount','unit','estimated'])

    for day in r['ResultsByTime']:
        for group in day['Groups']:
            c=group['Metrics']['UnblendedCost']

            # estimated が True の日は、まだ金額が確定していない (後で変わり得る)。
            w.writerow([
                day['TimePeriod']['Start'],
                group['Keys'][0],
                c['Amount'],
                c['Unit'],
                day['Estimated'],
            ])
