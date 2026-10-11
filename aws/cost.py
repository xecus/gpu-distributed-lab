"""Read-only Cost Explorer reconciliation. IAM ce:GetCostAndUsage required."""
import argparse,subprocess,json,csv
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--start',required=True);p.add_argument('--end',required=True)
p.add_argument('--run-id',required=True);p.add_argument('--out',default='runs/billing.csv');a=p.parse_args()
f={'Tags':{'Key':'RunId','Values':[a.run_id],'MatchOptions':['EQUALS']}}
cmd=['aws','ce','get-cost-and-usage','--region','us-east-1','--time-period',f'Start={a.start},End={a.end}',
 '--granularity','DAILY','--metrics','UnblendedCost','--filter',json.dumps(f),
 '--group-by','Type=DIMENSION,Key=SERVICE','--output','json']
r=json.loads(subprocess.check_output(cmd,text=True));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='') as file:
 w=csv.writer(file);w.writerow(['date','service','amount','unit','estimated'])
 for day in r['ResultsByTime']:
  for group in day['Groups']:
   c=group['Metrics']['UnblendedCost'];w.writerow([day['TimePeriod']['Start'],group['Keys'][0],c['Amount'],c['Unit'],day['Estimated']])
