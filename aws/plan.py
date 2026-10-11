"""Render EC2 request only. Never invokes RunInstances. One 8-GPU node per request."""
import argparse,json
from pathlib import Path
# Network cards per node. Confirm with: aws ec2 describe-instance-types --query 'InstanceTypes[0].NetworkInfo.MaximumNetworkCards'
INSTANCES={'p5.48xlarge':('H100 80GB',32),'p5e.48xlarge':('H200 141GB',32),'p5en.48xlarge':('H200 141GB',16),'p6-b200.48xlarge':('B200 180GB',8)}
p=argparse.ArgumentParser();p.add_argument('--ami',required=True);p.add_argument('--subnet',required=True)
p.add_argument('--sg',required=True);p.add_argument('--profile',required=True);p.add_argument('--placement',required=True)
p.add_argument('--instance-type',choices=sorted(INSTANCES),default='p5.48xlarge')
p.add_argument('--network-cards',type=int,help='override after checking describe-instance-types MaximumNetworkCards')
p.add_argument('--run-id',required=True);p.add_argument('--out',default='aws/launch.local.json');a=p.parse_args()
gpu,cards=INSTANCES[a.instance_type];cards=a.network_cards or cards
req={'ImageId':a.ami,'InstanceType':a.instance_type,'MinCount':1,'MaxCount':1,
 'IamInstanceProfile':{'Name':a.profile},'Placement':{'GroupName':a.placement},
 'MetadataOptions':{'HttpTokens':'required'},
 'BlockDeviceMappings':[{'DeviceName':'/dev/sda1','Ebs':{'VolumeSize':2000,'VolumeType':'gp3','DeleteOnTermination':True,'Encrypted':True}}],
 'NetworkInterfaces':[{'DeviceIndex':0 if card==0 else 1,'NetworkCardIndex':card,'SubnetId':a.subnet,
 'Groups':[a.sg],'InterfaceType':'efa','DeleteOnTermination':True} for card in range(cards)],
 'TagSpecifications':[{'ResourceType':t,'Tags':[{'Key':'Project','Value':'gpu-lab'},{'Key':'RunId','Value':a.run_id}]} for t in ['instance','volume']]}
Path(a.out).write_text(json.dumps(req,indent=2));print(f'Plan only: {a.out}; 1 node {a.instance_type}, 8x {gpu}, {cards} EFA interfaces. Review before explicit approval.')
