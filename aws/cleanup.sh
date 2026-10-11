#!/usr/bin/env bash
set -euo pipefail
# Terminate instances tagged with this RUN_ID. Without --terminate-confirmed it only lists them.
: "${AWS_REGION:?required}"; : "${RUN_ID:?required}"
# Dry-run inventory is default. Backup all rank shards FIRST.
ids=$(aws ec2 describe-instances --region "$AWS_REGION" \
 --filters "Name=tag:RunId,Values=$RUN_ID" 'Name=tag:Project,Values=gpu-lab' \
 'Name=instance-state-name,Values=pending,running,stopping,stopped' \
 --query 'Reservations[].Instances[].InstanceId' --output text)
echo "Target instance IDs: $ids"
if [[ "${1:-}" != '--terminate-confirmed' ]]; then
 echo 'Inventory only. Review IDs and backup checkpoint shards; then pass --terminate-confirmed.'; exit 0
fi
if [[ -z "$ids" || "$ids" == None ]]; then exit 0; fi
read -r -a instance_ids <<< "$ids"
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "${instance_ids[@]}"
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "${instance_ids[@]}"
# Report leftovers; never blindly delete untagged or shared resources.
aws ec2 describe-volumes --region "$AWS_REGION" --filters "Name=tag:RunId,Values=$RUN_ID"
aws ec2 describe-network-interfaces --region "$AWS_REGION" --filters "Name=tag:RunId,Values=$RUN_ID"
