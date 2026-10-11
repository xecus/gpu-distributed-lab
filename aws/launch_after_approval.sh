#!/usr/bin/env bash
set -euo pipefail
# Intentionally impossible to invoke accidentally. User approval is still required.
if [[ "${AWS_PAID_RESOURCE_APPROVAL:-}" != 'I_HAVE_EXPLICIT_USER_APPROVAL' ]]; then
 echo 'Refusing paid resource creation: obtain explicit user approval for region/type/count/rate/time first.' >&2; exit 2
fi
: "${AWS_REGION:?required}"
: "${RUN_ID:?required}"
: "${REQUEST_JSON:?reviewed launch JSON required}"
python - "$REQUEST_JSON" "$RUN_ID" <<'PY2'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['InstanceType'] in {'p5.48xlarge','p5e.48xlarge','p5en.48xlarge','p6-b200.48xlarge'} and r['MinCount']==r['MaxCount']==1
assert any(t['Key']=='RunId' and t['Value']==sys.argv[2] for s in r['TagSpecifications'] if s['ResourceType']=='instance' for t in s['Tags'])
PY2
aws ec2 run-instances --region "$AWS_REGION" --cli-input-json "file://$REQUEST_JSON" \
 --client-token "$RUN_ID-${NODE_LABEL:-0}" > "aws/launched-${NODE_LABEL:-0}.json"
