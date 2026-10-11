#!/usr/bin/env bash
# =============================================================================
# aws/launch_after_approval.sh — 承認済みの起動リクエストで EC2 インスタンスを 1 台起動する
#
# 目的:
#   aws/plan.py が作った JSON を使って、高額な GPU インスタンスを実際に起動する。
#   誤って実行しても起動しないよう、何重もの安全装置を入れてある。
#   (このスクリプトが条件を満たしても、利用者本人の明示的な承認は別途必要)
#
# 使い方:
#   AWS_PAID_RESOURCE_APPROVAL=I_HAVE_EXPLICIT_USER_APPROVAL \
#   AWS_REGION=us-east-1 RUN_ID=lab-20261001 REQUEST_JSON=aws/launch.local.json NODE_LABEL=0 \
#     bash aws/launch_after_approval.sh
#
# 読み取る環境変数:
#   AWS_PAID_RESOURCE_APPROVAL (必須) : 決まった合言葉 I_HAVE_EXPLICIT_USER_APPROVAL
#   AWS_REGION   (必須) : 起動するリージョン
#   RUN_ID       (必須) : 実験 ID。JSON のインスタンスタグと一致している必要がある
#   REQUEST_JSON (必須) : レビュー済みの起動リクエスト JSON (plan.py の出力)
#   NODE_LABEL          : ノード番号 (既定 0)。冪等性トークンと出力ファイル名に使う
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 安全装置 1: 承認の合言葉
#   長くて偶然には一致しない文字列を要求し、うっかり実行を防ぐ。
#   一致しなければ終了コード 2 で拒否する (tests/test_lab.py の test_aws_guard で確認)。
#   起動前に、リージョン・インスタンスタイプ・台数・料金・時間について
#   利用者の明示的な承認を得ておくこと。
# -----------------------------------------------------------------------------
if [[ "${AWS_PAID_RESOURCE_APPROVAL:-}" != 'I_HAVE_EXPLICIT_USER_APPROVAL' ]]; then
  echo 'Refusing paid resource creation: obtain explicit user approval for region/type/count/rate/time first.' >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# 安全装置 2: 必須の環境変数 (未設定なら即エラー終了)
# -----------------------------------------------------------------------------
: "${AWS_REGION:?required}"
: "${RUN_ID:?required}"
: "${REQUEST_JSON:?reviewed launch JSON required}"

# -----------------------------------------------------------------------------
# 安全装置 3: リクエスト JSON の中身を Python で検査する (どれかに違反すると assert で失敗)
#   - InstanceType が教材で想定する 4 種類のどれかであること
#   - MinCount と MaxCount がどちらも 1 であること (1 回で 1 台だけ起動)
#   - インスタンスのタグに RunId=$RUN_ID が含まれること
#     (cleanup.sh で確実に見つけて終了できるようにするため)
#   検査に失敗すると python が非 0 で終了し、set -e によりここでスクリプトが止まる。
# -----------------------------------------------------------------------------
python - "$REQUEST_JSON" "$RUN_ID" <<'PY2'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['InstanceType'] in {'p5.48xlarge','p5e.48xlarge','p5en.48xlarge','p6-b200.48xlarge'} and r['MinCount']==r['MaxCount']==1
assert any(t['Key']=='RunId' and t['Value']==sys.argv[2] for s in r['TagSpecifications'] if s['ResourceType']=='instance' for t in s['Tags'])
PY2

# -----------------------------------------------------------------------------
# 起動 (ここで初めて課金が発生する)
#   --client-token : 冪等性トークン。同じ RUN_ID・NODE_LABEL で再実行しても、
#                    (トークンの有効期間内であれば) 同じリクエストとみなされ、
#                    2 台目が起動されることはない。
#   起動結果 (インスタンス ID など) は aws/launched-<NODE_LABEL>.json に保存する。
# -----------------------------------------------------------------------------
aws ec2 run-instances --region "$AWS_REGION" --cli-input-json "file://$REQUEST_JSON" \
  --client-token "$RUN_ID-${NODE_LABEL:-0}" > "aws/launched-${NODE_LABEL:-0}.json"
