#!/usr/bin/env bash
# =============================================================================
# aws/cleanup.sh — この実験 (RUN_ID) で起動した EC2 インスタンスを終了 (削除) する
#
# 目的:
#   GPU インスタンスは起動しているだけで高額な料金がかかるので、実験が終わったら
#   確実に終了させる。ただし誤って別の実験のインスタンスを消さないよう、
#   Project=gpu-lab と RunId=<RUN_ID> の両方のタグが付いたものだけを対象にする。
#
# 使い方:
#   # 1. まずは一覧表示のみ (既定。何も削除しない)
#   AWS_REGION=us-east-1 RUN_ID=lab-20261001 bash aws/cleanup.sh
#
#   # 2. ID を確認し、チェックポイントをバックアップしてから、本当に終了させる
#   AWS_REGION=us-east-1 RUN_ID=lab-20261001 bash aws/cleanup.sh --terminate-confirmed
#
# 読み取る環境変数:
#   AWS_REGION (必須) : インスタンスのあるリージョン
#   RUN_ID     (必須) : plan.py で付けた RunId タグの値
# =============================================================================
set -euo pipefail

# 必須の環境変数。未設定なら即エラー終了する。
: "${AWS_REGION:?required}"
: "${RUN_ID:?required}"

# -----------------------------------------------------------------------------
# 対象インスタンスの一覧を取得する
#   既定は「一覧表示だけ (ドライラン)」。
#   インスタンスを終了するとローカルディスクのデータも消えるので、
#   全 rank のチェックポイントシャードを必ず先にバックアップすること。
#
#   絞り込み条件 (すべて満たすもの):
#     - RunId タグ = $RUN_ID
#     - Project タグ = gpu-lab
#     - 状態が pending / running / stopping / stopped (終了済みのものは除く)
# -----------------------------------------------------------------------------
ids=$(aws ec2 describe-instances --region "$AWS_REGION" \
  --filters "Name=tag:RunId,Values=$RUN_ID" 'Name=tag:Project,Values=gpu-lab' \
  'Name=instance-state-name,Values=pending,running,stopping,stopped' \
  --query 'Reservations[].Instances[].InstanceId' --output text)
echo "Target instance IDs: $ids"

# -----------------------------------------------------------------------------
# 安全装置: 第 1 引数が --terminate-confirmed でなければ、一覧を表示しただけで終了する
# -----------------------------------------------------------------------------
if [[ "${1:-}" != '--terminate-confirmed' ]]; then
  echo 'Inventory only. Review IDs and backup checkpoint shards; then pass --terminate-confirmed.'
  exit 0
fi

# 対象が 1 台もなければ何もせず終了する (aws CLI は空の結果を "None" と出すことがある)。
if [[ -z "$ids" || "$ids" == None ]]; then
  exit 0
fi

# -----------------------------------------------------------------------------
# インスタンスを終了し、完全に終了するまで待つ
#   スペース区切りの ID 文字列を配列に分解して渡す。
# -----------------------------------------------------------------------------
read -r -a instance_ids <<< "$ids"
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "${instance_ids[@]}"
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "${instance_ids[@]}"

# -----------------------------------------------------------------------------
# 残ったリソースを報告する (削除はしない)
#   DeleteOnTermination の設定漏れなどでボリュームが残ると課金が続くので、
#   一覧を表示して人間が確認する。タグのないリソースや共有リソースを
#   機械的に削除するのは危険なので、決して自動では消さない。
#   ※ plan.py は ENI にタグを付けないため、下の ENI 一覧は通常空になる。
#     ENI の残存はタグ以外の条件 (サブネット等) で別途確認すること。
# -----------------------------------------------------------------------------
aws ec2 describe-volumes --region "$AWS_REGION" --filters "Name=tag:RunId,Values=$RUN_ID"
aws ec2 describe-network-interfaces --region "$AWS_REGION" --filters "Name=tag:RunId,Values=$RUN_ID"
