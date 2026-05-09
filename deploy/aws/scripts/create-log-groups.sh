#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-1}"
AWS_CMD="${AWS_CMD:-aws}"

if ! command -v "$AWS_CMD" >/dev/null 2>&1; then
  if [ -x "$HOME/Library/Python/3.9/bin/aws" ]; then
    AWS_CMD="$HOME/Library/Python/3.9/bin/aws"
  else
    echo "aws CLI not found. Install AWS CLI or set AWS_CMD=/path/to/aws." >&2
    exit 1
  fi
fi

log_groups=(
  "/ecs/spartanguard/frontend"
  "/ecs/spartanguard/backend"
  "/ecs/spartanguard/gateway"
  "/ecs/spartanguard/mad-api"
  "/ecs/spartanguard/rlhf"
)

for group in "${log_groups[@]}"; do
  if "$AWS_CMD" logs describe-log-groups \
    --region "$AWS_REGION" \
    --log-group-name-prefix "$group" \
    --query "logGroups[?logGroupName=='$group'].logGroupName" \
    --output text | grep -qx "$group"; then
    echo "exists: $group"
  else
    "$AWS_CMD" logs create-log-group --region "$AWS_REGION" --log-group-name "$group"
    "$AWS_CMD" logs put-retention-policy --region "$AWS_REGION" --log-group-name "$group" --retention-in-days 30
    echo "created: $group"
  fi
done
