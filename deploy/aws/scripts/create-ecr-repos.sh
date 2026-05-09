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

repos=(
  "spartanguard/frontend"
  "spartanguard/backend"
  "spartanguard/gateway"
  "spartanguard/mad-api"
  "spartanguard/rlhf"
)

for repo in "${repos[@]}"; do
  if "$AWS_CMD" ecr describe-repositories --region "$AWS_REGION" --repository-names "$repo" >/dev/null 2>&1; then
    echo "exists: $repo"
  else
    "$AWS_CMD" ecr create-repository \
      --region "$AWS_REGION" \
      --repository-name "$repo" \
      --image-scanning-configuration scanOnPush=true \
      --encryption-configuration encryptionType=AES256 >/dev/null
    echo "created: $repo"
  fi
done
