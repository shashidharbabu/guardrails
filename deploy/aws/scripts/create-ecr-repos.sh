#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-west-1}"

repos=(
  "spartanguard/frontend"
  "spartanguard/backend"
  "spartanguard/gateway"
  "spartanguard/mad-api"
  "spartanguard/rlhf"
)

for repo in "${repos[@]}"; do
  if aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$repo" >/dev/null 2>&1; then
    echo "exists: $repo"
  else
    aws ecr create-repository \
      --region "$AWS_REGION" \
      --repository-name "$repo" \
      --image-scanning-configuration scanOnPush=true \
      --encryption-configuration encryptionType=AES256 >/dev/null
    echo "created: $repo"
  fi
done
