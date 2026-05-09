#!/usr/bin/env bash
set -euo pipefail

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-829108230080}"
AWS_REGION="${AWS_REGION:-us-west-1}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
AWS_CMD="${AWS_CMD:-aws}"

if ! command -v "$AWS_CMD" >/dev/null 2>&1; then
  if [ -x "$HOME/Library/Python/3.9/bin/aws" ]; then
    AWS_CMD="$HOME/Library/Python/3.9/bin/aws"
  else
    echo "aws CLI not found. Install AWS CLI or set AWS_CMD=/path/to/aws." >&2
    exit 1
  fi
fi

services=(
  "frontend:app/frontend:Dockerfile:spartanguard/frontend"
  "backend:.:app/backend/Dockerfile:spartanguard/backend"
  "gateway:.:gateway/Dockerfile:spartanguard/gateway"
  "mad-api:.:multi_agent_debate/full_FinalMAD_with_judge/Dockerfile:spartanguard/mad-api"
  "rlhf:.:rlhf/Dockerfile:spartanguard/rlhf"
)

"$AWS_CMD" ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

for service_spec in "${services[@]}"; do
  IFS=":" read -r service context dockerfile repo <<<"$service_spec"
  image="${REGISTRY}/${repo}:${IMAGE_TAG}"

  echo "building ${service} -> ${image}"
  docker build -f "$dockerfile" -t "$image" "$context"

  echo "pushing ${image}"
  docker push "$image"
done

echo "pushed tag: ${IMAGE_TAG}"
