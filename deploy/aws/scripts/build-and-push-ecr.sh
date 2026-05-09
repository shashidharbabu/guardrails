#!/usr/bin/env bash
set -euo pipefail

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-829108230080}"
AWS_REGION="${AWS_REGION:-us-west-1}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

services=(
  "frontend:app/frontend/Dockerfile:spartanguard/frontend"
  "backend:app/backend/Dockerfile:spartanguard/backend"
  "gateway:gateway/Dockerfile:spartanguard/gateway"
  "mad-api:multi_agent_debate/full_FinalMAD_with_judge/Dockerfile:spartanguard/mad-api"
  "rlhf:rlhf/Dockerfile:spartanguard/rlhf"
)

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

for service_spec in "${services[@]}"; do
  IFS=":" read -r service dockerfile repo <<<"$service_spec"
  image="${REGISTRY}/${repo}:${IMAGE_TAG}"

  echo "building ${service} -> ${image}"
  docker build -f "$dockerfile" -t "$image" .

  echo "pushing ${image}"
  docker push "$image"
done

echo "pushed tag: ${IMAGE_TAG}"
