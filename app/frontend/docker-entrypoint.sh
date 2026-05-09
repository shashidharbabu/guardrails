#!/bin/bash
set -e

# BACKEND_URL must be set — e.g. http://backend:8000
: "${BACKEND_URL:?BACKEND_URL environment variable is required}"

envsubst '${BACKEND_URL}' \
  < /etc/nginx/conf.d/guardrails.conf.template \
  > /etc/nginx/conf.d/guardrails.conf

exec nginx -g "daemon off;"
