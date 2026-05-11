# syntax=docker/dockerfile:1
# Multi-stage production frontend image.
# Stage 1: build static assets
FROM node:20-alpine AS builder

WORKDIR /build

# Install dependencies from lockfile for reproducible builds
COPY package.json package-lock.json ./
RUN npm ci --no-audit

COPY . .

ARG VITE_APP_ENV=production
ARG VITE_APP_VERSION=1.0.0
ARG VITE_MOCK_API=false

ENV VITE_APP_ENV=$VITE_APP_ENV
ENV VITE_APP_VERSION=$VITE_APP_VERSION
ENV VITE_MOCK_API=$VITE_MOCK_API

RUN npm run build


# Stage 2: Nginx runtime
FROM nginx:1.25-alpine AS runtime

LABEL maintainer="guardrails-platform"
LABEL description="Enterprise Guardrails Frontend"

RUN apk add --no-cache bash

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# nginx.conf uses ${BACKEND_URL} — substituted at container start via entrypoint
COPY nginx.conf /etc/nginx/conf.d/guardrails.conf.template

COPY --from=builder /build/dist /usr/share/nginx/html

RUN chown -R nginx:nginx /usr/share/nginx/html \
 && chmod -R 755 /usr/share/nginx/html

# Entrypoint: substitute BACKEND_URL then start nginx
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1

CMD ["/docker-entrypoint.sh"]
