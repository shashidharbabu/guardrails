#!/bin/bash

# Stop Airflow Services

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to docker directory and stop services
cd "$PROJECT_ROOT/docker"

echo "Stopping Airflow services..."
docker-compose down

echo ""
echo "✅ Airflow services stopped!"
echo ""

