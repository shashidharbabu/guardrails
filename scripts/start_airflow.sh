#!/bin/bash

# Start Airflow Services

set -e

echo "Starting Airflow services..."
export AIRFLOW_UID=$(id -u)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to docker directory and start services
cd "$PROJECT_ROOT/docker"

# Start services in detached mode
docker-compose up -d

echo ""
echo "✅ Airflow services started!"
echo ""
echo "Access Airflow UI at: http://localhost:8080"
echo "Username: airflow"
echo "Password: airflow"
echo ""
echo "To view logs: cd docker && docker-compose logs -f"
echo "To stop services: ./scripts/stop_airflow.sh"
echo ""

