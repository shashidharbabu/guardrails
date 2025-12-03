#!/bin/bash

# Start Airflow Services

set -e

echo "Starting Airflow services..."
export AIRFLOW_UID=$(id -u)

# Start services in detached mode
docker-compose up -d

echo ""
echo "✅ Airflow services started!"
echo ""
echo "Access Airflow UI at: http://localhost:8080"
echo "Username: airflow"
echo "Password: airflow"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"
echo ""

