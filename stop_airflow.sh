#!/bin/bash

# Stop Airflow Services

echo "Stopping Airflow services..."
docker-compose down

echo ""
echo "✅ Airflow services stopped!"
echo ""

