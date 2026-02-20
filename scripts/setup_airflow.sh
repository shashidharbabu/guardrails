#!/bin/bash

# Airflow Setup Script
# This script helps set up Airflow for the PII NER Pipeline

set -e

echo "=========================================="
echo "Airflow Setup for PII NER Pipeline"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p gcp_keys
echo "✅ Directories created"
echo ""

# Check for GCP service account key
if [ ! -f "gcp_keys/service-account-key.json" ]; then
    echo "⚠️  GCP Service Account key not found!"
    echo "   Please place your GCP service account JSON key at:"
    echo "   gcp_keys/service-account-key.json"
    echo ""
    echo "   You can create a service account key at:"
    echo "   https://console.cloud.google.com/iam-admin/serviceaccounts"
    echo ""
    read -p "Press Enter to continue after adding the key, or Ctrl+C to exit..."
fi

# Set Airflow UID
export AIRFLOW_UID=$(id -u)
echo "Setting AIRFLOW_UID to: $AIRFLOW_UID"
echo ""

# Note: Airflow database will be initialized automatically when services start
echo "ℹ️  Airflow database will be initialized automatically when services start"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure your GCP service account key is at: gcp_keys/service-account-key.json"
echo "2. Update config/pipeline_config.yaml with your GCS bucket names"
echo "3. Start Airflow with: ./scripts/start_airflow.sh"
echo "   Or manually: cd docker && docker-compose up -d"
echo "4. Access Airflow UI at: http://localhost:8080"
echo "   Username: airflow"
echo "   Password: airflow"
echo ""

