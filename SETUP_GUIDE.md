# Airflow Setup Guide

This guide will help you set up Apache Airflow for the PII NER Pipeline project.

## Option 1: Docker Setup (Recommended - Easiest)

### Step 1: Install Docker

**macOS:**
- Download and install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Launch Docker Desktop and ensure it's running

**Windows:**
- Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Launch Docker Desktop and ensure it's running

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 2: Prepare GCP Service Account Key

1. **Create Service Account** (if you don't have one):
   - Go to [GCP Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
   - Click "Create Service Account"
   - Name it (e.g., "airflow-pii-pipeline")
   - Click "Create and Continue"

2. **Grant Permissions**:
   - Add role: **Storage Object Admin**
   - Add role: **Storage Object Viewer**
   - Click "Continue" then "Done"

3. **Create Key**:
   - Click on the service account you just created
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose "JSON" format
   - Download the key file

4. **Place Key in Project**:
   ```bash
   mkdir -p gcp_keys
   cp ~/Downloads/your-service-account-key.json gcp_keys/service-account-key.json
   ```

### Step 3: Configure Pipeline

Edit `config/pipeline_config.yaml`:

```yaml
gcs:
  source_bucket: "your-actual-source-bucket"      # ← Change this
  destination_bucket: "your-actual-dest-bucket"   # ← Change this
  source_path: "raw/pii-masking-200k"
  destination_path: "processed"
  eda_output_path: "eda"
```

### Step 4: Run Setup Script

```bash
./setup_airflow.sh
```

This script will:
- Check Docker installation
- Create necessary directories
- Initialize Airflow database
- Set up the environment

### Step 5: Start Airflow

```bash
./start_airflow.sh
```

Or manually:
```bash
export AIRFLOW_UID=$(id -u)
docker-compose up -d
```

### Step 6: Access Airflow UI

1. Open browser: **http://localhost:8080**
2. Login:
   - Username: `airflow`
   - Password: `airflow`

### Step 7: Verify Setup

1. In Airflow UI, you should see the `pii_ner_pipeline` DAG
2. Toggle it ON (if it's paused)
3. Check that all tasks are visible

### Troubleshooting Docker Setup

**Issue: Permission denied**
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x *.sh
```

**Issue: Port 8080 already in use**
```bash
# Edit docker-compose.yml and change:
ports:
  - "8081:8080"  # Use different port
```

**Issue: Cannot connect to GCS**
- Verify service account key is at `gcp_keys/service-account-key.json`
- Check that the key has correct permissions
- Verify bucket names in `pipeline_config.yaml`

**View Logs:**
```bash
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
```

## Option 2: Local Installation (Without Docker)

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Airflow

```bash
# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Install Airflow
pip install apache-airflow[google]==2.7.0

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Step 3: Configure Airflow

Edit `~/airflow/airflow.cfg` or set environment variables:

```bash
export AIRFLOW__CORE__DAGS_FOLDER=/Users/spartan/Documents/Guardrails/dags
export AIRFLOW__CORE__PLUGINS_FOLDER=/Users/spartan/Documents/Guardrails/plugins
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Step 4: Set Up GCP Connection

```bash
airflow connections add google_cloud_default \
    --conn-type google-cloud-platform \
    --conn-extra '{"extra__google_cloud_platform__key_path": "/path/to/service-account-key.json", "extra__google_cloud_platform__project": "YOUR_PROJECT_ID"}'
```

### Step 5: Start Airflow

```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler
```

### Step 6: Access Airflow UI

Open **http://localhost:8080** and login with the credentials you created.

## Option 3: Google Cloud Composer (Production)

For production use, consider Google Cloud Composer:

1. **Create Composer Environment**:
   ```bash
   gcloud composer environments create pii-pipeline \
       --location us-central1 \
       --python-version 3 \
       --image-version composer-2.7.0-airflow-2.7.0
   ```

2. **Upload DAGs and Plugins**:
   ```bash
   gcloud composer environments storage dags import \
       --environment pii-pipeline \
       --location us-central1 \
       --source dags/
   
   gcloud composer environments storage plugins import \
       --environment pii-pipeline \
       --location us-central1 \
       --source plugins/
   ```

3. **Install Dependencies**:
   - Go to Composer UI → Environment → PyPI packages
   - Upload `requirements.txt` or install packages individually

4. **Configure GCP Connection**:
   - Composer automatically sets up `google_cloud_default` connection
   - Ensure service account has necessary permissions

## Next Steps

After Airflow is set up:

1. **Upload Dataset to GCS**:
   ```bash
   gsutil -m cp -r /path/to/dataset/* gs://your-source-bucket/raw/pii-masking-200k/
   ```

2. **Trigger the DAG**:
   - Go to Airflow UI
   - Find `pii_ner_pipeline` DAG
   - Click "Trigger DAG"

3. **Monitor Execution**:
   - Watch task progress in Airflow UI
   - Check logs for any errors
   - Verify outputs in GCS destination bucket

## Common Issues

### Issue: Module not found errors
**Solution**: Ensure plugins directory is correctly mounted/configured

### Issue: GCS authentication errors
**Solution**: 
- Verify service account key path
- Check key has Storage Object Admin role
- Verify GOOGLE_APPLICATION_CREDENTIALS is set

### Issue: DAG not appearing
**Solution**:
- Check DAG syntax: `python dags/pii_ner_pipeline.py` (should not error)
- Verify DAGs folder is correct
- Check Airflow logs for import errors

### Issue: Tasks failing
**Solution**:
- Check task logs in Airflow UI
- Verify GCS bucket names in config
- Ensure dataset exists in source bucket
- Check that all dependencies are installed

## Getting Help

If you encounter issues:
1. Check Airflow task logs
2. Verify all configuration files
3. Ensure GCP permissions are correct
4. Review the main README.md for more details

