# Quick Start Guide - PII NER Pipeline

Get the PII NER Pipeline up and running in about 15 minutes! This guide will walk you through setting up Apache Airflow with Docker and running your first pipeline.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (5 Steps)](#quick-setup-5-steps)
3. [Detailed Setup Instructions](#detailed-setup-instructions)
4. [Running Your First Pipeline](#running-your-first-pipeline)
5. [Verifying Results](#verifying-results)
6. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- [ ] **Docker Desktop** installed and running ([Download here](https://www.docker.com/products/docker-desktop))
- [ ] **Google Cloud Platform (GCP) Account** with billing enabled
- [ ] **GCP Project** created
- [ ] At least **4GB RAM** and **10GB disk space** available
- [ ] **Command-line access** (Terminal on Mac/Linux, PowerShell on Windows)

### Verify Docker Installation

```bash
docker --version
docker-compose --version
# or
docker compose version
```

If these commands work, you're good to go! ✅

---

## Quick Setup (5 Steps)

### Step 1: Clone/Download the Project

```bash
# Navigate to your desired directory
cd ~/Documents  # or wherever you want the project

# If using git:
git clone <repository-url> Guardrails
cd Guardrails

# Or extract the project folder if you received a zip file
```

### Step 2: Set Up Google Cloud Platform

#### 2.1 Create GCS Buckets

You'll need two GCS buckets (or one bucket with different paths). Create them in the [GCP Console](https://console.cloud.google.com/storage/browser):

```bash
# Option A: Using gcloud CLI (recommended)
gcloud storage buckets create gs://your-project-name-data --location=us-central1

# Option B: Via GCP Console
# 1. Go to https://console.cloud.google.com/storage/browser
# 2. Click "Create Bucket"
# 3. Name it (e.g., "your-project-name-data")
# 4. Choose location (e.g., us-central1)
# 5. Click "Create"
```

**Note:** You can use the same bucket with different paths, or create separate buckets. We'll configure this next.

#### 2.2 Create Service Account

1. Go to [GCP Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click **"Create Service Account"**
3. Enter details:
   - **Name:** `airflow-pii-pipeline`
   - **Description:** `Service account for PII NER Pipeline`
4. Click **"Create and Continue"**

#### 2.3 Grant Permissions

Grant these roles to the service account:
- **Storage Object Admin** (to read/write files)
- **Storage Object Viewer** (to list objects)

1. In the "Grant this service account access to project" section:
   - Click **"Add Role"**
   - Search for and select **"Storage Object Admin"**
   - Click **"Add Another Role"**
   - Search for and select **"Storage Object Viewer"**
2. Click **"Continue"** then **"Done"**

#### 2.4 Create and Download Service Account Key

1. Click on the service account you just created
2. Go to the **"Keys"** tab
3. Click **"Add Key"** → **"Create new key"**
4. Select **"JSON"** format
5. Click **"Create"** (the file will download automatically)

### Step 3: Place Service Account Key

```bash
# Make sure you're in the project directory
cd /path/to/Guardrails

# Create the gcp_keys directory (if it doesn't exist)
mkdir -p gcp_keys

# Copy your downloaded key file to the project
# Replace '~/Downloads/your-key-name-xxxxx.json' with your actual file path
cp ~/Downloads/your-key-name-xxxxx.json gcp_keys/service-account-key.json

# Verify the file exists
ls -la gcp_keys/service-account-key.json
```

**Important:** 
- The file **must** be named exactly `service-account-key.json`
- Keep this file secure! It's excluded from git by default

### Step 4: Configure the Pipeline

Edit the configuration file with your GCP bucket names:

```bash
# Open the config file in your favorite editor
nano config/pipeline_config.yaml
# or
code config/pipeline_config.yaml  # VS Code
# or
open -e config/pipeline_config.yaml  # Mac TextEdit
```

Update the `gcs` section:

```yaml
gcs:
  source_bucket: "your-actual-bucket-name"        # ← Change this!
  destination_bucket: "your-actual-bucket-name"   # ← Change this! (can be same bucket)
  source_path: "raw/pii-masking-200k"
  destination_path: "processed"
  eda_output_path: "eda"
```

**Example:**
```yaml
gcs:
  source_bucket: "my-company-data-bucket"
  destination_bucket: "my-company-data-bucket"  # Using same bucket
  source_path: "raw/pii-masking-200k"
  destination_path: "processed"
  eda_output_path: "eda"
```

**Optional:** Update the GCP project ID in `docker-compose.yml` if needed:

```bash
# Open docker-compose.yml and find line 16
# Update the project ID if different from 'guardrails-477420'
nano docker-compose.yml
```

Look for this line and update your project ID:
```yaml
AIRFLOW_CONN_GOOGLE_CLOUD_DEFAULT: 'google-cloud-platform://?extra__google_cloud_platform__key_path=/opt/airflow/gcp_keys/service-account-key.json&extra__google_cloud_platform__project=YOUR-PROJECT-ID'
```

### Step 5: Initialize and Start Airflow

```bash
# Make setup scripts executable (if needed)
chmod +x setup_airflow.sh start_airflow.sh stop_airflow.sh

# Run the setup script
./setup_airflow.sh
```

This will:
- ✅ Check Docker installation
- ✅ Create necessary directories
- ✅ Initialize Airflow database
- ✅ Set up the environment

**Important:** If you see a warning about the service account key, make sure you completed Step 3 correctly!

Then start Airflow:

```bash
./start_airflow.sh
```

This will start all services. Wait about 30-60 seconds for everything to initialize.

---

## Detailed Setup Instructions

### Understanding the Project Structure

```
Guardrails/
├── dags/
│   └── pii_ner_pipeline.py      # Main Airflow DAG
├── plugins/
│   └── pii_processing/
│       ├── data_loader.py       # GCS data loading
│       ├── eda.py               # EDA analysis
│       ├── transformers.py      # NER transformation
│       └── gcs_utils.py         # GCS utilities
├── config/
│   └── pipeline_config.yaml     # Pipeline configuration
├── gcp_keys/                     # Service account keys (create this)
│   └── service-account-key.json # Your GCP key (not in git)
├── logs/                         # Airflow logs (auto-created)
├── docker-compose.yml           # Docker configuration
├── requirements.txt             # Python dependencies
└── QUICK_START.md              # This file!
```

### What the Pipeline Does

The pipeline performs these tasks in sequence:

1. **download_from_huggingface** - Downloads raw dataset from Hugging Face and uploads to GCS
2. **load_raw_data** - Loads the dataset from GCS
3. **perform_eda** - Generates exploratory data analysis statistics and visualizations
4. **transform_data** - Converts data to NER format (BIO tagging) and creates train/val/test splits
5. **upload_processed_data** - Uploads transformed data back to GCS

---

## Running Your First Pipeline

### Access Airflow UI

1. Open your web browser
2. Go to: **http://localhost:8080**
3. Login with:
   - **Username:** `airflow`
   - **Password:** `airflow`

### Enable and Trigger the DAG

1. In the Airflow UI, find the **`pii_ner_pipeline`** DAG in the list
2. If the toggle switch on the left is **OFF** (gray), click it to turn it **ON** (green)
3. Click the **play button (▶️)** next to the DAG name, or click on the DAG name and then click **"Trigger DAG"**

### Monitor Execution

1. Click on the DAG name to see the **Graph View**
2. You'll see the pipeline tasks and their status:
   - 🟡 **Light yellow** = Queued
   - 🔵 **Dark yellow** = Running
   - 🟢 **Green** = Success
   - 🔴 **Red** = Failed
3. Click on any task to see details and logs
4. Wait for all tasks to complete (usually 5-15 minutes depending on dataset size)

---

## Verifying Results

### Check Airflow Logs

1. In the Airflow UI, click on a completed task
2. Click **"Log"** button to see execution logs
3. Look for success messages like:
   - `"Successfully loaded X rows"`
   - `"Data upload completed successfully"`

### Check GCS Bucket

Verify files were created in your GCS bucket:

**Using gcloud CLI:**
```bash
# List files in processed directory
gsutil ls gs://your-bucket-name/processed/

# List files in EDA directory
gsutil ls gs://your-bucket-name/eda/
```

**Using GCP Console:**
1. Go to [GCP Console → Cloud Storage](https://console.cloud.google.com/storage/browser)
2. Navigate to your bucket
3. You should see:
   - `processed/train.jsonl`
   - `processed/val.jsonl`
   - `processed/test.jsonl`
   - `eda/` folder with statistics and visualizations

### Expected Output Structure

```
gs://your-bucket-name/
├── raw/
│   └── pii-masking-200k/
│       └── train.jsonl                    # Raw data from Hugging Face
├── processed/
│   ├── train.jsonl                        # Training split (NER format)
│   ├── val.jsonl                          # Validation split (NER format)
│   └── test.jsonl                         # Test split (NER format)
└── eda/
    ├── statistics.json                    # EDA statistics
    ├── entity_distribution.png            # Entity distribution chart
    └── [other visualization files]
```

---

## Common Issues & Troubleshooting

### ❌ Issue: "Docker command not found"

**Solution:**
- Install Docker Desktop from https://www.docker.com/products/docker-desktop
- Make sure Docker Desktop is running (check system tray/menu bar)
- Restart your terminal after installation

### ❌ Issue: "Cannot login to Airflow UI" or "Invalid credentials"

**Solution:**
If the default credentials (`airflow`/`airflow`) don't work, you may need to create a user. This happens on first setup:

```bash
# Stop services first
docker-compose down

# Create Airflow user (run this once)
export AIRFLOW_UID=$(id -u)
docker-compose run --rm airflow-webserver airflow users create \
    --username airflow \
    --firstname Airflow \
    --lastname Admin \
    --role Admin \
    --email airflow@example.com \
    --password airflow

# Start services again
docker-compose up -d
```

Then try logging in again with username: `airflow`, password: `airflow`.

### ❌ Issue: "Port 8080 already in use"

**Solution:**
Edit `docker-compose.yml` and change the port mapping:

```yaml
ports:
  - "8081:8080"  # Changed from 8080:8080
```

Then access Airflow at http://localhost:8081

Or, find and stop what's using port 8080:
```bash
# Mac/Linux
lsof -ti:8080 | xargs kill -9

# Windows (PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8080).OwningProcess | Stop-Process
```

### ❌ Issue: "Service account key not found"

**Solution:**
```bash
# Verify the file exists
ls -la gcp_keys/service-account-key.json

# If missing, copy your key file again
cp ~/Downloads/your-key-file.json gcp_keys/service-account-key.json

# Verify the filename is exactly: service-account-key.json
```

### ❌ Issue: "Permission denied" or "Access Denied" for GCS

**Solutions:**
1. **Verify service account permissions:**
   - Go to [GCP Console → IAM & Admin → IAM](https://console.cloud.google.com/iam-admin/iam)
   - Find your service account
   - Ensure it has **Storage Object Admin** role

2. **Verify bucket names:**
   - Check `config/pipeline_config.yaml`
   - Ensure bucket names match exactly (case-sensitive)
   - Verify buckets exist in your GCP project

3. **Test GCS access manually:**
   ```bash
   # Set credentials
   export GOOGLE_APPLICATION_CREDENTIALS=./gcp_keys/service-account-key.json
   
   # Test access
   gsutil ls gs://your-bucket-name/
   ```

### ❌ Issue: "DAG not appearing in Airflow UI"

**Solutions:**
1. **Check DAG syntax:**
   ```bash
   python dags/pii_ner_pipeline.py
   ```
   This should run without errors.

2. **Check scheduler logs:**
   ```bash
   docker-compose logs -f airflow-scheduler
   ```
   Look for import errors or syntax issues.

3. **Restart services:**
   ```bash
   ./stop_airflow.sh
   ./start_airflow.sh
   ```

4. **Verify directory structure:**
   - Ensure `dags/` folder is mounted correctly
   - Check `docker-compose.yml` volumes section

### ❌ Issue: "Task failed: ModuleNotFoundError"

**Solutions:**
1. **Check dependencies are installed:**
   - View logs: `docker-compose logs airflow-webserver`
   - Dependencies should auto-install from `requirements.txt`

2. **Manually install if needed:**
   ```bash
   docker-compose exec airflow-webserver pip install -r /opt/airflow/requirements.txt
   docker-compose restart
   ```

### ❌ Issue: "Out of memory" or pipeline is slow

**Solutions:**
1. **Increase Docker memory:**
   - Docker Desktop → Settings → Resources
   - Increase Memory to at least 4GB (8GB recommended)
   - Apply & Restart

2. **Reduce dataset size for testing:**
   - Edit `config/pipeline_config.yaml`
   - Test with a smaller sample first

### ❌ Issue: "spaCy model not found" warning

**Note:** This is usually not critical. The pipeline will use a basic tokenizer as fallback.

**To fix (optional):**
```bash
docker-compose exec airflow-webserver python -m spacy download en_core_web_sm
docker-compose restart
```

### Viewing Logs

**All services:**
```bash
docker-compose logs -f
```

**Specific service:**
```bash
docker-compose logs -f airflow-webserver
docker-compose logs -f airflow-scheduler
```

**Task-specific logs:**
- Best viewed in Airflow UI → DAG → Task Instance → Log

---

## Next Steps

### Customize the Pipeline

1. **Adjust data splits:**
   Edit `config/pipeline_config.yaml`:
   ```yaml
   data_splits:
     train_ratio: 0.8
     val_ratio: 0.1
     test_ratio: 0.1
   ```

2. **Add custom entity types:**
   ```yaml
   ner:
     entity_types:
       - "PERSON"
       - "ORGANIZATION"
       - "YOUR_CUSTOM_ENTITY"
   ```

3. **Modify EDA settings:**
   ```yaml
   eda:
     generate_visualizations: true
     statistics_to_generate:
       - "dataset_size"
       - "entity_distribution"
       # Add more...
   ```

### Schedule the Pipeline

Edit `config/pipeline_config.yaml`:
```yaml
airflow:
  schedule_interval: "0 2 * * *"  # Run daily at 2 AM
  # or
  schedule_interval: "@daily"
  # or keep null for manual triggers only
```

### Production Deployment

For production, consider:
- **Google Cloud Composer** (managed Airflow)
- **Cloud Run** for containerized execution
- **Cloud Scheduler** for cron-based triggers

See `SETUP_GUIDE.md` for Cloud Composer setup instructions.

---

## Useful Commands

```bash
# Start Airflow
./start_airflow.sh

# Stop Airflow
./stop_airflow.sh

# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f airflow-scheduler

# Restart services
docker-compose restart

# Check service status
docker-compose ps

# Stop and remove everything (including data)
docker-compose down -v

# Rebuild containers
docker-compose up -d --build

# Access Airflow container shell
docker-compose exec airflow-webserver bash

# Check Airflow connections
docker-compose exec airflow-webserver airflow connections list
```

---

## Getting Help

If you're stuck:

1. **Check the logs:**
   - Airflow UI → Task logs
   - `docker-compose logs -f`

2. **Verify configuration:**
   - `config/pipeline_config.yaml`
   - `docker-compose.yml`
   - Service account key location

3. **Review detailed documentation:**
   - `README.md` - Full project documentation
   - `SETUP_GUIDE.md` - Detailed setup instructions

4. **Common fixes:**
   - Restart services: `./stop_airflow.sh && ./start_airflow.sh`
   - Verify GCP permissions
   - Check bucket names match exactly

---

## Quick Reference Checklist

Before running the pipeline, verify:

- [ ] Docker Desktop is running
- [ ] Service account key is at `gcp_keys/service-account-key.json`
- [ ] GCS buckets exist and are accessible
- [ ] `config/pipeline_config.yaml` has correct bucket names
- [ ] `docker-compose.yml` has correct project ID (if different)
- [ ] Airflow is accessible at http://localhost:8080
- [ ] DAG appears in Airflow UI
- [ ] All services are running: `docker-compose ps`

---

## Success! 🎉

If you've made it here and your pipeline ran successfully, congratulations! 

Your transformed NER dataset is now available in:
- **Processed splits:** `gs://your-bucket/processed/`
- **EDA results:** `gs://your-bucket/eda/`

You can now use these files to train your NER models!

---

**Questions or issues?** Check the logs, review the configuration, or consult the full `README.md` and `SETUP_GUIDE.md` files.
