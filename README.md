# PII Dataset EDA and NER Transformation Pipeline

This project provides an Apache Airflow DAG pipeline for performing Exploratory Data Analysis (EDA) and transforming the `ai4privacy/pii-masking-200k` dataset from Google Cloud Storage (GCS) into a Named Entity Recognition (NER) ready format.

## Overview

The pipeline performs the following operations:
1. **Load Raw Data**: Downloads and loads the PII dataset from a GCS bucket
2. **Exploratory Data Analysis**: Generates statistics and visualizations about the dataset
3. **Data Transformation**: Converts the dataset to NER format (BIO tagging) and creates train/val/test splits
4. **Upload Processed Data**: Uploads the transformed dataset back to GCS

## Project Structure

```
Guardrails/
├── dags/
│   └── pii_ner_pipeline.py          # Main Airflow DAG
├── plugins/
│   └── pii_processing/
│       ├── __init__.py
│       ├── data_loader.py           # GCS data loading utilities
│       ├── eda.py                   # EDA analysis functions
│       ├── transformers.py          # NER transformation logic
│       └── gcs_utils.py             # GCS upload/download helpers
├── config/
│   └── pipeline_config.yaml        # Pipeline configuration
├── gcp_keys/                        # GCP service account keys (create this)
├── logs/                            # Airflow logs (auto-created)
├── docker-compose.yml               # Docker Compose configuration
├── Dockerfile                       # Airflow Docker image
├── setup_airflow.sh                 # Setup script
├── start_airflow.sh                 # Start script
├── stop_airflow.sh                  # Stop script
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
└── README.md
```

## Prerequisites

1. **Docker and Docker Compose**
   - Docker Desktop or Docker Engine installed
   - Docker Compose v2.0+ (usually included with Docker Desktop)
   - At least 4GB RAM and 10GB disk space available

2. **Google Cloud Platform (GCP) Account**
   - GCP project with billing enabled
   - GCS buckets for source and destination data
   - Service account with appropriate permissions

3. **GCP Service Account Key**
   - JSON key file for authentication
   - Must have Storage Object Admin role

## Quick Start with Docker (Recommended)

### 1. Install Docker

If you don't have Docker installed:
- **macOS/Windows**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow [Docker installation guide](https://docs.docker.com/engine/install/)

### 2. Set Up GCP Service Account

1. Go to [GCP Console → IAM & Admin → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Create a new service account or use an existing one
3. Grant the following roles:
   - **Storage Object Admin** (for reading/writing to GCS)
   - **Storage Object Viewer** (for listing objects)
4. Create and download a JSON key file
5. Place the key file in the project directory:
   ```bash
   mkdir -p gcp_keys
   cp /path/to/your-service-account-key.json gcp_keys/service-account-key.json
   ```

### 3. Configure Pipeline

Edit `config/pipeline_config.yaml` with your GCS bucket names:

```yaml
gcs:
  source_bucket: "your-actual-source-bucket-name"
  destination_bucket: "your-actual-destination-bucket-name"
  source_path: "raw/pii-masking-200k"
```

### 4. Initialize and Start Airflow

Run the setup script:

```bash
./setup_airflow.sh
```

This will:
- Check Docker installation
- Create necessary directories
- Initialize Airflow database
- Set up environment

Then start Airflow:

```bash
./start_airflow.sh
```

Or manually:

```bash
export AIRFLOW_UID=$(id -u)
docker-compose up -d
```

### 5. Access Airflow UI

1. Open your browser and go to: **http://localhost:8080**
2. Login with:
   - **Username**: `airflow`
   - **Password**: `airflow`

### 6. Stop Airflow

When done, stop the services:

```bash
./stop_airflow.sh
```

Or manually:

```bash
docker-compose down
```

## Manual Setup (Without Docker)

If you prefer to install Airflow locally without Docker:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install spaCy Model (Optional but Recommended)

For better tokenization, install the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

### 3. Initialize Airflow

```bash
# Set Airflow home directory
export AIRFLOW_HOME=~/airflow

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

### 4. Configure GCP

#### Create GCS Buckets

```bash
# Create source bucket for raw data
gsutil mb -p YOUR_PROJECT_ID gs://your-source-bucket-name

# Create destination bucket for processed data
gsutil mb -p YOUR_PROJECT_ID gs://your-destination-bucket-name
```

#### Set Up Service Account

1. Create a service account in GCP Console
2. Grant the following roles:
   - Storage Object Admin (for reading/writing to GCS)
   - Storage Object Viewer (for listing objects)
3. Download the service account key JSON file
4. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

#### Configure Airflow Connection

In Airflow UI or using CLI:

```bash
airflow connections add google_cloud_default \
    --conn-type google-cloud-platform \
    --conn-extra '{"extra__google_cloud_platform__key_path": "/path/to/service-account-key.json", "extra__google_cloud_platform__project": "YOUR_PROJECT_ID"}'
```

### 4. Configure Pipeline

Edit `config/pipeline_config.yaml` with your settings:

```yaml
gcs:
  source_bucket: "your-source-bucket-name"
  destination_bucket: "your-destination-bucket-name"
  source_path: "raw/pii-masking-200k"
  destination_path: "processed"
  eda_output_path: "eda"
```

### 5. Prepare Dataset

Upload the PII dataset to your source GCS bucket. The dataset should be in one of the following formats:
- JSON
- JSONL
- Parquet
- CSV

Example:

```bash
# If downloading from Hugging Face
python -c "from datasets import load_dataset; ds = load_dataset('ai4privacy/pii-masking-200k'); ds.save_to_disk('local_path')"

# Upload to GCS
gsutil -m cp -r local_path/* gs://your-source-bucket-name/raw/pii-masking-200k/
```

## Usage

### Deploy to Airflow

1. **Cloud Composer**:
   - Upload the `dags/` and `plugins/` directories to your Composer environment's DAGs folder
   - Upload `config/` directory to the same location
   - Ensure `requirements.txt` is installed in the Composer environment

2. **Self-managed Airflow**:
   - Copy `dags/` to your Airflow `DAGS_FOLDER`
   - Copy `plugins/` to your Airflow `PLUGINS_FOLDER`
   - Copy `config/` to a location accessible by Airflow

### Run the DAG

1. Open the Airflow UI
2. Find the `pii_ner_pipeline` DAG
3. Toggle it ON
4. Click "Trigger DAG" to run manually

### Monitor Execution

- View task logs in the Airflow UI
- Check GCS buckets for output files
- Review EDA outputs in the `eda/` folder of the destination bucket

## Output Structure

After successful execution, your destination GCS bucket will contain:

```
gs://your-destination-bucket/
├── eda/
│   ├── statistics.json              # EDA statistics
│   ├── entity_distribution.png      # Entity distribution chart
│   ├── text_length_analysis.png     # Text length analysis
│   └── missing_values_heatmap.png   # Missing values visualization
└── processed/
    ├── train.jsonl                  # Training split (BIO format)
    ├── val.jsonl                    # Validation split (BIO format)
    └── test.jsonl                   # Test split (BIO format)
```

## NER Format

The transformed data is in JSONL format with BIO tagging. Each line contains:

```json
{
  "tokens": ["John", "Doe", "lives", "in", "New", "York"],
  "tags": ["B-PERSON", "I-PERSON", "O", "O", "B-LOCATION", "I-LOCATION"],
  "text": "John Doe lives in New York",
  "entities": [
    {"start": 0, "end": 9, "label": "PERSON"},
    {"start": 20, "end": 28, "label": "LOCATION"}
  ]
}
```

## Configuration Options

### Data Splits

Configure train/validation/test ratios in `pipeline_config.yaml`:

```yaml
data_splits:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  random_seed: 42
  stratify: true
```

### NER Settings

```yaml
ner:
  format: "BIO"  # or "IOB2"
  entity_types:
    - "PERSON"
    - "ORGANIZATION"
    # ... add more entity types
  tokenizer: "spacy"  # or "nltk"
  max_sequence_length: 512
```

### EDA Settings

```yaml
eda:
  generate_visualizations: true
  statistics_to_generate:
    - "dataset_size"
    - "entity_distribution"
    - "text_length_stats"
    - "missing_values"
    - "entity_co_occurrence"
```

## Troubleshooting

### Common Issues

1. **GCS Permission Errors**
   - Verify service account has Storage Object Admin role
   - Check that `GOOGLE_APPLICATION_CREDENTIALS` is set correctly

2. **Dataset Format Issues**
   - Ensure dataset is in supported format (JSON, JSONL, Parquet, CSV)
   - Check that text and entity columns exist in the dataset

3. **Memory Issues**
   - For large datasets, consider processing in chunks
   - Increase Airflow worker memory if using Cloud Composer

4. **Tokenization Errors**
   - Install spaCy model: `python -m spacy download en_core_web_sm`
   - Or switch to basic tokenizer in config

### Logs

Check Airflow task logs for detailed error messages:
- Airflow UI → DAG → Task Instance → Log

## Development

### Local Testing

Test individual modules:

```python
from plugins.pii_processing.data_loader import DataLoader
from plugins.pii_processing.eda import EDA
from plugins.pii_processing.transformers import NERTransformer

# Test data loading
loader = DataLoader()
data = loader.load_from_gcs(bucket_name="your-bucket", source_path="path/to/data")

# Test EDA
eda = EDA()
stats = eda.perform_eda(data, output_bucket="dest-bucket", output_path="eda")

# Test transformation
transformer = NERTransformer()
ner_examples = transformer.transform_to_ner_format(data)
```

### Adding Custom Entity Types

1. Update `config/pipeline_config.yaml`:
```yaml
ner:
  entity_types:
    - "YOUR_CUSTOM_ENTITY"
```

2. The transformer will automatically handle the new entity type in BIO tagging

## License

This project is provided as-is for processing PII datasets for NER model training.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review Airflow task logs
3. Verify GCP permissions and configuration

