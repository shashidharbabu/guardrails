"""
Airflow DAG for PII Dataset EDA and NER Transformation Pipeline

This DAG performs:
1. Load raw dataset from GCS
2. Perform Exploratory Data Analysis (EDA)
3. Transform data to NER format (BIO tagging)
4. Upload processed data back to GCS
"""
import os
import sys
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# Add plugins to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'plugins'))
from pii_processing.data_loader import DataLoader
from pii_processing.eda import EDA
from pii_processing.transformers import NERTransformer
from pii_processing.gcs_utils import GCSUtils

logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'pipeline_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Default arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email': CONFIG['airflow']['default_args'].get('email_on_failure', []),
    'email_on_failure': CONFIG['airflow']['default_args'].get('email_on_failure', False),
    'email_on_retry': CONFIG['airflow']['default_args'].get('email_on_retry', False),
    'retries': CONFIG['airflow']['default_args'].get('retries', 3),
    'retry_delay': timedelta(minutes=CONFIG['airflow']['default_args'].get('retry_delay_minutes', 5)),
}

# Create DAG
dag = DAG(
    'pii_ner_pipeline',
    default_args=default_args,
    description='PII Dataset EDA and NER Transformation Pipeline',
    schedule_interval=CONFIG['airflow'].get('schedule_interval'),
    start_date=days_ago(1),
    catchup=CONFIG['airflow'].get('catchup', False),
    max_active_runs=CONFIG['airflow'].get('max_active_runs', 1),
    tags=['pii', 'ner', 'eda', 'gcp'],
)


def download_from_huggingface(**context):
    """Download dataset from Hugging Face and upload raw files to GCS if missing."""
    logger.info("Starting Hugging Face download task")
    try:
        dataset_name = CONFIG.get('huggingface', {}).get('dataset_name', 'ai4privacy/pii-masking-200k')
        splits = CONFIG.get('huggingface', {}).get('splits', ['train'])
        download_if_missing = CONFIG.get('huggingface', {}).get('download_if_missing', True)

        source_bucket = CONFIG['gcs']['source_bucket']
        source_path = CONFIG['gcs']['source_path']

        data_loader = DataLoader(gcp_conn_id='google_cloud_default')
        gcs_utils = GCSUtils(gcp_conn_id='google_cloud_default')

        # Check existing files
        existing = set(gcs_utils.list_files(source_bucket, source_path))
        logger.info(f"Existing objects under {source_path}: {len(existing)}")

        temp_dir = tempfile.mkdtemp()
        uploaded = []

        for split in splits:
            target_object = f"{source_path}/{split}.jsonl"
            already_exists = any(obj.endswith(f"{split}.jsonl") for obj in existing)

            if already_exists and not download_if_missing:
                logger.info(f"Skipping {split}: file already exists and download_if_missing is False")
                continue

            logger.info(f"Loading split '{split}' from Hugging Face dataset '{dataset_name}'")
            try:
                ds = data_loader.load_huggingface_dataset(dataset_name=dataset_name, split=split)
            except ValueError as e:
                if "Unknown split" in str(e):
                    logger.warning(
                        "Split '%s' not found in dataset '%s'. Available splits may differ. Skipping.",
                        split,
                        dataset_name,
                    )
                    continue
                raise

            # Save to JSONL
            local_file = os.path.join(temp_dir, f"{split}.jsonl")
            logger.info(f"Saving split '{split}' to local file {local_file}")
            # Convert to pandas and write jsonl to avoid large memory overheads from datasets serialization
            df = ds.to_pandas()
            df.to_json(local_file, orient='records', lines=True, force_ascii=False)

            # Upload to GCS
            logger.info(f"Uploading {local_file} to gs://{source_bucket}/{target_object}")
            gcs_path = gcs_utils.upload_file(
                bucket_name=source_bucket,
                destination_object=target_object,
                source_file=local_file,
                content_type='application/jsonl'
            )
            uploaded.append(gcs_path)

        context['ti'].xcom_push(key='downloaded_files', value=uploaded)
        logger.info(f"Hugging Face download task completed, uploaded: {len(uploaded)} files")
        return {'status': 'success', 'uploaded_files': uploaded}

    except Exception as e:
        logger.error(f"Error downloading from Hugging Face: {e}")
        raise


def load_raw_data(**context):
    """Task 1: Load raw dataset from GCS"""
    logger.info("Starting data loading task")
    
    try:
        # Initialize data loader
        data_loader = DataLoader(gcp_conn_id='google_cloud_default')
        
        # Load data from GCS
        source_bucket = CONFIG['gcs']['source_bucket']
        source_path = CONFIG['gcs']['source_path']
        
        logger.info(f"Loading data from gs://{source_bucket}/{source_path}")
        data = data_loader.load_from_gcs(
            bucket_name=source_bucket,
            source_path=source_path,
            use_huggingface=False
        )
        
        # Validate schema
        validation_result = data_loader.validate_schema(data)
        if not validation_result['valid']:
            logger.warning(f"Schema validation issues: {validation_result['errors']}")
        
        # Store data in XCom for next tasks
        # In production, you might want to save to a temp location or use Airflow's XCom
        # For now, we'll save metadata and reload in next tasks
        context['ti'].xcom_push(key='data_shape', value={'rows': len(data), 'columns': list(data.columns)})
        context['ti'].xcom_push(key='source_bucket', value=source_bucket)
        context['ti'].xcom_push(key='source_path', value=source_path)
        
        # Save data reference (in production, save to temp GCS location)
        logger.info(f"Successfully loaded {len(data)} rows")
        return {'status': 'success', 'rows': len(data)}
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def perform_eda(**context):
    """Task 2: Perform Exploratory Data Analysis"""
    logger.info("Starting EDA task")
    
    try:
        # Get data info from previous task
        source_bucket = context['ti'].xcom_pull(key='source_bucket', task_ids='load_raw_data')
        source_path = context['ti'].xcom_pull(key='source_path', task_ids='load_raw_data')
        
        # Initialize modules
        data_loader = DataLoader(gcp_conn_id='google_cloud_default')
        eda = EDA(gcp_conn_id='google_cloud_default')
        
        # Reload data (in production, use shared storage or XCom for large data)
        logger.info("Reloading data for EDA")
        data = data_loader.load_from_gcs(
            bucket_name=source_bucket,
            source_path=source_path,
            use_huggingface=False
        )
        
        # Perform EDA
        destination_bucket = CONFIG['gcs']['destination_bucket']
        eda_output_path = CONFIG['gcs']['eda_output_path']
        
        stats = eda.perform_eda(
            data=data,
            output_bucket=destination_bucket,
            output_path=eda_output_path,
            generate_visualizations=CONFIG['eda']['generate_visualizations'],
            statistics_to_generate=CONFIG['eda']['statistics_to_generate']
        )
        
        # Store EDA results
        context['ti'].xcom_push(key='eda_stats', value=stats)
        logger.info("EDA completed successfully")
        return {'status': 'success', 'stats_keys': list(stats.keys())}
        
    except Exception as e:
        logger.error(f"Error performing EDA: {e}")
        raise


def transform_data(**context):
    """Task 3: Transform data to NER format"""
    logger.info("Starting data transformation task")
    
    try:
        # Get data info from previous task
        source_bucket = context['ti'].xcom_pull(key='source_bucket', task_ids='load_raw_data')
        source_path = context['ti'].xcom_pull(key='source_path', task_ids='load_raw_data')
        
        # Initialize modules
        data_loader = DataLoader(gcp_conn_id='google_cloud_default')
        
        # Reload data
        logger.info("Reloading data for transformation")
        data = data_loader.load_from_gcs(
            bucket_name=source_bucket,
            source_path=source_path,
            use_huggingface=False
        )
        
        # Initialize transformer
        transformer = NERTransformer(
            ner_format=CONFIG['ner']['format'],
            entity_types=CONFIG['ner']['entity_types'],
            tokenizer=CONFIG['ner']['tokenizer'],
            max_sequence_length=CONFIG['ner']['max_sequence_length']
        )
        
        # Detect text and entity columns (adjust based on actual dataset structure)
        text_column = 'text'
        entity_column = None
        annotations_column = None
        
        # Try to find appropriate columns
        for col in data.columns:
            if 'text' in col.lower() or 'content' in col.lower():
                text_column = col
            if 'entity' in col.lower() or 'annotation' in col.lower():
                if 'annotation' in col.lower():
                    annotations_column = col
                else:
                    entity_column = col
        
        logger.info(f"Using text_column: {text_column}, entity_column: {entity_column}, annotations_column: {annotations_column}")
        
        # Transform to NER format
        logger.info("Converting to NER format")
        ner_examples = transformer.transform_to_ner_format(
            data=data,
            text_column=text_column,
            entity_column=entity_column,
            annotations_column=annotations_column
        )
        
        # Clean data
        logger.info("Cleaning data")
        cleaned_examples = transformer.clean_data(
            ner_examples=ner_examples,
            min_text_length=CONFIG['data_cleaning']['min_text_length'],
            max_text_length=CONFIG['data_cleaning']['max_text_length'],
            remove_invalid_annotations=CONFIG['data_cleaning']['remove_invalid_annotations']
        )
        
        # Create splits
        logger.info("Creating train/val/test splits")
        splits = transformer.create_splits(
            ner_examples=cleaned_examples,
            train_ratio=CONFIG['data_splits']['train_ratio'],
            val_ratio=CONFIG['data_splits']['val_ratio'],
            test_ratio=CONFIG['data_splits']['test_ratio'],
            random_seed=CONFIG['data_splits']['random_seed'],
            stratify=CONFIG['data_splits']['stratify']
        )
        
        # Save splits to temp location
        import tempfile
        temp_dir = tempfile.mkdtemp()
        file_paths = transformer.save_splits(splits, temp_dir, format='jsonl')
        
        # Store file paths for upload task
        context['ti'].xcom_push(key='split_files', value=file_paths)
        context['ti'].xcom_push(key='temp_dir', value=temp_dir)
        context['ti'].xcom_push(key='split_sizes', value={
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        })
        
        logger.info("Data transformation completed successfully")
        return {'status': 'success', 'splits': list(splits.keys())}
        
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise


def upload_processed_data(**context):
    """Task 4: Upload processed data to GCS"""
    logger.info("Starting data upload task")
    
    try:
        # Get file paths from previous task
        split_files = context['ti'].xcom_pull(key='split_files', task_ids='transform_data')
        temp_dir = context['ti'].xcom_pull(key='temp_dir', task_ids='transform_data')
        
        if not split_files:
            raise ValueError("No split files found from transformation task")
        
        # Initialize GCS utils
        gcs_utils = GCSUtils(gcp_conn_id='google_cloud_default')
        
        # Upload each split
        destination_bucket = CONFIG['gcs']['destination_bucket']
        destination_path = CONFIG['gcs']['destination_path']
        
        uploaded_files = []
        for split_name, file_path in split_files.items():
            gcs_object_path = f"{destination_path}/{split_name}.jsonl"
            
            logger.info(f"Uploading {split_name} split to gs://{destination_bucket}/{gcs_object_path}")
            gcs_path = gcs_utils.upload_file(
                bucket_name=destination_bucket,
                destination_object=gcs_object_path,
                source_file=file_path
            )
            uploaded_files.append(gcs_path)
        
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
            logger.info("Cleaned up temp directory")
        
        context['ti'].xcom_push(key='uploaded_files', value=uploaded_files)
        logger.info("Data upload completed successfully")
        return {'status': 'success', 'uploaded_files': len(uploaded_files)}
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise


# Define tasks
load_data_task = PythonOperator(
    task_id='load_raw_data',
    python_callable=load_raw_data,
    dag=dag,
)

eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='upload_processed_data',
    python_callable=upload_processed_data,
    dag=dag,
)

# Define task dependencies
download_task = PythonOperator(
    task_id='download_from_huggingface',
    python_callable=download_from_huggingface,
    dag=dag,
)

download_task >> load_data_task >> eda_task >> transform_task >> upload_task

