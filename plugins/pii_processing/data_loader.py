"""
Data Loader Module for loading datasets from GCS buckets
Supports multiple formats: JSON, JSONL, Parquet
"""
import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
import pandas as pd
from datasets import Dataset, load_dataset
from pii_processing.gcs_utils import GCSUtils

logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading datasets from GCS"""
    
    def __init__(self, gcp_conn_id: str = 'google_cloud_default'):
        """
        Initialize DataLoader
        
        Args:
            gcp_conn_id: Airflow connection ID for GCP
        """
        self.gcs_utils = GCSUtils(gcp_conn_id=gcp_conn_id)
        self.temp_dir = None
    
    def load_from_gcs(
        self,
        bucket_name: str,
        source_path: str,
        format: Optional[str] = None,
        use_huggingface: bool = False
    ) -> Union[pd.DataFrame, Dataset]:
        """
        Load dataset from GCS bucket
        
        Args:
            bucket_name: Name of the GCS bucket
            source_path: Path to dataset in bucket (can be file or directory)
            format: File format (json, jsonl, parquet, csv). Auto-detect if None
            use_huggingface: If True, return HuggingFace Dataset, else pandas DataFrame
            
        Returns:
            pandas DataFrame or HuggingFace Dataset
        """
        try:
            # Create temporary directory for downloads
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Check if source_path is a file or directory
            if self.gcs_utils.file_exists(bucket_name, source_path):
                # Single file
                local_file = self._download_file(bucket_name, source_path)
                format = format or self._detect_format(source_path)
                data = self._load_file(local_file, format, use_huggingface)
            else:
                # Directory - list and download all files
                files = self.gcs_utils.list_files(bucket_name, source_path)
                if not files:
                    raise FileNotFoundError(f"No files found at {bucket_name}/{source_path}")
                
                logger.info(f"Found {len(files)} files in {source_path}")
                data = self._load_directory(bucket_name, files, format, use_huggingface)
            
            logger.info(f"Successfully loaded dataset with {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading dataset from GCS: {e}")
            raise
        finally:
            # Cleanup temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temp directory")
    
    def _download_file(self, bucket_name: str, source_path: str) -> str:
        """Download a single file from GCS"""
        filename = os.path.basename(source_path)
        local_file = os.path.join(self.temp_dir, filename)
        self.gcs_utils.download_file(bucket_name, source_path, local_file)
        return local_file
    
    def _detect_format(self, file_path: str) -> str:
        """Detect file format from extension"""
        extension = Path(file_path).suffix.lower()
        format_map = {
            '.json': 'json',
            '.jsonl': 'jsonl',
            '.parquet': 'parquet',
            '.csv': 'csv'
        }
        return format_map.get(extension, 'json')
    
    def _load_file(
        self,
        file_path: str,
        format: str,
        use_huggingface: bool
    ) -> Union[pd.DataFrame, Dataset]:
        """Load a single file"""
        logger.info(f"Loading {file_path} as {format}")
        
        if format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data) if isinstance(data, list) else pd.json_normalize(data)
        elif format == 'jsonl':
            df = pd.read_json(file_path, lines=True)
        elif format == 'parquet':
            df = pd.read_parquet(file_path)
        elif format == 'csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if use_huggingface:
            return Dataset.from_pandas(df)
        return df
    
    def _load_directory(
        self,
        bucket_name: str,
        files: list,
        format: Optional[str],
        use_huggingface: bool
    ) -> Union[pd.DataFrame, Dataset]:
        """Load multiple files from directory"""
        dataframes = []
        
        for file_path in files:
            try:
                local_file = self._download_file(bucket_name, file_path)
                file_format = format or self._detect_format(file_path)
                df = self._load_file(local_file, file_format, False)  # Always load as DataFrame first
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid files could be loaded")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} files into single dataset")
        
        if use_huggingface:
            return Dataset.from_pandas(combined_df)
        return combined_df
    
    def validate_schema(
        self,
        data: Union[pd.DataFrame, Dataset],
        required_columns: Optional[list] = None,
        expected_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset schema
        
        Args:
            data: Dataset to validate (DataFrame or Dataset)
            required_columns: List of required column names
            expected_types: Dict mapping column names to expected types
            
        Returns:
            Validation result dictionary
        """
        # Convert to DataFrame if needed
        if isinstance(data, Dataset):
            df = data.to_pandas()
        else:
            df = data
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'columns': list(df.columns),
            'row_count': len(df),
            'missing_columns': [],
            'type_mismatches': []
        }
        
        # Check required columns
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                validation_result['valid'] = False
                validation_result['missing_columns'] = missing
                validation_result['errors'].append(f"Missing required columns: {missing}")
        
        # Check column types
        if expected_types:
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type.lower():
                        validation_result['type_mismatches'].append({
                            'column': col,
                            'expected': expected_type,
                            'actual': actual_type
                        })
                        validation_result['warnings'].append(
                            f"Type mismatch for {col}: expected {expected_type}, got {actual_type}"
                        )
        
        # Check for empty dataset
        if len(df) == 0:
            validation_result['valid'] = False
            validation_result['errors'].append("Dataset is empty")
        
        # Check for missing values in critical columns
        if required_columns:
            for col in required_columns:
                if col in df.columns and df[col].isna().all():
                    validation_result['warnings'].append(f"Column {col} has all missing values")
        
        logger.info(f"Schema validation completed: {'PASSED' if validation_result['valid'] else 'FAILED'}")
        return validation_result
    
    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Load dataset directly from HuggingFace Hub
        
        Args:
            dataset_name: Name of the dataset (e.g., 'ai4privacy/pii-masking-200k')
            split: Dataset split to load (e.g., 'train', 'test')
            cache_dir: Directory to cache the dataset
            
        Returns:
            HuggingFace Dataset
        """
        try:
            logger.info(f"Loading dataset {dataset_name} from HuggingFace Hub")
            dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
            logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise

