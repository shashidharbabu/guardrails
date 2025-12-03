"""
GCS Utilities for uploading and downloading files to/from Google Cloud Storage
"""
import os
import logging
from typing import Optional, List
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
from airflow.providers.google.cloud.hooks.gcs import GCSHook

logger = logging.getLogger(__name__)


class GCSUtils:
    """Utility class for GCS operations"""
    
    def __init__(self, gcp_conn_id: str = 'google_cloud_default'):
        """
        Initialize GCS utilities
        
        Args:
            gcp_conn_id: Airflow connection ID for GCP
        """
        self.gcp_conn_id = gcp_conn_id
        self.hook = GCSHook(gcp_conn_id=gcp_conn_id)
    
    def download_file(
        self,
        bucket_name: str,
        source_object: str,
        destination_file: str,
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> str:
        """
        Download a file from GCS bucket
        
        Args:
            bucket_name: Name of the GCS bucket
            source_object: Path to object in bucket
            destination_file: Local path to save the file
            chunk_size: Size of chunks for downloading large files
            
        Returns:
            Path to downloaded file
        """
        try:
            logger.info(f"Downloading {bucket_name}/{source_object} to {destination_file}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_file) if os.path.dirname(destination_file) else '.', exist_ok=True)
            
            # Download file
            self.hook.download(
                bucket_name=bucket_name,
                object_name=source_object,
                filename=destination_file
            )
            
            logger.info(f"Successfully downloaded to {destination_file}")
            return destination_file
            
        except GoogleCloudError as e:
            logger.error(f"Error downloading file from GCS: {e}")
            raise
    
    def upload_file(
        self,
        bucket_name: str,
        destination_object: str,
        source_file: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to GCS bucket
        
        Args:
            bucket_name: Name of the GCS bucket
            destination_object: Path to object in bucket
            source_file: Local path to file to upload
            content_type: MIME type of the file
            
        Returns:
            GCS path to uploaded file
        """
        try:
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"Source file not found: {source_file}")
            
            logger.info(f"Uploading {source_file} to {bucket_name}/{destination_object}")
            
            # Determine content type if not provided
            if content_type is None:
                content_type = self._get_content_type(source_file)
            
            # Upload file
            self.hook.upload(
                bucket_name=bucket_name,
                object_name=destination_object,
                filename=source_file,
                mime_type=content_type
            )
            
            gcs_path = f"gs://{bucket_name}/{destination_object}"
            logger.info(f"Successfully uploaded to {gcs_path}")
            return gcs_path
            
        except (GoogleCloudError, FileNotFoundError) as e:
            logger.error(f"Error uploading file to GCS: {e}")
            raise
    
    def upload_directory(
        self,
        bucket_name: str,
        destination_prefix: str,
        source_directory: str,
        pattern: Optional[str] = None
    ) -> List[str]:
        """
        Upload all files from a directory to GCS
        
        Args:
            bucket_name: Name of the GCS bucket
            destination_prefix: Prefix path in bucket
            source_directory: Local directory to upload
            pattern: Optional glob pattern to filter files
            
        Returns:
            List of GCS paths to uploaded files
        """
        uploaded_files = []
        source_path = Path(source_directory)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_directory}")
        
        # Find files to upload
        if pattern:
            files = list(source_path.glob(pattern))
        else:
            files = [f for f in source_path.rglob('*') if f.is_file()]
        
        for file_path in files:
            # Calculate relative path
            relative_path = file_path.relative_to(source_path)
            destination_object = f"{destination_prefix}/{relative_path}".replace('\\', '/')
            
            try:
                gcs_path = self.upload_file(
                    bucket_name=bucket_name,
                    destination_object=destination_object,
                    source_file=str(file_path)
                )
                uploaded_files.append(gcs_path)
            except Exception as e:
                logger.error(f"Error uploading {file_path}: {e}")
                raise
        
        logger.info(f"Uploaded {len(uploaded_files)} files to {bucket_name}/{destination_prefix}")
        return uploaded_files
    
    def list_files(
        self,
        bucket_name: str,
        prefix: str,
        delimiter: Optional[str] = None
    ) -> List[str]:
        """
        List files in GCS bucket with given prefix
        
        Args:
            bucket_name: Name of the GCS bucket
            prefix: Prefix to filter objects
            delimiter: Optional delimiter for directory-like listing
            
        Returns:
            List of object names
        """
        try:
            objects = self.hook.list(
                bucket_name=bucket_name,
                prefix=prefix,
                delimiter=delimiter
            )
            return list(objects) if objects else []
        except GoogleCloudError as e:
            logger.error(f"Error listing files in GCS: {e}")
            raise
    
    def file_exists(
        self,
        bucket_name: str,
        object_name: str
    ) -> bool:
        """
        Check if a file exists in GCS bucket
        
        Args:
            bucket_name: Name of the GCS bucket
            object_name: Path to object in bucket
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            return self.hook.exists(bucket_name=bucket_name, object_name=object_name)
        except GoogleCloudError as e:
            logger.error(f"Error checking file existence in GCS: {e}")
            return False
    
    def delete_file(
        self,
        bucket_name: str,
        object_name: str
    ) -> bool:
        """
        Delete a file from GCS bucket
        
        Args:
            bucket_name: Name of the GCS bucket
            object_name: Path to object in bucket
            
        Returns:
            True if deleted successfully
        """
        try:
            self.hook.delete(bucket_name=bucket_name, object_name=object_name)
            logger.info(f"Deleted {bucket_name}/{object_name}")
            return True
        except GoogleCloudError as e:
            logger.error(f"Error deleting file from GCS: {e}")
            raise
    
    @staticmethod
    def _get_content_type(file_path: str) -> str:
        """
        Determine content type based on file extension
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        extension = Path(file_path).suffix.lower()
        content_types = {
            '.json': 'application/json',
            '.jsonl': 'application/jsonl',
            '.parquet': 'application/parquet',
            '.csv': 'text/csv',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.txt': 'text/plain',
            '.yaml': 'application/x-yaml',
            '.yml': 'application/x-yaml'
        }
        return content_types.get(extension, 'application/octet-stream')

