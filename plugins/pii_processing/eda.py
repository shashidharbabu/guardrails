"""
Exploratory Data Analysis (EDA) Module
Generates statistics and visualizations for PII dataset
"""
import os
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from pii_processing.gcs_utils import GCSUtils

logger = logging.getLogger(__name__)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class EDA:
    """Class for performing EDA on PII datasets"""
    
    def __init__(self, gcp_conn_id: str = 'google_cloud_default'):
        """
        Initialize EDA module
        
        Args:
            gcp_conn_id: Airflow connection ID for GCP
        """
        self.gcs_utils = GCSUtils(gcp_conn_id=gcp_conn_id)
        self.temp_dir = None
        self.stats = {}
    
    def perform_eda(
        self,
        data: Union[pd.DataFrame, Dataset],
        output_bucket: str,
        output_path: str,
        generate_visualizations: bool = True,
        statistics_to_generate: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive EDA on the dataset
        
        Args:
            data: Dataset to analyze (DataFrame or Dataset)
            output_bucket: GCS bucket to save EDA outputs
            output_path: Path in bucket to save outputs
            generate_visualizations: Whether to generate visualizations
            statistics_to_generate: List of statistics to generate
            
        Returns:
            Dictionary containing all statistics
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(data, Dataset):
                df = data.to_pandas()
            else:
                df = data
            
            # Create temp directory for outputs
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temp directory for EDA outputs: {self.temp_dir}")
            
            # Default statistics to generate
            if statistics_to_generate is None:
                statistics_to_generate = [
                    'dataset_size',
                    'entity_distribution',
                    'text_length_stats',
                    'missing_values',
                    'entity_co_occurrence'
                ]
            
            # Generate statistics
            if 'dataset_size' in statistics_to_generate:
                self.stats['dataset_size'] = self._get_dataset_size(df)
            
            if 'entity_distribution' in statistics_to_generate:
                self.stats['entity_distribution'] = self._get_entity_distribution(df)
            
            if 'text_length_stats' in statistics_to_generate:
                self.stats['text_length_stats'] = self._get_text_length_stats(df)
            
            if 'missing_values' in statistics_to_generate:
                self.stats['missing_values'] = self._get_missing_values(df)
            
            if 'entity_co_occurrence' in statistics_to_generate:
                self.stats['entity_co_occurrence'] = self._get_entity_co_occurrence(df)
            
            # Generate visualizations
            if generate_visualizations:
                self._generate_visualizations(df)
            
            # Save statistics to JSON
            stats_file = os.path.join(self.temp_dir, 'statistics.json')
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            # Upload all outputs to GCS
            self._upload_eda_outputs(output_bucket, output_path)
            
            logger.info("EDA completed successfully")
            return self.stats
            
        except Exception as e:
            logger.error(f"Error performing EDA: {e}")
            raise
        finally:
            # Cleanup temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temp directory")
    
    def _get_dataset_size(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset size statistics"""
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        logger.info(f"Dataset size: {stats['total_rows']} rows, {stats['total_columns']} columns")
        return stats
    
    def _get_entity_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get distribution of entity types"""
        # Try to find entity-related columns
        entity_columns = [col for col in df.columns if 'entity' in col.lower() or 'label' in col.lower()]
        
        if not entity_columns:
            logger.warning("No entity columns found for distribution analysis")
            return {'message': 'No entity columns found'}
        
        distribution = {}
        for col in entity_columns:
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts().to_dict()
                distribution[col] = {
                    'unique_count': df[col].nunique(),
                    'value_counts': value_counts,
                    'top_10': dict(list(value_counts.items())[:10])
                }
        
        return distribution
    
    def _get_text_length_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about text length"""
        # Try to find text columns
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'sentence' in col.lower()]
        
        if not text_columns:
            logger.warning("No text columns found for length analysis")
            return {'message': 'No text columns found'}
        
        stats = {}
        for col in text_columns:
            if df[col].dtype == 'object':
                lengths = df[col].astype(str).str.len()
                stats[col] = {
                    'mean': float(lengths.mean()),
                    'median': float(lengths.median()),
                    'std': float(lengths.std()),
                    'min': int(lengths.min()),
                    'max': int(lengths.max()),
                    'percentiles': {
                        '25th': float(lengths.quantile(0.25)),
                        '50th': float(lengths.quantile(0.50)),
                        '75th': float(lengths.quantile(0.75)),
                        '90th': float(lengths.quantile(0.90)),
                        '95th': float(lengths.quantile(0.95)),
                        '99th': float(lengths.quantile(0.99))
                    }
                }
        
        return stats
    
    def _get_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get missing value statistics"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_stats = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': missing[missing > 0].to_dict(),
            'missing_percentages': missing_pct[missing_pct > 0].to_dict(),
            'columns_without_missing': list(missing[missing == 0].index)
        }
        
        return missing_stats
    
    def _get_entity_co_occurrence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze entity co-occurrence patterns"""
        # This is a placeholder - actual implementation depends on dataset structure
        # For PII datasets, we might want to see which entities appear together
        entity_columns = [col for col in df.columns if 'entity' in col.lower() or 'label' in col.lower()]
        
        if len(entity_columns) < 2:
            return {'message': 'Insufficient entity columns for co-occurrence analysis'}
        
        # Simple co-occurrence analysis
        co_occurrence = {}
        # Implementation would depend on specific dataset structure
        return co_occurrence
    
    def _generate_visualizations(self, df: pd.DataFrame):
        """Generate visualization plots"""
        try:
            # Entity distribution plot
            entity_columns = [col for col in df.columns if 'entity' in col.lower() or 'label' in col.lower()]
            if entity_columns:
                self._plot_entity_distribution(df, entity_columns[0])
            
            # Text length distribution
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
            if text_columns:
                self._plot_text_length_distribution(df, text_columns[0])
            
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                self._plot_missing_values(df)
            
        except Exception as e:
            logger.warning(f"Error generating visualizations: {e}")
    
    def _plot_entity_distribution(self, df: pd.DataFrame, entity_col: str):
        """Plot entity type distribution"""
        try:
            value_counts = df[entity_col].value_counts().head(20)
            
            plt.figure(figsize=(14, 8))
            value_counts.plot(kind='bar')
            plt.title(f'Entity Distribution - {entity_col}', fontsize=16, fontweight='bold')
            plt.xlabel('Entity Type', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            output_file = os.path.join(self.temp_dir, 'entity_distribution.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved entity distribution plot to {output_file}")
        except Exception as e:
            logger.warning(f"Error plotting entity distribution: {e}")
    
    def _plot_text_length_distribution(self, df: pd.DataFrame, text_col: str):
        """Plot text length distribution"""
        try:
            lengths = df[text_col].astype(str).str.len()
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Histogram
            axes[0].hist(lengths, bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Text Length Distribution - {text_col}', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Text Length (characters)', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            
            # Box plot
            axes[1].boxplot(lengths, vert=True)
            axes[1].set_title(f'Text Length Box Plot - {text_col}', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Text Length (characters)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(self.temp_dir, 'text_length_analysis.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved text length analysis plot to {output_file}")
        except Exception as e:
            logger.warning(f"Error plotting text length distribution: {e}")
    
    def _plot_missing_values(self, df: pd.DataFrame):
        """Plot missing values heatmap"""
        try:
            missing_data = df.isnull()
            
            plt.figure(figsize=(12, max(6, len(df.columns) * 0.5)))
            sns.heatmap(missing_data, yticklabels=False, cbar=True, cmap='viridis')
            plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
            plt.xlabel('Columns', fontsize=12)
            plt.ylabel('Rows', fontsize=12)
            plt.tight_layout()
            
            output_file = os.path.join(self.temp_dir, 'missing_values_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved missing values heatmap to {output_file}")
        except Exception as e:
            logger.warning(f"Error plotting missing values: {e}")
    
    def _upload_eda_outputs(self, bucket_name: str, output_path: str):
        """Upload all EDA outputs to GCS"""
        if not self.temp_dir or not os.path.exists(self.temp_dir):
            return
        
        files = [f for f in os.listdir(self.temp_dir) if os.path.isfile(os.path.join(self.temp_dir, f))]
        
        for file in files:
            local_path = os.path.join(self.temp_dir, file)
            gcs_path = f"{output_path}/{file}"
            
            try:
                self.gcs_utils.upload_file(
                    bucket_name=bucket_name,
                    destination_object=gcs_path,
                    source_file=local_path
                )
                logger.info(f"Uploaded {file} to {bucket_name}/{gcs_path}")
            except Exception as e:
                logger.error(f"Error uploading {file}: {e}")

