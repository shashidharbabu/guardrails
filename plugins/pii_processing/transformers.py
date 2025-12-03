"""
Data Transformation Module for NER
Converts dataset to BIO/IOB2 tagging format and prepares for model training
"""
import os
import json
import logging
import re
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import spacy
from spacy.lang.en import English

logger = logging.getLogger(__name__)


class NERTransformer:
    """Class for transforming data to NER format"""
    
    def __init__(
        self,
        ner_format: str = 'BIO',
        entity_types: Optional[List[str]] = None,
        tokenizer: str = 'spacy',
        max_sequence_length: int = 512
    ):
        """
        Initialize NER Transformer
        
        Args:
            ner_format: NER tagging format ('BIO' or 'IOB2')
            entity_types: List of entity types to extract
            tokenizer: Tokenizer to use ('spacy' or 'nltk')
            max_sequence_length: Maximum sequence length for tokens
        """
        self.ner_format = ner_format.upper()
        self.entity_types = entity_types or []
        self.tokenizer_type = tokenizer
        self.max_sequence_length = max_sequence_length
        self.tokenizer = self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer"""
        if self.tokenizer_type == 'spacy':
            try:
                # Try to load a spacy model, fallback to basic English
                try:
                    return spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy model not found, using basic English tokenizer")
                    nlp = English()
                    nlp.add_pipe("sentencizer")
                    return nlp
            except Exception as e:
                logger.warning(f"Error loading spaCy: {e}, using basic tokenizer")
                nlp = English()
                nlp.add_pipe("sentencizer")
                return nlp
        else:
            # Basic whitespace tokenizer as fallback
            return None
    
    def transform_to_ner_format(
        self,
        data: Union[pd.DataFrame, Dataset],
        text_column: str = 'text',
        entity_column: Optional[str] = None,
        annotations_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transform dataset to NER format (BIO tagging)
        
        Args:
            data: Input dataset
            text_column: Name of column containing text
            entity_column: Name of column containing entity labels (if single column)
            annotations_column: Name of column containing entity annotations (if structured)
            
        Returns:
            List of dictionaries in NER format
        """
        # Convert to DataFrame if needed
        if isinstance(data, Dataset):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        logger.info(f"Transforming {len(df)} examples to NER format")
        
        ner_examples = []
        
        for idx, row in df.iterrows():
            try:
                text = str(row[text_column]) if text_column in row else str(row.iloc[0])
                
                # Extract entities from row
                entities = self._extract_entities(row, entity_column, annotations_column)
                
                # Convert to BIO format
                ner_example = self._convert_to_bio(text, entities)
                
                if ner_example:
                    ner_examples.append(ner_example)
                    
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Successfully transformed {len(ner_examples)} examples")
        return ner_examples
    
    def _extract_entities(
        self,
        row: pd.Series,
        entity_column: Optional[str],
        annotations_column: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Extract entity annotations from a row"""
        entities = []
        
        # Try different ways to extract entities
        if annotations_column and annotations_column in row:
            # Structured annotations (list of dicts with start, end, label)
            annotations = row[annotations_column]
            if isinstance(annotations, str):
                try:
                    annotations = json.loads(annotations)
                except:
                    pass
            if isinstance(annotations, list):
                entities = annotations
        elif entity_column and entity_column in row:
            # Single entity column - might need parsing
            entity_data = row[entity_column]
            if isinstance(entity_data, dict):
                entities = [entity_data]
            elif isinstance(entity_data, str):
                try:
                    entities = json.loads(entity_data)
                except:
                    # Try to parse as simple format
                    pass
        
        # Normalize entity format
        normalized_entities = []
        for entity in entities:
            if isinstance(entity, dict):
                normalized_entities.append({
                    'start': int(entity.get('start', entity.get('start_pos', 0))),
                    'end': int(entity.get('end', entity.get('end_pos', entity.get('start', 0) + 1))),
                    'label': str(entity.get('label', entity.get('entity_type', 'UNKNOWN')))
                })
        
        return normalized_entities
    
    def _convert_to_bio(
        self,
        text: str,
        entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Convert text and entities to BIO format"""
        # Clean text
        text = self._clean_text(text)
        if not text or len(text) < 1:
            return None
        
        # Tokenize text
        tokens, token_positions = self._tokenize(text)
        
        if not tokens:
            return None
        
        # Create BIO tags
        tags = ['O'] * len(tokens)
        
        # Map entities to tokens
        for entity in entities:
            start = entity['start']
            end = entity['end']
            label = entity['label']
            
            # Find tokens that overlap with entity span
            entity_tokens = []
            for i, (token_start, token_end) in enumerate(token_positions):
                if token_start < end and token_end > start:
                    entity_tokens.append(i)
            
            # Assign BIO tags
            if entity_tokens:
                for i, token_idx in enumerate(entity_tokens):
                    if i == 0:
                        tags[token_idx] = f'B-{label}' if self.ner_format == 'BIO' else f'I-{label}'
                    else:
                        tags[token_idx] = f'I-{label}'
        
        # Truncate if too long
        if len(tokens) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]
            tags = tags[:self.max_sequence_length]
        
        return {
            'tokens': tokens,
            'tags': tags,
            'text': text,
            'entities': entities
        }
    
    def _tokenize(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenize text and return tokens with their positions"""
        if self.tokenizer_type == 'spacy' and self.tokenizer:
            doc = self.tokenizer(text)
            tokens = [token.text for token in doc]
            positions = [(token.idx, token.idx + len(token.text)) for token in doc]
            return tokens, positions
        else:
            # Simple whitespace tokenization with position tracking
            tokens = []
            positions = []
            start = 0
            for match in re.finditer(r'\S+', text):
                tokens.append(match.group())
                positions.append((match.start(), match.end()))
            return tokens, positions
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def clean_data(
        self,
        ner_examples: List[Dict[str, Any]],
        min_text_length: int = 10,
        max_text_length: int = 10000,
        remove_invalid_annotations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Clean NER examples
        
        Args:
            ner_examples: List of NER examples
            min_text_length: Minimum text length
            max_text_length: Maximum text length
            remove_invalid_annotations: Whether to remove invalid annotations
            
        Returns:
            Cleaned list of NER examples
        """
        cleaned = []
        
        for example in ner_examples:
            text = example.get('text', '')
            
            # Check text length
            if len(text) < min_text_length or len(text) > max_text_length:
                continue
            
            # Check token count
            tokens = example.get('tokens', [])
            if len(tokens) == 0:
                continue
            
            # Validate tags match tokens
            tags = example.get('tags', [])
            if len(tags) != len(tokens):
                if remove_invalid_annotations:
                    continue
                else:
                    # Pad or truncate tags
                    if len(tags) < len(tokens):
                        tags.extend(['O'] * (len(tokens) - len(tags)))
                    else:
                        tags = tags[:len(tokens)]
                    example['tags'] = tags
            
            cleaned.append(example)
        
        logger.info(f"Cleaned {len(cleaned)} examples from {len(ner_examples)} original examples")
        return cleaned
    
    def create_splits(
        self,
        ner_examples: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        stratify: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split dataset into train/val/test sets
        
        Args:
            ner_examples: List of NER examples
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
            stratify: Whether to stratify based on entity distribution
            
        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        logger.info(f"Creating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Simple random split (stratification would require more complex logic)
        np.random.seed(random_seed)
        indices = np.random.permutation(len(ner_examples))
        
        train_end = int(len(ner_examples) * train_ratio)
        val_end = train_end + int(len(ner_examples) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        splits = {
            'train': [ner_examples[i] for i in train_indices],
            'val': [ner_examples[i] for i in val_indices],
            'test': [ner_examples[i] for i in test_indices]
        }
        
        logger.info(f"Created splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
        return splits
    
    def save_splits(
        self,
        splits: Dict[str, List[Dict[str, Any]]],
        output_dir: str,
        format: str = 'jsonl'
    ) -> Dict[str, str]:
        """
        Save splits to files
        
        Args:
            splits: Dictionary with train/val/test splits
            output_dir: Directory to save files
            format: File format ('jsonl' or 'json')
            
        Returns:
            Dictionary mapping split names to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        file_paths = {}
        
        for split_name, examples in splits.items():
            filename = f"{split_name}.{format}"
            filepath = os.path.join(output_dir, filename)
            
            if format == 'jsonl':
                with open(filepath, 'w', encoding='utf-8') as f:
                    for example in examples:
                        f.write(json.dumps(example, ensure_ascii=False) + '\n')
            else:  # json
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(examples, f, indent=2, ensure_ascii=False)
            
            file_paths[split_name] = filepath
            logger.info(f"Saved {split_name} split ({len(examples)} examples) to {filepath}")
        
        return file_paths

