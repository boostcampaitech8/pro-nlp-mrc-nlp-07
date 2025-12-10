"""
Simple data collator for padding preprocessed data.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForPreprocessedLM:
    """
    Data collator that pads preprocessed data (input_ids and labels).
    """
    
    tokenizer: PreTrainedTokenizerBase
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pad input_ids and labels to the same length.
        """
        # Get max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            
            # Calculate padding length
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            
            # Pad attention_mask
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            # Pad labels with -100
            padded_labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)
        
        # Convert to tensors
        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        
        return batch
