"""
LLM Ensemble Configuration
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnsembleConfig:
    """Configuration for LLM ensemble system"""
    
    # Paths
    test_dataset_path: str = field(
        default="./data/test_dataset",
        metadata={"help": "Path to test dataset containing questions and IDs"}
    )
    csv_folder: str = field(
        default="./csv",
        metadata={"help": "Folder containing answer CSV files"}
    )
    output_file: str = field(
        default="./ensemble_output.csv",
        metadata={"help": "Output CSV file path"}
    )
    fallback_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to high-scoring submission file to use as fallback"}
    )
    
    # LLM Model Settings
    model_name_or_path: str = field(
        default="beomi/Llama-3-Open-Ko-8B-Instruct-preview",
        metadata={"help": "LLM model name or path"}
    )
    
    # Generation Parameters
    max_new_tokens: int = field(
        default=50,
        metadata={"help": "Maximum number of tokens to generate"}
    )
    temperature: float = field(
        default=0.1,
        metadata={"help": "Temperature for generation (lower = more deterministic)"}
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to use sampling (False = greedy decoding)"}
    )
    
    max_retries: int = field(
        default=10,
        metadata={"help": "Maximum number of retries for invalid answers"}
    )
    
    # Processing
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for processing"}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to use (cuda/cpu)"}
    )
    
    # Validation
    strict_validation: bool = field(
        default=True,
        metadata={"help": "Whether to strictly validate answers against choices"}
    )
    
    # Logging
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print verbose logs"}
    )
    log_interval: int = field(
        default=10,
        metadata={"help": "Log progress every N questions"}
    )
