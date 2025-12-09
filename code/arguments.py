from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    hub_repo_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The repository id on Hugging Face Hub to push the model to. Format: 'model-name' (will be uploaded to NLP-07-ODQA organization) or 'organization/model-name'. Use --push_to_hub flag from TrainingArguments to enable upload."
        },
    )
    hub_model_repo_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Hugging Face Hub repository ID to download model from. Format: 'organization/model-name' or 'model-name' (will search in NLP-07-ODQA organization). Used in inference.py to download model from Hub."
        },
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": "Revision (branch or commit) of the Hugging Face Hub model to download. Default is 'main'. Can be a branch name like 'checkpoint-1000'."
        },
    )
    hub_model_cache_dir: Optional[str] = field(
        default="./models",
        metadata={
            "help": "Local directory to cache downloaded Hugging Face models. Models will be saved in subdirectories named after the repo_id."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    use_vllm: bool = field(
        default=False, 
        metadata={"help": "Whether to use vLLM for faster generation inference (requires vllm package)"}
    )
