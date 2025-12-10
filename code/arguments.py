# arguments.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="klue/roberta-large",
        metadata={"help": "Pretrained model path"}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default="../data/train_dataset")
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=4)

    max_seq_length: int = field(default=384)
    pad_to_max_length: bool = field(default=False)
    doc_stride: int = field(default=192)
    max_answer_length: int = field(default=30)

    eval_retrieval: bool = field(default=True)
    top_k_retrieval: int = field(default=20)

    add_negative_samples: bool = field(
        default=False,
        metadata={"help": "Hard negative sampling option"}
    )
