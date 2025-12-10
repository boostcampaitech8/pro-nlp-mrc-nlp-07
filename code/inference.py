"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import os
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
import torch
from arguments import DataTrainingArguments, ModelArguments
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
from retrieval import SparseRetrieval
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

# vLLM import (optional)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM is not installed. Install with 'pip install vllm' to use faster inference.")
    

MAX_INPUT_LENGTH = 2048


# 모델별 설정 딕셔너리 (확장 가능한 구조)
MODEL_CONFIGS = {
    "qwen": {
        "is_generation": True,
        "skip_config": True,  # Qwen3는 transformers>=4.51.0 필요하므로 config 로딩 건너뛰기
        "load_model_kwargs": {
            "torch_dtype": "auto",
            "device_map": "auto",
        },
        "load_tokenizer_kwargs": {},
        "generation_kwargs": {
            "max_new_tokens": 100,  # QA 태스크에 적합한 짧은 답변 길이로 제한
            "do_sample": False,  # Greedy decoding (메모리 효율적)
        },
        "generation_method": "qwen",  # Qwen 공식 예제 방식
    },
    "default": {
        "is_generation": False,
        "skip_config": False,
        "load_model_kwargs": {},
        "load_tokenizer_kwargs": {
            "use_fast": True,
        },
        "generation_kwargs": {},
        "generation_method": "extractive",
    },
}


def get_model_config(model_name: str) -> dict:
    """
    모델 이름을 기반으로 설정을 가져옵니다.
    
    Args:
        model_name: 모델 이름 또는 경로 (예: "Qwen/Qwen3-4B-Instruct-2507")
    
    Returns:
        모델 설정 딕셔너리
    """
    model_name_lower = model_name.lower()
    
    # Qwen 모델 체크
    if "qwen" in model_name_lower:
        return MODEL_CONFIGS["qwen"]
    
    # 기본값 (Extractive QA 모델)
    return MODEL_CONFIGS["default"]


def main():
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    # Hugging Face Hub에서 모델 다운로드 (옵션)
    if model_args.hub_model_repo_id:
        hub_repo_id = model_args.hub_model_repo_id
        # organization이 지정되지 않은 경우 NLP-07-ODQA organization에 있는 것으로 가정
        if "/" not in hub_repo_id:
            hub_repo_id = f"NLP-07-ODQA/{hub_repo_id}"
        
        # 모델 저장 경로 (repo_id의 마지막 부분을 폴더명으로 사용)
        cache_dir = model_args.hub_model_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        repo_name = hub_repo_id.split("/")[-1]
        revision = model_args.hub_model_revision
        if revision and revision != "main":
            # 브랜치가 있으면 폴더명에 포함
            local_model_dir = os.path.join(cache_dir, f"{repo_name}_{revision}")
        else:
            local_model_dir = os.path.join(cache_dir, repo_name)
        
        logger.info(f"Downloading model from Hugging Face Hub: {hub_repo_id} (revision: {revision})")
        logger.info(f"Model will be saved to: {local_model_dir}")
        
        try:
            # Hugging Face Hub에서 모델 다운로드
            # cache_dir를 지정하지 않으면 기본 캐시 디렉토리(~/.cache/huggingface/hub/)를 사용하고,
            # local_dir에 실제 파일을 복사합니다.
            snapshot_download(
                repo_id=hub_repo_id,
                revision=revision,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일 복사
            )
            logger.info(f"Successfully downloaded model to {local_model_dir}")
            
            # model_name_or_path를 다운로드한 로컬 경로로 업데이트
            model_args.model_name_or_path = local_model_dir
            logger.info(f"Using local model path: {model_args.model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to download model from Hugging Face Hub: {e}")
            logger.error("Falling back to original model_name_or_path")
            raise

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # 모델 설정 가져오기
    model_config = get_model_config(model_args.model_name_or_path)
    is_generation_model = model_config["is_generation"]
    
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Model type: {'Generation' if is_generation_model else 'Extractive QA'}")
    logger.info(f"Generation method: {model_config['generation_method']}")
    
    # Config 로딩 (일부 모델은 건너뛰기)
    config = None
    if not model_config["skip_config"]:
        config = AutoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path
            ),
        )
    
    # Tokenizer 경로 설정
    tokenizer_path = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )
    
    # 모델 및 Tokenizer 로딩
    vllm_model = None  # vLLM 모델 인스턴스
    if is_generation_model:
        # vLLM 사용 여부 확인
        use_vllm = data_args.use_vllm and VLLM_AVAILABLE
        if data_args.use_vllm and not VLLM_AVAILABLE:
            logger.warning("--use_vllm is set but vLLM is not installed. Falling back to transformers.")
            logger.warning("Install vLLM with: pip install vllm")
            use_vllm = False
        
        # Generation 모델 로드
        logger.info(f"Loading generation model: {model_args.model_name_or_path}")
        if use_vllm:
            logger.info("Using vLLM for faster inference")
            try:
                # vLLM으로 모델 로드
                vllm_model = LLM(
                    model=model_args.model_name_or_path,
                    max_model_len=MAX_INPUT_LENGTH ,  # context 길이 제한 (top-10 retrieval 대응)
                    gpu_memory_utilization=0.9,  # GPU 메모리 사용률
                    trust_remote_code=True,
                )
                # Tokenizer는 vLLM이 자동으로 로드하지만, 별도로도 필요할 수 있음
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    **model_config["load_tokenizer_kwargs"]
                )
                model = None  # vLLM 사용 시 transformers 모델은 사용하지 않음
                logger.info("Successfully loaded model with vLLM")
            except Exception as e:
                logger.error(f"Failed to load model with vLLM: {e}")
                logger.warning("Falling back to transformers")
                use_vllm = False
        
        if not use_vllm:
            # Transformers로 모델 로드
            try:
                # Tokenizer 로드
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    **model_config["load_tokenizer_kwargs"]
                )
                
                # Model 로드
                model = AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    **model_config["load_model_kwargs"]
                )
                
                logger.info(f"Successfully loaded generation model and tokenizer (transformers)")
            except Exception as e:
                logger.error(f"Failed to load model/tokenizer: {e}")
                raise RuntimeError(
                    f"Failed to load generation model. Error: {e}\n"
                    "Please try:\n"
                    "1. Check transformers version (Qwen3 requires transformers>=4.51.0)\n"
                    "2. Clear Hugging Face cache: rm -rf ~/.cache/huggingface\n"
                    "3. Check your internet connection\n"
                    "4. Or use a different model"
                )
    else:
        # 일반 Extractive QA 모델
        logger.info(f"Loading extractive QA model: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            **model_config["load_tokenizer_kwargs"]
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        if is_generation_model:
            run_mrc_generation(
                data_args, 
                training_args, 
                model_args, 
                datasets, 
                tokenizer, 
                model,
                model_config,
                vllm_model  # vLLM 모델 전달
            )
        else:
            run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def run_mrc_generation(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    model_config: dict,
    vllm_model=None,  # vLLM 모델 인스턴스
) -> NoReturn:
    """
    Generation 기반 QA를 수행하는 함수
    Generation 모델을 사용할 때 호출됩니다.
    
    Args:
        model_config: 모델별 설정 딕셔너리 (generation_method, generation_kwargs 포함)
    """
    import json
    from tqdm import tqdm
    
    logger.info("Using Generation-based QA")
    
    # 프롬프트 템플릿 함수
    def create_qa_messages(context: str, question: str) -> list:
        """QA 메시지 생성 (chat template용)"""
        return [
            {"role": "user", 
             "content": f"""다음 지문에는 질문에 대한 답이 포함되어 있습니다. 
             지문에서 직접 답을 찾아서 그대로 제시하세요. 지문에 없는 내용을 생성하거나 설명을 추가하지 마세요.
             정답은 한 단어 또는 짧은 구문으로 추출하세요.
             만약에 정답이 없으면 "없음"이라고 대답하세요.
             질문: {question}
             지문: {context}
             답변:"""}
        ]
    
    def prepare_qa_prompt(context: str, question: str, tokenizer) -> str:
        """QA 프롬프트 텍스트 준비 (chat template 시도, 실패 시 fallback)"""
        messages = create_qa_messages(context, question)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (AttributeError, TypeError):
            # Chat template이 없는 경우 fallback: messages의 content를 직접 사용
            text = messages[0]["content"]
        return text
    
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else None
    
    # Retrieval된 context 확인
    if len(datasets["validation"]) > 0:
        sample_context = datasets["validation"][0][context_column_name]
        logger.info(f"Sample retrieved context length: {len(sample_context)} characters")
        logger.info(f"Using retrieved contexts from sparse retrieval (top-k={data_args.top_k_retrieval})")
    
    # vLLM 사용 여부 확인
    use_vllm = vllm_model is not None
    if use_vllm:
        logger.info("Using vLLM for generation (faster inference)")
        # vLLM SamplingParams 설정
        generation_kwargs = model_config["generation_kwargs"].copy()
        sampling_params = SamplingParams(
            max_tokens=generation_kwargs.get("max_new_tokens", 100),  # 짧은 답변을 위해 100으로 제한
            temperature=0.0 if not generation_kwargs.get("do_sample", False) else 0.7,
            stop=[],  # stop strings는 나중에 처리
        )
    else:
        if model is not None:
            model.eval()
        logger.info("Using transformers for generation")
    
    predictions = {}
    
    # 토큰 길이 통계 수집
    token_lengths = []
    truncated_count = 0
    max_input_length = MAX_INPUT_LENGTH   # Transformers 경로에서 사용하는 max_length (top-10 retrieval 대응)
    
    logger.info(f"Processing {len(datasets['validation'])} examples with retrieved contexts...")
    logger.info(f"Max input token length: {max_input_length} (Transformers) / {vllm_model.max_model_len if use_vllm and vllm_model else 'N/A'} (vLLM)")
    
    # vLLM 사용 시 torch.no_grad() 불필요, transformers 사용 시 필요
    if use_vllm:
        # vLLM은 자체적으로 메모리 관리
        for example in tqdm(datasets["validation"]):
            question = example[question_column_name]
            context = example[context_column_name]
            example_id = example["id"]
            
            generation_kwargs = model_config["generation_kwargs"].copy()
            
            try:
                # QA 프롬프트 준비
                text = prepare_qa_prompt(context, question, tokenizer)
                
                # 토큰 길이 확인 (vLLM)
                encoded = tokenizer.encode(text, add_special_tokens=False)
                token_length = len(encoded)
                token_lengths.append(token_length)
                max_vllm_length = vllm_model.max_model_len if hasattr(vllm_model, 'max_model_len') else MAX_INPUT_LENGTH 
                
                if token_length > max_vllm_length:
                    truncated_count += 1
                    logger.warning(
                        f"Example {example_id}: Input token length ({token_length}) exceeds vLLM max_model_len ({max_vllm_length}). "
                        f"Context will be truncated. Context char length: {len(context)}"
                    )
                
                # vLLM 사용
                outputs = vllm_model.generate([text], sampling_params)
                generated_text = outputs[0].outputs[0].text
                answer = generated_text.strip()
                
                # 빈 답변 처리
                if not answer or len(answer) == 0:
                    answer = ""
                    
            except Exception as e:
                logger.warning(f"Error generating answer for example {example_id}: {e}")
                # 실패 시 빈 답변 반환
                answer = ""
            
                # 메모리 정리 (각 예제 처리 후)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                predictions[example_id] = answer
    else:
        # Transformers 사용 시 torch.no_grad() 필요
        with torch.no_grad():
            for example in tqdm(datasets["validation"]):
                question = example[question_column_name]
                context = example[context_column_name]
                example_id = example["id"]
                
                generation_kwargs = model_config["generation_kwargs"].copy()
                
                try:
                    # QA 프롬프트 준비
                    text = prepare_qa_prompt(context, question, tokenizer)
                    
                    # 토큰 길이 확인 (Transformers)
                    encoded_before = tokenizer.encode(text, add_special_tokens=False)
                    token_length_before = len(encoded_before)
                    token_lengths.append(token_length_before)
                    
                    # Context 길이 제한으로 메모리 사용량 감소
                    model_inputs = tokenizer(
                        [text], 
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_input_length,
                    ).to(model.device)
                    
                    # 잘렸는지 확인
                    token_length_after = model_inputs.input_ids.shape[1]
                    if token_length_before > max_input_length:
                        truncated_count += 1
                        logger.warning(
                            f"Example {example_id}: Input token length ({token_length_before}) exceeds max_length ({max_input_length}). "
                            f"Truncated to {token_length_after} tokens. Context char length: {len(context)}"
                        )
                    
                    # Generation
                    generated_ids = model.generate(
                        **model_inputs,
                        **generation_kwargs,
                        eos_token_id=tokenizer.eos_token_id,  # EOS 토큰으로 조기 종료
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    )
                    
                    # 입력 부분 제거하고 답변만 추출
                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    
                    if not answer or len(answer) == 0:
                        answer = ""
                        
                except Exception as e:
                    logger.warning(f"Error generating answer for example {example_id}: {e}")
                    # 실패 시 빈 답변 반환
                    answer = ""
                
                # 메모리 정리 (각 예제 처리 후)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                predictions[example_id] = answer
    
    # predictions.json 저장
    output_file = os.path.join(training_args.output_dir, "predictions.json")
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Predictions saved to {output_file}")
    
    # 토큰 길이 통계 출력
    if token_lengths:
        avg_length = np.mean(token_lengths)
        max_length = np.max(token_lengths)
        min_length = np.min(token_lengths)
        median_length = np.median(token_lengths)
        p95_length = np.percentile(token_lengths, 95)
        p99_length = np.percentile(token_lengths, 99)
        
        logger.info("=" * 60)
        logger.info("Input Token Length Statistics:")
        logger.info(f"  Total examples: {len(token_lengths)}")
        logger.info(f"  Truncated examples: {truncated_count} ({truncated_count/len(token_lengths)*100:.1f}%)")
        logger.info(f"  Average: {avg_length:.1f} tokens")
        logger.info(f"  Median: {median_length:.1f} tokens")
        logger.info(f"  Min: {min_length} tokens")
        logger.info(f"  Max: {max_length} tokens")
        logger.info(f"  95th percentile: {p95_length:.1f} tokens")
        logger.info(f"  99th percentile: {p99_length:.1f} tokens")
        max_vllm_len = vllm_model.max_model_len if use_vllm and vllm_model and hasattr(vllm_model, 'max_model_len') else None
        max_allowed_str = f"{max_input_length} tokens (Transformers)" if not use_vllm else (f"{max_vllm_len} tokens (vLLM)" if max_vllm_len else "N/A")
        logger.info(f"  Max allowed: {max_allowed_str}")
        
        if truncated_count > 0:
            logger.warning(
                f"⚠️  {truncated_count} examples ({truncated_count/len(token_lengths)*100:.1f}%) were truncated. "
                f"Consider reducing top_k_retrieval or increasing max_length."
            )
        logger.info("=" * 60)
    
    # 평가 (do_eval인 경우)
    if training_args.do_eval and answer_column_name:
        metric = evaluate.load("squad")
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in datasets["validation"]
        ]
        
        metrics = metric.compute(predictions=formatted_predictions, references=references)
        logger.info(f"Evaluation metrics: {metrics}")
        
        # 메트릭 저장
        metrics_file = os.path.join(training_args.output_dir, "eval_results.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Metrics saved to {metrics_file}")
    else:
        logger.info("No evaluation performed (do_eval=False or no answers in dataset)")


if __name__ == "__main__":
    main()
