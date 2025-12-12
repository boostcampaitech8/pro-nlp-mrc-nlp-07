import torch
import os
from unsloth import FastLanguageModel
from datasets import load_dataset, load_from_disk, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

# === [설정] ===
MODEL_NAME = "beomi/Llama-3-Open-Ko-8B" 
OUTPUT_DIR = "./models/llama3_mrc_lora_optimized"

# 데이터 경로 설정
ORG_DATA_PATH = "../data/train_dataset"          # 원본 데이터 (약 4000개)
KORQUAD_PATH = "../data/train_dataset_korquad"   # KorQuAD 데이터 (여기서 6000개 추출)

# 문맥 길이 (2048로 줄여서 속도 확보)
MAX_SEQ_LENGTH = 2048 

def main():
    # 1. 모델 로드 (4-bit 유지)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, 
        load_in_4bit = True, 
    )

    # 2. LoRA 어댑터 설정 (Rank 64 유지)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 64, 
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 2025,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. 데이터 로드 및 혼합 (핵심 수정 부분)
    print("Loading & Mixing Datasets...")

    # (A) 원본 데이터 로드 (전체 사용)
    try:
        # load_from_disk 시도
        org_dataset = load_from_disk(ORG_DATA_PATH)
        if "train" in org_dataset: org_dataset = org_dataset["train"]
    except:
        # json 로드 시도
        org_dataset = load_dataset("json", data_files={"train": os.path.join(ORG_DATA_PATH, "train.json")})["train"]
    
    print(f" - Original Data Count: {len(org_dataset)} (Keep All)")

    # (B) KorQuAD 데이터 로드 & 6000개 샘플링
    try:
        kor_dataset = load_from_disk(KORQUAD_PATH)
        if "train" in kor_dataset: kor_dataset = kor_dataset["train"]
    except:
        # 경로가 다르거나 파일 형태일 경우 대비
        try:
            kor_dataset = load_dataset("disk", KORQUAD_PATH)["train"]
        except:
             # 만약 KorQuAD 폴더가 없다면 원본만 쓰도록 예외처리 (안전장치)
             print("Warning: KorQuAD dataset not found. Using Original only.")
             kor_dataset = None

    if kor_dataset:
        sample_count = 6000
        if len(kor_dataset) > sample_count:
            kor_sampled = kor_dataset.shuffle(seed=2025).select(range(sample_count))
        else:
            kor_sampled = kor_dataset # 6000개가 안되면 다 씀
        print(f" - KorQuAD Sampled Count: {len(kor_sampled)}")
    else:
        kor_sampled = None

    # (C) 데이터 병합
    # 컬럼 충돌 방지를 위해 공통 필수 컬럼만 선택
    # 보통 context, question, answers는 무조건 있습니다.
    target_columns = ["context", "question", "answers"]
    
    # 원본 데이터 컬럼 필터링
    org_dataset = org_dataset.select_columns([c for c in target_columns if c in org_dataset.column_names])
    
    if kor_sampled:
        # KorQuAD 데이터 컬럼 필터링
        kor_sampled = kor_sampled.select_columns([c for c in target_columns if c in kor_sampled.column_names])
        # 병합
        train_data = concatenate_datasets([org_dataset, kor_sampled])
    else:
        train_data = org_dataset

    # 최종 셔플
    train_data = train_data.shuffle(seed=2025)
    print(f"Final Training Dataset Size: {len(train_data)}")

    # 프롬프트 포맷
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
지문을 읽고 질문에 대한 정답을 한 단어 또는 짧은 구문으로 추출하세요.

### Input:
지문: {context}
질문: {question}

### Response:
{answer}"""

    def formatting_prompts_func(examples):
        contexts = examples["context"]
        questions = examples["question"]
        answers = examples["answers"]
        
        texts = []
        for context, question, answer in zip(contexts, questions, answers):
            if len(answer["text"]) > 0:
                ans_text = answer["text"][0]
            else:
                ans_text = "" 

            text = alpaca_prompt.format(
                context=context,
                question=question,
                answer=ans_text,
            ) + tokenizer.eos_token 
            texts.append(text)
        return { "text" : texts, }

    # 데이터 매핑 (병합된 데이터 사용)
    train_dataset = train_data.map(formatting_prompts_func, batched=True, load_from_cache_file=False)

    # 4. 학습 설정 (안전한 배치 1 설정 유지)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 4,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 1,  # 배치 1 (안전)
            gradient_accumulation_steps = 16, # 누적 16 (총 배치 16)
            
            warmup_steps = 50, # 데이터가 줄었으므로 웜업도 줄임
            num_train_epochs = 1, # 1 에포크면 충분
            learning_rate = 2e-4, 
            fp16 = True, 
            logging_steps = 10,
            optim = "paged_adamw_8bit", 
            gradient_checkpointing = True,
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 2024,
            output_dir = OUTPUT_DIR,
            save_strategy = "epoch",
            report_to = "none",
            dataloader_num_workers = 0, 
        ),
    )

    print(f"LLM Fine-tuning Start... (Size: {len(train_dataset)})")
    trainer.train()
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training Finished! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()