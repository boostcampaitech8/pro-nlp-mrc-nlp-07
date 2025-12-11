# finetune_bge.py

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
import json
from typing import List
from tqdm.auto import tqdm
import os

# --- 1. 설정 변수 ---
SEED = 42
DATA_PATH = "../data"
HN_DATA_FILENAME = "train_with_hard_negatives.json" # 마이닝된 데이터 파일
MODEL_NAME = "BAAI/bge-m3"
TUNED_MODEL_PATH = "./models/bge-m3-finetuned-odqa"

# 학습 하이퍼파라미터
BATCH_SIZE = 16
NUM_EPOCHS = 1
WARMUP_STEPS = 50 
LEARNING_RATE = 5e-6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 시드 고정
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# --- 2. 모델 로드 및 데이터 준비 함수 ---
def get_model(model_name: str) -> SentenceTransformer:
    """SentenceTransformer 모델 로드 및 설정"""
    print(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = 512
    model.to(DEVICE)
    return model

def prepare_training_data(examples: List[dict]) -> List[InputExample]:
    """
    Hard Negative가 포함된 리스트를 SentenceTransformer 학습 형식(InputExample)으로 변환합니다.
    """
    train_examples = []
    QUERY_PREFIX = "query: " 
    
    print("Converting data to InputExample format...")
    for example in tqdm(examples):
        question = QUERY_PREFIX + example['question']
        positive = example['positive_context']
        negatives = example['negative_contexts']
        
        # 1. Positive 쌍 추가 (Anchor: Question, Positive: P+)
        # 레이블을 torch.tensor(1.0) 대신 float(1.0)으로 수정
        train_examples.append(InputExample(texts=[question, positive], label=1.0)) # <--- 수정
        
        # 2. Negative 쌍 추가 (Anchor: Question, Negative: P-)
        for neg in negatives:
             # 레이블을 torch.tensor(0.0) 대신 float(0.0)으로 수정
             train_examples.append(InputExample(texts=[question, neg], label=0.0)) # <--- 수정
             
    return train_examples
    """
    Hard Negative가 포함된 리스트를 SentenceTransformer 학습 형식(InputExample)으로 변환합니다.
    """
    train_examples = []
    # BGE 계열 모델은 Query에 'query: ' prefix를 필요로 합니다.
    QUERY_PREFIX = "query: " 
    
    print("Converting data to InputExample format...")
    for example in tqdm(examples):
        question = QUERY_PREFIX + example['question']
        positive = example['positive_context']
        negatives = example['negative_contexts']
        
        # 1. Positive 쌍 추가 (Anchor: Question, Positive: P+)
        # Loss 함수는 이 쌍의 유사도를 높이도록 학습합니다.
        train_examples.append(InputExample(texts=[question, positive], label=torch.tensor(1.0)))
        
        # 2. Negative 쌍 추가 (Anchor: Question, Negative: P-)
        # MultipleNegativesRankingLoss는 Anchor와 Negative의 유사도를 낮추도록 학습합니다.
        for neg in negatives:
             train_examples.append(InputExample(texts=[question, neg], label=torch.tensor(0.0)))
             
    return train_examples

# --- 3. 메인 학습 함수 ---
def run_finetuning():
    
    # 3.1. Hard Negative 데이터 로드
    hn_data_path = os.path.join(DATA_PATH, HN_DATA_FILENAME)
    print(f"Loading Hard Negative Data from {hn_data_path}...")
    
    try:
        with open(hn_data_path, 'r', encoding='utf-8') as f:
            df_train_list = json.load(f)
    except FileNotFoundError:
        print(f"🚨 Error: Hard Negative data file not found at {hn_data_path}")
        print("Please run 'mine_hard_negatives.py' first.")
        return

    num_negatives = len(df_train_list[0]['negative_contexts']) if df_train_list else 0
    print(f"Total training queries: {len(df_train_list)}")
    print(f"Negatives per query: {num_negatives}")
    
    # 3.2. 데이터 준비 및 DataLoader 생성
    train_examples = prepare_training_data(df_train_list)
    
    model = get_model(MODEL_NAME)
    
    # SentencesDataset과 DataLoader 준비
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    
    # 3.3. Loss 함수 정의 및 학습
    # MultipleNegativesRankingLoss는 Contrastive Learning의 한 형태입니다.
    train_loss = MultipleNegativesRankingLoss(model=model)

    print("\nStarting Fine-tuning...")
    
    # 학습 시작
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS, 
        output_path=TUNED_MODEL_PATH,
        show_progress_bar=True,
        save_best_model=True,
        optimizer_params={'lr': LEARNING_RATE}
    )
    
    print(f"\n✅ Fine-tuning completed. Best model saved to {TUNED_MODEL_PATH}")

if __name__ == "__main__":
    # 이 스크립트를 실행하기 전에 'pip install sentence-transformers' 필요
    run_finetuning()