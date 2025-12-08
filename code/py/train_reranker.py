import json
import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

# === [설정] ===
# 1. 베이스 모델: 성능이 입증된 Dongjin-kr 모델 사용
MODEL_NAME = "Dongjin-kr/ko-reranker" 

# 2. 학습 파라미터
BATCH_SIZE = 4 
NUM_EPOCHS = 2  # 데이터가 적으니(4k) 2 Epoch 정도로 늘려서 확실히 학습
OUTPUT_PATH = "./models/my_reranker_finetuned"
MAX_LENGTH = 512

def main():
    # 1. 데이터 로드
    # make_rerank_data.py로 만든 전체 데이터셋 파일
    data_path = "../data/reranker_train_data_origin.json" 
    print(f"학습 데이터 로드 중: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    train_data = all_data
    # 순서 섞기
    random.seed(2024)
    random.shuffle(train_data)

    print(f"=== [Data Strategy: Original Only Fine-tuning] ===")
    print(f"Base Model: {MODEL_NAME}")
    print(f"Training Samples: {len(train_data)} (Original Data Only)")
    print("====================================================")

    # 2. InputExample 생성
    train_examples = []
    for item in tqdm(train_data, desc="Preparing Batches"):
        query = item['question']
        positive = item['positive']
        negatives = item['negatives']
        
        # Positive (Label 1)
        train_examples.append(InputExample(texts=[query, positive], label=1.0))
        
        # Hard Negative (Label 0) - 상위 4개 사용
        for neg in negatives[:4]:
            train_examples.append(InputExample(texts=[query, neg], label=0.0))

    # 3. DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 4. 모델 초기화
    print(f"Loading Pre-trained Model: {MODEL_NAME}...")
    model = CrossEncoder(
        MODEL_NAME, 
        num_labels=1, 
        max_length=MAX_LENGTH
    )

    # 5. 학습 진행 (Low Learning Rate)
    print("Reranker Fine-tuning 시작...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        warmup_steps=100, # 웜업을 통해 학습 초반 불안정 방지
        output_path=OUTPUT_PATH,
        show_progress_bar=True,
        # [핵심] 학습률을 5e-6으로 매우 낮게 설정하여 기존 지식 보존 + 미세 조정
        optimizer_params={'lr': 5e-6} 
    )
    
    # 6. 저장
    model.save(OUTPUT_PATH)
    abs_path = os.path.abspath(OUTPUT_PATH)
    print(f"Reranker 파인튜닝 완료!")
    print(f"모델 저장 경로: {abs_path}")

if __name__ == "__main__":
    main()