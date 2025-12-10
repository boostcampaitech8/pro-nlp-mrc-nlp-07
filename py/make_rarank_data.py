import json
import os
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

# 변경: 일반 BERT/RoBERTa 대신 Reranking 전용 모델 사용
MODEL_NAME = "Dongjin-kr/ko-reranker" 
BATCH_SIZE = 4 
NUM_EPOCHS = 5 # 데이터가 많으므로 1 Epoch만으로 충분
OUTPUT_PATH = "./models/my_reranker"
MAX_LENGTH = 512

def main():
    # 1. 데이터 로드
    # make_rerank_data.py (또는 make_final_full_data 등)로 생성한 데이터 사용
    data_path = "../data/reranker_train_data.json" 
    print(f"학습 데이터 로드 중: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} 파일이 없습니다. make_rerank_data.py를 먼저 실행하세요.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 2. 데이터 전략: 전체 데이터 사용 (Shuffle)
    # 레퍼런스 코드의 전처리는 Inference용이므로, 학습용으로는 기존 방식을 유지합니다.
    train_data = all_data
    random.seed(42)
    random.shuffle(train_data)

    print(f"=== [Data Composition Strategy] ===")
    print(f"Model Base: {MODEL_NAME}")
    print(f"Total Training Data: {len(train_data)}")
    print("===================================")

    # 3. InputExample 생성 (Positive, Negative 쌍 만들기)
    train_examples = []
    
    for item in tqdm(train_data, desc="Preparing Batches"):
        query = item['question']
        positive = item['positive']
        negatives = item['negatives']
        
        # (Query, Positive) -> Label 1 (유사함)
        train_examples.append(InputExample(texts=[query, positive], label=1.0))
        
        # (Query, Hard Negative) -> Label 0 (유사하지 않음)
        # 데이터 생성 시 topk로 뽑은 오답들 중 상위 4개를 사용
        for neg in negatives[:4]:
            train_examples.append(InputExample(texts=[query, neg], label=0.0))

    # 4. DataLoader 설정
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    
    # 5. 모델 초기화 [핵심 변경 사항]
    # Dongjin-kr/ko-reranker 모델을 로드하여 Fine-tuning 합니다.
    print(f"Loading Base Model: {MODEL_NAME}...")
    model = CrossEncoder(
        MODEL_NAME, 
        num_labels=1, 
        max_length=MAX_LENGTH
    )

    # 6. 학습 진행
    print("Reranker 학습 시작...")
    # Smart Batching 등 SentenceTransformers의 최적화 기능 활용
    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_PATH,
        show_progress_bar=True,
        # 학습률은 모델이 깨지지 않도록 낮게 설정 (레퍼런스 모델이 이미 학습된 상태일 수 있으므로)
        optimizer_params={'lr': 1e-5} 
    )
    
    # 7. 모델 저장 및 확인
    model.save(OUTPUT_PATH)
    
    abs_path = os.path.abspath(OUTPUT_PATH)
    print(f"Reranker 학습 완료!")
    print(f"모델이 다음 경로에 강제로 저장되었습니다: {abs_path}")
    
    print("----- 파일 목록 확인 -----")
    os.system(f"ls -l {OUTPUT_PATH}")
    print("--------------------------")

if __name__ == "__main__":
    main()