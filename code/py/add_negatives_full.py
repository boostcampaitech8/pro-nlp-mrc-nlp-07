import os
import copy
import random
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset
from retrieval import ElasticSearchRetrieval

def main():
    # === 설정 ===
    # 1. 전체 Positive 데이터 (KorQuAD 합친 것)
    full_positive_path = "../data/train_dataset_korquad" 
    
    # 2. Negative 생성의 기준이 될 원본 데이터
    org_data_path = "../data/train_dataset"
    
    # 3. 최종 저장 경로
    final_data_path = "../data/train_dataset_full_100000" 
    
    # 4. Negative 설정 (원본 질문 1개당 8개 생성)
    NUM_NEGATIVES = 8 
    
    # ---------------------------------------------------------
    
    # 1. 데이터 로드
    print("Loading datasets...")
    
    # (A) 전체 학습 데이터 (Positive용) - 약 64,000개
    if not os.path.exists(full_positive_path):
        print(f"Error: {full_positive_path}가 없습니다. add_korquad.py를 먼저 실행하세요.")
        return
    full_dataset = load_from_disk(full_positive_path)
    train_dataset_positive = full_dataset['train']
    
    # (B) 원본 데이터 (Negative 생성 대상) - 약 4,000개
    org_dataset = load_from_disk(org_data_path)['train']
    
    print(f"Total Positive Samples: {len(train_dataset_positive)}")
    print(f"Original Samples for Negative Mining: {len(org_dataset)}")

    # 2. Retriever 초기화
    print("Initializing Retriever...")
    retriever = ElasticSearchRetrieval(
        data_path="../data",
        context_path="wikipedia_documents.json",
        rerank_model_path="Dongjin-kr/ko-reranker" 
    )

    # 3. Negative Mining (원본 데이터에 대해서만 수행)
    negative_samples = []
    print(f"Mining negatives from ORIGINAL dataset only ({len(org_dataset)} samples)...")
    
    for example in tqdm(org_dataset):
        query = example['question']
        ground_truth = example['context']
        
        # ES로 상위 30개 검색
        _, indices = retriever.search_es(query, topk=30)
        
        collected = 0
        for idx in indices:
            candidate_ctx = retriever.contexts[idx]
            
            # 정답 문서와 다르면 Negative로 간주
            if candidate_ctx.strip() != ground_truth.strip():
                new_example = copy.deepcopy(example)
                new_example['context'] = candidate_ctx
                
                # 정답 비우기
                new_example['answers'] = {'text': [], 'answer_start': []}
                
                # ID 변경
                new_example['id'] = new_example['id'] + f"_neg_{collected}"
                
                negative_samples.append(new_example)
                collected += 1
                
            if collected >= NUM_NEGATIVES:
                break
    
    # 4. 데이터셋 병합
    print("Converting Negative samples to Dataset...")
    neg_dataset = Dataset.from_list(negative_samples, features=train_dataset_positive.features)
    print(f"Created {len(neg_dataset)} negative samples.")

    print("Concatenating Full Positive + Original Negatives...")
    # 전체 긍정 데이터(6.4만) + 원본 기반 부정 데이터(3.2만)
    final_train_dataset = concatenate_datasets([train_dataset_positive, neg_dataset])
    
    # 섞기 (Shuffle)
    final_train_dataset = final_train_dataset.shuffle(seed=42)

    # Validation은 원본 유지
    final_dataset = DatasetDict({
        'train': final_train_dataset,
        'validation': full_dataset['validation']
    })
    
    print(f"Final Train Size: {len(final_train_dataset)}")
    
    # 5. 저장
    final_dataset.save_to_disk(final_data_path)
    print(f"Saved final dataset to {final_data_path}")

if __name__ == "__main__":
    main()