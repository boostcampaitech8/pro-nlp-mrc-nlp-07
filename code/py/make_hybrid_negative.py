import os
import copy
import random
import pickle
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset
from retrieval import ElasticSearchRetrieval  # 하이브리드 기능이 포함된 최신 retrieval.py

def main():
    # === [설정] ===
    # 1. 원본 데이터 (Negative 생성 대상)
    org_data_path = "../data/train_dataset"
    
    # 2. Positive 데이터 (KorQuAD 합친 것)
    full_positive_path = "../data/train_dataset_korquad" 
    
    # 3. 최종 저장 경로 (하이브리드 네거티브 버전)
    final_data_path = "../data/train_dataset_hybrid_neg_v1" 
    
    # 4. Negative 설정
    # ES와 DPR에서 각각 후보를 가져와 섞습니다.
    NUM_NEGATIVES = 8  # 질문당 최종적으로 저장할 오답 개수
    
    # ---------------------------------------------------------
    
    # 1. 데이터 로드
    print("Loading datasets...")
    org_dataset = load_from_disk(org_data_path)['train']
    full_positive = load_from_disk(full_positive_path)
    train_dataset_positive = full_positive['train']
    
    print(f"Original Samples for Mining: {len(org_dataset)}")

    # 2. Retriever 초기화 (Hybrid 모드)
    print("Initializing Hybrid Retriever...")
    # retrieval.py에 하이브리드 기능이 구현되어 있어야 합니다.
    # rerank_model_path는 로드만 하고 사용하지 않습니다 (검색만 수행)
    retriever = ElasticSearchRetrieval(
        data_path="../data",
        context_path="wikipedia_documents.json",
        rerank_model_path="Dongjin-kr/ko-reranker" 
    )

    # 3. Hybrid Negative Mining
    negative_samples = []
    print(f"Mining Negatives using Hybrid Search (ES + DPR)...")
    
    for example in tqdm(org_dataset):
        query = example['question']
        ground_truth = example['context'].strip()
        
        # [핵심] ES와 DPR에서 각각 후보군을 가져옵니다.
        # Reranker를 거치지 않은 순수 검색 결과여야 '어려운 오답'이 많습니다.
        es_indices = retriever.search_es(query, topk=20)
        dpr_indices = retriever.search_dense(query, topk=20)
        
        # 두 결과를 합칩니다 (중복 제거)
        candidate_indices = list(set(es_indices) | set(dpr_indices))
        
        collected = 0
        for idx in candidate_indices:
            candidate_ctx = retriever.contexts[idx]
            
            # 정답 문서와 내용이 다르면 Negative로 간주
            if candidate_ctx.strip() != ground_truth:
                new_example = copy.deepcopy(example)
                new_example['context'] = candidate_ctx
                
                # 정답 비우기 (Empty Answer) -> 모델에게 '답 없음'을 학습
                new_example['answers'] = {'text': [], 'answer_start': []}
                new_example['id'] = new_example['id'] + f"_neg_{collected}"
                
                negative_samples.append(new_example)
                collected += 1
            
            # 목표 개수 채우면 중단
            if collected >= NUM_NEGATIVES:
                break
    
    # 4. 데이터셋 병합 및 저장
    print("Converting & Merging datasets...")
    neg_dataset = Dataset.from_list(negative_samples, features=train_dataset_positive.features)
    print(f"Created {len(neg_dataset)} Hybrid Negatives.")

    final_train_dataset = concatenate_datasets([train_dataset_positive, neg_dataset])
    final_train_dataset = final_train_dataset.shuffle(seed=42)

    final_dataset = DatasetDict({
        'train': final_train_dataset,
        'validation': full_positive['validation']
    })
    
    print(f"Final Train Size: {len(final_train_dataset)}")
    final_dataset.save_to_disk(final_data_path)
    print(f"Saved to {final_data_path}")

if __name__ == "__main__":
    main()