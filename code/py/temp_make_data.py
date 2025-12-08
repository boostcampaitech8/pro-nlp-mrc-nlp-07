import os
import copy
import random
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset, Features
from retrieval import ElasticSearchRetrieval

def main():
    # === 설정 ===
    full_positive_path = "../data/train_dataset_korquad"
    org_data_path = "../data/train_dataset"
    final_data_path = "../data/train_dataset_final_balanced"
    NUM_NEGATIVES = 4
    
    print("Loading datasets...")
    full_dataset = load_from_disk(full_positive_path)
    train_dataset_positive = full_dataset['train']
    org_dataset = load_from_disk(org_data_path)['train']
    
    print("Initializing Retriever...")
    retriever = ElasticSearchRetrieval(
        data_path="../data",
        context_path="wikipedia_documents.json",
        rerank_model_path="Dongjin-kr/ko-reranker"
    )

    negative_samples = []
    print(f"Mining {NUM_NEGATIVES} negatives...")
    for example in tqdm(org_dataset):
        query = example['question']
        ground_truth = example['context']
        
        # [수정됨] Hybrid 리트리버는 indices 리스트만 반환하므로 변수 하나로 받습니다.
        indices = retriever.search_es(query, topk=30)
        
        collected = 0
        for idx in indices:
            candidate_ctx = retriever.contexts[idx]
            if candidate_ctx.strip() != ground_truth.strip():
                new_example = copy.deepcopy(example)
                new_example['context'] = candidate_ctx
                new_example['answers'] = {'text': [], 'answer_start': []}
                new_example['id'] = new_example['id'] + f"_neg_{collected}"
                negative_samples.append(new_example)
                collected += 1
            if collected >= NUM_NEGATIVES:
                break
    
    print(f"Mining done. Created {len(negative_samples)} negative samples.")

    # [수정됨] Features 처리 및 데이터셋 생성
    # 원본 데이터셋(train_dataset_positive)의 features 구조를 복사합니다.
    features = train_dataset_positive.features.copy()
    
    # '__index_level_0__' 컬럼이 있다면 제거하여 충돌 방지
    if '__index_level_0__' in features:
        del features['__index_level_0__']
    
    # Negative 데이터셋 생성
    neg_dataset = Dataset.from_list(negative_samples, features=features)
    
    # [수정됨] Positive 데이터와 Negative 데이터 병합 (Concatenate)
    print("Concatenating Positive and Negative datasets...")
    
    # Positive 데이터에서도 혹시 모를 충돌 방지를 위해 __index_level_0__ 제거
    if '__index_level_0__' in train_dataset_positive.features:
        train_dataset_positive = train_dataset_positive.remove_columns('__index_level_0__')
        
    final_train = concatenate_datasets([train_dataset_positive, neg_dataset])
    final_train = final_train.shuffle(seed=42)
    
    print(f"Final Train Size: {len(final_train)}")

    # 저장
    DatasetDict({
        'train': final_train,
        'validation': full_dataset['validation']
    }).save_to_disk(final_data_path)
    
    print(f"Saved to {final_data_path}")

if __name__ == "__main__":
    main()