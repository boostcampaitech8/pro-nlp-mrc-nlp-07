# code/make_final_reader_data.py

import os
import random
import copy
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Dataset
from retrieval import ElasticSearchRetrieval

def main():
    # === 설정 ===
    org_data_path = "../data/train_dataset"       # 원본 데이터
    korquad_data_path = "../data/train_dataset_korquad" # KorQuAD 합친 데이터
    final_data_path = "../data/train_dataset_final_optimized" # 최종 저장 경로
    
    # KorQuAD 샘플링 개수 (시간 절약을 위해 12,000개만 사용)
    # 원본(4k) + KorQuAD(12k) + Negative(알파) = 약 2만개 예상 -> 학습 2~3시간 컷 가능
    KORQUAD_SAMPLE_SIZE = 12000 
    NUM_NEGATIVES = 4  # 질문당 만들 Negative 개수
    
    # 1. 데이터 로드
    print("Loading datasets...")
    org_dataset = load_from_disk(org_data_path)['train'] # 원본은 100% 사용
    
    if os.path.exists(korquad_data_path):
        full_korquad = load_from_disk(korquad_data_path)['train']
        # 원본 데이터(약 4000개)를 제외한 순수 KorQuAD 부분만 추출 (인덱스 슬라이싱 가정)
        # add_korquad.py에서 원본 뒤에 붙였으므로 뒤쪽 데이터만 가져옴
        only_korquad = full_korquad.select(range(len(org_dataset), len(full_korquad)))
        
        # 랜덤 샘플링
        print(f"Sampling {KORQUAD_SAMPLE_SIZE} from KorQuAD...")
        indices = range(len(only_korquad))
        sampled_indices = random.sample(indices, KORQUAD_SAMPLE_SIZE)
        korquad_subset = only_korquad.select(sampled_indices)
    else:
        print("KorQuAD dataset not found! Using original only.")
        korquad_subset = None

    # 2. 1차 병합 (Positive 데이터셋)
    if korquad_subset:
        positive_dataset = concatenate_datasets([org_dataset, korquad_subset])
    else:
        positive_dataset = org_dataset
        
    print(f"Positive Dataset Size: {len(positive_dataset)}")

    # 3. Negative Mining (Elasticsearch 사용)
    print("Initializing Retriever for Negative Mining...")
    # Reranker 학습 중이어도 ES는 사용 가능하므로 기본 모델로 로드
    retriever = ElasticSearchRetrieval(
        data_path="../data",
        context_path="wikipedia_documents.json",
        rerank_model_path="klue/roberta-large" 
    )

    negative_samples = []
    print("Mining negatives...")
    
    # 원본 데이터에 대해서만 Negative를 만들거나, 전체에 대해 만들 수 있음
    # 효율을 위해 '원본 데이터'에 대해서만 집중적으로 Negative를 만듭니다 (기출 변형 집중)
    target_for_neg = org_dataset 
    
    for example in tqdm(target_for_neg):
        query = example['question']
        ground_truth = example['context']
        
        # ES로 상위 10~20개 검색
        _, indices = retriever.search_es(query, topk=20)
        
        collected = 0
        for idx in indices:
            candidate_ctx = retriever.contexts[idx]
            
            # 정답 문서와 다르면 Negative로 간주
            if candidate_ctx.strip() != ground_truth.strip():
                new_example = copy.deepcopy(example)
                new_example['context'] = candidate_ctx
                # 정답 비우기 (Empty Answer)
                new_example['answers'] = {'text': [], 'answer_start': []}
                new_example['id'] = new_example['id'] + f"_neg_{collected}"
                
                negative_samples.append(new_example)
                collected += 1
                
            if collected >= NUM_NEGATIVES:
                break
    
    # Negative 데이터셋 변환
    neg_dataset = Dataset.from_list(negative_samples, features=positive_dataset.features)
    print(f"Created {len(neg_dataset)} negative samples.")

    # 4. 최종 병합
    print("Finalizing dataset...")
    final_train_dataset = concatenate_datasets([positive_dataset, neg_dataset])
    
    # 섞기 (Shuffle)
    final_train_dataset = final_train_dataset.shuffle(seed=42)

    final_dataset = DatasetDict({
        'train': final_train_dataset,
        'validation': load_from_disk(org_data_path)['validation']
    })
    
    print(f"Final Train Size: {len(final_train_dataset)}")
    
    # 저장
    final_dataset.save_to_disk(final_data_path)
    print(f"Saved final dataset to {final_data_path}")

if __name__ == "__main__":
    main()