import os
import random
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict
from retrieval import ElasticSearchRetrieval

def main():
    # === [설정] ===
    org_data_path = "../data/train_dataset"
    output_path = "../data/train_dataset_concat_context"
    TOPK_RETRIEVAL = 5 
    
    print(f"Loading original dataset from {org_data_path}...")
    org_dataset = load_from_disk(org_data_path)
    train_dataset = org_dataset['train']
    
    print("Initializing Retriever (Hybrid + Rerank)...")
    retriever = ElasticSearchRetrieval(
        data_path="../data",
        context_path="wikipedia_documents.json",
        rerank_model_path="Dongjin-kr/ko-reranker", 
        dense_model_name="jhgan/ko-sroberta-multitask" 
    )

    new_data = []
    print(f"Augmenting Contexts (Concatenating Top-{TOPK_RETRIEVAL} passages)...")

    for example in tqdm(train_dataset):
        query = example['question']
        original_context = example['context']
        answer_text = example['answers']['text'][0]
        
        # 1. 검색 수행
        es_indices = retriever.search_es(query, topk=30)
        dpr_indices = retriever.search_dense(query, topk=30)
        candidates = list(set(es_indices) | set(dpr_indices))
        
        if not candidates:
            best_contexts = []
        else:
            _, rr_indices = retriever.rerank(query, candidates, topk=TOPK_RETRIEVAL)
            best_contexts = [retriever.contexts[i] for i in rr_indices]
        
        # 2. 정답 문서 포함 보장
        has_answer = False
        for ctx in best_contexts:
            if original_context.strip() in ctx.strip():
                has_answer = True
                break
        
        if not has_answer:
            if len(best_contexts) >= TOPK_RETRIEVAL:
                best_contexts[-1] = original_context
            else:
                best_contexts.append(original_context)

        # 3. 셔플 & 합치기
        random.shuffle(best_contexts)
        joined_context = " ".join(best_contexts)
        
        # 4. 정답 위치 재계산
        new_answer_start = joined_context.find(answer_text)
        
        if new_answer_start != -1:
            new_example = {
                "id": example["id"],
                "question": query,
                "context": joined_context,
                "title": example["title"],
                "answers": {
                    "text": [answer_text],
                    "answer_start": [new_answer_start]
                },
                "document_id": example.get("document_id", -1)
            }
            new_data.append(new_example)

    # 데이터셋 저장
    print(f"Processed {len(new_data)} samples.")
    
    # [수정됨] features 복사 후 불필요한 컬럼 제거
    # 원본 데이터셋의 features를 복사합니다.
    features = train_dataset.features.copy()
    
    # '__index_level_0__' 컬럼이 있다면 제거합니다. (오류 원인)
    if '__index_level_0__' in features:
        del features['__index_level_0__']
    
    # 수정된 features를 사용하여 데이터셋 생성
    final_dataset = Dataset.from_list(new_data, features=features)
    
    DatasetDict({
        'train': final_dataset,
        'validation': org_dataset['validation']
    }).save_to_disk(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()