import json
import os
import argparse
from tqdm import tqdm
from datasets import load_from_disk
from retrieval import ElasticSearchRetrieval

def main(args):
    # 1. 원본 데이터셋 로드
    # 대회에서 제공한 순수 train_dataset만 로드합니다.
    print(f"Loading ORIGINAL dataset from {args.dataset_name}...")
    if not os.path.exists(args.dataset_name):
        print(f"Error: {args.dataset_name} 경로가 존재하지 않습니다.")
        return

    org_dataset = load_from_disk(args.dataset_name)
    train_dataset = org_dataset['train']
    print(f"Total Original Samples: {len(train_dataset)}")

    # 2. Retriever 초기화 (Elasticsearch 연결)
    # Reranker 학습 전이라도 ES 검색은 가능하므로, 에러 방지용으로 기본 모델 이름만 넣어줍니다.
    print("Initializing Retriever for Negative Mining...")
    retriever = ElasticSearchRetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
        rerank_model_path="./models/my_reranker" # 로드만 하고 실제 Reranking은 안 함
    )

    # 3. Hard Negative Mining (오답 찾기)
    rerank_training_data = []
    
    print(f"Mining Hard Negatives (Top-{args.topk_search} search)...")
    for example in tqdm(train_dataset):
        query = example['question']
        ground_truth = example['context'] # 정답 문서 (Positive)

        # 3-1. Elasticsearch로 상위 N개 검색
        # Top-15 정도로 좁게 잡아야 '정답과 매우 유사한 강력한 오답(Hard Negative)'이 잡힙니다.
        _, indices = retriever.search_es(query, topk=args.topk_search)
        
        hard_negatives = []
        for idx in indices:
            candidate_context = retriever.contexts[idx]
            
            # 3-2. 정답 문서와 내용이 다른 경우에만 Negative로 추가
            if candidate_context.strip() != ground_truth.strip():
                hard_negatives.append(candidate_context)
            
            # 목표 개수만큼 모이면 중단
            if len(hard_negatives) >= args.num_negatives:
                break
        
        # 4. 저장할 데이터 구조 생성
        # (Negative가 하나라도 있어야 학습 가능)
        if len(hard_negatives) > 0:
            rerank_training_data.append({
                "question": query,
                "positive": ground_truth,
                "negatives": hard_negatives
            })

    # 5. 파일로 저장
    # 기존 파일과 섞이지 않게 파일명에 '_original'을 붙이는 것을 추천하거나,
    # train_reranker.py가 읽는 기본 파일명으로 덮어씁니다. (여기서는 기본 파일명 사용)
    output_path = os.path.join(args.data_path, "reranker_train_data_origin.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rerank_training_data, f, ensure_ascii=False, indent=4)
    
    print(f"데이터 생성 완료! 총 {len(rerank_training_data)}개의 학습 데이터가 저장되었습니다.")
    print(f"저장 경로: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # [중요] 원본 데이터셋 경로로 기본값 설정
    parser.add_argument("--dataset_name", type=str, default="../data/train_dataset")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--context_path", type=str, default="wikipedia_documents.json")
    
    # [설정 추천]
    # topk_search: 너무 넓게 잡으면(100) 쉬운 오답이 섞임. 15~20 추천.
    # num_negatives: 질문당 오답 몇 개 저장할지. 4~5개 추천.
    parser.add_argument("--topk_search", type=int, default=15)
    parser.add_argument("--num_negatives", type=int, default=5)
    
    args = parser.parse_args()
    main(args)