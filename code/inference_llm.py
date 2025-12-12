import torch
import pandas as pd
import json
import os
from tqdm import tqdm
from unsloth import FastLanguageModel
from datasets import load_dataset, load_from_disk
from retrieval import ElasticSearchRetrieval

# === [설정] ===
MODEL_PATH = "./models/llama3_mrc_lora_optimized"
TEST_DATA_PATH = "../data/test_dataset"
OUTPUT_CSV = "submission_llm.csv"

# [수정 1] 입력 길이 넉넉하게 (에러 방지)
MAX_SEQ_LENGTH = 9000

# [수정 2] 검색 문서 개수
TOP_K_RETRIEVAL = 6

# [추가] 검색 결과를 저장할 파일명 (캐시 파일)
RETRIEVAL_CACHE_FILE = f"./retrieval_results_top{TOP_K_RETRIEVAL}.csv"

def main():
    # ---------------------------------------------------------
    # 1. Retrieval 수행 (캐시 확인 로직 추가)
    # ---------------------------------------------------------
    
    # 만약 저장해둔 검색 결과 파일이 있다면? -> 로드 (시간 절약!)
    if os.path.exists(RETRIEVAL_CACHE_FILE):
        print(f"Found cached retrieval results! Loading from {RETRIEVAL_CACHE_FILE}...")
        retrieved_df = pd.read_csv(RETRIEVAL_CACHE_FILE)
        print(f"Loaded {len(retrieved_df)} items.")
        
    # 없다면? -> 검색 수행 후 저장
    else:
        print("No cache found. Running Retrieval from scratch...")
        
        # 리트리버 초기화
        retriever = ElasticSearchRetrieval(
            data_path="../data",
            context_path="wikipedia_documents.json",
            rerank_model_path="Dongjin-kr/ko-reranker"
        )
        
        # 테스트 데이터셋 로드
        try:
            dataset = load_from_disk(TEST_DATA_PATH)["validation"]
        except:
            print("Warning: load_from_disk failed. Trying load_dataset...")
            dataset = load_dataset(TEST_DATA_PATH)["validation"]

        # 검색 실행
        retrieved_df = retriever.retrieve(dataset, topk=TOP_K_RETRIEVAL)
        
        # [중요] 결과 저장 (다음번엔 이걸 씁니다)
        print(f"Saving retrieval results to {RETRIEVAL_CACHE_FILE}...")
        retrieved_df.to_csv(RETRIEVAL_CACHE_FILE, index=False)
        print("Retrieval Done & Saved!")

    # ---------------------------------------------------------
    # 2. LLM 모델 로드 (Inference 모드)
    # ---------------------------------------------------------
    print(f"Loading LLM Model from {MODEL_PATH}...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = MODEL_PATH,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("학습이 완료되었는지, 경로가 맞는지 확인해주세요.")
        return

    # 3. 프롬프트 포맷
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
지문을 읽고 질문에 대한 정답을 한 단어 또는 짧은 구문으로 추출하세요.

### Input:
지문: {context}
질문: {question}

### Response:
"""

    results = []
    print("Generating Answers with LLM...")
    
    # DataFrame을 순회하며 추론
    for idx, row in tqdm(retrieved_df.iterrows(), total=len(retrieved_df)):
        # 1. 입력 텍스트 생성
        input_text = alpaca_prompt.format(
            context=row["context"],
            question=row["question"]
        )
        
        # 2. 토큰화
        inputs = tokenizer([input_text], return_tensors="pt").to("cuda")
        
        # 3. 생성 (Generate)
        try:
            outputs = model.generate(
                **inputs, 
                max_new_tokens = 32,
                use_cache = True,
                pad_token_id = tokenizer.eos_token_id
            )
            
            # 4. 정답 파싱
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if "### Response:\n" in decoded:
                answer = decoded.split("### Response:\n")[-1].strip()
            else:
                answer = decoded.replace(input_text, "").strip()
        except Exception as e:
            print(f"Error generating for ID {row['id']}: {e}")
            answer = "" # 에러 발생 시 빈칸 처리
        
        results.append({"id": row["id"], "text": answer})

    # 4. 저장
    print(f"Saving final submission to {OUTPUT_CSV}...")
    df = pd.DataFrame(results)
    df = df.rename(columns={"text": "PredictionString"})
    df.to_csv(OUTPUT_CSV, index=False)
    print("Inference Finished!")

if __name__ == "__main__":
    main()