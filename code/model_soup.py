import torch
import os
from transformers import AutoModelForQuestionAnswering, AutoConfig, AutoTokenizer

# ====================================================
# [설정] 체크포인트 경로 및 가중치 설정
# ====================================================
# 경로를 본인 상황에 맞게 수정하세요.
MODELS = [
    {
        "path": "./models/roberta_large_tapt_qa", # 3에포크 (Best)
        "weight": 0.7  # 가중치 70%
    },
    {
        "path": "./models/roberta_large_tapt_qa_5EP/checkpoint-8960", # 4에포크
        "weight": 0.15 # 가중치 15%
    },
    {
        "path": "./models/roberta_large_tapt_qa_5EP/checkpoint-17920", # 5에포크
        "weight": 0.15 # 가중치 15%
    }
]
OUTPUT_DIR = "./models/roberta_soup_model"
# ====================================================

def main():
    print(f"Weighted Model Soup 시작! (총 {len(MODELS)}개 모델)")

    # 1. 첫 번째 모델(Best) 로드
    base_info = MODELS[0]
    print(f"Loading Base Model (Weight {base_info['weight']}): {base_info['path']}")
    base_model = AutoModelForQuestionAnswering.from_pretrained(base_info['path'])
    soup_dict = base_model.state_dict()
    
    # 베이스 모델에 가중치 곱하기
    for key in soup_dict:
        soup_dict[key] = soup_dict[key] * base_info['weight']

    # 2. 나머지 모델들 가중치 곱해서 더하기
    for info in MODELS[1:]:
        print(f"Adding Model (Weight {info['weight']}): {info['path']}")
        model = AutoModelForQuestionAnswering.from_pretrained(info['path'])
        new_dict = model.state_dict()
        
        for key in soup_dict:
            soup_dict[key] += new_dict[key] * info['weight']

    # (가중치 합이 1.0이므로 나누기 과정 불필요)

    # 3. 저장
    print(f"Saving Weighted Soup to {OUTPUT_DIR}...")
    base_model.load_state_dict(soup_dict)
    base_model.save_pretrained(OUTPUT_DIR)
    
    # Config & Tokenizer 저장
    config = AutoConfig.from_pretrained(base_info['path'])
    config.save_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(base_info['path'])
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("완료!")

if __name__ == "__main__":
    main()