import os
import re
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict
from bs4 import BeautifulSoup
from tqdm import tqdm

# HTML 태그 제거 함수
def clean_html(text):
    # BeautifulSoup을 이용해 텍스트만 추출 (separator=' '로 단어 붙음 방지)
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    # 연속된 공백 하나로 통일
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def main():
    # === 설정 ===
    # 1. 입력: 이전에 합친 (원본 + KorQuAD 1.0) 데이터
    prev_data_path = "../data/train_dataset_korquad"
    
    # 2. 출력: 최종 합체 데이터 (원본 + K1 + K2)
    output_path = "../data/train_dataset_korquad_all"
    
    print(f"Loading previous dataset from {prev_data_path}...")
    if not os.path.exists(prev_data_path):
        print("Error: 이전 단계 데이터가 없습니다. add_korquad.py를 먼저 실행하세요.")
        return
    prev_dataset = load_from_disk(prev_data_path)
    
    print("Downloading KorQuAD 2.0...")
    # HuggingFace에서 KorQuAD 2.0 로드
    korquad_v2 = load_dataset("squad_kor_v2")
    
    # --- [전처리 로직] ---
    print("Preprocessing KorQuAD 2.0 (Removing HTML & Re-indexing)...")
    
    valid_samples = []
    
    # tqdm으로 진행상황 표시
    for example in tqdm(korquad_v2['train']):
        original_context = example['context']
        answer_text = example['answers']['text'][0] # 첫 번째 정답 텍스트
        
        # 1. HTML 태그 제거
        cleaned_context = clean_html(original_context)
        
        # 2. 정답 위치 재검색
        # 태그가 사라지면서 글자 위치가 바뀌었으므로, 정답 텍스트로 다시 위치를 찾습니다.
        new_answer_start = cleaned_context.find(answer_text)
        
        # 3. 유효성 검사
        # 정답이 Cleaned Context 안에 그대로 살아있는 경우만 사용 (표 복잡도 등으로 사라졌으면 폐기)
        if new_answer_start != -1:
            # 데이터 구조 맞추기
            new_sample = {
                'id': example['id'],
                'title': example['title'],
                'context': cleaned_context,
                'question': example['question'],
                'answers': {
                    'text': [answer_text],
                    'answer_start': [new_answer_start]
                },
                'document_id': -1 # 더미 ID
            }
            valid_samples.append(new_sample)
    
    print(f"KorQuAD 2.0: Original {len(korquad_v2['train'])} -> Valid Preprocessed {len(valid_samples)}")
    
    # 리스트를 Dataset 객체로 변환
    from datasets import Dataset
    features = prev_dataset['train'].features # 기존 데이터셋과 feature 맞춤
    
    # K2 데이터셋 생성 (타입 강제 변환 포함)
    k2_dataset = Dataset.from_list(valid_samples).cast(features)

    # 4. 병합
    print("Concatenating all datasets...")
    final_train = concatenate_datasets([prev_dataset['train'], k2_dataset])
    
    final_dataset = DatasetDict({
        'train': final_train,
        'validation': prev_dataset['validation'] # 검증셋은 대회 원본 유지
    })
    
    print(f"Previous Size: {len(prev_dataset['train'])}")
    print(f"Final Size with KorQuAD 2.0: {len(final_dataset['train'])}")
    
    # 5. 저장
    final_dataset.save_to_disk(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()