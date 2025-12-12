import pandas as pd
import re
import os
import json

# === [설정] ===
# 입력 파일 경로 (원본 예측 파일 사용)
INPUT_CSV_PATH = "./outputs/submission_hybrid_neg_3ep/predictions_submit.csv"

# 출력 파일 경로
OUTPUT_DIR = "./outputs/submission_hybrid_neg_3ep"
OUTPUT_SAFE_CSV = os.path.join(OUTPUT_DIR, "predictions_safe.csv")
OUTPUT_SAFE_JSON = os.path.join(OUTPUT_DIR, "predictions_safe.json")

def preprocess_safe(text):
    text = str(text).strip()
    
    # 1. 공백/줄바꿈 정리 (안전함)
    text = text.replace("\n", " ").replace("\\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    
    # 2. [핵심] 안전한 조사만 제거 (Risk 최소화)
    # '이, 가, 로, 에, 와, 과, 의'는 삭제하지 않음 (단어의 일부일 가능성 높음)
    safe_josas = ['은', '는', '을', '를', '에게', '에서', '으로', '로서', '로써']
    
    if len(text) > 1:
        for josa in safe_josas:
            if text.endswith(josa):
                text = text[:-len(josa)]
                break
                
    # 3. 마침표 제거 (선택 사항: 보통 정답 끝에 .은 불필요함)
    if text.endswith("."):
        text = text[:-1]

    return text

def main():
    print(f"Loading CSV from {INPUT_CSV_PATH}...")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: 파일이 없습니다. {INPUT_CSV_PATH}")
        return

    # 데이터 로드
    try:
        df = pd.read_csv(INPUT_CSV_PATH, sep='\t', header=None, names=['id', 'prediction'])
    except:
        df = pd.read_csv(INPUT_CSV_PATH, sep='\t', header=None, names=['id', 'prediction'], engine='python')

    print(f"Original Data Sample:\n{df.head(3)}\n")

    # --- Safe 전처리 적용 ---
    df_safe = df.copy()
    df_safe['prediction'] = df_safe['prediction'].apply(preprocess_safe)
    
    # 저장
    df_safe.to_csv(OUTPUT_SAFE_CSV, sep='\t', index=False, header=False)
    
    # JSON 변환
    dict_safe = dict(zip(df_safe['id'], df_safe['prediction']))
    with open(OUTPUT_SAFE_JSON, 'w', encoding='utf-8') as f:
        json.dump(dict_safe, f, indent=4, ensure_ascii=False)

    print(f" Safe Preprocessing Done!")
    print(f"   - JSON: {OUTPUT_SAFE_JSON}")

    # --- 변경된 내용 확인 ---
    print("\n=== 변경된 샘플 (Top 5) ===")
    diff_count = 0
    for i in range(len(df)):
        orig = df.iloc[i]['prediction']
        safe = df_safe.iloc[i]['prediction']
        
        if orig != safe:
            diff_count += 1
            if diff_count <= 5:
                print(f"ID: {df.iloc[i]['id']}")
                print(f"  Before: {orig}")
                print(f"  After : {safe}")
                print("-" * 30)
    
    print(f"Total changed samples: {diff_count}")

if __name__ == "__main__":
    main()