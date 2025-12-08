import json
import pandas as pd
from collections import defaultdict
from glob import glob
from difflib import SequenceMatcher
import os

# ==========================================
# 설정 구간
# ==========================================
INPUT_FOLDER = "./results_soft"        # JSON 파일들이 있는 폴더
OUTPUT_FILE = "final_submission_soft_filled.csv" # 최종 저장될 파일명

# 유사도 기준 (0.8 이상이면 같은 답변으로 간주)
SIMILARITY_THRESHOLD = 0.8  

# [핵심] 모델별 가중치 설정 (파일명 일부 포함 시 적용)
# 예: 'uomnf97'이 파일명에 포함되면 점수 1.5배, 'roberta'는 1.0배
# 설정하지 않은 파일은 기본값(1.0)이 적용됩니다.
MODEL_WEIGHTS = {
    "65.42": 1.0,           # 예: n=4 모델 (기준)
    "71.25": 3.0, # 예: TAPT+n=8 모델 (에이스) -> 가중치 3배
    "64.17": 1.0          # 예: 성능 낮은 모델 -> 가중치 절반
}

# ==========================================

def get_weight(filename):
    """파일명에 따라 가중치를 반환하는 함수"""
    base_name = os.path.basename(filename)
    for key, weight in MODEL_WEIGHTS.items():
        if key in base_name:
            return weight
    return 1.0 # 기본 가중치

def similar(a, b):
    # 유사도 검사
    return SequenceMatcher(None, a, b).ratio() > SIMILARITY_THRESHOLD

def main():
    # 1. 파일 목록 확인
    files = glob(os.path.join(INPUT_FOLDER, "*.json"))
    ensemble_predictions = defaultdict(list)

    if not files:
        print(f"'{INPUT_FOLDER}' 폴더에 JSON 파일이 없습니다.")
        return

    print(f"총 {len(files)}개의 JSON 파일을 읽습니다...")
    print(f"적용된 가중치 설정: {MODEL_WEIGHTS}")

    # 2. JSON 파일 로드 및 데이터 수집 (가중치 적용)
    for file_path in files:
        weight = get_weight(file_path)
        print(f" - Reading: {os.path.basename(file_path)} (Weight: {weight})")
        
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                predictions = json.load(f)
                for key, value in predictions.items():
                    # nbest 리스트인 경우
                    if isinstance(value, list):
                        # 각 예측값에 파일 가중치를 함께 저장
                        for pred in value:
                            pred['file_weight'] = weight
                            ensemble_predictions[key].append(pred)
                    
                    # 단일 텍스트인 경우 (확률 1.0 * 가중치)
                    else:
                        ensemble_predictions[key].append({
                            "text": value, 
                            "probability": 1.0, 
                            "file_weight": weight
                        })
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    print("\nWeighted Soft voting 계산 중...")
    final_results = []

    # 3. 그룹화 및 점수 계산
    for key, predictions in ensemble_predictions.items():
        grouped_predictions = defaultdict(list)
        
        # (1) 유사한 답변끼리 그룹화
        for pred in predictions:
            text = pred.get("text", "") if isinstance(pred, dict) else pred
            
            if not text: continue # 빈 답변 무시
                
            added = False
            for group_key in grouped_predictions:
                if similar(text, group_key):
                    grouped_predictions[group_key].append(pred)
                    added = True
                    break
            if not added:
                grouped_predictions[text].append(pred)
        
        # (2) 그룹별 가중치 점수 합산
        group_scores = {}
        for group_key, group_preds in grouped_predictions.items():
            # 점수 = (모델 확률) * (모델 가중치)
            score = sum(p.get("probability", 0.0) * p.get("file_weight", 1.0) for p in group_preds)
            group_scores[group_key] = score
        
        # (3) 최고 점수 답변 선택
        if group_scores:
            best_text = max(group_scores, key=group_scores.get)
        else:
            best_text = ""
        
        final_results.append({'id': key, 'text': best_text})

    # 4. 결과 저장
    print("결과 저장 중...")
    df = pd.DataFrame(final_results)
    
    # id 순서로 정렬 (선택 사항)
    # df = df.sort_values(by='id')
    
    df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
    
    print(f"완료! '{OUTPUT_FILE}' 파일이 생성되었습니다.")
    
    empty_count = len(df[df['text'] == ""])
    print(f"-> 최종 결과에서 빈칸 개수: {empty_count}개")

if __name__ == "__main__":
    main()