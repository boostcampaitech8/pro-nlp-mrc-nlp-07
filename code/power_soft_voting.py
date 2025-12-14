import json
import pandas as pd
from collections import defaultdict
from glob import glob
from difflib import SequenceMatcher
import os

# ==========================================
# [설정] 파라미터 튜닝 구간
# ==========================================
INPUT_FOLDER = "./results_soft"        # JSON 파일들이 있는 폴더
OUTPUT_FILE = "final_submission_power_ensemble.csv" # 최종 저장될 파일명

# 1. Power 설정
# 1.0 = 일반 합산, 2.0 = 제곱 합산 (확신하는 답에 가산점), 3.0 = 세제곱
POWER = 2.0 

# 2. 모델별 가중치 설정
MODEL_WEIGHTS = {
    "71.25": 1.5,   # 예: predictions_submit_71.25.json
    "69.58": 1.0,   # 예: predictions_submit_65.42.json
    # 파일명에 위 키워드가 없으면 기본값 1.0 적용
}

# 유사도 기준 (0.8 이상이면 같은 답변으로 간주)
SIMILARITY_THRESHOLD = 0.8  
# ==========================================

def get_weight(filename):
    """파일명에 따라 가중치를 반환하는 함수"""
    base_name = os.path.basename(filename)
    for key, weight in MODEL_WEIGHTS.items():
        if key in base_name:
            return weight
    return 1.0 # 설정에 없으면 기본 가중치

def similar(a, b):
    # 유사도 검사
    return SequenceMatcher(None, a, b).ratio() > SIMILARITY_THRESHOLD

def main():
    # 1. 파일 목록 확인
    files = glob(os.path.join(INPUT_FOLDER, "*.json"))
    ensemble_predictions = defaultdict(list)

    if not files:
        print(f"Error: '{INPUT_FOLDER}' 폴더에 JSON 파일이 없습니다.")
        return

    print(f"총 {len(files)}개의 JSON 파일을 읽습니다...")
    print(f"적용된 전략: Power({POWER}) + Weights({MODEL_WEIGHTS})")

    # 2. JSON 파일 로드 및 데이터 수집
    for file_path in files:
        weight = get_weight(file_path)
        print(f" - Reading: {os.path.basename(file_path)} (Weight: {weight})")
        
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                predictions = json.load(f)
                for key, value in predictions.items():
                    # nbest 리스트인 경우
                    if isinstance(value, list):
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

    print("\nPower Soft Voting 계산 중...")
    final_results = []

    # 3. 그룹화 및 점수 계산 (핵심 로직)
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
        
        # (2) 그룹별 점수 계산 [Power * Weight]
        group_scores = {}
        for group_key, group_preds in grouped_predictions.items():
            # 점수 = (확률 ^ POWER) * (가중치)
            # 확률을 제곱하면 0.9 -> 0.81 (보존), 0.5 -> 0.25 (대폭 하락) 효과 발생
            score = sum(
                (p.get("probability", 0.0) ** POWER) * p.get("file_weight", 1.0) 
                for p in group_preds
            )
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
    
    # 제출 형식에 맞춰 저장 (헤더 없음, 탭 구분)
    df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
    
    print(f"완료! 결과가 '{OUTPUT_FILE}'에 저장되었습니다.")
    
    # 빈칸 체크
    empty_count = len(df[df['text'] == ""])
    if empty_count > 0:
        print(f"경고: 빈칸(No Answer)이 {empty_count}개 존재합니다.")

if __name__ == "__main__":
    main()