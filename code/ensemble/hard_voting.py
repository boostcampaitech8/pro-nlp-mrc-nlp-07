import pandas as pd
import os
from glob import glob
from collections import Counter

# ==========================================
# 설정 구간
# ==========================================
INPUT_FOLDER = "./results_hard"       # 앙상블할 파일들이 있는 폴더
OUTPUT_FILE = "final_submission_hard.csv" # 최종 저장될 파일명

# 우선순위 파일명 리스트 (확장자 포함 정확히)
priority_order = [
    'predictions_submit_65.42.csv', 
    'predictions_submit_64.17.csv', 
    'predictions_submit_64.17(2).csv',
    'predictions_submit_60.42.csv'
]
# ==========================================

def read_submission_file(file_path):
    """
    제출 파일(탭 구분, 헤더 없음)을 읽어오는 함수
    """
    try:
        # 헤더가 없으므로 header=None, 구분자는 탭(\t)
        # names=['id', 'text']로 컬럼명 강제 지정
        df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'text'], quoting=3)
        # 데이터가 모두 문자열로 처리되도록 변환
        df['text'] = df['text'].fillna("").astype(str)
        return df
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {e}")
        return None

def hard_voting_with_priority(prediction_list, priority_order):
    final_result = {}
    
    if not prediction_list:
        return {}
        
    # 기준이 될 ID 리스트 (첫 번째 파일 기준)
    target_ids = list(prediction_list[0]['preds'].keys())
    
    for question_id in target_ids:
        answers = []
        for model_data in prediction_list:
            ans = model_data['preds'].get(question_id)
            if ans: # 빈 문자열이 아니면 추가
                answers.append(ans)
        
        answer_counts = Counter(answers)
        
        if answer_counts:
            max_count = max(answer_counts.values())
            top_answers = [ans for ans, count in answer_counts.items() if count == max_count]
            
            if len(top_answers) == 1:
                final_result[question_id] = top_answers[0]
            else:
                # 동점 처리: 우선순위 확인
                found = False
                for priority_model in priority_order:
                    # 해당 모델의 데이터 찾기
                    p_data = next((item for item in prediction_list if item['filename'] == priority_model), None)
                    if p_data:
                        p_ans = p_data['preds'].get(question_id)
                        if p_ans in top_answers:
                            final_result[question_id] = p_ans
                            found = True
                            break
                
                # 우선순위 모델에서도 답이 안 나오면 그냥 첫 번째 것 선택
                if not found:
                    final_result[question_id] = top_answers[0]
        else:
            final_result[question_id] = ""
            
    return final_result

def main():
    files = glob(os.path.join(INPUT_FOLDER, "*.csv"))
    if not files:
        print(f"'{INPUT_FOLDER}' 폴더에 파일이 없습니다.")
        return

    print(f"총 {len(files)}개의 파일을 발견했습니다.")
    prediction_list = []

    # 파일 읽기
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Reading: {filename}...")
        
        df = read_submission_file(file_path)
        if df is not None:
            preds = dict(zip(df['id'], df['text']))
            prediction_list.append({'filename': filename, 'preds': preds})
            print(f"  -> 성공! ({len(preds)}건)")

    # 하드 보팅 수행
    print("\n하드 보팅 계산 중...")
    final_dict = hard_voting_with_priority(prediction_list, priority_order)
    
    # 저장 (탭 구분, 헤더 없음)
    print("결과 저장 중...")
    submission_df = pd.DataFrame(list(final_dict.items()), columns=['id', 'text'])
    
    # to_csv 옵션 중요: sep='\t', header=False, index=False
    submission_df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
    
    print(f"완료! '{OUTPUT_FILE}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()