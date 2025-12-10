import pandas as pd
import os

# ==========================================
# [설정] 수정할 파일 경로
# ==========================================
INPUT_FILE = "./results_hard/predictions_submit_tapt_3EP.csv"
OUTPUT_FILE = "predictions_submit_tapt_3EP_final_fixed.csv"

def fix_text_logic(text):
    """
    텍스트의 따옴표, 괄호 짝을 맞추고 불필요한 중복을 제거하는 함수
    """
    if not isinstance(text, str):
        return ""
    
    # 공백 제거
    text = text.strip()

    # 1. 3중 따옴표 (""" or ''') -> 1개로 축소
    # 예: """연금""" -> "연금"
    while '"""' in text:
        text = text.replace('"""', '"')
    while "'''" in text:
        text = text.replace("'''", "'")

    # 2. 쌍따옴표 (") 짝 맞추기
    # 따옴표 개수가 홀수일 때만 작동 (이미 짝이 맞으면 건드리지 않음)
    if text.count('"') % 2 != 0:
        if text.startswith('"'): # 시작은 있는데 끝이 없으면
            text += '"'
        elif text.endswith('"'): # 끝은 있는데 시작이 없으면
            text = '"' + text

    # 3. 홑따옴표 (') 짝 맞추기
    if text.count("'") % 2 != 0:
        if text.startswith("'"):
            text += "'"
        elif text.endswith("'"):
            text = "'" + text

    # 4. 꺾쇠 괄호 (《 》) 짝 맞추기
    has_open = '《' in text
    has_close = '》' in text
    
    # 여는 건 있는데 닫는 게 없으면 -> 뒤에 추가
    if has_open and not has_close:
        text += '》'
    # 닫는 건 있는데 여는 게 없으면 -> 앞에 추가
    elif has_close and not has_open:
        text = '《' + text

    return text

def main():
    print(f"📂 파일 읽기 및 복구 시작: {INPUT_FILE}")
    
    if not os.path.exists(INPUT_FILE):
        print("❌ 파일을 찾을 수 없습니다.")
        return

    data = []
    
    # 1. 파일을 Raw Text로 한 줄씩 읽기 (파싱 에러 방지)
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"   -> 총 {len(lines)}줄 읽음")
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # 탭으로 분리 (첫 번째 탭 기준)
            parts = line.split('\t', 1)
            
            if len(parts) >= 2:
                row_id = parts[0]
                original_text = parts[1]
            else:
                # 탭이 없으면(깨진 경우) 그냥 빈칸 처리하거나 넘김
                row_id = parts[0]
                original_text = ""
            
            # 2. 텍스트 수정 로직 적용
            fixed_text = fix_text_logic(original_text)
            
            # 수정된 내용 로그 출력 (확인용)
            if original_text != fixed_text:
                print(f"   [수정] {original_text}  ->  {fixed_text}")
                
            data.append({'id': row_id, 'text': fixed_text})

        # 3. 결과 저장
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
        
        print(f"\n✅ 복구 완료! 저장된 파일: {OUTPUT_FILE}")
        print(f"   -> 데이터 개수: {len(df)}")

    except Exception as e:
        print(f"❌ 처리 중 에러 발생: {e}")

if __name__ == "__main__":
    main()