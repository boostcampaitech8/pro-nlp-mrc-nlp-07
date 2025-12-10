import pandas as pd
import os

# ==========================================
# [설정] 파일 경로
# ==========================================
INPUT_FILE = "./results_hard/predictions_submit_tapt_3EP_cleaned2.csv"  # 문제가 되는 파일
OUTPUT_FILE = "predictions_submit_tapt_3EP_cleaned3.csv"          # 저장할 파일명

def fix_structure_keep_content():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    print(f"📂 파일 읽기 (Raw Line Mode): {INPUT_FILE}")
    
    data = []
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"   -> 총 {len(lines)}줄 읽음")
        
        for i, line in enumerate(lines):
            # 1. 줄바꿈 문자 제거
            line = line.strip()
            if not line: continue # 빈 줄 건너뜀
            
            # 2. 첫 번째 탭(\t)을 기준으로 딱 한 번만 나눔
            # (정답 내용 안에 탭이 있어도, 그건 내용으로 침)
            parts = line.split('\t', 1)
            
            if len(parts) == 2:
                # 정상: ID와 정답이 모두 있음
                row_id, row_text = parts
            else:
                # 비정상(에러 원인): 탭이 없어서 쪼개지지 않음 -> 정답을 빈칸으로 처리
                row_id = parts[0]
                row_text = "" 
                # print(f"⚠️ 경고: {i+1}번째 줄 포맷 복구 (ID: {row_id})")

            # 3. 데이터 저장 (어떤 처리도 하지 않음)
            data.append({'id': row_id, 'text': row_text})

        # 4. DataFrame 생성
        df = pd.DataFrame(data)
        
        # 5. 저장 (탭 구분, 헤더 없음)
        # quoting=3 (csv.QUOTE_NONE) 옵션은 쓰지 않습니다. (안전하게 pandas 기본값 사용)
        # 정답 안에 따옴표가 있어도 그대로 유지됩니다.
        df.to_csv(OUTPUT_FILE, sep='\t', header=False, index=False)
        
        print(f"💾 복구 및 저장 완료: {OUTPUT_FILE}")
        print("   (내용 수정 없이 구조만 맞췄습니다.)")
        
        # 확인용 미리보기
        print("\n[미리보기 (따옴표 유지 확인)]")
        print(df.head().to_string(index=False, header=False))

    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    fix_structure_keep_content()