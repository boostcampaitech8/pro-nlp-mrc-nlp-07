import os
from huggingface_hub import login
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# ======================================================
# 1. Hugging Face 토큰 (Write 권한 필수!)
HF_TOKEN = "hf_@@@@@" 

# 2. 업로드할 로컬 모델 폴더 (학습 완료된 폴더 경로)
LOCAL_MODEL_PATH = "./models/roberta_large_tapt_qa" 

# 3. 허깅페이스에 올라갈 이름 (팀이름/모델명)
HUB_MODEL_ID = "NLP-07-ODQA/roberta-large-tapt-n8-hybrid" 
# ======================================================

def upload_to_huggingface():
    print("🔑 Hugging Face 로그인 시도 중...")
    try:
        # 코드에서 강제 로그인
        login(token=HF_TOKEN)
        print("✅ 로그인 성공!")
    except Exception as e:
        print(f"❌ 로그인 실패: {e}")
        print("토큰이 정확한지 확인해주세요.")
        return

    print(f"📂 로컬 모델 로딩 중... ({LOCAL_MODEL_PATH})")
    try:
        # 모델과 토크나이저 불러오기
        model = AutoModelForQuestionAnswering.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print(f"'{LOCAL_MODEL_PATH}' 경로에 학습된 모델 파일들이 있는지 확인해주세요.")
        return

    print(f"🚀 팀 저장소(NLP-07-ODQA)로 업로드 시작... ({HUB_MODEL_ID})")
    print("용량이 커서 시간이 조금 걸릴 수 있습니다...")

    try:
        # 허깅페이스로 푸시
        model.push_to_hub(HUB_MODEL_ID)
        tokenizer.push_to_hub(HUB_MODEL_ID)
        
        print("------------------------------------------------------")
        print("🎉 업로드 완료!")
        print(f"👉 확인 링크: https://huggingface.co/{HUB_MODEL_ID}")
        print("------------------------------------------------------")
    except Exception as e:
        print(f"❌ 업로드 중 에러 발생: {e}")
        print("혹시 권한 오류(403)라면, 팀장님께 멤버 초대를 받았는지 확인해보세요.")

if __name__ == "__main__":
    upload_to_huggingface()