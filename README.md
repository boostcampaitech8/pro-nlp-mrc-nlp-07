# LLM Ensemble System

LLM을 사용한 앙상블 시스템입니다. 여러 CSV 파일의 답변을 집계하고, LLM을 사용하여 가장 적절한 답변을 선택합니다.

## 📁 폴더 구조

```
llm-ensemble/
├── data/
│   └── test_dataset/     # 질문과 ID가 포함된 테스트 데이터셋
├── csv/                  # 답변 CSV 파일들을 여기에 넣으세요
│   ├── answer1.csv
│   ├── answer2.csv
│   └── answer3.csv
├── config.py            # 설정 파일
├── ensemble.py          # 메인 앙상블 스크립트
└── README.md
```

## 📋 입력 데이터 형식

### 1. 테스트 데이터셋 (`./data/test_dataset`)
- HuggingFace datasets 형식
- 필드: `id`, `question`

### 2. 답변 CSV 파일들 (`./csv/*.csv`)
- 탭으로 구분된 CSV 파일
- 헤더 없음
- 형식: `id\tanswer`
- 예시:
  ```
  mrc-1-000653	사만
  mrc-1-001113	냉전 종식
  mrc-0-002191	대통령인 빌헬름 미클라스
  ```

## 🚀 사용 방법

### 1. CSV 파일 준비
답변 CSV 파일들을 `./csv` 폴더에 넣으세요:
```bash
cp /path/to/answer1.csv ./csv/
cp /path/to/answer2.csv ./csv/
cp /path/to/answer3.csv ./csv/
```

### 2. 기본 실행
```bash
python ensemble.py
```

### 3. 옵션을 사용한 실행
```bash
# 다른 모델 사용
python ensemble.py --model beomi/Llama-3-Open-Ko-8B-Instruct-preview

# 출력 파일 지정
python ensemble.py --output ./my_output.csv

# Temperature 조정 (더 다양한 답변)
python ensemble.py --temperature 0.3

# CSV 폴더 지정
python ensemble.py --csv_folder ./my_csv_folder
```

## ⚙️ 설정

`config.py`에서 다음 설정을 변경할 수 있습니다:

```python
# 모델 설정
model_name_or_path = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

# 생성 파라미터
max_new_tokens = 50        # 생성할 최대 토큰 수
temperature = 0.1          # 낮을수록 더 결정적
do_sample = False          # False = greedy decoding

# 경로
test_dataset_path = "./data/test_dataset"
csv_folder = "./csv"
output_file = "./ensemble_output.csv"
```

## 📊 출력

### 1. 결과 CSV 파일 (`ensemble_output.csv`)
- 탭으로 구분
- 헤더 없음
- 형식: `id\tanswer`

### 2. 디버그 파일 (`ensemble_output_debug.json`)
- 각 질문의 선택지 목록
- 검증 결과
- 통계 정보

## 🔍 작동 방식

1. **데이터 로드**: 테스트 데이터셋과 모든 CSV 파일 로드
2. **답변 집계**: 각 질문 ID에 대해 모든 CSV에서 답변 수집
3. **중복 제거**: 동일한 답변은 하나만 보기에 포함
4. **프롬프트 생성**: 다음 형식으로 프롬프트 생성
   ```
   다음 중 아래 질문에 대한 답으로 가장 적절한 것을 보기 중에서 고르시오. 
   답변을 작성할 때는 부가적인 내용을 덧붙이지 말고, 반드시 보기 중에 선택한 것 만을 그대로 작성하시오.

   질문: (질문 내용)

   보기:
   - (답변1)
   - (답변2)
   - (답변3)
   ```
5. **LLM 호출**: 각 프롬프트에 대해 LLM이 답변 생성
6. **답변 검증**: 생성된 답변이 보기 중 하나와 일치하는지 확인
7. **결과 저장**: 최종 답변을 CSV 파일로 저장

## 📈 검증 통계

실행 후 다음과 같은 통계가 출력됩니다:
```
Validation Statistics:
  Total questions: 600
  Valid answers: 580 (96.7%)
  Invalid answers: 15 (2.5%)
  No choices found: 5
```

- **Valid answers**: LLM 답변이 보기 중 하나와 일치
- **Invalid answers**: LLM 답변이 보기와 일치하지 않음 (그래도 답변은 저장됨)
- **No choices found**: 해당 질문 ID에 대한 답변이 CSV에 없음

## 💡 팁

1. **단일 답변**: 모든 CSV에서 동일한 답변만 있으면 LLM을 호출하지 않고 바로 사용
2. **검증**: `strict_validation=True`로 설정하면 답변 검증 수행
3. **로깅**: 처음 몇 개 예제의 프롬프트와 답변이 출력되어 확인 가능
4. **GPU**: CUDA가 있으면 자동으로 GPU 사용

## 🐛 문제 해결

### CSV 파일을 찾을 수 없음
```
ValueError: No CSV files found in ./csv
```
→ `./csv` 폴더에 `.csv` 파일이 있는지 확인

### 메모리 부족
→ `config.py`에서 `batch_size`를 줄이거나 더 작은 모델 사용

### 답변이 보기와 일치하지 않음
→ `temperature`를 낮추거나 (0.1 → 0.05) 프롬프트 확인
