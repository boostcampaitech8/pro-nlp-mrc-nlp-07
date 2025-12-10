# 📊 inference.py 테스트 및 성능 평가 가이드

## 🎯 개요

이 가이드는 `inference.py`를 사용하여 ODQA 시스템의 성능을 평가하는 방법을 설명합니다.

## 📈 측정 가능한 지표

### 1. **Retrieval 성능**
- **Recall@K**: Top-K 문서 중 정답 문서가 포함된 비율
- **Exact Match Recall**: 검색된 문서가 정답 문서와 정확히 일치하는 비율

### 2. **Reader (MRC) 성능**
- **EM (Exact Match)**: 예측 답변이 정답과 정확히 일치하는 비율
- **F1 Score**: 예측 답변과 정답 간의 토큰 레벨 F1 점수

### 3. **End-to-End 성능**
- Retrieval + Reader의 전체 파이프라인 성능

---

## 🚀 실행 방법

### **방법 1: End-to-End 평가 (EM, F1 측정)**

정답이 있는 validation 데이터로 전체 시스템 평가:

```bash
# TF-IDF 사용
python inference.py \
    --model_name_or_path klue/bert-base \
    --dataset_name ./data/train_dataset \
    --output_dir ./outputs/eval_tfidf/ \
    --do_eval \
    --eval_retrieval \
    --sparse_retrieval_type tfidf \
    --top_k_retrieval 10

# Evaluation bm25+
python inference.py \
    --model_name_or_path ./models/train_dataset/ \
    --dataset_name ../data/train_dataset \
    --output_dir ./outputs/eval_bm25plus/ \
    --do_eval \
    --eval_retrieval \
    --sparse_retrieval_type bm25plus \
    --top_k_retrieval 100

# Prediction bm25+
python inference.py \
    --model_name_or_path ./models/train_dataset/ \
    --dataset_name ../data/test_dataset \
    --output_dir ./outputs/predict_bm25plus_top10/ \
    --do_predict \
    --sparse_retrieval_type bm25plus \
    --top_k_retrieval 10

# Train Command

python train.py --output_dir ./models/train_dataset --model_name_or_path "HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned" --do_train

# Evaluation Command
python inference.py \
    --model_name_or_path ./models/train_dataset \
    --dataset_name ./data/train_dataset \
    --output_dir ./outputs/eval_sparse-bm25_top25/ \
    --do_eval \
    --eval_retrieval \
    --sparse_retrieval_type bm25 \
    --top_k_retrieval 25

# Prediction Command
python inference.py \
    --model_name_or_path ./models/train_dataset/ \
    --dataset_name ./data/test_dataset \
    --output_dir ./outputs/predict_sparse-bm25_top5/ \
    --do_predict \
    --sparse_retrieval_type bm25 \
    --top_k_retrieval 5
```


**출력 결과**:
```
test_exact_match: 65.4
test_f1: 72.3
test_samples: 600
```

결과는 `./outputs/eval_*/test_results.json`에 저장됩니다.

---

### **방법 2: Retrieval만 평가 (Recall 측정)**

새로 생성한 `evaluate_retrieval.py` 스크립트 사용:

```bash
# 단일 Retriever 평가
python evaluate_retrieval.py \
    --retriever_type tfidf \
    --top_k 10

# 두 Retriever 비교
python evaluate_retrieval.py \
    --retriever_type both \
    --top_k 10

# 여러 Top-K 값으로 종합 비교
python evaluate_retrieval.py --compare
```

**출력 예시**:
```
============================================================
Evaluating TFIDF Retrieval with Top-10
============================================================

Total Questions: 600
Top-10 Exact Match Recall: 78.50%
Top-10 Contains Recall: 85.30%
============================================================
```

---

### **방법 3: Prediction 모드 (정답 없는 테스트)**

```bash
python inference.py \
    --model_name_or_path klue/bert-base \
    --dataset_name ./data/test_dataset \
    --output_dir ./outputs/predictions/ \
    --do_predict \
    --eval_retrieval \
    --sparse_retrieval_type bm25 \
    --top_k_retrieval 10
```

결과는 `./outputs/predictions/predictions.json`에 저장됩니다.

---

## 📊 성능 비교 실험

### **실험 1: Retriever 비교 (TF-IDF vs BM25)**

```bash
# 스크립트 실행
bash compare_retrievers.sh
```

`compare_retrievers.sh` 내용:
```bash
#!/bin/bash

echo "Comparing TF-IDF and BM25 Retrievers"

for TOP_K in 1 5 10 20
do
    echo "================================"
    echo "Testing with Top-K = $TOP_K"
    echo "================================"
    
    # TF-IDF
    python inference.py \
        --model_name_or_path klue/bert-base \
        --dataset_name ./data/train_dataset \
        --output_dir ./outputs/tfidf_k${TOP_K}/ \
        --do_eval \
        --eval_retrieval \
        --sparse_retrieval_type tfidf \
        --top_k_retrieval $TOP_K
    
    # BM25
    python inference.py \
        --model_name_or_path klue/bert-base \
        --dataset_name ./data/train_dataset \
        --output_dir ./outputs/bm25_k${TOP_K}/ \
        --do_eval \
        --eval_retrieval \
        --sparse_retrieval_type bm25 \
        --top_k_retrieval $TOP_K
done

echo "Results saved in ./outputs/"
```


---

## 📁 결과 파일 위치

### **Evaluation 결과**
```
./outputs/eval_*/
├── test_results.json       # EM, F1 점수
├── eval_results.json       # 추가 평가 지표
└── all_results.json        # 전체 결과
```

### **Prediction 결과**
```
./outputs/predictions/
└── predictions.json        # 각 질문에 대한 예측 답변
```

### **결과 파일 예시**

`test_results.json`:
```json
{
  "test_exact_match": 65.4,
  "test_f1": 72.3,
  "test_samples": 600
}
```

`predictions.json`:
```json
{
  "mrc-0-000": "서울",
  "mrc-0-001": "1945년",
  ...
}
```

---

## 🔍 결과 분석 방법

### **Python으로 결과 읽기**

```python
import json

# EM, F1 점수 확인
with open('./outputs/eval_tfidf/test_results.json', 'r') as f:
    results = json.load(f)
    print(f"EM: {results['test_exact_match']:.2f}")
    print(f"F1: {results['test_f1']:.2f}")

# 예측 결과 확인
with open('./outputs/predictions/predictions.json', 'r') as f:
    predictions = json.load(f)
    for qid, answer in list(predictions.items())[:5]:
        print(f"{qid}: {answer}")
```

### **결과 비교 스크립트**

```python
import json
import pandas as pd

def compare_results(result_dirs):
    """여러 실험 결과를 비교합니다."""
    data = []
    for name, dir_path in result_dirs.items():
        with open(f'{dir_path}/test_results.json', 'r') as f:
            results = json.load(f)
            data.append({
                'Experiment': name,
                'EM': results['test_exact_match'],
                'F1': results['test_f1']
            })
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    return df

# 사용 예시
results = compare_results({
    'TF-IDF (k=5)': './outputs/tfidf_k5',
    'TF-IDF (k=10)': './outputs/tfidf_k10',
    'BM25 (k=5)': './outputs/bm25_k5',
    'BM25 (k=10)': './outputs/bm25_k10',
})
```

---

## 💡 성능 개선 팁

### **1. Top-K 값 조정**
```bash
# 더 많은 문서를 검색하면 Recall은 높아지지만 노이즈도 증가
--top_k_retrieval 20  # 기본값: 10
```

### **2. Retriever 선택**
```bash
# BM25가 일반적으로 TF-IDF보다 성능이 좋음
--sparse_retrieval_type bm25
```

### **3. FAISS 사용 (TF-IDF만)**
```bash
# 속도 향상 (정확도는 약간 감소 가능)
--use_faiss \
--num_clusters 64
```

### **4. 모델 Fine-tuning**
```bash
# 먼저 train.py로 학습
python train.py \
    --model_name_or_path klue/bert-base \
    --dataset_name ./data/train_dataset \
    --output_dir ./models/finetuned/ \
    --do_train \
    --num_train_epochs 3

# Fine-tuned 모델로 inference
python inference.py \
    --model_name_or_path ./models/finetuned/ \
    --do_eval
```

---

## 🐛 문제 해결

### **BM25 사용 시 에러**
```bash
# rank-bm25 설치 필요
pip install rank-bm25
```

### **메모리 부족**
```bash
# Batch size 줄이기
--per_device_eval_batch_size 8  # 기본값: 16
```

### **속도 개선**
```bash
# Worker 수 조정
--preprocessing_num_workers 4
--dataloader_num_workers 4
```

---

## 📌 요약

### **빠른 테스트**
```bash
python evaluate_retrieval.py --compare
```

### **전체 평가**
```bash
python inference.py \
    --do_eval \
    --eval_retrieval \
    --sparse_retrieval_type bm25 \
    --top_k_retrieval 10
```

### **결과 확인**
```bash
cat ./outputs/eval_*/test_results.json
```
