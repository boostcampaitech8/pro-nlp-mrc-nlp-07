# 📊 inference.py 테스트 및 성능 평가 가이드

## 🎯 개요

이 가이드는 `inference.py`를 사용하여 ODQA 시스템의 성능을 평가하는 방법을 설명합니다.

---

## 🚀 실행 방법

### **방법 1: End-to-End 평가 (EM, F1 측정)**

정답이 있는 validation 데이터로 전체 시스템 평가:

```bash

python inference.py \
    --output_dir ./outputs/rag_result \
    --dataset_name ./data/test_dataset \
    --model_name_or_path "NLP-07-ODQA/roberta-large-tapt-n8-hybrid" \
    --do_predict
```
