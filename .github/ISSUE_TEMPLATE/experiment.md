---
name: 실험 이슈
about: 새로운 실험을 시작할 때 사용하는 템플릿
title: '[EXP] 실험명'
labels: experiment
assignees: ''
---

## 실험 상세

### 실험 목적
<!-- 이 실험의 목적과 기대 효과를 설명해주세요 -->

### 실험 설명
<!-- 실험에 대한 상세 설명을 작성해주세요 -->


## 데이터 처리 및 증강

### 원본 데이터
- **데이터셋**: 
- **데이터 개수**: 
- **데이터 전처리 방법**: 

### 데이터 증강
- **증강 방법**: 
  - 예: 원본 데이터(4000) + KorQuAD(1) + Negative(n=8)
- **증강된 데이터 개수**: 
- **증강 상세 설정**: 


## Retriever 구성

### Retriever 타입
- [ ] Sparse Retrieval (TF-IDF, BM25)
- [ ] Dense Retrieval (DPR, Dense Embedding)
- [ ] Hybrid Retrieval
- [ ] Elasticsearch
- [ ] 기타: 

### Retriever 설정
- **모델/방법**: 
- **Top-K**: 
- **하이퍼파라미터**: 
- **추가 설정**: 


## Reader 구성

### 모델 정보
- **베이스 모델**: 
  - 예: klue-roberta-large
- **모델 경로/저장소**: 

### 학습 설정
- **Pre-training/TAPT**: 
- **학습률 (learning_rate)**: 
- **배치 사이즈 (batch_size)**: 
- **에폭 (epochs)**: 
- **최대 시퀀스 길이 (max_seq_length)**: 
- **기타 하이퍼파라미터**: 


## 실험 계획

### 실험 단계
- [ ] 데이터 전처리 및 증강
- [ ] Retriever 학습/구성
- [ ] Reader 학습
- [ ] 모델 평가
- [ ] 결과 분석
- [ ] Hugging Face Hub 업로드

### 예상 결과
<!-- 예상되는 성능 지표나 결과를 작성해주세요 -->


## 관련 정보

### 브랜치
<!-- 실험 브랜치명 (예: ke-12) -->

### 참고 자료
<!-- 관련 논문, 문서, 이전 실험 링크 등 -->


## 체크리스트

- [ ] 실험 브랜치 생성 완료
- [ ] 실험 폴더 생성 완료
- [ ] 실험 계획 수립 완료
- [ ] 필요한 데이터/리소스 준비 완료

