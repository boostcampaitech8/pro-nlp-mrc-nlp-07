# ODQA(Open-Domain Question Answering Competition)


---


## 개요

| 항목 | 내용 |
|---|---|
| 프로젝트 주제 | 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정입니다. |
| 핵심 기능 | 질문에 관련된 문서를 찾는 Retriever와 찾아온 문서에서 질문에 대한 정답을 찾는 Reader로 구성됩니다. |
| 프로젝트 구성 | 질문에 관련된 문서를 찾는 Retriever와 찾아온 문서에서 질문에 대한 정답을 찾는 Reader로 구성됩니다. |
| 평가 지표 | Exact Match (EM) Score, F1 Score(참고용) |
| 진행 기간 | 2025.12.03 ~ 2025.12.11 |

<img width="801" height="309" alt="image" src="https://github.com/user-attachments/assets/6258bf8c-61bb-4075-862f-d6a4412927f3" />

---

## 최종 결과 (Public / Private)
<img width="987" height="121" alt="image" src="https://github.com/user-attachments/assets/15eeedf6-56c0-4408-8755-51aa7f2c217e" />
<img width="966" height="119" alt="image" src="https://github.com/user-attachments/assets/05dd7977-59b2-4ed9-b91b-32486ea8b965" />

---

## 팀원

| 이름 | 역할 |
|---|---|
| 가을 | Reader model 중 Decoder-only LLM 성능 비교 실험 |
| 박신지 | Sparse Retriever tokenizer 실험 및 Dense Retriever 단독 성능 비교 실험 |
| 박희권 | Sparse Retriever, Reader base model 비교 |
| 이준영 | Retriever 성능 개선 실험, Dense 파인튜닝 실험 |
| 이형석 | 데이터 증강 기반 TAPT 및 Reader model 개선 실험, Ensemble |

---

## Wrap-Up Report / 문서

[MRC-NLP-07 Wrap Up Report.pdf](https://github.com/user-attachments/files/25171997/MRC-NLP-07.Wrap.Up.Report.pdf)

---

## 폴더 구조

```text
pro-nlp-mrc-nlp-07/
├── .github/
├── code/
├── data/
├── .gitignore
└── README.md
```
### `.github/`
- GitHub Actions 기반 CI 워크플로우 및 자동화 설정을 관리합니다.

### `code/`
- MRC 파이프라인의 핵심 로직이 구현된 메인 소스 코드 디렉터리입니다.

### `data/`
- 원본 데이터, 전처리 데이터 및 문서 코퍼스를 저장합니다.

> 📌 본 프로젝트의 주요 구현 코드와 실험 관련 로직은 아래 경로에 정리되어 있습니다.  
> https://github.com/boostcampaitech8/pro-nlp-mrc-nlp-07/tree/lhs-06/code



