# 프로젝트 시작 가이드

## Git Clone 및 초기 설정

### 1. 프로젝트 클론
```bash
cd /data/ephemeral/home/{자신의 캠퍼 아이디}/
git clone https://{git username}:{PAT (github token (classic 추천))}@github.com/boostcampaitech8/pro-nlp-mrc-nlp-07.git
cd pro-nlp-mrc-nlp-07
```

### 2. Git 사용자 정보 설정
각 프로젝트 폴더마다 독립적으로 사용자 정보를 설정하기 위해 `--local` 옵션을 사용합니다.

```bash
git config --local user.name "사용자이름"
git config --local user.email "이메일@example.com"
```

> **참고**: `--local` 옵션을 사용하면 해당 저장소에만 설정이 적용되므로, 다른 폴더의 프로젝트에서는 다른 깃허브 계정을 사용할 수 있습니다.

설정 확인:
```bash
git config --local --list
```

## 브랜치 전략

### 기본 브랜치
- default 브랜치: `feedback`
- **작업 브랜치: `main` 브랜치에서 세부 브랜치를 나눠서 작업합니다.**

### 브랜치 생성 및 작업 흐름

1. **main 브랜치로 전환**
```bash
git checkout main
git pull origin main
```

2. **실험 브랜치 생성**
브랜치명 형식: `{팀원들의 이니셜}-{issue num}`

```bash
git checkout -b ke-12
git checkout -b xyz-15
git checkout -b abc-20
```

3. **실험 폴더 생성**
브랜치를 생성한 후, 각 실험별로 폴더를 만듭니다.
폴더명 형식: `{팀원들의 이니셜}-{issue num}` (브랜치명과 동일)

```bash
mkdir ke-12
mkdir xyz-15
mkdir abc-20
```

### 브랜치 및 폴더 예시
- 브랜치: `ke-12` → 폴더: `ke-12/`
- 브랜치: `xyz-15` → 폴더: `xyz-15/`
- 브랜치: `abc-20` → 폴더: `abc-20/`

### 브랜치 작업 흐름
```bash
# 1. main 브랜치 최신화
git checkout main
git pull origin main

# 2. 새 실험 브랜치 생성
git checkout -b ke-25

# 3. 실험 폴더 생성
mkdir ke-25

# 4. 작업 및 커밋
# ... 코드 작성 ...
git add .
git commit -m "[FEAT] ke-25 실험 코드 추가"

# 5. 브랜치 푸시
git push origin ksh-25
```

## 커밋 메시지 작성 가이드

### 커밋 메시지 형식
커밋 메시지는 다음 형식을 따라 작성합니다:

```
[분류] 제목

- 디테일
```

### 커밋 타입 (분류)
- `[FEAT]`: 새로운 기능 추가
- `[EXP]`: 새 실험 추가 (모델, 하이퍼파라미터 등)
- `[MODEL]`: 모델 아키텍처 변경 또는 새로운 모델 구현
- `[DATA]`: 데이터 전처리, 증강, 분석 관련
- `[CONFIG]`: 설정 파일 변경 (하이퍼파라미터, 경로 등)
- `[RESULT]`: 실험 결과 기록 및 로그
- `[EVAL]`: 평가 메트릭, 평가 스크립트 관련
- `[FIX]`: 버그 수정
- `[DOCS]`: 문서 수정 (README, 주석 등)
- `[STYLE]`: 코드 포맷팅, 세미콜론 누락 등
- `[REFACTOR]`: 코드 리팩토링
- `[TEST]`: 테스트 코드 추가 및 수정
- `[CHORE]`: 빌드 업무 수정, 패키지 매니저 설정 등
- `[COMMENT]`: 주석 추가 및 수정
- `[RENAME]`: 파일 또는 폴더명 변경
- `[REMOVE]`: 파일 삭제

### 커밋 메시지 예시
```
[EXP] exp-1: BERT-base baseline 실험 추가

- BERT-base 모델 학습 코드 구현
- 학습률 3e-5, 배치 사이즈 16 설정
```

```
[MODEL] RoBERTa-large 모델 아키텍처 추가

- RoBERTa-large 기반 MRC 모델 구현
- 커스텀 헤드 레이어 추가
```

```
[DATA] 데이터 전처리 파이프라인 개선

- 컨텍스트 길이 최적화 (512 → 384)
- 토큰화 전략 변경 (문장 단위 분할)
```

```
[RESULT] exp-2 실험 결과 기록

- EM Score: 85.2, F1 Score: 91.5
- 학습 loss 곡선 및 평가 결과 저장
```

```
[FIX] 데이터 로더 메모리 누수 문제 수정

- 배치 처리 시 불필요한 텐서 메모리 해제
- 데이터셋 크기 제한 로직 개선
```

## 이슈와 커밋 메시지 연결 방법

### GitHub 이슈 연결
커밋 메시지에서 이슈를 언급하면 자동으로 연결됩니다.

**형식:**
- `#이슈번호` - 커밋 메시지에 포함
- `Closes #이슈번호` - 이슈를 자동으로 닫음
- `Fixes #이슈번호` - 버그 이슈를 자동으로 닫음
- `Resolves #이슈번호` - 이슈 해결을 표시

**예시:**
```
[EXP] exp-3: 데이터 증강 실험 추가 #12

- Back-translation 기반 데이터 증강 구현
- 증강된 데이터셋으로 모델 학습
```

```
[FIX] 학습 중 메모리 부족 오류 수정 Fixes #15

- 배치 사이즈 동적 조정 로직 추가
- 그래디언트 누적 기법 적용
```

### 여러 이슈 언급
여러 이슈를 동시에 언급할 수 있습니다:
```
[EVAL] 평가 스크립트 개선 Closes #10, #11

- Exact Match 및 F1 Score 계산 함수 최적화
- 대용량 데이터셋 평가 지원 추가
```

