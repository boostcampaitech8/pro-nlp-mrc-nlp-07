# Readme

## 소개

Open-Domain Question Answering (ODQA) 대회를 위한 베이스라인 코드 

## 설치 방법

### 요구 사항

```
# data (51.2 MB)
tar -xzf data.tar.gz

# 필요한 파이썬 패키지 설치. 
pip install -r requirements.txt
```

### 환경 변수 설정

Hugging Face Hub에 모델을 업로드하기 위해 `.env` 파일을 생성하고 토큰을 설정합니다.

`.env`파일은 `pro-nlp-mrc-nlp-07` 폴더 바로 아래에 생성합니다.

```bash
# 프로젝트 루트 디렉토리에 .env 파일 생성
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

> **참고**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다. 각 팀원이 자신의 토큰을 설정해야 합니다.

## 파일 구성


### 저장소 구조

```bash
../data/                 # 전체 데이터. 아래 상세 설명
./assets/                # readme 에 필요한 이미지 저장 
requirements.txt         # 요구사항 설치 파일
retrieval.py             # sparse retreiver 모듈 제공 
arguments.py             # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음
trainer_qa.py            # MRC 모델 학습에 필요한 trainer 제공.
utils_qa.py              # 기타 유틸 함수 제공 

train.py                 # MRC, Retrieval 모델 학습 및 평가 
inference.py		     # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
```

## 데이터 소개

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```bash
../data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

# 훈련, 평가, 추론

### train

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 

roberta 모델을 사용할 경우 tokenizer 사용시 아래 함수의 옵션을 수정해야합니다.
베이스라인은 klue/bert-base로 진행되니 이 부분의 주석을 해제하여 사용해주세요 ! 
tokenizer는 train, validation (train.py), test(inference.py) 전처리를 위해 호출되어 사용됩니다.
(tokenizer의 return_token_type_ids=False로 설정해주어야 함)

```python
# train.py
def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )
```

```bash
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train

# Hugging Face Hub에 모델 업로드 (학습 완료 후 자동 업로드)
# 모델명만 지정하면 자동으로 NLP-07-ODQA organization에 업로드됩니다
python train.py \
  --output_dir ./models/train_dataset \
  --do_train \
  --push_to_hub \
  --hub_repo_id your-model-name

# 또는 organization을 포함하여 직접 지정할 수도 있습니다
python train.py \
  --output_dir ./models/train_dataset \
  --do_train \
  --push_to_hub \
  --hub_repo_id NLP-07-ODQA/your-model-name
```

> **참고**: 
> - `--push_to_hub` 옵션을 사용하면 학습 완료 후 자동으로 Hugging Face Hub에 모델과 토크나이저가 업로드됩니다.
> - `.env` 파일에 `HF_TOKEN`이 설정되어 있어야 합니다.
> - `--hub_repo_id`에 organization이 포함되지 않으면 자동으로 `NLP-07-ODQA` organization에 업로드됩니다.
> - `--hub_repo_id`를 지정하지 않으면 `TrainingArguments`의 `--hub_model_id`를 사용합니다.
> - 모델은 [NLP-07-ODQA organization](https://huggingface.co/NLP-07-ODQA)에서 확인할 수 있습니다.

### eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# mrc 모델 평가 (train_dataset 사용)
python train.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
```

### inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

```bash
# ODQA 실행 (로컬 모델 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python inference.py --output_dir ./outputs/test_dataset/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/train_dataset/ --do_predict

# ODQA 실행 (Hugging Face Hub에서 모델 다운로드)
# 모델명만 지정하면 자동으로 NLP-07-ODQA organization에서 찾습니다
python inference.py \
  --output_dir ./outputs/test_dataset/ \
  --dataset_name ../data/test_dataset/ \
  --hub_model_repo_id 20251207-test \
  --do_predict

# 특정 브랜치(체크포인트)에서 모델 다운로드
python inference.py \
  --output_dir ./outputs/test_dataset/ \
  --dataset_name ../data/test_dataset/ \
  --hub_model_repo_id 20251207-test \
  --hub_model_revision checkpoint-1000 \
  --do_predict

# 전체 경로로 지정 (다른 organization의 모델)
python inference.py \
  --output_dir ./outputs/test_dataset/ \
  --dataset_name ../data/test_dataset/ \
  --hub_model_repo_id NLP-07-ODQA/20251207-test \
  --do_predict
```

> **참고**: 
> - `--hub_model_repo_id`를 사용하면 Hugging Face Hub에서 모델을 자동으로 다운로드하여 `./models/` 디렉토리에 저장합니다.
> - 모델명만 지정하면 자동으로 `NLP-07-ODQA` organization에서 찾습니다.
> - `--hub_model_revision`으로 특정 브랜치(체크포인트)를 지정할 수 있습니다 (기본값: `main`).
> - 다운로드한 모델은 `./models/{모델명}` 또는 `./models/{모델명}_{브랜치명}` 폴더에 저장됩니다.

### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

## Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 기존 파일이 load 됩니다.

2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. `./outputs/` 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.

4. **모델 공유**: 학습된 모델은 Hugging Face Hub에 업로드하여 팀원들과 공유합니다. GitHub에는 모델 체크포인트가 커밋되지 않도록 `.gitignore`에 설정되어 있습니다.

5. **데이터 보안**: 데이터세트(`data/`)는 절대 GitHub에 커밋하지 않습니다. `.gitignore`에 포함되어 있지만, 실수로 커밋하지 않도록 주의하세요.