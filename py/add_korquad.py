import os
from datasets import load_dataset, load_from_disk, concatenate_datasets, Value, Features, Sequence, DatasetDict

def main():
    # 1. 경로 설정
    org_data_path = "../data/train_dataset"
    new_data_path = "../data/train_dataset_korquad"
    
    print(f"Loading original dataset from {org_data_path}...")
    # 기존 데이터 로드
    org_dataset = load_from_disk(org_data_path)
    
    print("Downloading KorQuAD 1.0...")
    # KorQuAD 1.0 다운로드
    korquad = load_dataset("squad_kor_v1")
    
    # 2. 데이터 형식 맞추기 (Schema Alignment)
    def transform_korquad(example):
        # document_id가 없으므로 임의의 값(-1) 부여
        example['document_id'] = -1 
        return example

    # KorQuAD 데이터에 부족한 컬럼 추가
    korquad_train = korquad['train'].map(transform_korquad)
    korquad_dev = korquad['validation'].map(transform_korquad) 
    
    # 불필요한 컬럼 제거 및 타입 통일
    features = org_dataset['train'].features.copy()
    if '__index_level_0__' in features:
        features.pop('__index_level_0__')

    # 3. 데이터 병합 (Concatenate)
    print("Concatenating datasets...")
    
    combined_train_dataset = concatenate_datasets([
        org_dataset['train'],
        korquad_train.cast(features),
        korquad_dev.cast(features)
    ])
    
    # [수정된 부분] DatasetDict로 명시적으로 묶어줍니다.
    combined_dataset = DatasetDict({
        'train': combined_train_dataset,
        'validation': org_dataset['validation']
    })
    
    print(f"Original Train Size: {len(org_dataset['train'])}")
    print(f"Augmented Train Size: {len(combined_dataset['train'])}")
    
    # 4. 저장
    print(f"Saving new dataset to {new_data_path}...")
    combined_dataset.save_to_disk(new_data_path)
    print("Done!")

if __name__ == "__main__":
    main()