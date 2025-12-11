#!/usr/bin/env python3
"""
Predictions JSON 파일에서 "**"를 제거하는 후처리 스크립트
"""

import json
import argparse
import os
from pathlib import Path


def remove_asterisks(text: str) -> str:
    """
    텍스트에서 "**"를 제거합니다.
    
    Args:
        text: 원본 텍스트
    
    Returns:
        "**"가 제거된 텍스트
    """
    if not isinstance(text, str):
        return text
    
    # "**텍스트**" 형식을 "텍스트"로 변환
    return text.replace("*", "")


def postprocess_predictions(input_file: str, output_file: str = None, in_place: bool = False):
    """
    Predictions JSON 파일을 후처리합니다.
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_file: 출력 JSON 파일 경로 (None이면 input_file에 덮어쓰기)
        in_place: True이면 원본 파일에 덮어쓰기
    """
    # 입력 파일 확인
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # JSON 파일 읽기
    print(f"Reading predictions from: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    # 후처리: 모든 값에서 "*" 제거
    processed_predictions = {}
    total_count = 0
    modified_count = 0
    
    for key, value in predictions.items():
        total_count += 1
        original_value = value
        processed_value = remove_asterisks(value)
        
        if original_value != processed_value:
            modified_count += 1
        
        processed_predictions[key] = processed_value
    
    # 출력 파일 경로 결정
    if in_place:
        output_file = input_file
    elif output_file is None:
        # 기본값: 원본 파일명에 "_processed" 추가
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_processed{input_path.suffix}")
    
    # 결과 저장
    print(f"Saving processed predictions to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_predictions, f, ensure_ascii=False, indent=4)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("Post-processing Statistics:")
    print(f"  Total predictions: {total_count}")
    print(f"  Modified predictions: {modified_count} ({modified_count/total_count*100:.1f}%)")
    print(f"  Unchanged predictions: {total_count - modified_count}")
    print("=" * 60)
    
    # CSV 파일도 함께 처리 (같은 디렉토리에 있는 경우)
    csv_file = input_file.replace(".json", "_submit.csv")
    if os.path.exists(csv_file):
        print(f"\nProcessing CSV file: {csv_file}")
        process_csv_file(csv_file, csv_file.replace(".csv", "_processed.csv") if not in_place else csv_file, in_place)
    
    print(f"\n✅ Post-processing completed!")
    print(f"   Output file: {output_file}")


def process_csv_file(input_csv: str, output_csv: str, in_place: bool):
    """
    CSV 파일도 함께 처리합니다.
    
    Args:
        input_csv: 입력 CSV 파일 경로
        output_csv: 출력 CSV 파일 경로
        in_place: True이면 원본 파일에 덮어쓰기
    """
    import csv
    
    if in_place:
        output_csv = input_csv
    
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    
    # 각 행의 두 번째 컬럼(답변)에서 "**" 제거
    processed_rows = []
    for row in rows:
        if len(row) >= 2:
            row[1] = remove_asterisks(row[1])
        processed_rows.append(row)
    
    with open(output_csv, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(processed_rows)
    
    print(f"   CSV file saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove '**' from predictions JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 원본 파일에 덮어쓰기
  python postprocess_predictions.py predictions.json --in-place
  
  # 새 파일로 저장
  python postprocess_predictions.py predictions.json -o predictions_processed.json
  
  # 기본값 (predictions_processed.json으로 저장)
  python postprocess_predictions.py predictions.json
        """
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input predictions JSON file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: <input>_processed.json)"
    )
    
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file (use with caution!)"
    )
    
    args = parser.parse_args()
    
    try:
        postprocess_predictions(
            input_file=args.input_file,
            output_file=args.output,
            in_place=args.in_place
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
