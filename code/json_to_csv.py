"""
predictions.json 파일을 CSV 형식으로 변환하는 스크립트
"""

import json
import csv
import argparse
import os
from pathlib import Path


def convert_json_to_csv(json_path: str, output_path: str = None, delimiter: str = '\t'):
    """
    predictions.json 파일을 CSV 형식으로 변환
    
    Args:
        json_path: 입력 JSON 파일 경로
        output_path: 출력 CSV 파일 경로 (None이면 자동 생성)
        delimiter: CSV 구분자 (기본값: 탭)
    """
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # 출력 경로 설정
    if output_path is None:
        json_dir = os.path.dirname(json_path)
        json_basename = os.path.basename(json_path).replace('.json', '')
        output_path = os.path.join(json_dir, f"{json_basename}.csv")
    
    # CSV 파일 작성 (베이스라인 코드와 동일하게 정렬하지 않고 딕셔너리 순서 유지)
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        
        # 헤더는 없이 id와 answer만 작성 (제출 형식에 맞춤)
        # 베이스라인 코드처럼 predictions.items()를 그대로 순회
        for id, answer in predictions.items():
            # 줄바꿈 문자를 공백으로 치환 (CSV 형식 유지)
            answer_cleaned = answer.replace('\n', ' ').replace('\r', ' ')
            writer.writerow([id, answer_cleaned])
    
    print(f"✅ Converted {len(predictions)} predictions from {json_path}")
    print(f"✅ Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert predictions.json to CSV format')
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to predictions.json file'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: same directory as JSON with .csv extension)'
    )
    parser.add_argument(
        '--delimiter',
        '-d',
        type=str,
        default='\t',
        help='CSV delimiter (default: tab)'
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.json_path):
        print(f"❌ Error: File not found: {args.json_path}")
        return
    
    # 변환 실행
    convert_json_to_csv(args.json_path, args.output, args.delimiter)


if __name__ == "__main__":
    main()
