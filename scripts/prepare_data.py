#!/usr/bin/env python3
"""
Data Preparation Script - 데이터셋 병합 및 저장
"""
import os
import sys
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def create_fixed_standard_format(dataset, dataset_name, max_samples=1000):
    """표준 형식으로 데이터 변환"""
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    standardized_data = []

    if dataset_name == 'argilla_customer':
        for item in dataset:
            instruction = item.get('user-message', '')
            context = item.get('context','')
            response = item.get('response-suggestion','')

            if instruction.strip() and response.strip():
                if context.strip():
                    full_instruction = f"Context : {context[:200]}...\n\nQuestion : {instruction}"
                else :
                    full_instruction = instruction

                standardized_data.append({
                    'instruction': full_instruction,
                    'response': response,
                    'source': 'argilla_customer',
                    'domain': 'rag_customer_support',
                    'has_context': bool(context.strip())
                })

    elif dataset_name == 'argilla_synthetic':
        for item in dataset:
            instruction = item.get('prompt','')
            response = item.get('completion', '')
            system_prompt = item.get('system_prompt', '')

            if instruction.strip() and response.strip():
                standardized_data.append({
                    'instruction': instruction,
                    'response': response,
                    'source': 'argilla_synthetic',
                    'domain': 'synthetic_customer_support',
                    'has_context': False,
                    'system_prompt': system_prompt
                })

    elif dataset_name == 'bitext':
        for item in dataset:
            instruction = item.get('instruction','')
            response = item.get('response', '')

            if isinstance(response, list) and len(response) > 0 :
                response = response[0]

            if instruction.strip() and response.strip():
                standardized_data.append({
                    'instruction': instruction,
                    'response': response,
                    'source': 'bitext',
                    'domain': 'customer_support',
                    'has_context': False
                })

    print(f"표준화 완료: {len(standardized_data)}개 샘플")
    return standardized_data


def merge_fixed_datasets(sample_limit=1000):
    """데이터셋 병합"""
    datasets_info = {
        'argilla_customer': 'argilla/customer_assistant',
        'argilla_synthetic': 'argilla/synthetic-sft-customer-support-single-turn',
        'bitext': 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
    }

    all_data = []

    for name, path in datasets_info.items():
        print(f"\n{name} 처리중...")
        try:
            dataset = load_dataset(path, split=f'train[:{sample_limit}]')
            standardized = create_fixed_standard_format(dataset, name, sample_limit)
            all_data.extend(standardized)
        except Exception as e:
            print(f"{name} 처리 실패: {str(e)}")

    if all_data:
        unified_dataset = Dataset.from_list(all_data)
        print(f"\n수정된 통합 데이터셋 생성 완료")
        print(f"   - 총 샘플: {len(unified_dataset)}")
        print(f"   - 컬럼: {unified_dataset.column_names}")

    return unified_dataset


def main():
    """메인 실행 함수"""
    print("=== Customer Support 데이터 준비 시작 ===\n")
    
    # 1. 데이터 디렉토리 생성
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # 2. 데이터셋 병합 (전체 데이터)
    print("전체 데이터셋 병합 중...")
    full_dataset = merge_fixed_datasets(sample_limit=10000)  # 전체 데이터
    
    # 3. 학습/평가 데이터 분리
    print("\n학습/평가 데이터 분리 중...")
    # 90% train, 10% test
    train_test = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    test_dataset = train_test['test']
    
    print(f"학습 데이터: {len(train_dataset)}개")
    print(f"평가 데이터: {len(test_dataset)}개")
    
    # 4. 데이터 저장
    print("\n데이터 저장 중...")
    
    # 전체 데이터 저장 (선택사항)
    full_dataset.save_to_disk(str(data_dir / "unified_customer_support"))
    full_dataset.to_json(str(data_dir / "unified_customer_support.json"))
    
    # 학습 데이터 저장
    train_dataset.save_to_disk(str(data_dir / "train_dataset"))
    train_dataset.to_json(str(data_dir / "train_dataset.json"))
    
    # 테스트 데이터 저장
    test_dataset.save_to_disk(str(data_dir / "test_dataset"))
    test_dataset.to_json(str(data_dir / "test_dataset.json"))
    
    # 5. 작은 디버그용 데이터셋 생성
    print("\n디버그용 작은 데이터셋 생성 중...")
    debug_size = min(100, len(train_dataset))
    debug_dataset = train_dataset.select(range(debug_size))
    debug_dataset.to_json(str(data_dir / "debug_dataset.json"))
    
    # 6. 데이터 통계 저장
    stats = {
        "total_samples": len(full_dataset),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "debug_samples": debug_size,
        "sources": train_dataset['source'].unique().tolist() if hasattr(train_dataset['source'], 'unique') else list(set(train_dataset['source'])),
        "domains": train_dataset['domain'].unique().tolist() if hasattr(train_dataset['domain'], 'unique') else list(set(train_dataset['domain'])),
    }
    
    with open(data_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print("\n=== 데이터 준비 완료! ===")
    print(f"\n저장된 파일들:")
    print(f"  - {data_dir}/unified_customer_support/ (전체 데이터)")
    print(f"  - {data_dir}/train_dataset.json (학습 데이터)")
    print(f"  - {data_dir}/test_dataset.json (평가 데이터)")
    print(f"  - {data_dir}/debug_dataset.json (디버그용)")
    print(f"  - {data_dir}/dataset_stats.json (통계)")
    
    # 샘플 출력
    print("\n=== 데이터 샘플 ===")
    sample = train_dataset[0]
    print(f"Instruction: {sample['instruction'][:100]}...")
    print(f"Response: {sample['response'][:100]}...")
    print(f"Source: {sample['source']}")
    print(f"Domain: {sample['domain']}")


if __name__ == "__main__":
    main()