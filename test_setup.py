"""
Test script to verify the LLM Reader training setup.
Tests data formatting, collator, and model loading.
"""

import sys
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from data_formatter import create_formatter
from data_collator import DataCollatorForAnswerOnlyLM

def test_data_formatter():
    """Test the prompt formatter."""
    print("="*50)
    print("Testing Data Formatter")
    print("="*50)
    
    formatter = create_formatter(template_type="instruction")
    
    # Test single example
    example = {
        "question": "대한민국의 수도는 어디인가요?",
        "context": "대한민국의 수도는 서울특별시입니다. 서울은 한국의 정치, 경제, 문화의 중심지입니다.",
        "answers": {
            "text": ["서울특별시"],
            "answer_start": [11]
        }
    }
    
    formatted = formatter.format_example(example, include_answer=True)
    
    print("\n[Prompt]")
    print(formatted["prompt"])
    print("\n[Answer]")
    print(formatted["answer"])
    print("\n[Full Text]")
    print(formatted["full_text"])
    print("\n✓ Data formatter test passed!\n")
    
    return formatted

def test_data_collator(formatted_example):
    """Test the data collator."""
    print("="*50)
    print("Testing Data Collator")
    print("="*50)
    
    # Load tokenizer (using a small model for testing)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create collator
    collator = DataCollatorForAnswerOnlyLM(
        tokenizer=tokenizer,
        max_length=512,
        padding=True,
    )
    
    # Prepare features
    features = [formatted_example, formatted_example]  # Batch of 2
    
    # Collate
    batch = collator(features)
    
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Check that prompt tokens are masked
    labels = batch['labels'][0]
    masked_count = (labels == -100).sum().item()
    total_count = labels.shape[0]
    
    print(f"\nMasked tokens (prompt): {masked_count}/{total_count}")
    print(f"Answer tokens: {total_count - masked_count}")
    
    print("\n✓ Data collator test passed!\n")

def test_model_loading():
    """Test model loading."""
    print("="*50)
    print("Testing Model Loading")
    print("="*50)
    
    model_name = "Qwen/Qwen2.5-0.5B"  # Small model for testing
    
    print(f"\nLoading model: {model_name}")
    
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
    )
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    print("\n✓ Model loading test passed!\n")

def test_dataset_loading():
    """Test dataset loading and formatting."""
    print("="*50)
    print("Testing Dataset Loading")
    print("="*50)
    
    try:
        datasets = load_from_disk("./data/train_dataset")
        print(f"\nDataset loaded successfully!")
        print(f"Train samples: {len(datasets['train'])}")
        print(f"Validation samples: {len(datasets['validation'])}")
        
        # Test formatting on a small subset
        formatter = create_formatter(template_type="instruction")
        
        def format_examples(examples):
            return formatter.format_batch(examples, include_answer=True)
        
        # Format first 5 examples
        small_dataset = datasets["train"].select(range(5))
        formatted_dataset = small_dataset.map(
            format_examples,
            batched=True,
            remove_columns=small_dataset.column_names,
        )
        
        print(f"\nFormatted dataset columns: {formatted_dataset.column_names}")
        print(f"\nSample formatted example:")
        print(f"Prompt length: {len(formatted_dataset[0]['prompt'])} chars")
        print(f"Answer: {formatted_dataset[0]['answer']}")
        
        print("\n✓ Dataset loading test passed!\n")
        
    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}")
        print("Make sure ./data/train_dataset exists")

def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("LLM Reader Training Setup Test")
    print("="*50 + "\n")
    
    try:
        # Test 1: Data Formatter
        formatted = test_data_formatter()
        
        # Test 2: Data Collator
        test_data_collator(formatted)
        
        # Test 3: Model Loading
        test_model_loading()
        
        # Test 4: Dataset Loading (optional, requires data)
        test_dataset_loading()
        
        print("="*50)
        print("All tests completed successfully! ✓")
        print("="*50)
        print("\nYou can now run training with:")
        print("python train.py \\")
        print("  --model_name_or_path Qwen/Qwen2.5-0.5B \\")
        print("  --output_dir ./models/llm-reader \\")
        print("  --do_train \\")
        print("  --per_device_train_batch_size 4 \\")
        print("  --num_train_epochs 3 \\")
        print("  --learning_rate 2e-5 \\")
        print("  --prompt_template_type instruction")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
