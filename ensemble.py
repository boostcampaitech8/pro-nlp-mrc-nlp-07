"""
LLM Ensemble System
Aggregates predictions from multiple CSV files and uses LLM to select the best answer.
"""

import os
import json
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple
from collections import defaultdict
from config import EnsembleConfig


class LLMEnsemble:
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.fallback_df = None
        
    def load_model(self):
        """Load LLM model and tokenizer"""
        print(f"Loading model: {self.config.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        
        if self.config.device == "cpu":
            self.model.to(self.config.device)
        
        self.model.eval()
        print("Model loaded successfully!")
        
    def load_test_dataset(self) -> pd.DataFrame:
        """Load test dataset with questions and IDs"""
        print(f"Loading test dataset from: {self.config.test_dataset_path}")
        
        dataset = load_from_disk(self.config.test_dataset_path)
        df = dataset["validation"].to_pandas()
        
        print(f"Loaded {len(df)} questions")
        return df
    
    def load_csv_files(self) -> List[pd.DataFrame]:
        """Load all CSV files from csv folder"""
        csv_files = [f for f in os.listdir(self.config.csv_folder) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.config.csv_folder}")
        
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        dfs = []
        for csv_file in csv_files:
            csv_path = os.path.join(self.config.csv_folder, csv_file)
            df = pd.read_csv(csv_path, sep='\t', header=None, names=['id', 'answer'])
            dfs.append(df)
            print(f"  - {csv_file}: {len(df)} answers")
        
        return dfs
    
    def aggregate_answers(self, question_id: str, csv_dfs: List[pd.DataFrame]) -> List[str]:
        """Aggregate and deduplicate answers for a given question ID"""
        answers = []
        
        for df in csv_dfs:
            matching_rows = df[df['id'] == question_id]
            if not matching_rows.empty:
                answer = matching_rows.iloc[0]['answer']
                if pd.notna(answer):  # Check if not NaN
                    answer = str(answer).strip()
                    if answer and answer not in answers:  # Deduplicate
                        answers.append(answer)
        
        return answers
    
    def create_prompt(self, question: str, choices: List[str]) -> str:
        """Create prompt in the specified format"""
        prompt = "다음 중 아래 질문에 대한 답으로 가장 적절한 것을 보기 중에서 고르시오. "
        prompt += "답변을 작성할 때는 부가적인 내용을 덧붙이지 말고, 반드시 보기 중에 선택한 것 만을 그대로 작성하시오.\n\n"
        prompt += f"질문: {question}\n\n"
        prompt += "보기:\n"
        
        for choice in choices:
            prompt += f"- {choice}\n"
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract answer
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from generated text
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text.strip()
        
        # Clean up answer - take first line
        answer = answer.split('\n')[0].strip()
        
        # Remove leading dash if present
        if answer.startswith('-'):
            answer = answer[1:].strip()
        
        return answer
    
    def validate_answer(self, answer: str, choices: List[str]) -> Tuple[bool, str]:
        """
        Validate if answer matches one of the choices
        Returns: (is_valid, matched_choice)
        """
        answer_clean = answer.strip().lower()
        
        # Exact match
        for choice in choices:
            if answer_clean == choice.strip().lower():
                return True, choice
        
        # Partial match - check if answer is contained in choice or vice versa
        for choice in choices:
            choice_clean = choice.strip().lower()
            if answer_clean in choice_clean or choice_clean in answer_clean:
                return True, choice
        
        return False, answer
    
    def run_ensemble(self):
        """Main ensemble process"""
        # Load model
        self.load_model()
        
        # Load data
        test_df = self.load_test_dataset()
        csv_dfs = self.load_csv_files()
        
        # Process each question
        results = []
        validation_stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'no_choices': 0
        }
        
        print(f"\nProcessing {len(test_df)} questions...")
        
        for idx, row in test_df.iterrows():
            question_id = row['id']
            question = row['question']
            
            # Aggregate answers from all CSVs
            choices = self.aggregate_answers(question_id, csv_dfs)
            
            validation_stats['total'] += 1
            
            if not choices:
                # No answers found for this question
                validation_stats['no_choices'] += 1
                if self.config.verbose and idx < 5:
                    print(f"\nWarning: No answers found for {question_id}")
                results.append({
                    'id': question_id,
                    'answer': '',
                    'choices': [],
                    'is_valid': False
                })
                continue
            
            if len(choices) == 1:
                # Only one answer, use it directly
                answer = choices[0]
                is_valid = True
                if self.config.verbose and idx < 5:
                    print(f"\n[{idx}] {question_id}: Only 1 choice, using directly")
            else:
                # Multiple choices, use LLM
                base_prompt = self.create_prompt(question, choices)
                prompt = base_prompt
                
                if self.config.verbose and idx < 3:
                    print(f"\n{'='*80}")
                    print(f"Example Prompt [{idx}]:")
                    print(prompt)
                    print(f"{'='*80}")
                
                # Retry loop
                for attempt in range(self.config.max_retries + 1):
                    answer = self.generate_answer(prompt)
                    
                    # Validate answer
                    is_valid, matched_choice = self.validate_answer(answer, choices)
                    
                    if is_valid:
                        answer = matched_choice  # Use the exact matched choice
                        if attempt > 0 and self.config.verbose:
                             print(f"  Success on attempt {attempt + 1}")
                        break
                    
                    # If invalid, prepare for retry
                    if attempt < self.config.max_retries:
                        if self.config.verbose:
                             print(f"  Attempt {attempt + 1} invalid: '{answer}'. Retrying...")
                        
                        # Add feedback to prompt for next attempt
                        prompt = base_prompt + f"\n\n이전 답변 '{answer}'은(는) 보기에 없습니다. 반드시 위 보기 목록 중에서 하나를 골라 그대로 작성하시오."
                
                if self.config.verbose and idx < 5:
                    print(f"\n[{idx}] {question_id}")
                    print(f"  Choices ({len(choices)}): {choices[:3]}{'...' if len(choices) > 3 else ''}")
                    print(f"  LLM Answer: {answer}")
                    print(f"  Valid: {is_valid}")
                if not is_valid:
                    print(f"  Failed after {self.config.max_retries + 1} attempts")
                    
                    # Fallback logic
                    if self.config.fallback_file:
                        if self.fallback_df is None:
                             # Load fallback file if not loaded
                             print(f"  Loading fallback file: {self.config.fallback_file}")
                             self.fallback_df = pd.read_csv(self.config.fallback_file, sep='\t', header=None, names=['id', 'answer'])
                        
                        fallback_row = self.fallback_df[self.fallback_df['id'] == question_id]
                        if not fallback_row.empty:
                            fallback_answer = fallback_row.iloc[0]['answer']
                            # Check if fallback answer is in choices (optional, but good to know)
                            # is_fallback_valid, _ = self.validate_answer(str(fallback_answer), choices)
                            
                            answer = str(fallback_answer).strip()
                            is_valid = True # Mark as valid since we trusted the fallback
                            print(f"  Using fallback answer: {answer}")
                        else:
                            print(f"  Warning: Question ID {question_id} not found in fallback file")

            
            # Update stats
            if is_valid:
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
            
            results.append({
                'id': question_id,
                'answer': answer,
                'choices': choices,
                'is_valid': is_valid
            })
            
            # Log progress
            if (idx + 1) % self.config.log_interval == 0:
                print(f"Processed {idx + 1}/{len(test_df)} questions...")
        
        # Save results
        self.save_results(results, validation_stats)
        
    def save_results(self, results: List[Dict], validation_stats: Dict):
        """Save results to CSV file"""
        # Create output dataframe
        output_df = pd.DataFrame([
            {'id': r['id'], 'answer': r['answer']}
            for r in results
        ])
        
        # Save to CSV (tab-separated, no header)
        output_df.to_csv(
            self.config.output_file,
            sep='\t',
            header=False,
            index=False
        )
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {self.config.output_file}")
        print(f"\nValidation Statistics:")
        print(f"  Total questions: {validation_stats['total']}")
        print(f"  Valid answers: {validation_stats['valid']} ({validation_stats['valid']/validation_stats['total']*100:.1f}%)")
        print(f"  Invalid answers: {validation_stats['invalid']} ({validation_stats['invalid']/validation_stats['total']*100:.1f}%)")
        print(f"  No choices found: {validation_stats['no_choices']}")
        print(f"{'='*80}")
        
        # Save detailed results for debugging
        debug_file = self.config.output_file.replace('.csv', '_debug.json')
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'stats': validation_stats
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Debug info saved to: {debug_file}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Ensemble System")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--csv_folder", type=str, default=None, help="CSV folder path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--fallback", type=str, default=None, help="Fallback CSV file path")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Create config
    config = EnsembleConfig()
    
    # Override with command line arguments
    if args.model:
        config.model_name_or_path = args.model
    if args.csv_folder:
        config.csv_folder = args.csv_folder
    if args.output:
        config.output_file = args.output
    if args.fallback:
        config.fallback_file = args.fallback
    if args.temperature is not None:
        config.temperature = args.temperature
    
    # Run ensemble
    ensemble = LLMEnsemble(config)
    ensemble.run_ensemble()


if __name__ == "__main__":
    main()
