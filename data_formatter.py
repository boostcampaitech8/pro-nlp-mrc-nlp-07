"""
Data formatting utilities for LLM-based Question Answering.
Converts dataset features (question, context, answers) into prompt-based format.
"""

from typing import Dict, List, Optional


class PromptFormatter:
    """Formats QA data into prompts for CasualLM training."""
    
    def __init__(self, template_type: str = "instruction"):
        """
        Initialize the prompt formatter.
        
        Args:
            template_type: Type of template ('simple', 'instruction', 'chat')
        """
        self.template_type = template_type
        self.templates = {
            "simple": self._simple_template,
            "instruction": self._instruction_template,
            "chat": self._chat_template,
        }
        
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}. Choose from {list(self.templates.keys())}")
    
    def _simple_template(self, question: str, context: str, answer: Optional[str] = None) -> Dict[str, str]:
        """Simple Q&A format."""
        prompt = f"질문: {question}\n문맥: {context}\n답변:"
        
        if answer is not None:
            full_text = f"{prompt} {answer}"
            return {
                "prompt": prompt,
                "answer": answer,
                "full_text": full_text,
            }
        return {"prompt": prompt}
    
    def _instruction_template(self, question: str, context: str, answer: Optional[str] = None) -> Dict[str, str]:
        """Instruction-based format with clear sections."""
        prompt = (
            "[INSTRUCTION]\n"
            "주어진 문맥(Context)을 주의 깊게 읽고, 뒤따르는 질문(Question)에 대해 답하세요.\n"
            "답변은 반드시 문맥 내에서 찾은 구문이나 단어여야 하며, 가능한 한 짧고 간결한 명사구 형태로 제시되어야 합니다.\n\n"
            f"[CONTEXT]\n{context}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "[ANSWER]\n"
        )
        
        if answer is not None:
            full_text = f"{prompt}{answer}"
            return {
                "prompt": prompt,
                "answer": answer,
                "full_text": full_text,
            }
        return {"prompt": prompt}
    
    def _chat_template(self, question: str, context: str, answer: Optional[str] = None) -> Dict[str, str]:
        """Chat format for chat-tuned models."""
        system_msg = "주어진 문맥을 읽고 질문에 정확하게 답변하세요."
        user_msg = f"문맥: {context}\n\n질문: {question}"
        
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
        if answer is not None:
            full_text = f"{prompt}{answer}<|im_end|>"
            return {
                "prompt": prompt,
                "answer": answer,
                "full_text": full_text,
            }
        return {"prompt": prompt}
    
    def format_example(self, example: Dict, include_answer: bool = True) -> Dict[str, str]:
        """
        Format a single example into prompt format.
        
        Args:
            example: Dictionary containing 'question', 'context', and optionally 'answers'
            include_answer: Whether to include the answer in the formatted text
            
        Returns:
            Dictionary with 'prompt', 'answer' (if include_answer), and 'full_text' (if include_answer)
        """
        question = example["question"]
        context = example["context"]
        
        # Extract answer text if available and requested
        answer = None
        if include_answer and "answers" in example:
            answers_data = example["answers"]
            if isinstance(answers_data, dict) and "text" in answers_data:
                # Handle both list and single value cases
                answer_text = answers_data["text"]
                if isinstance(answer_text, list) and len(answer_text) > 0:
                    answer = answer_text[0]
                elif isinstance(answer_text, str):
                    answer = answer_text
        
        formatter = self.templates[self.template_type]
        return formatter(question, context, answer)
    
    def format_batch(self, examples: Dict[str, List], include_answer: bool = True) -> Dict[str, List[str]]:
        """
        Format a batch of examples.
        
        Args:
            examples: Dictionary with lists of 'question', 'context', and optionally 'answers'
            include_answer: Whether to include answers in the formatted text
            
        Returns:
            Dictionary with lists of 'prompt', 'answer', and 'full_text'
        """
        num_examples = len(examples["question"])
        
        prompts = []
        answers = []
        full_texts = []
        
        for i in range(num_examples):
            example = {
                "question": examples["question"][i],
                "context": examples["context"][i],
            }
            
            if "answers" in examples:
                example["answers"] = examples["answers"][i]
            
            formatted = self.format_example(example, include_answer=include_answer)
            
            prompts.append(formatted["prompt"])
            if include_answer:
                answers.append(formatted.get("answer", ""))
                full_texts.append(formatted["full_text"])
        
        result = {"prompt": prompts}
        if include_answer:
            result["answer"] = answers
            result["full_text"] = full_texts
        
        return result


def create_formatter(template_type: str = "instruction") -> PromptFormatter:
    """
    Factory function to create a PromptFormatter.
    
    Args:
        template_type: Type of template to use
        
    Returns:
        PromptFormatter instance
    """
    return PromptFormatter(template_type=template_type)
