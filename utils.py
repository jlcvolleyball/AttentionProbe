"""
Utility functions for the AttentionProbe application.
"""

from typing import List, Tuple, Optional
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


def find_difference(tokens1: List[str], tokens2: List[str]) -> int:
    """
    Find the index where two token lists differ.
    
    Args:
        tokens1: First list of tokens
        tokens2: Second list of tokens
        
    Returns:
        Index where tokens differ, or -1 if they are identical
    """
    if len(tokens1) != len(tokens2):
        raise ValueError("Token lists must have the same length")
    
    for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
        if t1 != t2:
            return i
    return -1


def validate_sentence(sentence: str, required_pronouns: List[str], max_count: int = 1) -> bool:
    """
    Validate that a sentence contains exactly one of the required pronouns.
    
    Args:
        sentence: Input sentence to validate
        required_pronouns: List of pronouns that should be present
        max_count: Maximum number of pronouns allowed
        
    Returns:
        True if sentence is valid, False otherwise
    """
    sentence_lower = sentence.lower()
    
    # Check if any required pronoun is present
    found_pronouns = [pronoun for pronoun in required_pronouns if f" {pronoun} " in sentence_lower]
    
    if not found_pronouns:
        return False
    
    # Check that no more than max_count pronouns are present
    total_count = sum(sentence_lower.count(f" {pronoun} ") for pronoun in required_pronouns)
    
    return total_count <= max_count


def generate_contrast_prompt(original_prompt: str, pronouns: List[str]) -> str:
    """
    Generate a contrasting prompt by replacing one pronoun with another.
    
    Args:
        original_prompt: Original prompt containing a pronoun
        pronouns: List of pronouns to swap between
        
    Returns:
        New prompt with pronoun swapped
    """
    original_lower = original_prompt.lower()
    
    for i, pronoun in enumerate(pronouns):
        if f" {pronoun} " in original_lower:
            # Find the other pronoun to swap with
            other_pronoun = pronouns[(i + 1) % len(pronouns)]
            pronoun_idx = original_lower.index(f" {pronoun} ")
            
            # Create new prompt with swapped pronoun
            new_prompt = (
                original_prompt[:pronoun_idx] + 
                f" {other_pronoun} " + 
                original_prompt[pronoun_idx + len(pronoun) + 2:]
            )
            return new_prompt
    
    raise ValueError(f"No pronoun from {pronouns} found in prompt")


class ModelManager:
    """Manages T5 model loading and inference."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load the T5 model and tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        if self.config is None:
            self.config = T5Config.from_pretrained(self.model_name)
            
    def generate_response(self, prompt: str, max_length: int = 20) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            
        Returns:
            Generated response text
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(outputs[0][1:-1])
    
    def get_attention_outputs(self, prompts: List[str], max_length: int = 20):
        """
        Generate attention outputs for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum length of generated response
            
        Returns:
            Model outputs with attention information
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)
        
        return self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=max_length
        ) 