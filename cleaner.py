import re
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"

class LLMTranscriptCleaner:
    def __init__(self, model_name=MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the transcript cleaner with an LLM model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        
        # Basic cleanup patterns for pre-processing
        self.basic_patterns = [
            (r'\s+', ' '),  # normalize whitespace
            (r'\b(\w+)( \1\b)+', r'\1'),  # remove immediate word repetitions
        ]

    def _basic_cleanup(self, text: str) -> str:
        """Apply basic cleanup patterns."""
        for pattern, replacement in self.basic_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()

    def _create_prompt(self, text: str) -> str:
        """Create a prompt for the LLM that encourages journalistic-style cleanup."""
        return f"""As a professional transcriptionist, clean up the following spoken text to make it more readable while maintaining its meaning. Remove speech disfluencies, false starts, and filler words. Format numbers and dates consistently. Maintain the speaker's language as closely as possible but make it flow naturally as if they had written it, not spoken it.".

Original text: {text}

Cleaned text:"""

    def _process_with_llm(self, text: str) -> str:
        """Process the text using the LLM."""
        prompt = self._create_prompt(text)
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=len(text) + 50,  # Allow some buffer for expanded text
            temperature=0.3,  # Lower temperature for more consistent outputs
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        # Decode and extract the cleaned text
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the cleaned portion after our prompt
        cleaned_text = full_output.split("Cleaned text:")[-1].strip()
        return cleaned_text

    def clean(self, text: str) -> str:
        """
        Clean a transcript using both basic rules and LLM processing.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned transcript text
        """
        # First apply basic cleanup
        text = self._basic_cleanup(text)
        
        # Then process with LLM
        cleaned_text = self._process_with_llm(text)
        return cleaned_text

class BatchTranscriptCleaner:
    def __init__(self, cleaner: LLMTranscriptCleaner):
        self.cleaner = cleaner

    def process_batch(self, texts: List[str], batch_size: int = 10) -> List[str]:
        """
        Process a batch of transcripts.
        
        Args:
            texts: List of transcript texts to clean
            batch_size: Number of transcripts to process at once
            
        Returns:
            List of cleaned transcripts
        """
        cleaned_texts = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned_batch = [self.cleaner.clean(text) for text in batch]
            cleaned_texts.extend(cleaned_batch)
        return cleaned_texts

# Example usage
if __name__ == "__main__":
    # Test cases that demonstrate various speech patterns
    test_cases = [
        "twenty twenty uh five twenty twenty five like you know",
        "the the way forward um I mean the path ahead",
        "we need to uh we need to focus on on the main issues",
        "December twenty third two thousand and uh twenty three",
        "they invested fifty uh thousand I mean five hundred thousand dollars"
    ]
    
    # Initialize the cleaner with a small model for testing
    cleaner = LLMTranscriptCleaner(model_name=MODEL_NAME)
    batch_cleaner = BatchTranscriptCleaner(cleaner)
    
    # Process test cases
    cleaned_texts = batch_cleaner.process_batch(test_cases)
    
    # Print results
    for original, cleaned in zip(test_cases, cleaned_texts):
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}\n")
