import re
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen1.5-4B-Chat"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8"

class LLMTranscriptCleaner:
    def __init__(self, model_name=MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu", low_cpu_mem_usage=True):
        """
        Initialize the transcript cleaner with an LLM model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.device = device
        # Initialize the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, low_cpu_mem_usage=low_cpu_mem_usage).to(device)
        
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
        return f"""As a professional transcriptionist, clean up the following spoken text while preserving its exact meaning and the speaker's style of expression. Your task is to:

1. Remove only pure disfluencies like 'uh', 'um', repeated words, and false starts
2. Remove excessive punctuation like '...' and normalize spacing and grammar
3. Remove repeated words or phrases that could be due to a damaged recording
4. Keep all hedging phrases, qualifiers, and expressions of uncertainty (like 'I think', 'I believe', 'probably', 'maybe')
5. Keep all parenthetical expressions and asides
6. Format numbers and dates consistently
7. Preserve the exact meaning and level of certainty expressed by the speaker

Original text: {text}

Cleaned text: """

    def _process_with_llm(self, text: str, prompt_template) -> str:
        """Process the text using the LLM."""
        prompt = self._create_prompt(text)
        
        # Tokenize and generate
        # Calculate appropriate max length based on input
        test_encoding = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = test_encoding.input_ids.size(1)
        max_length = min(prompt_length + 200, 32000)  # Add buffer, cap at model's context window

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = encoded['attention_mask'].to(self.device)

        outputs = self.model.generate(
            encoded.input_ids.to(self.device),
            attention_mask=attention_mask,
            max_new_tokens=len(text) + 50,  # Allow some buffer for expanded text
            temperature=0.15,  # Lower temperature for more consistent outputs
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
        "And... Ruth believe joined a year A year or two after that, some people are surprised to know that I have... never joined. because I we moved here to Fort Collins, when I was visiting faculty member, at the business school. CSU. and I had already known that there was lot of emotion. -based decision. making going on. on and the Dean had asked me to undertake a couple of kind of kind of politicized initiatives. And so the... So last thing I wanted to do was to... get things further. complicated by some religious organization. and the The winds have blown. various directions. in the couple of decades. since. I'm still in a state of mild state mild tension with part of what goes on in the congregation. you know, uncomfortable. I'm comfortable with where I am. and I'm delighted with where Ruth is. in the relationship. and so that's where we",
    ]
    
    # Initialize the cleaner with a small model for testing
    cleaner = LLMTranscriptCleaner(model_name=MODEL_NAME, low_cpu_mem_usage=True)
    batch_cleaner = BatchTranscriptCleaner(cleaner)
    
    # Process test cases
    cleaned_texts = batch_cleaner.process_batch(test_cases)
    
    # Print results
    for original, cleaned in zip(test_cases, cleaned_texts):
        print(f"Original: {original}")
        print(f"Cleaned:  {cleaned}\n")
