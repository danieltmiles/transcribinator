import asyncio
import re

import torch

from utils import load_quantized_llm_model, quantized_generate_from_prompt

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def get_answer(generated: str, start_delim: str, end_delim: str) -> str:
    begin_indexes = [i for i in range(len(generated)) if generated.startswith(start_delim, i)]
    end_indexes = [i for i in range(len(generated)) if generated.startswith(end_delim, i)]
    return generated[begin_indexes[-1] + len(start_delim):end_indexes[-1]]

def quantized_generate_from_messages(messages: list, model, tokenizer, model_type: str, **kwargs) -> str:
    """
    Generate text from a conversation history using chat templates.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "user", "content": "..."}]
        model: The loaded model
        tokenizer: The tokenizer
        model_type: Either "mlx" or "gguf"
        **kwargs: Additional generation parameters
    
    Returns:
        Generated text response
    """
    # Apply chat template to convert conversation to proper format
    if hasattr(tokenizer, 'apply_chat_template'):
        # Most modern tokenizers support this
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback: manually format as simple conversation
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f"System: {content}\n\n"
            elif role == 'user':
                prompt += f"User: {content}\n\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n\n"
        prompt += "Assistant: "
    
    return quantized_generate_from_prompt(prompt, model, tokenizer, model_type, **kwargs)

def main():
    try:
        model, tokenizer, model_type = load_quantized_llm_model(
            device,
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-MLX-4bit",
            "/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-GGUF/Olmo-3-32B-Think-Q4_K_M.gguf",
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/Qwen3-32B-GGUF",
        )
        
        # Read transcript once
        transcript = ""
        with open("Portland City Council Meeting AM Session 04⧸24⧸24 [6I7SlDJt17E]_transcript.txt", "r") as fl:
            transcript = fl.read()
        
        # Initialize conversation with transcript as context
        conversation = []
        
        # System message (optional) sets the assistant's behavior
        conversation.append({
            "role": "system",
            "content": "You are a political analyst helping to extract information from city council meeting transcripts."
        })
        
        # First user message establishes the transcript as context
        conversation.append({
            "role": "user",
            "content": f"""I'm going to provide you with a city council meeting transcript. Please read it carefully as I'll be asking you questions about it.

```transcript
{transcript}
```

Please confirm you've read the transcript and are ready to analyze it."""
        })
        
        # Get confirmation (optional, but helps establish context)
        confirmation = quantized_generate_from_messages(conversation, model, tokenizer, model_type)
        print("Assistant confirmation:", confirmation[:200], "...\n")
        
        # Add assistant's response to conversation history
        conversation.append({
            "role": "assistant",
            "content": confirmation
        })
        
        # Now ask for the political issues analysis
        conversation.append({
            "role": "user",
            "content": """Now, please determine all political issues being discussed in the transcript. 
Output a list of issues in graph format:
| Person -> Supports/Opposes -> Issue |

For example:
| Mayor Hales -> Supports -> Tenant Protections |

Please explain your thinking then indicate your graph-formatted response in a text block starting with ```graph"""
        })
        
        generated = quantized_generate_from_messages(conversation, model, tokenizer, model_type)
        print("Issues analysis:", generated[:500], "...\n")
        
        # Add to conversation history
        conversation.append({
            "role": "assistant",
            "content": generated
        })
        
        # Parse the graph
        start_delim = "```graph\n"
        end_delim = "```"
        answer = get_answer(generated, start_delim, end_delim)
        print("Extracted graph:\n", answer, "\n")
        
        pat = re.compile(r"^\s*|\s*([^|]*) -> (.*) -> ([^|]*)\s*|\s*$")
        
        # Create a base conversation context that stops after the graph extraction
        # This prevents the context from growing with each topic description
        base_conversation = conversation.copy()
        
        for entity, relationship, topic in pat.findall(answer):
            print(f"{entity=} {relationship=} {topic=}")
            
            if entity == "Person" or not (entity and relationship and topic):
                continue
            
            # EFFICIENT APPROACH: Use temporary concatenation instead of appending
            # This keeps context constant size rather than growing with each topic
            # Since each topic description is independent, we don't need previous descriptions
            description_generated = quantized_generate_from_messages(
                base_conversation + [{
                    "role": "user",
                    "content": f"""You identified the topic "{topic}" from the transcript. Please create a detailed description of this topic based on the information in the transcript. Explain your thinking then write your description in a text block starting with ```description"""
                }],
                model, tokenizer, model_type
            )
            
            print(f"\n{'='*80}")
            print(f"Topic: {topic}")
            print(f"Description: {description_generated}")
            print('='*80 + "\n")
            
            # Note: We're NOT appending to conversation here because each topic 
            # description is independent and doesn't need to see other topics

    finally:
        pass

if __name__ == '__main__':
    main()
