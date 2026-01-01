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
    elif hasattr(model, "create_chat_completion"):
        return model.create_chat_completion(messages)
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
    model, tokenizer, model_type = load_quantized_llm_model(
        device,
        #"/Users/dmiles/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF",
        #"/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-MLX-4bit",
        "/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-GGUF/Olmo-3-32B-Think-Q4_K_M.gguf",
        # "/Users/dmiles/.lmstudio/models/lmstudio-community/Qwen3-32B-GGUF",
    )
    # Read transcript once
    transcript = ""
    with open("Portland City Council Meeting AM Session 04⧸24⧸24 [6I7SlDJt17E]_transcript.txt", "r") as fl:
        transcript = fl.read()
    header_pat = re.compile(r"^\[\d\d:\d\d:\d\d - \d\d:\d\d:\d\d\].*:$", re.MULTILINE)
    headers = header_pat.findall(transcript)
    header_idxes = [transcript.index(x) for x in headers]
    sections = [transcript[header_idxes[i]:header_idxes[i+1]] for i in range(len(header_idxes)-1)]
    
    # Create sliding window segments with ~50% overlap
    transcript_segments = []
    i = 0
    while i < len(sections):
        # Build current segment starting at index i
        transcript_segment = ""
        section_start_idx = i
        section_count = 0
        
        # Accumulate sections until we exceed 4000 characters
        while i < len(sections):
            transcript_segment += sections[i]
            i += 1
            section_count += 1
            if len(transcript_segment) > 6000:
                break
        
        transcript_segments.append(transcript_segment)
        
        # Rewind to approximately the midpoint of this segment for overlap
        # Calculate how many sections to go back (about half)
        rewind_amount = section_count // 2
        i = section_start_idx + rewind_amount
        
        # Edge case: if we're at the end and rewinding would repeat the last segment
        # just break to avoid infinite loop
        if i >= len(sections) or rewind_amount == 0:
            break

    for transcript_segment in transcript_segments:
        try:

            # Initialize conversation with transcript as context
            conversation = []

            # System message (optional) sets the assistant's behavior
            conversation.append({
                "role": "system",
                "content": "You are a political analyst helping to extract information from city council meeting transcripts."
            })

#             # First user message establishes the transcript as context
#             conversation.append({
#                 "role": "user",
#                 "content": f"""I'm going to provide you with a city council meeting transcript. Please read it carefully as I'll be asking you questions about it.
#
# ```transcript
# {transcript_segment}
# ```
#
# Please confirm you've read the transcript and are ready to analyze it."""
#             })
#
#             # Get confirmation (optional, but helps establish context)
#             print("reading transcript")
#             confirmation = quantized_generate_from_messages(conversation, model, tokenizer, model_type)
#             #print("Assistant confirmation:", confirmation[:200], "...\n")
#
#             # Add assistant's response to conversation history
#             conversation.append({
#                 "role": "assistant",
#                 "content": confirmation
#             })

            # Now ask for the political issues analysis
            conversation.append({
                "role": "user",
                "content": """Extract all political issues as relationships in this exact format:
```graph
| Speaker -> Position -> Issue |
```

Rules:
- One relationship per line
- No additional explanation
- Maximum 15 relationships
"""
            })

            answer_tries = 3
            answer: str = ""
            generated: str = ""
            while answer_tries > 0:
                print("determining issues")
                generated = quantized_generate_from_messages(conversation, model, tokenizer, model_type)
                # Parse the graph
                try:
                    answer = get_answer(generated, start_delim="```graph\n", end_delim="```")
                    print(answer)
                except IndexError:
                    # try again
                    print(f"failed to find ```graph block in generated text:\n{generated}")
                    answer_tries -= 1
                    continue
                break
            if not generated or not answer:
                raise ValueError("tried to generate answer too many times")
            # Add to conversation history
            conversation.append({
                "role": "assistant",
                "content": generated
            })

            pat = re.compile(r"^\s*|\s*([^|]*) -> (.*) -> ([^|]*)\s*|\s*$")

            # Create a base conversation context that stops after the graph extraction
            # This prevents the context from growing with each topic description
            base_conversation = conversation.copy()

            seen_topics = set()
            for entity, relationship, topic in pat.findall(answer):
                if entity == "Person" or not (entity and relationship and topic):
                    continue
                if topic in seen_topics:
                    continue
                seen_topics.add(topic)
                print(f"{entity=} {relationship=} {topic=}")

                tries_left = 3
                description = ""
                while tries_left > 0:
                    description_generated = quantized_generate_from_messages(
                        base_conversation + [{
                            "role": "user",
                            "content": f"""You identified the topic "{topic}" from the transcript.
Please create a detailed description of this topic based on the information in the transcript in this exact format:.
```description
description goes here
```
"""
                        }],
                        model, tokenizer, model_type
                    )
                    try:
                        description = get_answer(description_generated, start_delim="```description", end_delim="```")
                    except IndexError:
                        tries_left -= 1
                        continue
                if not description:
                    raise ValueError("could not generate description in 3 tries")
                print(description)

                # Note: We're NOT appending to conversation here because each topic
                # description is independent and doesn't need to see other topics

        finally:
            pass

if __name__ == '__main__':
    main()
