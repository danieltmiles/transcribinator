import asyncio
import re

import torch

from utils import load_quantized_llm_model, quantized_generate_from_prompt

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

def get_answer(generated: str, start_delim: str, end_delim: str) -> str:
    begin_indexes = [i for i in range(len(generated)) if generated.startswith(start_delim, i)]
    end_indexes = [i for i in range(len(generated)) if generated.startswith(end_delim, i)]
    return generated[begin_indexes[-1] + len(start_delim):end_indexes[-1]]

def main():
    try:
        model, tokenizer, model_type = load_quantized_llm_model(
            device,
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF",
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/Qwen2.5-7B-Instruct-GGUF",
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-MLX-4bit",
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit",
            # "/Users/dmiles/.lmstudio/models/lmstudio-community/QwQ-32B-MLX-4bit",
            "/Users/dmiles/.lmstudio/models/lmstudio-community/Qwen3-32B-GGUF"
        )
        
        # ... existing prompt and generation code ...
        
        prompt_template = """Please read the following text and determine all political
issues being discussed. Output a list of issues in graph format:
| Person -> Supports/Opposes -> Issue |
For example:
| Mayor Hales -> Supports -> Tenant Protections |.
Please explain your thinking then indicate your graph-formatted response in a text
block starting with, ```graph.

```
{transcript}
```

"""
        transcript = ""
        with open("Portland City Council Meeting AM Session 04⧸24⧸24 [6I7SlDJt17E]_transcript.txt", "r") as fl:
            transcript = fl.read()
        prompt = prompt_template.format(transcript=transcript)
        generated = quantized_generate_from_prompt(prompt, model, tokenizer, model_type)
        start_delim = "```graph\n"
        end_delim = "```"
        answer = get_answer(generated, start_delim, end_delim)
        print(answer)
        pat = re.compile(r"^\s*|\s*([^|]*) -> (.*) -> ([^|]*)\s*|\s*$")
        for entity, relationship, topic in pat.findall(answer):
            print(f"{entity=} {relationship=} {topic=}")
            if entity == "Person":
                print("skipping Person")
                continue
            if not (entity and relationship and topic):
                print("skipping on a blank")
                continue
            new_prompt = f"""I have identified the topic, `{topic}` from the following transcript. Please review the text
and create a detailed description of the topic from the information in the transcript. Please explain your thinking then
write your description in a text block starting with ```description.
```transcript
{transcript}
```
"""
            print(f"{topic=}")
            description_generated = quantized_generate_from_prompt(new_prompt, model, tokenizer, model_type)
            print(description_generated)
            print("========================================================================================")

    finally:
        pass

if __name__ == '__main__':
    main()
