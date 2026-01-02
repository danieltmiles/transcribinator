import llama_cpp.llama
from llama_cpp import Llama
def main():
    model_name = "lmstudio-community/gpt-oss-20b-GGUF"
    llm: llama_cpp.llama.Llama = Llama.from_pretrained(repo_id=model_name, filename="gpt-oss-20b-MXFP4.gguf")
    print(type(llm))
    print(llm.__class__.__name__)
    llm.close()
    model = Llama(
        model_path="/Users/dmiles/.lmstudio/models/lmstudio-community/Olmo-3-32B-Think-GGUF/Olmo-3-32B-Think-Q4_K_M.gguf",
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=8192,  # Context window size
        verbose=False
    )
    print(type(model))
    print(model.__class__.__name__)
    model.close()

if __name__ == "__main__":
    main()