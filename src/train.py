import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

print("model loaded")