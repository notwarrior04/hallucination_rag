import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"
print(f"Testing T5 load: {model_name}")
try:
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
