import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

model_name = "google/flan-t5-base"
bi_model_name = "all-MiniLM-L6-v2"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Generator loaded.")

print(f"Loading {bi_model_name}...")
bi_encoder = SentenceTransformer(bi_model_name)
print("Bi-encoder loaded.")
