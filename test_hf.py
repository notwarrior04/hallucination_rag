from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("google/flan-t5-base")
    print(f"Model Type: {config.model_type}")
except Exception as e:
    print(f"Error: {e}")
