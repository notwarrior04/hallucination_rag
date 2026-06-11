from data.dataset_loader import DatasetLoader
import json

loader = DatasetLoader()
print("--- HoVer (Test Split) ---")
try:
    hover = loader.load_hover(split="test", max_samples=1)
    if hover:
        print(f"Sample: {hover[0]}")
        print(f"Label type: {type(hover[0].get('label'))}")
        print(f"Label value: {hover[0].get('label')}")
    else:
        print("HoVer empty")
except Exception as e:
    print(f"HoVer error: {e}")
