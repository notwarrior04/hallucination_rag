# inspect_fever.py

from datasets import load_dataset
import pprint

ds = load_dataset(
    "copenlu/fever_gold_evidence",
    split="train"
)

sample = ds[0]

print(sample.keys())
print()
pprint.pprint(sample)