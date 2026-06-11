from data.dataset_loader import DatasetLoader

loader = DatasetLoader()

fever = loader.load_fever_pairs(
    split="train",
    max_pairs=20
)

for x in fever[:10]:
    print()
    print("CLAIM:", x["claim"])
    print("LABEL:", x["label"])
    print("EVIDENCE:", x["evidence"])