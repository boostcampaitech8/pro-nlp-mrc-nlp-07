from datasets import load_from_disk
import json

# dataset = load_from_disk("./data/test_dataset")
# for i in list(dataset['validation']):
#     print(i)

# with open("./data/wikipedia_documents.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# print(json.dumps(data, indent=4, ensure_ascii=False))

dataset = load_from_disk("./data/train_dataset")
for i in list(dataset['train']):
    print(i['answers']['text'][0])