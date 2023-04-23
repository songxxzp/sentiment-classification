import json


def process_fn(sample, tokenizer=None):
    sample = sample.strip().rstrip()
    label, sentence = sample.split('\t')
    word_list = sentence.split()
    if tokenizer is not None:
        token_list = list(tokenizer.encode(word_list))
    else:
        token_list = None
    return {
        "label": int(label),
        "token_list": token_list,
        "sentence": "".join(word_list)
    }


def load_data(train_set="./Dataset/train.txt", valid_set="./Dataset/validation.txt", test_set="./Dataset/test.txt"):
    with open(train_set, "r", encoding="utf-8") as file:
        dataset_train = list(map(process_fn, file.readlines()))
    with open(valid_set, "r", encoding="utf-8") as file:
        dataset_valid = list(map(process_fn, file.readlines()))
    with open(test_set, "r", encoding="utf-8") as file:
        dataset_test = list(map(process_fn, file.readlines()))
    return dataset_train, dataset_valid, dataset_test


def convert_data_to_json():
    dataset_train, dataset_valid, dataset_test = load_data()
    with open("./Dataset/train.json", "w", encoding="utf-8") as file:
        data_list = []
        for sample in dataset_train:
            data_list.append(
                {
                    "text": sample["sentence"],
                    "label": sample["label"]
                }
            )
        json.dump({"data": data_list}, file, indent=2, ensure_ascii=False)
    with open("./Dataset/valid.json", "w", encoding="utf-8") as file:
        data_list = []
        for sample in dataset_valid:
            data_list.append(
                {
                    "text": sample["sentence"],
                    "label": sample["label"]
                }
            )
        json.dump({"data": data_list}, file, indent=2, ensure_ascii=False)
    with open("./Dataset/test.json", "w", encoding="utf-8") as file:
        data_list = []
        for sample in dataset_test:
            data_list.append(
                {
                    "text": sample["sentence"],
                    "label": sample["label"]
                }
            )
        json.dump({"data": data_list}, file, indent=2, ensure_ascii=False)
