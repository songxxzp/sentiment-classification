import json
import torch
from gensim.models import KeyedVectors
from typing import Dict, List
from functools import partial

from dataset import CustomDataset


class Tokenizer:
    def __init__(self, index_to_key: List):
        self.index_to_key = index_to_key
        self.key_to_index = {}
        for index, key in enumerate(index_to_key):
            self.key_to_index[key] = index

        self.unk = self.key_to_index["<unk>"]
        self.pad = self.key_to_index["<pad>"]

    def encode(self, word_list):
        token_list = list(map(lambda word:self.key_to_index[word] if word in self.key_to_index else self.unk, word_list))
        return token_list

    def decode(self, token_list):
        word_list = list(map(lambda token:self.index_to_key[token] if token in self.index_to_key else "", token_list))
        return word_list

    def max_token_id(self):
        return len(self.index_to_key)


def load_word_vector(path='./Dataset/wiki_word2vec_50.bin'):
    word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)
    pretrained_embedding = torch.cat([torch.zeros((1, 50)), torch.tensor(word2vec_model.vectors), torch.zeros((1, 50))])
    vocab = ["<pad>"] + word2vec_model.index_to_key + ["<unk>"]
    return pretrained_embedding, vocab


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
        "tokens": token_list,
        "word_list": word_list,
        "sentence": "".join(word_list)
    }


def collate_fn(samples, tokenizer=None):
    pad = 0
    if tokenizer is not None:
        pad = tokenizer.pad

    max_length = max([len(sample["tokens"]) for sample in samples])

    return {
        "labels": torch.tensor([sample["label"] for sample in samples]),
     	"tokens": torch.tensor([sample["tokens"] + [pad] * (max_length - len(sample["tokens"])) for sample in samples]),
    }


def build_dataset(tokenizer, train_data_path="./Dataset/train.txt", valid_data_path="./Dataset/validation.txt", test_data_path="./Dataset/test.txt"):
    dataset_train = CustomDataset(train_data_path, process_fn=partial(process_fn, tokenizer=tokenizer))
    dataset_valid = CustomDataset(valid_data_path, process_fn=partial(process_fn, tokenizer=tokenizer))
    dataset_test = CustomDataset(test_data_path, process_fn=partial(process_fn, tokenizer=tokenizer))
    return dataset_train, dataset_valid, dataset_test


def metric_accuracy(results: torch.Tensor, labels: torch.Tensor) -> float:
    acc = (results.argmax(-1) == labels).sum().item() / labels.size(0)
    return acc


def metric_f1(results, labels) -> float:
    TP = ((results.argmax(-1) == 1) & (labels == 1)).sum().item()
    FN = ((results.argmax(-1) == 0) & (labels == 1)).sum().item()
    FP = ((results.argmax(-1) == 1) & (labels == 0)).sum().item()
    TN = ((results.argmax(-1) == 0) & (labels == 0)).sum().item()
    precision = TP / (TP + FP) if (TP + FP) else 1
    recall = TP / (TP + FN) if (TP + FN) else 1
    f1_score =  2 / (1 / precision + 1 / recall)
    return f1_score

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
