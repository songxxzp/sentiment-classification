import os
# os.environ["HF_DATASETS_OFFLINE"]="1"
# os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from functools import partial

import evaluate

from utils import load_data


def compute_metrics(eval_pred, metric):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_bert_classifier(model_name="hfl/chinese-roberta-wwm-ext", log_path="./log/roberta", save_path="./model/roberta", num_labels=2, batch_size=1, train_data_file = "./Dataset/train.json", valid_data_file = "./Dataset/valid.json", test_data_file = "./Dataset/test.json", device=torch.device("cpu")):
    data_train = load_dataset("json", data_files=train_data_file, field="data", split="train").shuffle()
    data_valid = load_dataset("json", data_files=valid_data_file, field="data", split="train")
    data_test = load_dataset("json", data_files=test_data_file, field="data", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, padding=True)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
    tokenized_data = {
        "train": data_train.map(preprocess_function),
        "valid": data_valid.map(preprocess_function),
        "test": data_test.map(preprocess_function)
    }

    print("train/test data:")
    print(tokenized_data["train"])
    print(tokenized_data["valid"])
    print(tokenized_data["test"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    print("loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device=device)

    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    training_args = TrainingArguments(
        logging_dir=log_path,
        output_dir=save_path,
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=4,
        do_eval=True,
        evaluation_strategy="epoch",
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["valid"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, metric=metric)
    )

    trainer.train()
    trainer.save_model(save_path)

    result = trainer.evaluate(eval_dataset=tokenized_data["test"])
    print(result)


if __name__ == "__main__":
    # model_name = "hfl/chinese-roberta-wwm-ext"
    # model_name = "bert-base-chinese"
    train_bert_classifier(model_name="hfl/chinese-roberta-wwm-ext", batch_size=32, log_path="./log/roberta", save_path="./model/roberta", device="cuda" if torch.cuda.is_available() else "cpu")
    train_bert_classifier(model_name="bert-base-chinese", batch_size=32, log_path="./log/bert", save_path="./model/bert", device="cuda" if torch.cuda.is_available() else "cpu")
