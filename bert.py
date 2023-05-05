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
import evaluate

from utils import load_data

model_name = "hfl/chinese-roberta-wwm-ext"
# model_name = "bert-base-chinese"
num_labels = 2
train_data_file = "./Dataset/train.json"
valid_data_file = "./Dataset/valid.json"
test_data_file = "./Dataset/test.json"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        output_dir="./roberta",
        learning_rate=2e-5,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=40,
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
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("./model/roberta")

    result = trainer.evaluate(eval_dataset=tokenized_data["test"])
    print(result)
