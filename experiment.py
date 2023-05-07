import torch
import json
import datetime
import os
import copy

import torch.nn.functional as F
import torch.nn as nn

from train import Trainer
from models import TextCNN, TextLSTM, TextGRU, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, metric_f1, metric_accuracy, setup_random_seed

setup_random_seed(10728)
batch_size = 128
max_epoch = 50
device = torch.device("cuda")

# experiment 0
"""
base_models = [
    TextCNN,
    TextLSTM,
    TextGRU,
    MLP
]

base_models_args = [
    [{"convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]}],
    [{"num_layers": 2, "bidirectional": True}],
    [{"num_layers": 2, "bidirectional": True}],
    [{"inner_hidden_size": 200, "dropout": 0.5}]
]

dropouts = [0, 0.5]
optimizers = [torch.optim.SGD, torch.optim.AdamW]  # torch.optim.Adam
lrs = [1e-5, 1e-4, 1e-3]
early_stops = [0, 3]
pools = [F.max_pool1d, F.avg_pool1d]
embedding_requires_grads = [False, True]
"""

# experiment 1
base_models = [
    TextCNN,
    TextLSTM,
    TextGRU,
    MLP
]

base_models_args = [
    [
        {"act": F.leaky_relu, "convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]},
        {"act": F.sigmoid, "convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]},
        {"act": F.tanh, "convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]},
        {"convs": [{"out_channels":50, "kernel_size":3}]},
        {"convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]},
        {"convs": [{"out_channels":10, "kernel_size":5}, {"out_channels":10, "kernel_size":4}, {"out_channels":10, "kernel_size":3}, {"out_channels":10, "kernel_size":2}, {"out_channels":10, "kernel_size":1}]},
        {"convs": [{"out_channels":10, "kernel_size":4}, {"out_channels":10, "kernel_size":3}, {"out_channels":5, "kernel_size":2}]},
        {"convs": [{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]},
        {"convs": [{"out_channels":40, "kernel_size":4}, {"out_channels":40, "kernel_size":3}, {"out_channels":20, "kernel_size":2}]}
    ],
    [
        {"num_layers": 1, "bidirectional": False},
        {"num_layers": 1, "bidirectional": True},
        {"num_layers": 2, "bidirectional": False},
        {"num_layers": 2, "bidirectional": True},
        {"num_layers": 4, "bidirectional": False},
        {"num_layers": 4, "bidirectional": True},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 25},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 50},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 100},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 200}
    ],
    [
        {"num_layers": 1, "bidirectional": False},
        {"num_layers": 1, "bidirectional": True},
        {"num_layers": 2, "bidirectional": False},
        {"num_layers": 2, "bidirectional": True},
        {"num_layers": 4, "bidirectional": False},
        {"num_layers": 4, "bidirectional": True},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 25},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 50},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 100},
        {"num_layers": 2, "bidirectional": True, "hidden_size": 200}
    ],
    [
        {"inner_hidden_size": 200, "dropout": 0.0},
        {"inner_hidden_size": 200, "dropout": 0.1},
        {"inner_hidden_size": 200, "dropout": 0.5},
        {"inner_hidden_size": 200, "dropout": 0.9},
        {"inner_hidden_size": 200, "dropout": 1.0},
        {"inner_hidden_size": 50, "dropout": 0.5},
        {"inner_hidden_size": 100, "dropout": 0.5},
        {"inner_hidden_size": 200, "dropout": 0.5},
        {"inner_hidden_size": 400, "dropout": 0.5},
        {"inner_hidden_size": 800, "dropout": 0.5},
        {"inner_hidden_size": 200, "dropout": 0.5, "act": F.leaky_relu},
        {"inner_hidden_size": 200, "dropout": 0.5, "act": F.sigmoid},
        {"inner_hidden_size": 200, "dropout": 0.5, "act": F.tanh},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.uniform_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.normal_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.kaiming_uniform_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.kaiming_normal_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.xavier_uniform_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.xavier_normal_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.orthogonal_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.zeros_},
        {"inner_hidden_size": 200, "dropout": 0.5, "init": nn.init.ones_}
    ]
]

dropouts = [0]
optimizers = [torch.optim.AdamW]
lrs = [1e-4]
early_stops = [3]
pools = [F.avg_pool1d]
embedding_requires_grads = [True]


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_model_path = None
    save_result_path = None
    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)
    train_dataset, valid_dataset, test_dataset = build_dataset(tokenizer=tokenizer)

    results = {}

    for base_model, base_model_args in zip(base_models, base_models_args):
        results[base_model.__name__] = {}
        for base_model_arg in base_model_args:
            base_model_arg_dict = copy.deepcopy(base_model_arg)
            for k, v in base_model_arg.items():
                if k in ["act", "init"]:
                    base_model_arg[k] = v.__name__
            results[base_model.__name__][json.dumps(base_model_arg)] = {}
            arg_start_time = datetime.datetime.now().strftime("%H%M%S")
            for dropout in dropouts:
                results[base_model.__name__][json.dumps(base_model_arg)][dropout] = {}
                for optimizer_cls in optimizers:
                    results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__] = {}
                    for lr in lrs:
                        results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__][lr] = {}
                        for early_stop_epoch in early_stops:
                            results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__][lr][early_stop_epoch] = {}
                            for pool in pools:
                                results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__][lr][early_stop_epoch][pool.__name__] = {}
                                for embedding_requires_grad in embedding_requires_grads:
                                    results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__][lr][early_stop_epoch][pool.__name__][embedding_requires_grad] = {}
                                    print(base_model.__name__, json.dumps(base_model_arg), dropout, optimizer_cls.__name__, lr, pool.__name__, embedding_requires_grad)

                                    info = {
                                        "model": base_model.__name__,
                                        "model args": json.dumps(base_model_arg),
                                        "optimizer": optimizer_cls.__name__,
                                        "pooling": pool.__name__,
                                        "dropout": dropout,
                                        "lr": lr,
                                        "early_stop_epoch": early_stop_epoch,
                                        "froze_embedding": not embedding_requires_grad
                                    }

                                    log_dir = "./runs/{}".format(start_time)
                                    for key, value in info.items():
                                        if key in ["dropout", "lr", "early_stop_epoch", "froze_embedding"]:
                                            log_dir = os.path.join(log_dir, str(key) + "=" + str(value))
                                        elif key in ["model", "optimizer", "pooling"]:
                                            log_dir = os.path.join(log_dir, str(value))
                                        else:
                                            log_dir = os.path.join(log_dir, arg_start_time)

                                    model = Classifier(base_model(**base_model_arg_dict), dropout=dropout, vocab_size=len(vocab), pretrained_embedding=pretrained_embedding, pool=pool, embedding_requires_grad=embedding_requires_grad).to(device)

                                    optimizer = optimizer_cls(model.parameters(), lr=lr)
                                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

                                    trainer = Trainer(train_dataset, valid_dataset, test_dataset, tokenizer, early_stop_strategy="val_loss", early_stop_epoch=early_stop_epoch, device=device)

                                    result = trainer.train(
                                        model=model,
                                        epoch=max_epoch,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_size=batch_size,
                                        save_model_path=save_model_path,
                                        save_result_path=save_result_path,
                                        info=info,
                                        log_dir=log_dir
                                    )

                                    results[base_model.__name__][json.dumps(base_model_arg)][dropout][optimizer_cls.__name__][lr][early_stop_epoch][pool.__name__][embedding_requires_grad] = result
                                    with open("./results/results_{}".format(start_time), "w", encoding="utf-8") as file:
                                        json.dump(results, file, indent=2)

    with open("./results/results_{}".format(start_time), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
