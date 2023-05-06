import torch
import json
import datetime

from train import Trainer
from models import TextCNN, TextLSTM, TextGRU, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, metric_f1, metric_accuracy, setup_random_seed

setup_random_seed(10728)
batch_size = 128
max_epoch = 100
device = torch.device("cuda")

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
    [{"inner_hidden_size": 200, "dropout": 0.5}],
]

dropouts = [0, 0.5]
optimizers = [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
lrs = [1e-5, 1e-4, 1e-3]
early_stops = [0, 3]
# schedulers = [(torch.optim.lr_scheduler.LRScheduler, {}), (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": max_epoch})]


if __name__ == "__main__":
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model_path = None
    save_result_path = None
    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)
    train_dataset, valid_dataset, test_dataset = build_dataset(tokenizer=tokenizer)

    results = {}

    for base_model, base_model_args in zip(base_models, base_models_args):
        results[base_model.__name__] = {}
        for base_model_arg in base_model_args:
            results[json.dumps(base_model_arg)] = {}
            for dropout in dropouts:
                results[dropout] = {}
                for optimizer_cls in optimizers:
                    results[optimizer_cls.__name__] = {}
                    for lr in lrs:
                        results[lr] = {}
                        for early_stop_epoch in early_stops:
                            print(base_model.__name__, json.dumps(base_model_arg), dropout, optimizer_cls.__name__, lr)

                            results[early_stop_epoch] = {}

                            model = Classifier(base_model(**base_model_arg), dropout=dropout, vocab_size=len(vocab), pretrained_embedding=pretrained_embedding).to(device)

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
                                save_result_path=save_result_path
                            )

                            results[early_stop_epoch] = result

    with open("./results/results_{}".format(start_time), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
