import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from functools import partial
from torch.utils.data import dataset, dataloader

from models import TextCNN, TextLSTM, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, collate_fn, metric_f1, metric_accuracy


class Trainer:
    def __init__(self, early_stop_epoch=0, early_stop_strategy=None):
        self.logger = []
        self.early_stop_epoch = early_stop_epoch
        self.early_stop_strategy = early_stop_strategy

    def log(self, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
        self.logger.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )

    def early_stop(self):
        if self.early_stop_strategy is None:
            return False

        if len(self.logger) > self.early_stop_epoch:
            stop = True
            if self.early_stop_strategy == "val_loss":
                for i in range(len(self.logger) - self.early_stop_epoch, len(self.logger)):
                    if not(self.logger[i - 1]["val_loss"] < self.logger[i]["val_loss"]):
                        stop = False
                        break
            elif self.early_stop_strategy == "val_acc":
                for i in range(len(self.logger) - self.early_stop_epoch, len(self.logger)):
                    if not(self.logger[i - 1]["val_acc"] > self.logger[i]["val_acc"]):
                        stop = False
                        break
            elif self.early_stop_strategy == "val_f1":
                for i in range(len(self.logger) - self.early_stop_epoch, len(self.logger)):
                    if not(self.logger[i - 1]["val_acc"] > self.logger[i]["val_acc"]):
                        stop = False
                        break
            else:
                stop = False
            return stop

        return False

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion=F.cross_entropy, device=torch.device("cuda")):
    model.train()
    train_loss = 0.0
    results, labels = [], []

    for _, sample in enumerate(tqdm.tqdm(dataloader)):
        inputs, label = sample["tokens"].to(device), sample["labels"].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * label.size(0)
        results.append(output)
        labels.append(label)

    train_loss /= len(dataloader.dataset)
    scheduler.step()
    train_acc = metric_accuracy(torch.cat(results), torch.cat(labels))
    f1 = metric_f1(torch.cat(results), torch.cat(labels))
    return train_loss, train_acc, f1


def test(model, dataloader, criterion=F.cross_entropy, device=torch.device("cuda")):
    model.eval()
    val_loss = 0.0
    results, labels = [], []

    for _, sample in enumerate(tqdm.tqdm(dataloader)):
        inputs, label = sample["tokens"].to(device), sample["labels"].to(device)
        output = model(inputs)
        loss = criterion(output, label)
        val_loss += loss.item() * label.size(0)
        results.append(output)
        labels.append(label)

    val_loss /= len(dataloader.dataset)
    val_acc = metric_accuracy(torch.cat(results), torch.cat(labels))
    f1 = metric_f1(torch.cat(results), torch.cat(labels))
    return val_loss, val_acc, f1


if __name__ == "__main__":
    batch_size = 256
    epoch = 100
    device = torch.device("cuda")

    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)

    print(pretrained_embedding.shape)

    dataset_train, dataset_valid, dataset_test = build_dataset(tokenizer=tokenizer)

    collate_fn = partial(collate_fn, tokenizer=tokenizer)

    dataloader_train = dataloader.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloder_valid = dataloader.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloder_test = dataloader.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # TextLSTM(num_layers=2, bidirectional=True)
    # MLP()
    # TextCNN(convs=[{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}])

    model = Classifier(TextCNN(convs=[{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]), vocab_size=len(vocab), hidden_size=50, pretrained_embedding=pretrained_embedding).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    trainer = Trainer(early_stop_strategy="val_loss", early_stop_epoch=5)

    for e in range(epoch):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            dataloader=dataloader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        print(e + 1, train_loss, train_acc, train_f1)
        val_loss, val_acc, val_f1 = test(model, dataloder_valid, device=device)
        print(e + 1, val_loss, val_acc, val_f1)

        trainer.log(e + 1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1)

        if trainer.early_stop():
            epoch = e + 1
            print("early stop on epoch {}".format(epoch))
            break

    test_loss, test_acc, test_f1 = test(model, dataloder_test, device=device)
    print(epoch, test_loss, test_acc, test_f1)
    
