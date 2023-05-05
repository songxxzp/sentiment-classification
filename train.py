import torch
import torch.nn.functional as F
import numpy as np
import tqdm

from functools import partial
from torch.utils.data import dataset, dataloader

from models import TextCNN, TextLSTM, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, collate_fn, metric_f1, metric_accuracy


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


def early_stop(e, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
    return False  # TODO


if __name__ == "__main__":
    batch_size = 256
    epoch = 40
    device = torch.device("cuda")

    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)

    print(pretrained_embedding.shape)

    dataset_train, dataset_valid, dataset_test = build_dataset(tokenizer=tokenizer)

    collate_fn = partial(collate_fn, tokenizer=tokenizer)

    dataloader_train = dataloader.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloder_valid = dataloader.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloder_test = dataloader.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = Classifier(TextLSTM(num_layers=2, bidirectional=True), vocab_size=len(vocab), hidden_size=50, pretrained_embedding=pretrained_embedding).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

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

        if early_stop(e, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
            print("early stop")
            epoch = e + 1

    test_loss, test_acc, test_f1 = test(model, dataloder_test, device=device)
    print(epoch, test_loss, test_acc, test_f1)
    
