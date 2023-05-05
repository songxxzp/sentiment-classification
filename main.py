import torch

from train import Trainer
from models import TextCNN, TextLSTM, TextGRU, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, collate_fn, metric_f1, metric_accuracy

if __name__ == "__main__":
    batch_size = 256
    epoch = 100
    device = torch.device("cuda")
    save_model_path = "./TextCNN.pt"
    save_result_path = "./TextCNN.json"

    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)

    train_dataset, valid_dataset, test_dataset = build_dataset(tokenizer=tokenizer)

    # TextLSTM(num_layers=2, bidirectional=True)
    # TextGRU(num_layers=2, bidirectional=True)
    # MLP()
    # TextCNN(convs=[{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}])

    model = Classifier(TextGRU(num_layers=2, bidirectional=True), vocab_size=len(vocab), hidden_size=50, pretrained_embedding=pretrained_embedding).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    trainer = Trainer(train_dataset, valid_dataset, test_dataset, tokenizer, early_stop_strategy="val_loss", early_stop_epoch=5, device=device)

    trainer.train(
        model=model,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        save_model_path=save_model_path,
        save_result_path=save_result_path
    )

