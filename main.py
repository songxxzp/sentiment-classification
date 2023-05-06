import torch

from train import Trainer
from models import TextCNN, TextLSTM, TextGRU, MLP, Classifier
from utils import load_word_vector, Tokenizer, build_dataset, metric_f1, metric_accuracy, setup_random_seed

if __name__ == "__main__":
    setup_random_seed(10728)

    batch_size = 128
    epoch = 100
    device = torch.device("cuda")
    save_model_path = "./TextCNN.pt"
    save_result_path = "./TextCNN.json"

    pretrained_embedding, vocab = load_word_vector()
    tokenizer = Tokenizer(vocab)

    train_dataset, valid_dataset, test_dataset = build_dataset(tokenizer=tokenizer)

    # TextLSTM(num_layers=2, bidirectional=True)
    # TextGRU(num_layers=2, bidirectional=True)
    # MLP(inner_hidden_size=200, dropout=0.5)
    # TextCNN(convs=[{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}])

    model = Classifier(TextCNN(convs=[{"out_channels":20, "kernel_size":4}, {"out_channels":20, "kernel_size":3}, {"out_channels":10, "kernel_size":2}]), dropout=0, vocab_size=len(vocab), pretrained_embedding=pretrained_embedding).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    trainer = Trainer(train_dataset, valid_dataset, test_dataset, tokenizer, early_stop_strategy="val_loss", early_stop_epoch=3, device=device)

    trainer.train(
        model=model,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=batch_size,
        save_model_path=save_model_path,
        save_result_path=save_result_path
    )

