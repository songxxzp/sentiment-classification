import torch
from torch.utils.data import dataset, dataloader
from typing import Dict


class CustomDataset(dataset.Dataset):
    def __init__(self, path="./Dataset/train.txt", process_fn=lambda x:x) -> None:
        super().__init__()
        with open(path, "r", encoding="utf-8") as file:
            self.data = list(map(process_fn, file.readlines()))

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
