import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, LSTM

class MLP(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=50, inner_hidden_size=200, dropout=0.5) -> None:
        super().__init__()
        # I'm the believer of Universal Approximation theorem

        self.dense_1 = Linear(embedding_dim, inner_hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.act_1 = F.relu()
        self.dense_2 = Linear(inner_hidden_size, hidden_size)
        self.dropout_2 = nn.Dropout(dropout)
        self.act_2 = F.relu()

        nn.init.kaiming_uniform_(self.dense_1.weight)
        nn.init.kaiming_uniform_(self.dense_2.weight)
        with torch.no_grad():
            self.dense_1.bias.zero_()
            self.dense_2.bias.zero_()

    def forward(self, hidden_state) -> torch.Tensor:
        hidden_state = self.dense_1(hidden_state)
        hidden_state = self.dropout_1(hidden_state)
        hidden_state = self.act_1(hidden_state)
        hidden_state = self.dense_2(hidden_state)
        hidden_state = self.dropout_2(hidden_state)
        hidden_state = self.act_2(hidden_state)
        return hidden_state


class TextCNN(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=50):
        super(TextCNN, self).__init__()
        conv_1 = Conv1d(in_channels=embedding_dim, out_channels=2, kernel_size=4)
        pool_1 = nn.MaxPool1d(kernel_size=2)
        conv_2 = Conv1d(in_channels=embedding_dim, out_channels=2, kernel_size=3)
        pool_2 = nn.MaxPool1d(kernel_size=2)
        conv_3 = Conv1d(in_channels=embedding_dim, out_channels=2, kernel_size=2)
        pool_3 = nn.MaxPool1d(kernel_size=2)
        self.convs = nn.ModuleList([conv_1, conv_2, conv_3])
        self.pools = nn.ModuleList([pool_1, pool_2, pool_3])

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state.shape = [batch_size, seq_lenth, hidden_size]
        hidden_state = hidden_state.transpose(0, 2, 1)  # [batch_size, hidden_size, seq_lenth]
        hidden_state = torch.cat([pool(F.relu(conv(hidden_state))) for conv, pool in zip(self.convs, self.pools)])
        return hidden_state


class TextLSTM(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=50, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = hidden_state.transpose(1, 0, 2)
        output, (hidden_state, cell_state) = self.lstm(hidden_state)
        return hidden_state


class Classifier(nn.Module):
    def __init__(self, base_model: nn.Module, vocab_size, embedding_dim=50, hidden_size=50, num_classs=2, pretrained_embedding: torch.Tensor=None, embedding_requires_grad=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        print(self.embedding.weight.shape)
        if pretrained_embedding is not None:
            self.embedding.weight.data = pretrained_embedding.clone().detach()
        self.embedding.weight.requires_grad_(embedding_requires_grad)
        self.model = base_model
        self.lm_head = Linear(hidden_size, num_classs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor, return_logits=True) -> torch.Tensor:
        hidden_state = self.embedding(inputs)
        hidden_state = self.model(hidden_state)
        hidden_state = self.lm_head(hidden_state)
        if return_logits:
            return hidden_state
        return self.softmax(hidden_state)
