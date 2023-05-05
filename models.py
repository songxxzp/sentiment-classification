import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear, LSTM

class MLP(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=50, inner_hidden_size=200, dropout=0.5, act=F.leaky_relu, init=nn.init.kaiming_uniform_) -> None:
        super().__init__()
        # I'm the believer of Universal Approximation theorem

        self.dense_1 = Linear(embedding_dim, inner_hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dense_2 = Linear(inner_hidden_size, hidden_size)
        self.dropout_2 = nn.Dropout(dropout)

        self.act = act

        init(self.dense_1.weight)
        init(self.dense_2.weight)
        with torch.no_grad():
            self.dense_1.bias.zero_()
            self.dense_2.bias.zero_()

    def forward(self, hidden_state) -> torch.Tensor:
        hidden_state = self.dense_1(hidden_state)
        hidden_state = self.dropout_1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dense_2(hidden_state)
        hidden_state = self.dropout_2(hidden_state)
        hidden_state = self.act(hidden_state)  # [batch_size, seq_length, hidden_size]
        return hidden_state


class TextCNN(nn.Module):
    def __init__(self, embedding_dim=50, act=F.leaky_relu, convs=[{"out_channels":2, "kernel_size":4}, {"out_channels":2, "kernel_size":3}, {"out_channels":2, "kernel_size":2}]):
        super(TextCNN, self).__init__()
        self.convs = nn.ModuleList([Conv1d(in_channels=embedding_dim, **conv) for conv in convs])
        self.global_max_pool = lambda x: F.max_pool1d(x, x.shape[-1])
        self.act = act

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state.shape = [batch_size, seq_lenth, embedding_dim]
        hidden_state = hidden_state.transpose(2, 1)  # [batch_size, hidden_size, seq_lenth]
        hidden_state = torch.cat([self.global_max_pool(self.act(conv(hidden_state))) for conv in self.convs], dim=1)
        hidden_state = hidden_state.transpose(1, 2)  # [batch_size, out_channels, hidden_size]
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
        # hidden_state.shape = [batch_size, seq_length, embedding_dim]
        hidden_state = hidden_state.transpose(1, 0)  # [seq_length, batch_size, embedding_dim]
        output, (hidden_state, cell_state) = self.lstm(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)  # [batch_size, num_layers * bidirectional, embedding_dim]
        return hidden_state


class TextGRU(nn.Module):
    def __init__(self, embedding_dim=50, hidden_size=50, num_layers=1, bidirectional=False):
        super().__init__()
        self.GRU = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state.shape = [batch_size, seq_length, embedding_dim]
        hidden_state = hidden_state.transpose(1, 0)  # [seq_length, batch_size, embedding_dim]
        output, hidden_state = self.GRU(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)  # [batch_size, num_layers * bidirectional, embedding_dim]
        return hidden_state


class Classifier(nn.Module):
    def __init__(self, base_model: nn.Module, vocab_size, embedding_dim=50, hidden_size=50, num_classs=2, pretrained_embedding: torch.Tensor=None, embedding_requires_grad=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            assert pretrained_embedding.shape == self.embedding.weight.shape
            self.embedding.weight.data = pretrained_embedding.clone().detach()
        self.embedding.weight.requires_grad_(embedding_requires_grad)
        self.model = base_model
        self.lm_head = Linear(hidden_size, num_classs)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: torch.Tensor, return_logits=True) -> torch.Tensor:
        hidden_state = self.embedding(inputs)
        hidden_state = self.model(hidden_state)  # [batch_size, ..., embedding_dim]
        hidden_state = hidden_state.transpose(2, 1)  # [batch_size, embedding_dim, ...]
        hidden_state = F.max_pool1d(hidden_state, hidden_state.shape[-1]).squeeze(-1)
        hidden_state = self.lm_head(hidden_state)  # [batch_size, num_classs]
        if return_logits:
            return hidden_state
        return self.softmax(hidden_state)
