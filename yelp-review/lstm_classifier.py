import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM модель для классификации текстов"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=256,
        seed=42,
        output_dim=2,
        n_layers=2,
        dropout=0.2,
        pad_idx=None,
    ):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.seed = seed
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )

        # Dropout слой
        self.dropout = nn.Dropout(dropout)

        # Полносвязный слой для классификации
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Инициализация весов
        self.init_weights()

    def init_weights(self):
        """Инициализация весов"""
        generator = torch.Generator().manual_seed(self.seed)
        nn.init.xavier_normal_(self.embedding.weight, generator=generator)
        nn.init.xavier_normal_(self.fc.weight, generator=generator)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, attention_mask):
        """
        x: [batch_size, seq_len] - индексы токенов
        attention_mask: [batch_size, seq_len] - маска (1 для реальных токенов, 0 для паддинга)
        """
        embedded = self.embedding(x)

        mask = attention_mask.unsqueeze(-1).float()
        embedded = embedded * mask

        output, (hidden, cell) = self.lstm(embedded)

        lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(x.size(0), device=x.device)
        last_hidden = output[batch_indices, lengths, :]

        out = self.fc(last_hidden)

        return out
