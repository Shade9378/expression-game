from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalGRU(nn.Module):
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim, 7))

    def forward(
        self, embeddings: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1] < 1:
            raise ValueError("Expected batch, genuine sequence length, embedding dimension")
        sequence = (
            pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
            if lengths is not None
            else embeddings
        )
        _, hidden = self.gru(sequence)
        return self.head(hidden[-1])
