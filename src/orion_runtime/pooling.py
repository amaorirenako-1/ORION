from __future__ import annotations

import torch


class MeanPooling(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class AttentionPooling(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(hidden_states).squeeze(-1)
        scores = scores.masked_fill(~attention_mask.bool(), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


def create_pooling(mode: str, hidden_size: int) -> torch.nn.Module:
    if mode == "mean":
        return MeanPooling()
    if mode == "attention":
        return AttentionPooling(hidden_size)
    raise ValueError(f"Unsupported pooling mode: {mode}")

