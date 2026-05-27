from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class BackboneOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class BackboneAdapter(Protocol):
    hidden_size: int

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        ...

    def encode(self, sequences: list[str], device: torch.device) -> BackboneOutput:
        ...

