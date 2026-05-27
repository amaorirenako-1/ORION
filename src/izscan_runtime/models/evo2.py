from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..config import ModelConfig
from ..utils import get_torch_dtype
from .base import BackboneOutput


class Evo2FrozenBackbone(torch.nn.Module):
    """Frozen Evo2 embedding extractor."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        try:
            from evo2 import Evo2
        except ImportError as exc:
            raise RuntimeError("Evo2 is required for model.family=evo2. Install/load the Evo2 environment.") from exc

        self.config = config
        self.layer_name = config.evo2_layer_name
        self.model: Any = Evo2(config.name)
        self.hidden_size = 4096
        self.torch_dtype = get_torch_dtype(config.torch_dtype)

        internal = getattr(self.model, "model", None)
        if isinstance(internal, torch.nn.Module):
            for param in internal.parameters():
                param.requires_grad = False
            internal.eval()

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def _encode_one(self, sequence: str, device: torch.device) -> torch.Tensor:
        tokenized = self.model.tokenizer.tokenize(sequence)
        input_ids = torch.tensor(tokenized, dtype=torch.int, device=device).unsqueeze(0)
        with torch.no_grad():
            _, embeddings = self.model(
                input_ids,
                return_embeddings=True,
                layer_names=[self.layer_name],
            )
        hidden = embeddings[self.layer_name]
        return hidden.squeeze(0).to(dtype=self.torch_dtype)

    def encode(self, sequences: list[str], device: torch.device) -> BackboneOutput:
        hidden_list = [self._encode_one(seq, device) for seq in sequences]
        max_len = max(item.shape[0] for item in hidden_list)
        padded = []
        masks = []
        for hidden in hidden_list:
            length = hidden.shape[0]
            if length < max_len:
                hidden = F.pad(hidden, (0, 0, 0, max_len - length))
            padded.append(hidden)
            mask = torch.zeros(max_len, device=device, dtype=torch.bool)
            mask[:length] = True
            masks.append(mask)
        return BackboneOutput(
            hidden_states=torch.stack(padded, dim=0),
            attention_mask=torch.stack(masks, dim=0),
        )

