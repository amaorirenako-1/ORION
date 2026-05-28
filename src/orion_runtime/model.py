from __future__ import annotations

import torch

from .config import HeadConfig, LoraConfig, ModelConfig, PoolingConfig
from .models import Evo2FrozenBackbone, HFSequenceBackbone
from .pooling import create_pooling


def create_backbone(model_config: ModelConfig, lora_config: LoraConfig) -> torch.nn.Module:
    family = model_config.family.lower()
    if family == "evo2":
        if lora_config.enabled:
            raise ValueError("Evo2 LoRA is not enabled in this inference runtime.")
        return Evo2FrozenBackbone(model_config)
    if family in {"caduceus", "ntv3"}:
        return HFSequenceBackbone(model_config, lora_config)
    raise ValueError(f"Unsupported ORION model family: {model_config.family}")


class BinaryHead(torch.nn.Module):
    def __init__(self, input_dim: int, config: HeadConfig):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class IZModel(torch.nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoraConfig,
        pooling_config: PoolingConfig,
        head_config: HeadConfig,
    ):
        super().__init__()
        self.backbone = create_backbone(model_config, lora_config)
        hidden_size = int(self.backbone.hidden_size)
        self.pooling = create_pooling(pooling_config.mode, hidden_size)
        self.head = BinaryHead(hidden_size, head_config)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        if hasattr(self.backbone, "trainable_parameters"):
            params.extend(self.backbone.trainable_parameters())
        params.extend(self.pooling.parameters())
        params.extend(self.head.parameters())
        return [param for param in params if param.requires_grad]

    def _classifier_dtype(self) -> torch.dtype:
        for module in (self.pooling, self.head):
            for param in module.parameters(recurse=True):
                return param.dtype
        return torch.float32

    def forward(self, batch: dict, device: torch.device) -> torch.Tensor:
        encoded = self.backbone.encode(batch["sequences"], device)
        hidden_states = encoded.hidden_states
        classifier_dtype = self._classifier_dtype()
        if hidden_states.is_floating_point() and hidden_states.dtype != classifier_dtype:
            hidden_states = hidden_states.to(dtype=classifier_dtype)
        pooled = self.pooling(hidden_states, encoded.attention_mask)
        return self.head(pooled)
