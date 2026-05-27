from __future__ import annotations

import inspect
import os
from typing import Any

import torch

from ..config import LoraConfig, ModelConfig
from ..utils import get_torch_dtype
from .base import BackboneOutput


class HFSequenceBackbone(torch.nn.Module):
    """HuggingFace custom-code sequence backbone for Caduceus and NTv3."""

    def __init__(self, config: ModelConfig, lora: LoraConfig):
        super().__init__()
        try:
            from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for HuggingFace IZ backbones.") from exc

        self.config = config
        self.lora = lora
        self.torch_dtype = get_torch_dtype(config.torch_dtype)
        extra_model_kwargs = config.extra_model_kwargs or {}
        if config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        try:
            self.model = AutoModel.from_pretrained(
                config.name,
                trust_remote_code=config.trust_remote_code,
                local_files_only=config.local_files_only,
                torch_dtype=self.torch_dtype,
                **extra_model_kwargs,
            )
        except Exception:
            self.model = AutoModelForMaskedLM.from_pretrained(
                config.name,
                trust_remote_code=config.trust_remote_code,
                local_files_only=config.local_files_only,
                torch_dtype=self.torch_dtype,
                **extra_model_kwargs,
            )

        self.frozen = config.freeze_backbone and not lora.enabled
        if lora.enabled:
            self._attach_lora()
        elif self.frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.hidden_size = self._infer_hidden_size()

    def _attach_lora(self) -> None:
        try:
            from peft import LoraConfig as PeftLoraConfig
            from peft import get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required when lora.enabled=true: pip install peft") from exc

        peft_config = PeftLoraConfig(
            r=self.lora.r,
            lora_alpha=self.lora.alpha,
            lora_dropout=self.lora.dropout,
            target_modules=self.lora.target_modules,
            bias="none",
            **self._dora_kwargs(PeftLoraConfig),
        )
        self.model = get_peft_model(self.model, peft_config)

    def _dora_kwargs(self, peft_lora_config_cls: Any) -> dict[str, Any]:
        if not getattr(self.lora, "use_dora", False):
            return {}
        signature = inspect.signature(peft_lora_config_cls)
        if "use_dora" not in signature.parameters:
            raise RuntimeError("This PEFT version does not support use_dora=true. Upgrade peft or use a LoRA checkpoint.")
        return {"use_dora": True}

    def _infer_hidden_size(self) -> int:
        for attr in ("hidden_size", "d_model", "embed_dim"):
            value = getattr(getattr(self.model, "config", None), attr, None)
            if isinstance(value, int):
                return value
        for param in self.model.parameters():
            if param.dim() == 2:
                return int(param.shape[-1])
        raise RuntimeError("Unable to infer hidden size from HuggingFace backbone.")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _normalize_for_model(self, sequence: str) -> str:
        multiple = self.config.length_multiple
        if not multiple:
            return sequence
        remainder = len(sequence) % multiple
        if remainder == 0:
            return sequence
        return sequence + "N" * (multiple - remainder)

    def _select_hidden(self, outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            layer = self.config.hidden_layer
            if isinstance(layer, str):
                layer = -1 if layer.lower() == "last" else int(layer)
            return outputs.hidden_states[layer]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        raise RuntimeError("Backbone output has no hidden states or last_hidden_state.")

    def encode(self, sequences: list[str], device: torch.device) -> BackboneOutput:
        normalized = [self._normalize_for_model(seq) for seq in sequences]
        inputs = self.tokenizer(
            normalized,
            return_tensors="pt",
            padding=True,
            truncation=self.config.tokenizer_truncation,
            add_special_tokens=self.config.add_special_tokens,
            pad_to_multiple_of=self.config.length_multiple,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        signature = inspect.signature(self.model.forward)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        if not accepts_kwargs:
            allowed = set(signature.parameters)
            inputs = {key: value for key, value in inputs.items() if key in allowed}
        model_kwargs = dict(inputs)
        if accepts_kwargs or "output_hidden_states" in signature.parameters:
            model_kwargs["output_hidden_states"] = True
        self.model.to(device)
        if self.frozen:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**model_kwargs)
        else:
            outputs = self.model(**model_kwargs)
        hidden = self._select_hidden(outputs)
        mask = inputs.get("attention_mask")
        if mask is None or mask.shape[-1] != hidden.shape[1]:
            mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
        else:
            mask = mask.bool()
        return BackboneOutput(hidden_states=hidden, attention_mask=mask)
