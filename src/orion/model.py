from __future__ import annotations

import os
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from .config import load_json_or_yaml


def _update_dataclass_tolerant(obj: Any, values: dict[str, Any], section: str) -> None:
    allowed = {field.name for field in fields(obj)}
    quiet_sections = {"data", "training"}
    for key, value in values.items():
        if key in allowed:
            setattr(obj, key, value)
        elif section in quiet_sections:
            continue
        else:
            warnings.warn(f"Ignoring unknown config key: {section}.{key}", RuntimeWarning)


def load_pipeline_config(path: str | Path):
    from orion_runtime.config import PipelineConfig

    raw = load_json_or_yaml(path)
    config = PipelineConfig()
    for section, values in raw.items():
        if not hasattr(config, section):
            warnings.warn(f"Ignoring unknown config section: {section}", RuntimeWarning)
            continue
        if isinstance(values, dict):
            _update_dataclass_tolerant(getattr(config, section), values, section)
    return config


def configure_model_cache(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_MODULES_CACHE", os.path.join(cache_dir, "huggingface", "modules"))


def _extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "state_dict", "trainable_state_dict", "model"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint_obj and all(hasattr(value, "shape") for value in checkpoint_obj.values()):
            return checkpoint_obj
    raise ValueError("Unable to find a model state_dict in checkpoint.")


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state:
        return state
    if not any(key.startswith("module.") for key in state):
        return state
    return {key.removeprefix("module."): value for key, value in state.items()}


class OrionPredictor:
    """Prediction wrapper for the vendored ORION inference runtime."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: str = "cuda",
        torch_dtype: str | None = None,
    ):
        from orion_runtime.model import IZModel
        from orion_runtime.utils import resolve_device

        self.config = load_pipeline_config(config_path)
        configure_model_cache(getattr(self.config.experiment, "cache_dir", None))
        if device:
            self.config.model.device = device
        if torch_dtype:
            self.config.model.torch_dtype = torch_dtype
        self.device = resolve_device(self.config.model.device)
        self.model = IZModel(self.config.model, self.config.lora, self.config.pooling, self.config.head)
        checkpoint_obj = torch.load(Path(checkpoint_path), map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(checkpoint_obj))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            warnings.warn(f"Unexpected checkpoint keys were ignored: {len(unexpected)}", RuntimeWarning)
        critical_missing = [
            key
            for key in missing
            if key.startswith(("pooling.", "head."))
            or "lora_" in key
            or "lora_magnitude_vector" in key
        ]
        if critical_missing:
            preview = ", ".join(critical_missing[:5])
            warnings.warn(
                f"Missing classifier/LoRA keys after checkpoint load: {len(critical_missing)}. "
                f"First keys: {preview}",
                RuntimeWarning,
            )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, sequences: list[str]) -> list[dict[str, float]]:
        batch = {"sequences": sequences}
        logits = self.model(batch, self.device).detach().float().cpu()
        probs = torch.sigmoid(logits)
        return [
            {"logit": float(logit), "prob": float(prob)}
            for logit, prob in zip(logits.tolist(), probs.tolist(), strict=False)
        ]


def load_predictor(
    model_backend: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str,
    torch_dtype: str | None,
):
    backend = model_backend.lower()
    if backend != "orion":
        raise ValueError(f"Unsupported model backend: {model_backend}")
    return OrionPredictor(checkpoint_path, config_path, device=device, torch_dtype=torch_dtype)
