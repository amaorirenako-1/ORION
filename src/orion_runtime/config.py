from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    name: str = "p0_experiment"
    output_dir: str = "outputs/p0_experiment"
    seed: int = 42
    cache_dir: str | None = None


@dataclass
class DataConfig:
    splits_dir: str = "../IZ_Evo2_Phase1/data/splits_no_flank"
    max_length: int = 32768
    max_length_by_bucket: dict[str, int] | None = None
    window_stride: int | None = None
    train_sliding_windows: bool = False
    eval_sliding_windows: bool = False
    batch_size: int = 1
    num_workers: int = 0
    center_crop: bool = True
    pad_char: str = "N"
    pad_to_max_length: bool = True
    train_random_crop: bool = False
    bucket_batching: bool = False
    drop_last: bool = False


@dataclass
class ModelConfig:
    family: str = "evo2"
    name: str = "evo2_7b"
    trust_remote_code: bool = True
    local_files_only: bool = False
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    hidden_layer: int | str = -1
    evo2_layer_name: str = "blocks.28.mlp.l3"
    freeze_backbone: bool = True
    length_multiple: int | None = None
    add_special_tokens: bool = False
    tokenizer_truncation: bool = False
    extra_model_kwargs: dict[str, Any] | None = None


@dataclass
class LoraConfig:
    enabled: bool = False
    use_dora: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


@dataclass
class PoolingConfig:
    mode: str = "mean"


@dataclass
class HeadConfig:
    hidden_dim: int = 512
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    max_steps: int = 1000
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    grad_clip_norm: float = 1.0
    eval_steps: int = 200
    save_steps: int = 500
    bf16: bool = True
    fp16: bool = False
    log_steps: int = 10
    eval_at_start: bool = False


@dataclass
class PipelineConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise KeyError(f"Unknown config key: {type(obj).__name__}.{key}")


def load_config(path: str | Path, overrides: list[str] | None = None) -> PipelineConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs: pip install pyyaml") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = PipelineConfig()
    for section, values in raw.items():
        if not hasattr(config, section):
            raise KeyError(f"Unknown config section: {section}")
        if values is not None:
            _update_dataclass(getattr(config, section), values)

    for override in overrides or []:
        if "=" not in override or "." not in override:
            raise ValueError(f"Override must look like section.key=value, got: {override}")
        dotted, value = override.split("=", 1)
        section, key = dotted.split(".", 1)
        target = getattr(config, section)
        current = getattr(target, key)
        if isinstance(current, bool):
            parsed: Any = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(current, int) and value.lower() != "none":
            parsed = int(value)
        elif isinstance(current, float):
            parsed = float(value)
        elif value.lower() == "none":
            parsed = None
        else:
            parsed = value
        setattr(target, key, parsed)

    return config

