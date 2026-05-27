from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import write_json


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    checkpoint: Path
    config: Path
    threshold: float | None = None
    description: str = ""
    model_backend: str = "orion"


def load_json_or_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to read YAML files.") from exc
        return yaml.safe_load(text) or {}
    import json

    return json.loads(text)


def load_checkpoint_registry(path: str | Path) -> tuple[str | None, dict[str, CheckpointSpec]]:
    raw = load_json_or_yaml(path)
    default_label = raw.get("default_checkpoint")
    checkpoint_block = raw.get("checkpoints", {})
    specs: dict[str, CheckpointSpec] = {}
    for label, values in checkpoint_block.items():
        specs[label] = CheckpointSpec(
            label=label,
            checkpoint=Path(values["checkpoint"]).expanduser(),
            config=Path(values["config"]).expanduser(),
            threshold=values.get("threshold"),
            description=values.get("description", ""),
            model_backend=values.get("model_backend", "orion"),
        )
    return default_label, specs


def resolve_checkpoint_spec(
    checkpoint_registry: str | Path | None,
    checkpoint_label: str | None,
    checkpoint: str | Path | None,
    model_config: str | Path | None,
) -> CheckpointSpec:
    if checkpoint and model_config:
        label = checkpoint_label or "direct_checkpoint"
        return CheckpointSpec(label=label, checkpoint=Path(checkpoint).expanduser(), config=Path(model_config).expanduser())
    if checkpoint or model_config:
        raise ValueError("--checkpoint and --model-config must be provided together.")
    if not checkpoint_registry:
        raise ValueError("Provide --checkpoint-registry, or provide --checkpoint and --model-config.")
    default_label, specs = load_checkpoint_registry(checkpoint_registry)
    label = checkpoint_label or default_label
    if not label:
        raise ValueError("No checkpoint label was provided and registry has no default_checkpoint.")
    if label not in specs:
        choices = ", ".join(sorted(specs))
        raise KeyError(f"Checkpoint label not found: {label}. Available labels: {choices}")
    return specs[label]


def write_resolved_run_config(path: str | Path, values: dict[str, Any]) -> None:
    write_json(path, values)
