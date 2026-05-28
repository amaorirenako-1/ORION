# Selected Checkpoint Import and Independent Runtime Guide

This guide starts from importing the four selected checkpoints into an independent `ORION` layout. The scanner should depend only on this package, imported checkpoint files, matching `resolved_config.json` files, and a reference FASTA. It should not depend on earlier evaluation scripts, analysis scripts, attention scripts, or old experiment configs.

## Decoupling Rules

Copy only:

- Checkpoint weights, such as `best.pt` or `step_2800.pt`.
- The matching `resolved_config.json` from each training output directory.
- Optional model cards, checksum files, and threshold records.

Do not copy or depend on:

- Earlier evaluation or analysis scripts.
- Old training YAML configs.
- `external_tests/details`, prediction files, attention outputs, or temporary analysis directories.
- Genome FASTA files, bedGraph tracks, or run outputs.

The inference runtime is vendored in `src/orion_runtime/`. After model import, the CLI locates weights through `configs/checkpoints.local.json`.

ORION checkpoints do not include the full NTv3 650M pretrained backbone. The runtime host must prepare the NTv3 650M base model and point `model.name` in a local runtime config to that directory.

## Recommended Layout

```bash
PROJECT=/path/to/orion
MODEL_ROOT=/path/to/orion_checkpoint
```

Keep model files outside the Git working tree:

```text
/path/to/orion/
  configs/checkpoints.example.json
  configs/checkpoints.local.json        # local registry, do not commit
  src/

/path/to/orion_checkpoint/
  gc100_best/checkpoint.pt
  gc100_best/resolved_config.json
  gc100_2800/checkpoint.pt
  gc100_2800/resolved_config.json
  random100_3200/checkpoint.pt
  random100_3200/resolved_config.json
  random_psm_2200/checkpoint.pt
  random_psm_2200/resolved_config.json
  checksums.sha256
```

## Import Commands

```bash
PROJECT=/path/to/orion
MODEL_ROOT=/path/to/orion_checkpoint

mkdir -p "$MODEL_ROOT/gc100_best"
mkdir -p "$MODEL_ROOT/gc100_2800"
mkdir -p "$MODEL_ROOT/random100_3200"
mkdir -p "$MODEL_ROOT/random_psm_2200"
```

```bash
SOURCE_GC100=/path/to/gc100_training_output
SOURCE_RANDOM100=/path/to/random100_training_output
SOURCE_RANDOM_PSM=/path/to/random_psm_training_output

cp "$SOURCE_GC100/checkpoints/best.pt" \
  "$MODEL_ROOT/gc100_best/checkpoint.pt"
cp "$SOURCE_GC100/resolved_config.json" \
  "$MODEL_ROOT/gc100_best/resolved_config.json"

cp "$SOURCE_GC100/checkpoints/step_2800.pt" \
  "$MODEL_ROOT/gc100_2800/checkpoint.pt"
cp "$SOURCE_GC100/resolved_config.json" \
  "$MODEL_ROOT/gc100_2800/resolved_config.json"

cp "$SOURCE_RANDOM100/checkpoints/step_3200.pt" \
  "$MODEL_ROOT/random100_3200/checkpoint.pt"
cp "$SOURCE_RANDOM100/resolved_config.json" \
  "$MODEL_ROOT/random100_3200/resolved_config.json"

cp "$SOURCE_RANDOM_PSM/checkpoints/step_2200.pt" \
  "$MODEL_ROOT/random_psm_2200/checkpoint.pt"
cp "$SOURCE_RANDOM_PSM/resolved_config.json" \
  "$MODEL_ROOT/random_psm_2200/resolved_config.json"
```

Verify checksums:

```bash
find "$MODEL_ROOT" -maxdepth 2 -type f | sort
sha256sum "$MODEL_ROOT"/*/checkpoint.pt "$MODEL_ROOT"/*/resolved_config.json > "$MODEL_ROOT/checksums.sha256"
cat "$MODEL_ROOT/checksums.sha256"
```

## Prepare NTv3 Base Model and Local Config

The released `resolved_config.json` should stay sanitized and may contain:

```json
"name": "/path/to/NTv3_650M_pre"
```

That placeholder is not directly runnable. Prepare the NTv3 650M base model on the runtime host:

```bash
BASE_MODEL=/path/to/NTv3_650M_pre
ls "$BASE_MODEL"
```

The directory should contain HuggingFace model files such as `config.json`, tokenizer files, custom modeling Python files, and model weights.

Generate local runtime configs:

```bash
MODEL_ROOT=/path/to/orion_checkpoint
BASE_MODEL=/path/to/NTv3_650M_pre

for cfg in "$MODEL_ROOT"/*/resolved_config.json; do
  local_cfg="${cfg%.json}.local.json"
  python - "$cfg" "$local_cfg" "$BASE_MODEL" <<'PY'
import json
import sys
from pathlib import Path

src, dst, base_model = map(Path, sys.argv[1:])
obj = json.loads(src.read_text())
obj["model"]["name"] = str(base_model)
obj["model"]["local_files_only"] = True
dst.write_text(json.dumps(obj, indent=2) + "\n")
print(dst)
PY
done
```

Point `config` in `configs/checkpoints.local.json` to `resolved_config.local.json`. Do not publish local configs if they contain internal paths.

## Local Registry

```bash
cd "$PROJECT"
cp configs/checkpoints.example.json configs/checkpoints.local.json
```

Edit `configs/checkpoints.local.json` so each label points to the imported model directory. This file contains local absolute paths and should not be committed.

## Smoke Test

```bash
conda create -n orion python=3.11 -y
conda activate orion
cd "$PROJECT"
pip install -r requirements.txt
pip install -e .
orion --version
```

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --regions chr1:900000-1500000 \
  --output-dir output/smoke_gc100_best \
  --output-prefix smoke_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --batch-size 1
```

Peak calling is model-independent:

```bash
orion call-peaks \
  --score-track output/smoke_gc100_best/smoke_gc100_best.prob.bedGraph \
  --output-dir output/smoke_gc100_best/peaks \
  --output-prefix smoke_gc100_best \
  --method scipy \
  --peak-min-score 0.70 \
  --peak-prominence 0.05
```

## Release Packaging

Publish code and models separately:

- GitHub repository: code, README files, example registry, and docs.
- Release asset or external storage: imported `orion_checkpoint/` model directory, checksum file, and model card.

Create a model package:

```bash
cd /path/to/model_release_parent
tar -czf orion_models_v1.0.0.tar.gz orion_checkpoint
sha256sum orion_models_v1.0.0.tar.gz
```

Release notes should record the tool tag, model package checksum, checkpoint labels, recommended default checkpoint `gc100_best`, window size `30000 bp`, default score `prob`, and strict peak score cutoff `0.70`.
