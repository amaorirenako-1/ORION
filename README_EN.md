# ORION

`ORION` (Origin Recognition and Initiation-zone Omics Network) is an independent command-line tool for whole-genome initiation-zone (IZ) scoring. It scans a reference FASTA with fixed genomic windows, writes per-window model scores and a bedGraph score track, and can call high-confidence candidate IZ peaks from any bedGraph track.

This project does not depend on earlier evaluation scripts or experiment configs. A minimal inference runtime is vendored in this package. A released model must include both its checkpoint and matching `resolved_config.json`, otherwise the backbone, LoRA, pooling, and head settings cannot be reconstructed reliably.

## Features

- `orion predict`: score sliding windows from a genome FASTA.
- `orion call-peaks`: call candidate IZ peaks from a bedGraph score track; this step is independent of the model.
- `orion run`: run prediction and peak calling in one command.
- Default window size: `30000 bp`.
- Default score: `prob`.
- Default checkpoint label: `gc100_best`, with support for alternative labels or direct checkpoint/config paths.
- Supports arbitrary genome FASTA files. K562/hg19 is the first intended use case.
- No bigWig dependency.

## Installation

Recommended fresh server environment:

```bash
conda create -n orion python=3.11 -y
conda activate orion
cd /path/to/orion
pip install -r requirements.txt
pip install -e .
```

Optional MACS3 peak-calling support:

```bash
pip install "MACS3>=3.0"
```

The default `requirements.txt` pins `transformers<4.52` to reduce the risk of torch/transformers flex-attention API mismatches. If your NTv3 runtime has a separately validated version set, test and replace those pins in a clean environment.

## Model Registry

If you are using the four final selected checkpoints, first follow [docs/CHECKPOINT_IMPORT_EN.md](docs/CHECKPOINT_IMPORT_EN.md) to import the checkpoint weights and matching `resolved_config.json` files into an independent model directory. This import process copies only released model artifacts and does not depend on earlier evaluation scripts, attention scripts, or old experiment configs.

### Prepare the NTv3 Base Model

The released ORION checkpoints contain LoRA/DoRA adapter weights and the classifier head, but not the full NTv3 650M backbone. The runtime host must prepare the NTv3 650M pretrained base model and point `model.name` in the config to that directory.

Check the base model directory:

```bash
BASE_MODEL=/path/to/NTv3_650M_pre
ls "$BASE_MODEL"
```

It should contain the HuggingFace model files, such as `config.json`, tokenizer files, custom modeling Python files, and model weight files.

The public `resolved_config.json` uses a placeholder:

```json
"name": "/path/to/NTv3_650M_pre"
```

For local execution, generate `resolved_config.local.json` instead of modifying the released config in place:

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

Then point `config` in `configs/checkpoints.local.json` to `resolved_config.local.json`.

Copy and edit the example registry:

```bash
cp configs/checkpoints.example.json configs/checkpoints.local.json
```

Example:

```json
{
  "default_checkpoint": "gc100_best",
  "checkpoints": {
    "gc100_best": {
      "description": "Default high-confidence K562 checkpoint",
      "model_backend": "orion",
      "checkpoint": "/path/to/orion_checkpoint/gc100_best/checkpoint.pt",
      "config": "/path/to/orion_checkpoint/gc100_best/resolved_config.local.json",
      "threshold": 0.463
    }
  }
}
```

Fields:

- `default_checkpoint`: label used when `--checkpoint-label` is omitted.
- `checkpoint`: model weights file.
- `config`: local runtime config for the checkpoint. Prefer `resolved_config.local.json`, where `model.name` points to the local NTv3 650M base model path.
- `threshold`: recorded calibrated classification threshold; peak calling does not use it by default.
- `model_backend`: currently `orion`.

## Genome Scoring

K562/hg19 example:

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --output-dir /path/to/orion_runs/k562_hg19_gc100_best \
  --output-prefix k562_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --batch-size 4
```

Scan selected chromosomes:

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --sequence-names chr1,chr2,chr3 \
  --output-dir output/chr1_chr2_chr3
```

Scan selected regions:

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --regions chr1:0-10000000,chr2:5000000-12000000 \
  --output-dir output/regions
```

Use a checkpoint directly:

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint /path/to/orion_checkpoint/gc100_best/checkpoint.pt \
  --model-config /path/to/orion_checkpoint/gc100_best/resolved_config.local.json \
  --output-dir output/direct_model
```

## `predict` Arguments

- `--fasta`: reference genome FASTA.
- `--output-dir`: output directory.
- `--checkpoint-registry`: checkpoint registry JSON/YAML.
- `--checkpoint-label`: checkpoint label from the registry; defaults to `default_checkpoint`.
- `--checkpoint`: direct checkpoint path, used together with `--model-config`.
- `--model-config`: direct resolved model config path.
- `--device`: torch device, such as `cuda`, `cuda:0`, or `cpu`. Defaults to `cuda`, falling back to CPU if CUDA is unavailable.
- `--torch-dtype`: override model dtype from config, such as `float16` or `bfloat16`.
- `--window-size`: genomic window length. Default: `30000`.
- `--stride`: sliding-window stride. Default: `30000`.
- `--batch-size`: inference batch size.
- `--max-n-frac`: skip windows with a higher `N` fraction. Default: `0.20`.
- `--sequence-names`: comma-separated FASTA sequence names.
- `--regions`: comma-separated regions, e.g. `chr1:0-1000000`.
- `--regions-file`: one sequence name or `chrom:start-end` region per line.
- `--score`: score written to bedGraph. Choices: `prob`, `logit`. Default: `prob`.
- `--track-bin-size`: center-bin width for bedGraph. Default: `stride`.
- `--output-prefix`: output file prefix.
- `--write-sequences`: include raw window sequences in the TSV. Not recommended for whole-genome runs.
- `--no-terminal-window`: disable the extra terminal window that covers the end of each region.

## `predict` Outputs

For `--output-prefix k562_gc100_best`:

- `k562_gc100_best.window_predictions.tsv`
  - One row per scored window.
  - Key columns: `chrom`, `start`, `end`, `coord_key`, `prob`, `logit`, `score_name`, `score`, `n_fraction`, `window_length`.
- `k562_gc100_best.prob.bedGraph`
  - bedGraph score track.
  - To avoid overlapping bedGraph intervals, each window is represented by a center-bin. Full-window coordinates remain available in the TSV.
- `k562_gc100_best.scan_summary.json`
  - Run parameters, model information, regions, scored-window count, and output paths.
- `k562_gc100_best.resolved_run_config.json`
  - Reproducible run snapshot.

## Peak Calling

Run peak calling from any bedGraph:

```bash
orion call-peaks \
  --score-track /path/to/orion_runs/k562_hg19_gc100_best/k562_gc100_best.prob.bedGraph \
  --output-dir /path/to/orion_runs/k562_hg19_gc100_best/peaks \
  --output-prefix k562_gc100_best \
  --method scipy \
  --peak-min-score 0.70 \
  --peak-prominence 0.05 \
  --peak-min-distance 30000 \
  --peak-min-width 5000 \
  --peak-max-gap 30000 \
  --smooth-bins 5
```

Run scoring and peak calling together:

```bash
orion run \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --output-dir /path/to/orion_runs/k562_hg19_gc100_best_run \
  --output-prefix k562_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --method scipy \
  --peak-min-score 0.70
```

The default peak-calling parameters are intentionally stricter than a calibrated classifier threshold: `--peak-min-score 0.70` plus local prominence filtering for the `scipy` method. This aims to produce high-confidence candidate IZ regions rather than every likely positive window.

For broader IZ-style regions, use either the built-in `threshold` method or the optional MACS3 broad caller:

```bash
orion call-peaks \
  --score-track /path/to/orion_runs/k562_hg19_gc100_best/k562_gc100_best.prob.bedGraph \
  --output-dir /path/to/orion_runs/k562_hg19_gc100_best/peaks_broad \
  --output-prefix k562_gc100_best \
  --method macs3-broad \
  --peak-min-score 0.70 \
  --macs3-broad-link-score 0.50 \
  --peak-min-width 30000 \
  --peak-max-gap 30000 \
  --macs3-broad-max-gap 90000 \
  --macs3-fill-gaps-score 0
```

`macs3` uses `macs3 bdgpeakcall`, which is a single-cutoff peak caller. `macs3-broad` uses `macs3 bdgbroadcall`, which links strong regions through weaker nearby signal. For IZ analysis, broad candidate regions are often easier to interpret than narrow summit-like peaks, because ORION scores 30 kb windows rather than base-pair-resolution binding events. MACS3 bedGraph callers expect continuous tracks; if the ORION bedGraph has gaps from skipped windows, use `--macs3-fill-gaps-score 0` to create a temporary gap-filled MACS3 input.

## `call-peaks` Arguments

- `--score-track`: input bedGraph with `chrom start end score`.
- `--output-dir`: output directory.
- `--method`: `scipy`, `threshold`, `macs3`, or `macs3-broad`.
- `--peak-min-score`: minimum score for candidate peaks. Default: `0.70`.
- `--peak-prominence`: local prominence for `scipy`. Default: `0.05`.
- `--peak-min-distance`: minimum summit distance for `scipy`. Default: `30000 bp`.
- `--peak-min-width`: minimum peak width. Default: `5000 bp`.
- `--peak-max-gap`: maximum gap used for peak extension/merging. Default: `30000 bp`.
- `--smooth-bins`: rolling-mean bins before peak calling. Default: `5`.
- `--macs3-broad-link-score`: weak-link cutoff for `macs3-broad` / `bdgbroadcall`. Default: `0.50`.
- `--macs3-broad-max-gap`: level-2 weak-region max gap for `macs3-broad` / `bdgbroadcall`. Default: `90000 bp`.
- `--macs3-no-trackline`: pass `--no-trackline` to MACS3.
- `--macs3-verbose`: MACS3 verbose level.
- `--macs3-fill-gaps-score`: write a temporary MACS3 input bedGraph with gaps between adjacent bins filled by this score, usually `0`.
- `--macs3-cutoff-analysis`: run MACS3 `bdgpeakcall` cutoff analysis instead of writing peaks. Only valid with `--method macs3`.
- `--macs3-cutoff-analysis-steps`: number of MACS3 cutoff-analysis steps.
- `--output-prefix`: output file prefix.

## `call-peaks` Outputs

- `*.peaks.bed`
  - BED-like peak file.
  - First six columns: `chrom start end peak_id peak_score strand`.
  - Extra columns: `summit_start`, `summit_end`, `mean_score`, `n_bins`, `method`, `cutoff`, `prominence`.
- `*.peaks_summary.tsv`
  - Full per-peak statistics.
- `*.peak_calling_summary.json`
  - Input, parameters, peak count, and output paths.
- With `--method macs3`, the tool calls external `macs3 bdgpeakcall` and writes `*.macs3_bdgpeakcall.bed`.
- With `--method macs3-broad`, the tool calls external `macs3 bdgbroadcall` and writes `*.macs3_bdgbroadcall.gappedPeak`.

## Recommended Workflow

1. Test the environment on `chr22` or a small region.
2. Inspect `prob` values in `window_predictions.tsv`.
3. Run whole-genome K562/hg19 prediction.
4. Run `call-peaks` with the strict defaults.
5. If candidate peaks are too sparse, lower `--peak-min-score` gradually, for example `0.65` then `0.60`, and record the chosen cutoff.

## FASTA Index

The tool uses `pyfaidx`. If no `.fai` index exists next to the FASTA, `pyfaidx` will try to create one. If the reference directory is read-only, run `samtools faidx genome.fa` first or copy the FASTA to a writable location.
