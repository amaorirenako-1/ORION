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

Optional MACS3 support:

```bash
pip install "MACS3>=3.0"
```

The default `requirements.txt` pins `transformers<4.52` to reduce the risk of torch/transformers flex-attention API mismatches. If your NTv3 runtime has a separately validated version set, test and replace those pins in a clean environment.

## Model Registry

If you are using the four final selected checkpoints, first follow [docs/CHECKPOINT_IMPORT_EN.md](docs/CHECKPOINT_IMPORT_EN.md) to import the checkpoint weights and matching `resolved_config.json` files into an independent model directory. This import process copies only released model artifacts and does not depend on earlier evaluation scripts, attention scripts, or old experiment configs.

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
      "checkpoint": "/data01/share/cxsy1/orion_checkpoint/gc100_best/best.pt",
      "config": "/data01/share/cxsy1/orion_checkpoint/gc100_best/resolved_config.json",
      "threshold": 0.463
    }
  }
}
```

Fields:

- `default_checkpoint`: label used when `--checkpoint-label` is omitted.
- `checkpoint`: model weights file.
- `config`: matching `resolved_config.json` or YAML.
- `threshold`: recorded calibrated classification threshold; peak calling does not use it by default.
- `model_backend`: currently `orion`.

## Genome Scoring

K562/hg19 example:

```bash
orion predict \
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --output-dir /data01/share/cxsy1/orion/k562_hg19_gc100_best \
  --output-prefix k562_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --batch-size 4
```

Scan selected chromosomes:

```bash
orion predict \
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --sequence-names chr1,chr2,chr3 \
  --output-dir output/chr1_chr2_chr3
```

Scan selected regions:

```bash
orion predict \
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --regions chr1:0-10000000,chr2:5000000-12000000 \
  --output-dir output/regions
```

Use a checkpoint directly:

```bash
orion predict \
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint /data01/share/cxsy1/orion_checkpoint/gc100_best/best.pt \
  --model-config /data01/share/cxsy1/orion_checkpoint/gc100_best/resolved_config.json \
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
  --score-track /data01/share/cxsy1/orion/k562_hg19_gc100_best/k562_gc100_best.prob.bedGraph \
  --output-dir /data01/share/cxsy1/orion/k562_hg19_gc100_best/peaks \
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
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --output-dir /data01/share/cxsy1/orion/k562_hg19_gc100_best_run \
  --output-prefix k562_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --method scipy \
  --peak-min-score 0.70
```

The default peak-calling parameters are intentionally stricter than a calibrated classifier threshold: `--peak-min-score 0.70` plus local prominence filtering for the `scipy` method. This aims to produce high-confidence candidate IZ regions rather than every likely positive window.

## `call-peaks` Arguments

- `--score-track`: input bedGraph with `chrom start end score`.
- `--output-dir`: output directory.
- `--method`: `scipy`, `threshold`, or `macs3`.
- `--peak-min-score`: minimum score for candidate peaks. Default: `0.70`.
- `--peak-prominence`: local prominence for `scipy`. Default: `0.05`.
- `--peak-min-distance`: minimum summit distance for `scipy`. Default: `30000 bp`.
- `--peak-min-width`: minimum peak width. Default: `5000 bp`.
- `--peak-max-gap`: maximum gap used for peak extension/merging. Default: `30000 bp`.
- `--smooth-bins`: rolling-mean bins before peak calling. Default: `5`.
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

## Recommended Workflow

1. Test the environment on `chr22` or a small region.
2. Inspect `prob` values in `window_predictions.tsv`.
3. Run whole-genome K562/hg19 prediction.
4. Run `call-peaks` with the strict defaults.
5. If candidate peaks are too sparse, lower `--peak-min-score` gradually, for example `0.65` then `0.60`, and record the chosen cutoff.

## FASTA Index

The tool uses `pyfaidx`. If no `.fai` index exists next to the FASTA, `pyfaidx` will try to create one. If the reference directory is read-only, run `samtools faidx genome.fa` first or copy the FASTA to a writable location.
