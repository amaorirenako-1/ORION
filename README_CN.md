# ORION

`ORION`（Origin Recognition and Initiation-zone Omics Network）是一个独立的全基因组 initiation zone (IZ) 扫描工具。它把训练好的 IZ 模型封装为命令行程序，支持在任意参考基因组 FASTA 上按固定窗口运行，输出每个窗口的模型分数和 bedGraph 轨道，并可进一步从分数轨道中调用高置信 IZ peak。

本项目不依赖先前的评估脚本或实验 config。推理所需的最小模型 runtime 已经 vendored 到本包中；发布模型时必须同时提供 checkpoint 和该 checkpoint 对应的 `resolved_config.json`，否则无法可靠恢复 backbone、LoRA、pooling 和 head 设置。

## 主要功能

- `orion predict`：在 FASTA 上按滑动窗口预测 IZ 概率。
- `orion call-peaks`：从任意 bedGraph 分数轨道调用候选 IZ peak；这一步与模型解耦。
- `orion run`：先预测，再对预测得到的 bedGraph 进行 peak calling。
- 默认窗口大小为 `30000 bp`。
- 默认输出分数为 `prob`。
- 默认 checkpoint label 为 `gc100_best`，也支持 registry 中的其他 checkpoint 或直接指定 checkpoint/config。
- 支持任意基因组 FASTA；K562/hg19 只是推荐的首个使用场景。
- 不依赖 bigWig 输出。

## 安装

建议在服务器新建环境测试：

```bash
conda create -n orion python=3.11 -y
conda activate orion
cd /path/to/orion
pip install -r requirements.txt
pip install -e .
```

如果需要使用 MACS3 的 `bdgpeakcall`：

```bash
pip install "MACS3>=3.0"
```

注意：`requirements.txt` 将 `transformers` 限制在 `<4.52`，这是为了降低 `torch` 与新版 `transformers` 中 flex attention API 不匹配的风险。如果你的 NTv3 环境已经验证过另一组版本，可以在独立环境中按该版本替换。

## 模型文件准备

如果你要使用我们最终选定的四个 checkpoint，请先按照 [docs/CHECKPOINT_IMPORT_CN.md](docs/CHECKPOINT_IMPORT_CN.md) 把 checkpoint 和对应的 `resolved_config.json` 导入到独立模型目录中。该流程只复制模型权重和 resolved config，不依赖此前的评估脚本、attention 脚本或旧实验配置。

### 准备 NTv3 预训练基模

ORION 发布的 checkpoint 是 LoRA/DoRA 增量权重和分类头权重，不包含完整 NTv3 650M backbone。因此运行端必须提前准备 NTv3 650M 预训练基模，并让 `resolved_config.json` 中的 `model.name` 指向该目录。

先确认基模目录存在：

```bash
BASE_MODEL=/path/to/NTv3_650M_pre
ls "$BASE_MODEL"
```

目录中应包含 HuggingFace 模型所需文件，例如 `config.json`、tokenizer 文件、custom modeling Python 文件，以及权重文件。

公开发布的 `resolved_config.json` 使用占位路径：

```json
"name": "/path/to/NTv3_650M_pre"
```

本地运行时建议生成 `resolved_config.local.json`，不要直接改发布版 config：

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

之后 `configs/checkpoints.local.json` 中的 `config` 字段应指向 `resolved_config.local.json`。

推荐准备一个 checkpoint registry，例如复制并修改：

```bash
cp configs/checkpoints.example.json configs/checkpoints.local.json
```

`configs/checkpoints.local.json` 的格式如下：

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

字段含义：

- `default_checkpoint`：未指定 `--checkpoint-label` 时使用的模型标签。
- `checkpoint`：模型权重文件，支持原训练输出的 `.pt`。
- `config`：与该 checkpoint 对应的本地运行 config。推荐指向 `resolved_config.local.json`，其中 `model.name` 已替换为本机 NTv3 650M 基模路径。
- `threshold`：分类任务的推荐校准阈值，只作为记录；默认 peak calling 不直接使用它。
- `model_backend`：目前支持 `orion`。

## 全基因组扫描

K562/hg19 示例：

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

只扫描部分染色体：

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --sequence-names chr1,chr2,chr3 \
  --output-dir output/chr1_chr2_chr3
```

只扫描指定区间：

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --regions chr1:0-10000000,chr2:5000000-12000000 \
  --output-dir output/regions
```

也可以直接指定模型，不使用 registry：

```bash
orion predict \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint /path/to/orion_checkpoint/gc100_best/checkpoint.pt \
  --model-config /path/to/orion_checkpoint/gc100_best/resolved_config.local.json \
  --output-dir output/direct_model
```

## `predict` 参数说明

- `--fasta`：参考基因组 FASTA。
- `--output-dir`：输出目录。
- `--checkpoint-registry`：checkpoint registry JSON/YAML。
- `--checkpoint-label`：registry 中的 checkpoint 标签；不填时使用 `default_checkpoint`。
- `--checkpoint`：直接指定 checkpoint 路径，必须和 `--model-config` 同时使用。
- `--model-config`：直接指定 resolved model config 路径。
- `--device`：推理设备，例如 `cuda`、`cuda:0`、`cpu`。默认 `cuda`，无 GPU 时会回退到 CPU。
- `--torch-dtype`：覆盖 config 中的模型 dtype，例如 `float16`、`bfloat16`。
- `--window-size`：窗口长度，默认 `30000`。
- `--stride`：窗口步长，默认 `30000`。
- `--batch-size`：推理 batch size。
- `--max-n-frac`：窗口中 `N` 的最大比例，默认 `0.20`；超过则跳过。
- `--sequence-names`：逗号分隔的 FASTA sequence 名，例如 `chr1,chr2`。
- `--regions`：逗号分隔区间，例如 `chr1:0-1000000,chr2:50000-90000`。
- `--regions-file`：每行一个 sequence 名或 `chrom:start-end` 区间。
- `--score`：写入 bedGraph 的分数列，默认 `prob`，也可选 `logit`。
- `--track-bin-size`：bedGraph 中心 bin 的宽度，默认等于 `--stride`。
- `--output-prefix`：输出文件前缀。
- `--write-sequences`：在 TSV 中写出原始窗口序列；全基因组运行时不建议开启。
- `--no-terminal-window`：默认每个 region 末端额外补一个 terminal window，以覆盖 region 末端；该参数可关闭。

## `predict` 输出说明

假设 `--output-prefix k562_gc100_best`：

- `k562_gc100_best.window_predictions.tsv`
  - 每个预测窗口一行。
  - 主要列：
    - `chrom/start/end`：窗口坐标，0-based half-open。
    - `coord_key`：`chrom:start-end`。
    - `prob`：sigmoid 后的 IZ 概率。
    - `logit`：模型原始 logit。
    - `score_name`：写入 bedGraph 的分数类型。
    - `score`：实际使用的分数。
    - `n_fraction`：窗口中 `N` 比例。
    - `window_length`：窗口长度。
- `k562_gc100_best.prob.bedGraph`
  - bedGraph 分数轨道。
  - 为避免 overlapping bedGraph，默认写每个窗口中心附近的 center-bin，bin 宽度为 `--track-bin-size`，默认等于 stride。
- `k562_gc100_best.scan_summary.json`
  - 运行参数、模型信息、region 列表、预测窗口数和输出路径。
- `k562_gc100_best.resolved_run_config.json`
  - 当前运行的可追溯配置快照。

## Peak calling

`call-peaks` 可以独立使用，只需要输入 bedGraph：

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

也可以将预测和 peak calling 串联：

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

默认 peak calling 比分类校准阈值更严格：默认 `--peak-min-score 0.70`，并且 `scipy` 方法还要求 `--peak-prominence 0.05`。这样输出的是高置信 IZ 候选区域，而不是所有可能为阳性的窗口。若后续目标是提高召回率，可以降低 `--peak-min-score`，但需要在结果注释中说明。

## `call-peaks` 参数说明

- `--score-track`：输入 bedGraph，4 列：`chrom start end score`。
- `--output-dir`：输出目录。
- `--method`：`scipy`、`threshold` 或 `macs3`。
- `--peak-min-score`：候选 peak 的最低分数，默认 `0.70`。
- `--peak-prominence`：`scipy` 方法使用的峰突出度，默认 `0.05`。
- `--peak-min-distance`：`scipy` 方法中 summit 的最小间距，默认 `30000 bp`。
- `--peak-min-width`：最小 peak 宽度，默认 `5000 bp`。
- `--peak-max-gap`：合并或延伸 peak 时允许跨越的最大 gap，默认 `30000 bp`。
- `--smooth-bins`：peak calling 前对分数做 rolling mean 的 bin 数，默认 `5`。
- `--output-prefix`：输出文件前缀。

## `call-peaks` 输出说明

假设 `--output-prefix k562_gc100_best`：

- `k562_gc100_best.peaks.bed`
  - BED-like peak 文件。
  - 前 6 列为 `chrom start end peak_id peak_score strand`。
  - 额外列包括 `summit_start/summit_end/mean_score/n_bins/method/cutoff/prominence`。
- `k562_gc100_best.peaks_summary.tsv`
  - 每个 peak 的完整统计表。
- `k562_gc100_best.peak_calling_summary.json`
  - peak calling 参数、输入轨道、peak 数量和输出路径。
- 如果 `--method macs3`，会调用外部 `macs3 bdgpeakcall`，输出 `*.macs3_bdgpeakcall.bed`。

## 推荐运行策略

1. 先用 `--sequence-names chr22` 或一个小区间验证环境、模型路径和输出格式。
2. 确认 `window_predictions.tsv` 中 `prob` 分布合理。
3. 对 K562/hg19 全基因组运行 `orion predict`。
4. 用默认严格参数运行 `orion call-peaks`，得到高置信候选 IZ。
5. 若候选过少，再逐步降低 `--peak-min-score`，例如 `0.65`、`0.60`，并在结果中标注阈值。

## 常见问题

### 为什么 peak cutoff 默认是 0.70，而不是校准阈值？

分类阈值用于区分正负样本；peak calling 目标是从全基因组中标注更可信的 IZ 候选区域。全基因组扫描会产生大量窗口，因此这里默认使用更严格的 cutoff 和局部 prominence，减少低置信区域进入候选 peak。

### bedGraph 为什么不是完整窗口？

窗口之间可能重叠。标准 bedGraph 不应有 overlapping intervals，因此本工具默认把每个窗口分数写到窗口中心附近的 center-bin。完整窗口分数仍保留在 `window_predictions.tsv` 中。

### FASTA 需要索引吗？

本工具使用 `pyfaidx` 读取 FASTA。若同目录下没有 `.fai`，`pyfaidx` 会尝试自动建立索引；如果参考基因组目录不可写，请先运行 `samtools faidx genome.fa`，或把 FASTA 放到可写目录。

### mESC 或其他物种能用吗？

可以，只要提供对应参考基因组 FASTA，并确认模型输入序列长度、物种差异和 checkpoint 适用范围。当前默认 checkpoint 是基于 K562/hg19 场景选择的，跨物种使用时应重新校准阈值并谨慎解释。
