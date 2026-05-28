# 选定 checkpoint 导入与独立运行教程

本文档从四个最终选定 checkpoint 的导入开始，说明如何把旧训练结果整理成 `ORION` 可独立使用的模型目录。核心原则是：新工具运行时只依赖本项目代码、导入后的 checkpoint、对应的 `resolved_config.json` 和参考基因组 FASTA；不依赖此前的评估脚本、阈值分析脚本、attention 脚本或旧实验 config。

## 1. 解耦原则

只复制这些内容：

- 每个 checkpoint 的权重文件，例如 `best.pt`、`step_2800.pt`。
- 每个 checkpoint 对应训练输出目录下的 `resolved_config.json`。
- 可选：模型卡、校验值、阈值记录表。

不要复制或依赖这些内容：

- 之前的 `evaluate_checkpoint_triplet.py`、`analyze_iz_score_cluster_bins.py`、attention 分析脚本。
- 旧项目中的 `configs/*.yaml` 训练配置。
- `external_tests/details`、预测结果、attention 输出、临时分析结果。
- 全基因组 FASTA、bedGraph、运行输出目录。

本项目已经包含独立推理 runtime：`src/orion_runtime/`。导入模型后，CLI 通过 `configs/checkpoints.local.json` 找到权重和 `resolved_config.json`。

同时需要注意：ORION checkpoint 不包含完整 NTv3 650M 预训练 backbone。运行端必须提前准备 NTv3 650M 基模，并在本地运行版 config 中把 `model.name` 指向该基模目录。

## 2. 推荐目录布局

假设代码仓库放在：

```bash
PROJECT=/path/to/orion
```

建议把模型文件放在仓库外部，避免误提交大文件：

```bash
MODEL_ROOT=/path/to/orion_checkpoint
```

最终结构：

```text
/path/to/orion/
  configs/
    checkpoints.example.json
    checkpoints.local.json        # 本地 registry，不提交
  src/
  README_CN.md
  README_EN.md

/path/to/orion_checkpoint/
  gc100_best/
    checkpoint.pt
    resolved_config.json
  gc100_2800/
    checkpoint.pt
    resolved_config.json
  random100_3200/
    checkpoint.pt
    resolved_config.json
  random_psm_2200/
    checkpoint.pt
    resolved_config.json
  checksums.sha256
```

## 3. 创建模型目录

```bash
PROJECT=/path/to/orion
MODEL_ROOT=/path/to/orion_checkpoint

mkdir -p "$MODEL_ROOT/gc100_best"
mkdir -p "$MODEL_ROOT/gc100_2800"
mkdir -p "$MODEL_ROOT/random100_3200"
mkdir -p "$MODEL_ROOT/random_psm_2200"
```

## 4. 导入四个选定 checkpoint

先设置每类 checkpoint 对应的训练输出目录，下面的路径只是占位示例：

```bash
SOURCE_GC100=/path/to/gc100_training_output
SOURCE_RANDOM100=/path/to/random100_training_output
SOURCE_RANDOM_PSM=/path/to/random_psm_training_output
```

### 4.1 gc100_best

```bash
cp "$SOURCE_GC100/checkpoints/best.pt" \
  "$MODEL_ROOT/gc100_best/checkpoint.pt"

cp "$SOURCE_GC100/resolved_config.json" \
  "$MODEL_ROOT/gc100_best/resolved_config.json"
```

### 4.2 gc100_2800

```bash
cp "$SOURCE_GC100/checkpoints/step_2800.pt" \
  "$MODEL_ROOT/gc100_2800/checkpoint.pt"

cp "$SOURCE_GC100/resolved_config.json" \
  "$MODEL_ROOT/gc100_2800/resolved_config.json"
```

### 4.3 random100_3200

```bash
cp "$SOURCE_RANDOM100/checkpoints/step_3200.pt" \
  "$MODEL_ROOT/random100_3200/checkpoint.pt"

cp "$SOURCE_RANDOM100/resolved_config.json" \
  "$MODEL_ROOT/random100_3200/resolved_config.json"
```

### 4.4 random_psm_2200

```bash
cp "$SOURCE_RANDOM_PSM/checkpoints/step_2200.pt" \
  "$MODEL_ROOT/random_psm_2200/checkpoint.pt"

cp "$SOURCE_RANDOM_PSM/resolved_config.json" \
  "$MODEL_ROOT/random_psm_2200/resolved_config.json"
```

## 5. 校验导入结果

```bash
find "$MODEL_ROOT" -maxdepth 2 -type f | sort
sha256sum "$MODEL_ROOT"/*/checkpoint.pt "$MODEL_ROOT"/*/resolved_config.json > "$MODEL_ROOT/checksums.sha256"
cat "$MODEL_ROOT/checksums.sha256"
```

如果后续要公开发布模型包，`checksums.sha256` 应当一起发布，方便用户确认文件没有损坏。

## 6. 准备 NTv3 预训练基模和本地 config

发布版 `resolved_config.json` 应保持清洗状态，通常包含：

```json
"name": "/path/to/NTv3_650M_pre"
```

这是占位符，不能直接运行。运行端需要准备 NTv3 650M 预训练基模：

```bash
BASE_MODEL=/path/to/NTv3_650M_pre
ls "$BASE_MODEL"
```

确认目录中有 HuggingFace 模型文件，例如：

```text
config.json
tokenizer_config.json
modeling_*.py
*.safetensors 或 pytorch_model*.bin
```

然后为每个 checkpoint 生成本地运行版 config：

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

本地 `configs/checkpoints.local.json` 应指向 `resolved_config.local.json`。不要把 `resolved_config.local.json` 上传到公开仓库或 release，除非它不包含任何内部路径。

## 7. 创建本地 checkpoint registry

进入项目目录：

```bash
cd "$PROJECT"
cp configs/checkpoints.example.json configs/checkpoints.local.json
```

把 `configs/checkpoints.local.json` 改成下面的形式：

```json
{
  "default_checkpoint": "gc100_best",
  "checkpoints": {
    "gc100_best": {
      "description": "Selected default checkpoint from final K562 811 comparison.",
      "model_backend": "orion",
      "checkpoint": "/path/to/orion_checkpoint/gc100_best/checkpoint.pt",
      "config": "/path/to/orion_checkpoint/gc100_best/resolved_config.local.json",
      "threshold": 0.463
    },
    "gc100_2800": {
      "description": "GC-matched checkpoint at step 2800.",
      "model_backend": "orion",
      "checkpoint": "/path/to/orion_checkpoint/gc100_2800/checkpoint.pt",
      "config": "/path/to/orion_checkpoint/gc100_2800/resolved_config.local.json",
      "threshold": 0.463
    },
    "random100_3200": {
      "description": "Random-negative checkpoint at step 3200.",
      "model_backend": "orion",
      "checkpoint": "/path/to/orion_checkpoint/random100_3200/checkpoint.pt",
      "config": "/path/to/orion_checkpoint/random100_3200/resolved_config.local.json",
      "threshold": 0.463
    },
    "random_psm_2200": {
      "description": "Random/PSM mixed-negative checkpoint at step 2200.",
      "model_backend": "orion",
      "checkpoint": "/path/to/orion_checkpoint/random_psm_2200/checkpoint.pt",
      "config": "/path/to/orion_checkpoint/random_psm_2200/resolved_config.local.json",
      "threshold": 0.463
    }
  }
}
```

`threshold` 是分类校准阈值记录，不等同于全基因组 peak calling 的严格阈值。当前 peak calling 默认使用 `--peak-min-score 0.70`，目的是得到更高置信候选区域。

`configs/checkpoints.local.json` 是本地路径文件，不应提交到 GitHub。提交仓库时只保留 `configs/checkpoints.example.json`。

## 8. 新环境安装

```bash
conda create -n orion python=3.11 -y
conda activate orion
cd "$PROJECT"
pip install -r requirements.txt
pip install -e .
orion --version
```

确认 `configs/checkpoints.local.json` 指向的是 `resolved_config.local.json`，并且其中的 `model.name` 可以被当前环境读取。若模型依赖本地 HuggingFace custom code，先确认对应 NTv3 基模目录存在。

## 9. 小区间 smoke test

先不要直接跑全基因组，先用一个小区间确认模型能加载、FASTA 能索引、输出格式正常：

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

检查输出：

```bash
head output/smoke_gc100_best/smoke_gc100_best.window_predictions.tsv
head output/smoke_gc100_best/smoke_gc100_best.prob.bedGraph
cat output/smoke_gc100_best/smoke_gc100_best.scan_summary.json
```

## 10. 对四个 checkpoint 分别测试

```bash
for ckpt in gc100_best gc100_2800 random100_3200 random_psm_2200; do
  orion predict \
    --fasta /path/to/reference/hg19.fa \
    --checkpoint-registry configs/checkpoints.local.json \
    --checkpoint-label "$ckpt" \
    --regions chr1:900000-1500000 \
    --output-dir "output/smoke_${ckpt}" \
    --output-prefix "smoke_${ckpt}" \
    --window-size 30000 \
    --stride 30000 \
    --score prob \
    --batch-size 1
done
```

## 11. 独立 peak calling 测试

`call-peaks` 不需要 checkpoint，也不加载模型。它只读取 bedGraph：

```bash
orion call-peaks \
  --score-track output/smoke_gc100_best/smoke_gc100_best.prob.bedGraph \
  --output-dir output/smoke_gc100_best/peaks \
  --output-prefix smoke_gc100_best \
  --method scipy \
  --peak-min-score 0.70 \
  --peak-prominence 0.05 \
  --peak-min-distance 30000 \
  --peak-min-width 5000 \
  --peak-max-gap 30000 \
  --smooth-bins 5
```

## 12. K562/hg19 全基因组运行

确认 smoke test 正常后，再运行全基因组：

```bash
orion run \
  --fasta /path/to/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --output-dir /path/to/orion_runs/k562_hg19_gc100_best \
  --output-prefix k562_hg19_gc100_best \
  --window-size 30000 \
  --stride 30000 \
  --score prob \
  --batch-size 4 \
  --method scipy \
  --peak-min-score 0.70 \
  --peak-prominence 0.05
```

如果显存不足，先把 `--batch-size` 降到 `1` 或 `2`。

## 13. 发布时如何保持解耦

推荐发布两类内容：

1. GitHub 仓库：只放工具代码、README、示例 registry、教程。
2. Release 或外部存储中的模型包：放 `orion_checkpoint/` 模型目录、`checksums.sha256`、模型卡。

不要把本地绝对路径写入公开的 `configs/checkpoints.example.json`。公开示例里使用 `/path/to/...` 占位；用户下载模型后自己创建 `configs/checkpoints.local.json`。

如果用 GitHub Release 上传模型包：

```bash
cd /path/to/model_release_parent
tar -czf orion_models_v1.0.0.tar.gz orion_checkpoint
sha256sum orion_models_v1.0.0.tar.gz
```

Release notes 中记录：

- 工具版本 tag。
- 模型包文件名。
- 模型包 SHA256。
- 四个 checkpoint label 和来源。
- 推荐默认 checkpoint：`gc100_best`。
- 推荐窗口：`30000 bp`。
- 默认输出分数：`prob`。
- peak calling 默认严格阈值：`0.70`。
