# 模型与工具发布教程

本文档从服务器上的 Git 配置开始，说明如何把 `IZ Origin Scanner` 与模型文件发布到 GitHub，并在后续维护中保持可复现。

## 1. 在服务器上配置 Git

检查 Git：

```bash
git --version
```

设置用户名和邮箱：

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
git config --global init.defaultBranch main
```

建议开启常用显示：

```bash
git config --global core.editor "vim"
git config --global pull.rebase false
```

## 2. 配置 GitHub SSH Key

生成 SSH key：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

一路回车即可使用默认路径 `~/.ssh/id_ed25519`。查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

在 GitHub 网页中进入：

`Settings -> SSH and GPG keys -> New SSH key`

粘贴公钥。测试连接：

```bash
ssh -T git@github.com
```

第一次连接会要求确认 fingerprint，输入 `yes`。看到认证成功提示即可。

## 3. 准备发布目录

进入项目根目录：

```bash
cd /path/to/iz_origin_scanner
```

建议最终结构：

```text
iz_origin_scanner/
  src/
    izscan/
    iz_p0/
  configs/
    checkpoints.example.json
  docs/
    MODEL_PUBLISHING_CN.md
  README_CN.md
  README_EN.md
  requirements.txt
  pyproject.toml
  .gitignore
```

不要把本地运行输出、全基因组 FASTA、bedGraph、临时结果目录提交到 GitHub。

## 4. 处理模型权重文件

模型权重通常较大，不建议直接提交到普通 Git 仓库。可选方案：

### 方案 A：GitHub Release 附件

适合中等大小模型文件。流程：

1. 代码仓库只提交工具代码、README、示例 registry。
2. 在 GitHub 创建 release，例如 `v0.1.0`.
3. 上传模型文件包，例如：

```text
izscan_models_v0.1.0/
  gc100_best/
    best.pt
    resolved_config.json
    model_card.md
  gc100_2800/
    step_2800.pt
    resolved_config.json
```

4. 在 README 中提供 release 下载链接和校验值。

### 方案 B：Git LFS

适合需要把模型文件和仓库绑定管理的情况。安装并启用：

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.safetensors"
git add .gitattributes
```

再添加模型文件：

```bash
git add released_models/gc100_best/best.pt
git add released_models/gc100_best/resolved_config.json
```

注意 GitHub LFS 有配额限制，模型较大时要提前确认。

### 方案 C：Hugging Face Hub 或实验室服务器

适合较大模型或需要长期公开分发。GitHub 仓库只保存代码和下载说明，模型文件放在 Hugging Face Hub、Zenodo、机构对象存储或实验室服务器。

## 5. 写模型卡

建议每个 checkpoint 目录放一个 `model_card.md`，至少说明：

- 模型名称和版本。
- 训练数据来源和物种/细胞系。
- 正负样本定义。
- 推荐使用场景，例如 K562/hg19。
- 推荐窗口大小，当前为 `30 kb`。
- 输出分数含义，当前默认 `prob`。
- 分类校准阈值。
- peak calling 推荐阈值，例如严格高置信 `0.70`。
- 已知限制：跨物种、跨细胞系、低质量 FASTA 区域、N-rich 区域等。
- 复现实验的 commit hash。

## 6. 初始化 Git 仓库

```bash
cd /path/to/iz_origin_scanner
git init
git status
git add .
git commit -m "Initial release of IZ whole-genome scanner"
```

如果使用 Git LFS，先确认 `.gitattributes` 已经提交。

## 7. 在 GitHub 创建仓库

在 GitHub 网页创建新仓库，例如：

```text
iz-origin-scanner
```

可以先建 private 仓库，确认无敏感路径和未授权数据后再公开。

把远程地址加入本地仓库：

```bash
git remote add origin git@github.com:<your_org_or_user>/iz-origin-scanner.git
git branch -M main
git push -u origin main
```

## 8. 创建版本标签

建议使用语义化版本：

- `v0.1.0`：首次可用版本。
- `v0.1.1`：bug fix。
- `v0.2.0`：新增功能但保持大体兼容。
- `v1.0.0`：稳定公开版本。

创建 tag：

```bash
git tag -a v0.1.0 -m "Initial IZ scanner release"
git push origin v0.1.0
```

在 GitHub Releases 页面基于 `v0.1.0` 创建 release，并上传模型包或写明模型下载地址。

## 9. 发布前测试清单

在干净环境中测试：

```bash
conda create -n izscan_release_test python=3.11 -y
conda activate izscan_release_test
git clone git@github.com:<your_org_or_user>/iz-origin-scanner.git
cd iz-origin-scanner
pip install -r requirements.txt
pip install -e .
izscan --version
```

用一个小 region 测试模型推理：

```bash
izscan predict \
  --fasta /home/cxsy1/reference/hg19.fa \
  --checkpoint-registry configs/checkpoints.local.json \
  --checkpoint-label gc100_best \
  --regions chr22:0-300000 \
  --output-dir output/release_smoke_test \
  --output-prefix smoke \
  --batch-size 1
```

测试 peak calling：

```bash
izscan call-peaks \
  --score-track output/release_smoke_test/smoke.prob.bedGraph \
  --output-dir output/release_smoke_test/peaks \
  --output-prefix smoke
```

检查：

- 是否能 import 和执行 CLI。
- registry 中路径是否被正确解析。
- `window_predictions.tsv` 是否有 `prob/logit`。
- `bedGraph` 是否为 4 列且无明显坐标异常。
- `peaks_summary.tsv` 是否包含参数和 peak 统计。
- README 命令是否能直接复用。

## 10. 后续维护流程

### 日常开发

```bash
git checkout -b fix/some-issue
# 修改代码和文档
python -m compileall src
git status
git add .
git commit -m "Fix peak caller edge case"
git push -u origin fix/some-issue
```

在 GitHub 上开 Pull Request，确认测试通过后合并。

### 更新模型

1. 为新模型建立独立目录，例如 `released_models/gc100_best_v2/`。
2. 保存 checkpoint、`resolved_config.json`、模型卡和校验值。
3. 更新 `configs/checkpoints.example.json`，加入新 label。
4. 在 README 中说明推荐默认 checkpoint 是否改变。
5. 新建 release，例如 `v0.2.0`。

### 记录变更

建议新增 `CHANGELOG.md`，每次 release 记录：

- Added：新增功能。
- Changed：默认参数或输出格式变化。
- Fixed：bug 修复。
- Model：新增或替换 checkpoint。
- Known issues：已知问题。

### 保持可复现

每次发布至少记录：

- Git commit hash。
- Python 版本。
- `pip freeze` 或 `conda env export`。
- checkpoint 文件校验值：

```bash
sha256sum best.pt resolved_config.json
```

将校验值写入 release notes 或模型卡。

## 11. 不应公开的内容

发布前检查是否包含：

- 个人绝对路径。
- 未授权数据集。
- 大型 FASTA、bedGraph、临时输出。
- token、密码、SSH private key。
- 内部服务器地址或账号。

可用命令快速检查：

```bash
git status
git diff --cached
grep -R "token\\|password\\|/data01/share" -n README_CN.md README_EN.md configs docs src || true
```

如果 README 中需要保留 `/data01/share` 示例路径，应明确它们只是示例，不是必须存在的公开路径。

