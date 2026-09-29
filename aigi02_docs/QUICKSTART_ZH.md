# 从零运行：Linux / Conda / GPU 完整手册

以下命令供你的远程服务器执行。交付过程中没有在当前环境创建 Conda 环境、下载大模型或启动 GPU 训练。先完成小规模验收，再启动长训练；脚本遇到缺文件、配置不匹配、坏码流或非有限损失会直接报错，不会偷偷退回测试模型。

## 1. 获取代码并确认原版完整

```bash
git clone https://github.com/nadaojiudaifei/AIGI02.git
cd AIGI02
# 或解压完整源码 ZIP，进入包含 aigi02/ 和 models/ 的 AIGI02 目录。
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

禁止在原版 `models/`、`ELIC/`、`configs/` 中修改文件来完成新方法；新增内容使用 `aigi02*` 命名空间。原版跟踪的 `.pyc` 也保留，所以推荐全程禁止新写入字节码。

## 2. 建立主环境

```bash
conda create -n aigi02 python=3.12 -y
conda activate aigi02
python -m pip install --upgrade pip wheel setuptools==69.5.1
# CUDA 12.8 对应组合；先确认服务器驱动支持，不能只看本机 CUDA toolkit。
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-aigi02.txt
python -m pip check
python -m aigi02 verify-upstream
python -m pytest aigi02_tests -q -rs
```

不要接着安装原版 `requirements.txt`：其中 NumPy 2.4.3 与 CompressAI 1.2.8 的 NumPy `<2.0` 要求冲突；原版 PyIQA 0.1.14.1 又固定 Transformers 4.37.2，不能直接与新 SANA 文本模型环境合并。主环境采用 NumPy 1.26.4、PyIQA 0.1.15、Transformers 4.56.2，并限制 OpenCV 版本。完整 GPU/预训练依赖组合仍需在你的服务器验收；`pip check` 通过也不等于模型结果已复现。

PyIQA、LPIPS、TorchMetrics 等首次构造指标时可能下载各自权重。完全离线运行时还要预先填充其缓存，单独准备 SANA 并不能覆盖评测资产。主后端不要求 xformers。

仅运行轻量机制测试时可另用 CPU PyTorch，再安装 `requirements-aigi02-test.txt`。不要拿 tiny 后端计算论文曲线。

## 3. 准备模型：在线与离线两条路径

### 3.1 在线下载并固定版本

```bash
# 需要授权的 HF 仓库应先在本机终端登录。不要把 token 写进 YAML 或提交 Git。
hf auth login

python -m aigi02 hub-download \
  --repo Efficient-Large-Model/Sana_600M_1024px_diffusers \
  --output aigi02_assets/sana

# 先列出官方 DiT-IC 文件，明确选择对应质量档和 rank 的 merged 权重、ELIC 权重。
python -c "from huggingface_hub import HfApi; print('\n'.join(HfApi().list_repo_files('JunqiShi/DiT-IC')))"

# 下述两个 filename 必须换成上一条实际列出的路径：
python -m aigi02 hub-download --repo JunqiShi/DiT-IC \
  --filename ACTUAL_MERGED_FILE.pt --output aigi02_assets/ditic
python -m aigi02 hub-download --repo JunqiShi/DiT-IC \
  --filename ACTUAL_ELIC_FILE.pth --output aigi02_assets/ditic
```

下载器先解析 revision 为实际提交 SHA，并写入 `aigi02_asset_lock.json`；复现实验时保存该记录。SANA 需要 `vae/`、`transformer/`、`scheduler/`、`tokenizer/` 和 `text_encoder/`。原版 DiT-IC 推理不读文本不代表新方法也不需要文本编码器。

使用 DINO 判别器或自然图像 caption 时：

```bash
python -m aigi02 hub-download --repo facebook/dinov2-with-registers-small \
  --output aigi02_assets/dino
python -m aigi02 hub-download --repo Salesforce/blip-image-captioning-base \
  --output aigi02_assets/blip
# 可选 CLIP 图文/图像相似度：
python -m aigi02 hub-download --repo openai/clip-vit-base-patch32 \
  --output aigi02_assets/clip
```

新后端 warm start 读取**完整合并版** DiT-IC 权重；若只有原版训练产生的 raw LoRA checkpoint，先用原版 `merge.py` 按 `ReadMe.md` 合并。不可把嵌套的 `model/ema` checkpoint 当作平坦 merged 权重，加载器会明确拒绝。

### 3.2 已下载到本地

编辑 `aigi02_configs/idea1.yaml`、`idea2.yaml`：

```yaml
assets:
  sana: /srv/models/sana
  text: /srv/models/sana
  ditic: /srv/models/ditic/actual_merged.pt
  elic: null
  discriminator: /srv/models/dino
  local_files_only: true
  revision: null
```

对于 `oracle_sana.yaml`，保持 `ditic: null`，把 `elic` 填成真实 ELIC 路径，使 oracle 使用原始预训练 SANA 而非已适配的压缩解码器。从 SANA 开始训练也需要 ELIC 权重。

自己的 AIGI02 checkpoint 包含视觉网络及新增模块的全部状态，重新加载时不再要求原始 DiT-IC/ELIC 初始化权重；但仍需要匹配的 SANA 结构配置与文本编码器。换服务器时：

```bash
python -m aigi02 decode --checkpoint /srv/checkpoints/last.pt \
  --asset-config aigi02_configs/local_assets.example.yaml \
  --input image.aig --output restored.png --device cuda
```

先编辑 example YAML 中的路径。保持文本模型、软件版本和熵计算环境一致；格式会检测模型/文本指纹不匹配。

对于 GitHub 模型代码或直接 HTTPS 权重：

```bash
python -m aigi02 git-download --url https://github.com/OWNER/REPO.git \
  --revision FULL_COMMIT_SHA --output aigi02_assets/external_source
python -m aigi02 url-download --url https://HOST/path/model.pt \
  --sha256 EXPECTED_64_HEX_SHA256 --output aigi02_assets/model.pt
```

代码仓库克隆不会自动执行其中的脚本；Git LFS 权重需要按其官方说明取得真实文件。直接 URL 下载必须提供校验和。

## 4. 数据统一为 JSONL 清单

一行一个样本：

```json
{"id":"sample001","image":"images/001.png","prompt":"a red bicycle beside a wall","prompt_kind":"true_prompt","generator":"stable-diffusion"}
```

`prompt_kind` 只能为 `true_prompt`、`caption`、`empty`。自然图像的 caption 不是生成原始提示词，必须分开报告。`id` 只允许字母、数字、下划线、点、横线，不允许路径；不能重复。`image`、`maps` 支持相对清单的路径。

### 4.1 小规模自然图像 / HF 流式导出

```bash
python -m aigi02 prepare-hf --repo danjacobellis/kodak --split validation \
  --limit 24 --output aigi02_data/kodak --prompt-kind empty

# 先小规模验证，再扩大。该镜像与论文原始训练采样协议要分别记录。
python -m aigi02 prepare-hf --repo danjacobellis/LSDIR_raw --split train \
  --limit 1000 --output aigi02_data/lsdir_small --prompt-kind empty
# 大规模导出：显式设置更大上限；--limit 0 才表示全量。
python -m aigi02 prepare-hf --repo danjacobellis/LSDIR_raw --split train \
  --limit 50000 --output aigi02_data/lsdir_large --prompt-kind empty
```

对于任意支持的 HF parquet/image 数据集，传入实际 `--image-column` 和 `--prompt-column`。包含列表型 caption 的数据需先明确选取/合并策略，不能默默转字符串。原版项目自己的 `datasets/` 会遮蔽 HF 同名包，因此导出器在隔离子进程中运行。

`datasets.save_to_disk` 的本地数据：

```bash
python -m aigi02 prepare-hf --repo /srv/datasets/saved_dataset \
  --local-disk --split train --limit 1000 --image-column image \
  --prompt-column prompt --prompt-kind true_prompt \
  --output aigi02_data/local_hf_export
```

### 4.2 小规模 / 大规模 AI 生成图像：DiffusionDB

DiffusionDB 使用显式原始 ZIP 分片，不依赖旧式 HF Python 数据集加载脚本：

```bash
python -m aigi02 prepare-diffusiondb --parts 1 \
  --output aigi02_data/diffusiondb --manifest aigi02_data/diffusiondb/small.jsonl
python -m aigi02 prepare-diffusiondb --parts 1,2,3,4,5,6,7,8,9,10 \
  --output aigi02_data/diffusiondb --manifest aigi02_data/diffusiondb/large.jsonl
```

分片含原始生成提示词，导入时保留 `true_prompt` 和生成 seed。已有解压分片可放入相应 `part-000001/` 等目录，或保留 HF 缓存并用 `--offline`。不要默认下载数百万张；先检查容量、许可、内容过滤和目标实验规模。

JourneyDB 等需接受访问条款的数据不绕过授权。取得官方数据及标注后可本地导入；例如 JSONL 包含 `img_path` 和 `prompt`：

```bash
python -m aigi02 prepare-folder --root /srv/datasets/journeydb/images \
  --metadata /srv/datasets/journeydb/annotations.jsonl \
  --image-column img_path --prompt-column prompt --prompt-kind true_prompt \
  --output aigi02_data/journeydb.jsonl
```

### 4.3 原论文基准 / 已有文件夹

Kodak、DIV2K validation、CLIC 2020 test 均可使用原始官方图像目录导入：

```bash
python -m aigi02 prepare-folder --root /srv/datasets/DIV2K_valid_HR \
  --prompt-kind empty --output aigi02_data/div2k/manifest.jsonl
python -m aigi02 prepare-folder --root /srv/datasets/CLIC2020_test \
  --prompt-kind empty --output aigi02_data/clic2020/manifest.jsonl
```

官方基准尺寸/内容不能用另一年度 CLIC、下采样 DIV2K 或训练子集冒充。MLIC-Train-100K 的分卷压缩包需要完整下载全部分卷并从第一卷解压；本项目不把“下载一个分卷”标为准备完成。统一文件夹导入不限制数据集来源。

### 4.4 caption、合并和分组切分

```bash
python -m aigi02 caption --manifest aigi02_data/lsdir_small/manifest.jsonl \
  --model aigi02_assets/blip --output aigi02_data/lsdir_small/captions.jsonl --device cuda
python -m aigi02 merge-manifests \
  --manifest aigi02_data/lsdir_small/captions.jsonl \
  --manifest aigi02_data/diffusiondb/large.jsonl \
  --output aigi02_data/mixed.jsonl
python -m aigi02 split --manifest aigi02_data/mixed.jsonl \
  --output aigi02_data/splits --validation 0.05 --test 0.05 --seed 2026
```

切分按原文件 SHA-256 与规范化提示词建立连通分组；同图不同提示词、同提示词不同图及其传递关联不跨 split。也支持 `--holdout-generator NAME`。这不是感知近重复检测，也不是自动保证跨生成器独立；缺失 generator 标注须先补充。极小数据集可能没有 validation/test 文件，要扩大数据或显式建立非空清单。

## 5. 计算训练教师图与 oracle 评测图

先修改 `oracle_sana.yaml` 中的模型路径。默认有限积分 8 个 log-SNR 节点、每节点 2 次噪声采样，使用同一噪声比较真实/空提示词。

```bash
python -m aigi02 cache-maps --config aigi02_configs/oracle_sana.yaml \
  --manifest aigi02_data/splits/train.jsonl --output aigi02_data/maps_train \
  --output-manifest aigi02_data/splits/train.maps.jsonl
```

训练采用固定 resize+中心裁剪以与缓存精确对齐，默认 256×256；这与原版随机增强协议不同。修改提示词、图像、crop 或 λ 后必须重新缓存，读取器会检查一致性。每个码率工作点分别生成缓存。

区域评测与 oracle 编码消融需要**全分辨率图**：

```bash
python -m aigi02 cache-maps --config aigi02_configs/oracle_sana.yaml \
  --manifest aigi02_data/splits/test.jsonl --output aigi02_data/maps_test_full \
  --output-manifest aigi02_data/splits/test.oracle.jsonl --full-resolution
```

oracle 编码只在编码端使用该缓存；独立解码仍不访问它。预缓存 oracle 的耗时不能从编码器成本中“消失”，本实现单独标识该时间未计入在线 encode_ms。

## 6. 先验收，再完整训练

```bash
python -m aigi02 doctor --config aigi02_configs/idea2.yaml
python aigi02_scripts/gpu_acceptance.py --config aigi02_configs/idea2.yaml \
  --image /srv/datasets/example.png --prompt "a landscape with trees" \
  --output aigi02_runs/gpu_acceptance
```

验收会实际执行生产模型前向/反向、真实 rANS 编码、重新加载 checkpoint 后解码，并记录 GPU 型号和结果。不执行这个脚本，就不能宣称 GPU 验收通过。它仍不是完整数据集训练。

完整分阶段训练：

```bash
bash aigi02_scripts/run_pipeline.sh aigi02_configs/idea1.yaml aigi02_runs/idea1 1
bash aigi02_scripts/run_pipeline.sh aigi02_configs/idea2.yaml aigi02_runs/idea2 1
# 两卡：
bash aigi02_scripts/run_pipeline.sh aigi02_configs/idea2.yaml aigi02_runs/idea2_ddp 2
# 中断后：
bash aigi02_scripts/run_pipeline.sh aigi02_configs/idea2.yaml aigi02_runs/idea2_ddp 2 --resume
```

默认阶段：思路一 `maps → codec → gan`；思路二 `maps → codec → field_pretrain → flow_teacher → distill → gan`。关闭 realism 则跳过 GAN，关闭 distillation 则跳过蒸馏。新方法 `lambda_rd=256` 表示 `R + λD` 中失真权重，**不是原 DiT-IC 配置的 `lrd=0.5`，不能按数值直接对应**。

每阶段保存完整 checkpoint、解析后的配置、训练参数名和 JSONL 日志。默认步数是可执行起点，不是已经验证的最佳训练配方。输入 crop、batch、累积步数应按显存调整；文本编码器在 CPU，需关注系统内存。每个 checkpoint 包含视觉基础模型，磁盘占用会明显大于仅保存 LoRA。

单阶段和组件独立训练：

```bash
python -m aigi02 train --config aigi02_configs/idea2.yaml \
  --set train.stage=codec --set train.steps=1000 \
  --set train.output=aigi02_runs/codec_trial

python -m aigi02 train --config aigi02_configs/idea2.yaml \
  --set train.stage=codec --set train.trainable=allocation \
  --set train.init=aigi02_runs/idea2/codec/last.pt \
  --set train.output=aigi02_runs/allocation_only
```

`train.init` 是跨阶段初始化；`train.resume` 会恢复优化器、数据偏移、随机数和分布式状态。精确续训要求原清单与 world_size 不变，不允许悄悄用变更数据继续旧优化器。

## 7. 编码、解码、评测

```bash
python -m aigi02 roundtrip --checkpoint aigi02_runs/idea2/gan/last.pt \
  --manifest aigi02_data/kodak/manifest.jsonl \
  --output aigi02_results/idea2_kodak --device cuda

python -m aigi02 evaluate --manifest aigi02_data/kodak/manifest.jsonl \
  --recon aigi02_results/idea2_kodak/rec \
  --rates aigi02_results/idea2_kodak/rates.jsonl \
  --output aigi02_results/idea2_kodak/metrics --device cuda --distribution
```

独立解码示例在 README。`--beta` 只改变生成场强度，不需要重新编码；训练范围默认 `[0,2]`，范围外属于外推实验。tiny 结果不能替代正式结果。更多批量对照、消融和严格码率协议见实验手册。

## 8. 可选：严格旧版指标环境

为对齐原论文指定 PyIQA 版本，可在**单独环境**中重评已经保存的重构图：

```bash
conda create -n aigi02_eval_legacy python=3.12 -y
conda activate aigi02_eval_legacy
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-aigi02-legacy-eval.txt
python -m pip check
# 在同一源码目录运行 evaluate；不要在这个环境运行 SANA 训练。
```

所有对比方法使用同一指标环境和同一套权重缓存。评测输出会记录实际软件版本。不能一行采用新版本指标、另一行直接抄论文旧版本数值然后宣称同协议比较。

## 9. 常见错误

`SELECT_*` 未替换：列出官方 HF 文件并填真实路径。`Missing text component`：补齐 tokenizer/text_encoder。`Bitstream ... mismatch`：使用编码时同一 checkpoint、文本权重及运行配置。`stale map cache`：重新生成该图/提示词/crop/λ 的缓存。`Output already contains ...`：选择新输出目录或明确 resume，避免混结果。`No module named datasets`/加载本地 datasets：用提供的 `prepare-hf` 子进程入口。`Non-finite loss`：查看实际阶段日志，不应把 NaN 替换成 0 继续出实验表。CUDA OOM：从 batch=1、crop=256、累积梯度开始排查，空间 hook 路径不支持原生 gradient checkpointing。
