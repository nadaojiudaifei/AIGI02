# 实验、对照与消融协议

## 1. 先区分四种结论

“代码存在”“CPU 机制通过”“预训练 GPU 验收通过”“完整数据集率失真结果复现”是四个不同状态。本工程不会以 tiny 网络结果代替 DiT-IC/SANA 结果，不从论文表格复制数字填进新方法输出，也不把一次前向当作完整训练。

原 DiT-IC 原始源码和已发布 `results/` 文件均保留。它们是上游作者的历史结果，不是本项目重新运行的结果。新的所有评测输出写入 `aigi02_results/`；不存在的实验不会生成假分数。

## 2. 原版 DiT-IC 基线

同一清单、同一输入图像，不以“关掉新功能”作为原版替代。真正基线调用原 `models.DiT_IC.Codec`、原熵编解码、原 scheduler 和原合成路径：

```bash
python -m aigi02 baseline --config configs/inference_merge.yaml \
  --checkpoint aigi02_assets/ditic/ACTUAL_Q3_MERGED.pt \
  --sana aigi02_assets/sana --manifest aigi02_data/kodak/manifest.jsonl \
  --output aigi02_results/baseline_q3_kodak --device cuda
python -m aigi02 evaluate --manifest aigi02_data/kodak/manifest.jsonl \
  --recon aigi02_results/baseline_q3_kodak/rec \
  --rates aigi02_results/baseline_q3_kodak/rates.jsonl \
  --output aigi02_results/baseline_q3_kodak/metrics --device cuda --distribution
```

为使原版码流也能独立解码，外层增加尺寸、seed、模型指纹和完整性检查；同时保存两种码率：`bpp_original_ditic_container` 是原版 `write_body` 容器的真实大小，`bpp_total` 包括新外层。比较时不允许新方法算全部文件，而原版只算理想 likelihood，或相反。

```bash
python -m aigi02 baseline-decode --config configs/inference_merge.yaml \
  --checkpoint aigi02_assets/ditic/ACTUAL_Q3_MERGED.pt --sana aigi02_assets/sana \
  --input aigi02_results/baseline_q3_kodak/bin/SAMPLE_ID.bin \
  --output baseline_decoded.png --device cuda
```

基线保留原版 reflect padding；新方法采用 replicate padding。这一协议差异已记录，输入恰为 256 倍数时不产生边界 padding 差异。过小图像不能满足原版 reflect 规则时，明确报错，不静默改写原算法。

### 从训练开始复现原版

```bash
python -m aigi02 prepare-upstream \
  --train-manifest aigi02_data/splits/train.jsonl \
  --validation-manifest aigi02_data/splits/validation.jsonl \
  --sana /srv/models/sana --elic /srv/models/elic.pth \
  --merged /srv/models/ditic/actual_merged.pt \
  --output aigi02_runs/upstream_configs --batch 8 --steps 100001

python -m aigi02 upstream-train \
  --config aigi02_runs/upstream_configs/train_256_nogan.yaml --nproc 2 --execute
```

生成器创建清单对应的 symlink 图像视图和**额外配置副本**。覆盖原版 256/512、GAN/non-GAN；提供 merged 权重时也生成原版 merge finetune 配置。第二阶段需在新增配置里填入前阶段实际 checkpoint。不要凭示例路径假定权重已存在。

低/高 LoRA rank 对比可在新增原版配置中分别设置 VAE/DiT 为 32/64 与 64/128；使用匹配 rank 的原版权重和合并方式。四个质量档使用官方实际文件，不猜测 q 编号对应的 λ。训练集规模、原版随机增强、步数、batch、初始化与论文保持一致后，才可写“训练协议复现”；仅用新混合数据微调属于迁移实验。

## 3. 主实验矩阵

建议保持三个轴分离：

| 轴 | 内容 |
|---|---|
| 方法 | 原版 DiT-IC、思路一、思路二；需要时加 texture-only，不称其为官方 CADC |
| 数据 | Kodak、DIV2K validation、CLIC 2020 test；独立 AIGI 测试集；跨生成器 holdout |
| 码率/提示协议 | 至少四个真实工作点；计入提示词与明确免费共享提示词单列；true prompt/caption/empty 单列 |

`lambda_rd` 是训练权重，不是目标 bpp。先用验证集测真实码率，再调整工作点训练；以实际相邻率失真点插值或匹配码率比较，不按新旧 λ 的数值硬匹配。不要在测试集上调 λ、beta、引导系数再报告“独立测试”。

配置 `aigi02_configs/experiments.example.yaml` 中的数据清单和 checkpoint 全部替换为真实路径。默认包含 4 个原版档、4 个思路一档、4 个思路二档与四个测试集；只生成计划不会训练或计算分数。

```bash
python -m aigi02 matrix --spec aigi02_configs/experiments.example.yaml \
  --output aigi02_runs/plans/main.json
python -m aigi02 run-matrix --plan aigi02_runs/plans/main.json
python -m aigi02 run-matrix --plan aigi02_runs/plans/main.json --execute
# 仅跳过有成功状态记录且产物仍存在的任务：
python -m aigi02 run-matrix --plan aigi02_runs/plans/main.json --execute --resume
python -m aigi02 aggregate --glob 'aigi02_results/**/summary.json' \
  --output aigi02_results/all_measured_results.csv
```

每个任务保存 argv、日志、退出状态。失败后停止，不将缺失结果补成 0。完整矩阵耗费的训练/推理/数据下载由用户显式执行。

## 4. 指标定义与版本

全局：PSNR、LPIPS、DISTS、MS-SSIM、CLIPIQA、MUSIQ、NIQE；FID/256 使用原版非偏移与半 patch 偏移采样函数；KID/256 仅在超过 50 张图时启用，记录真实 patch 数和 subset_size。所有图像边长不足 256 时拒绝 distribution 指标，不将上采样后的分数伪装成原分辨率指标。

保存的 PNG 重构与清单 `id` 一一配对，不依赖排序后文件名的偶然对应。`bpp_total_pixel_weighted` 与逐图均值同时报告，大小混合的图像集上这两者不等价。解码时间包括真实熵解码和生成器；encode_ms 的测量含编码路径中的设备搬运，不可与仅 GPU kernel 时间直接比较。

可选 CLIP-I、CLIP-T 和 DINO：

```bash
python -m aigi02 evaluate --manifest aigi02_data/splits/test.jsonl \
  --recon aigi02_results/my_run/rec --rates aigi02_results/my_run/rates.jsonl \
  --output aigi02_results/my_run/semantic_metrics --device cuda \
  --semantic-config aigi02_configs/semantic_assets.example.yaml
```

这三项输出为归一化 embedding 的 cosine，分别命名 `clip_image_cosine`、`clip_text_cosine`、`dino_cls_cosine`；不把它们称为经过特定缩放的官方 CLIPScore。主环境 PyIQA 0.1.15 与原版 0.1.14.1 不同；严格旧版比较请用运行手册中的独立评测环境统一重评所有重构图。输出记录实际指标软件版本。

### 区域评测

使用全分辨率 oracle 清单（不是 256 裁剪缓存）作为 evaluate 的 manifest。按固定 oracle S 的上四分位阈值划分 high/low 区域，报告面积比例、区域 PSNR 与空间 LPIPS。所有方法复用同一 oracle，不能各自拿自己的预测图选择“表现好”的区域。

当前提供全局 DISTS，不将“涂黑低重要性区域再算 DISTS”当作严格区域 DISTS。常数 S 可能使 high 占整图，此时 low 指标为空而非伪造数值。

## 5. 18 组消融和 5 类组件独立训练

```bash
python -m aigi02 ablation-configs --config aigi02_configs/idea2.yaml \
  --output aigi02_configs/my_ablations
```

| 配置 | 实际变化 |
|---|---|
| full | 完整开关，包含可选 GAN 阶段 |
| texture_only | 仅纹理尺度，关语义分配与生成场；不是官方 CADC 复现 |
| no_token_weights | 蒸馏图预测器词权重改均匀 |
| no_importance | 不用 S 保护项，w=1 |
| no_conditional_score | c/e 置零 |
| no_texture | 移除纹理尺度基项 |
| no_correction | 关闭可学习尺度修正 |
| no_prompt_entropy | 熵模型不接入文本条件 |
| no_prompt_decoder | 生成器文本条件置空，仍有 latent prompt |
| no_waterfill | 关闭早期水位分配辅助损失 |
| no_cell | 关闭量化单元一致性损失 |
| fixed_variance_schedule | 使用固定的原版风格方差映射，不用 SNR 场和空间调制 |
| snr_only | 去掉语义 beta 偏置与 rho 修正 |
| no_field_residual | 保留 SNR/语义偏置，rho=0 |
| no_spatial_adaln | 不把局部时间注入各层 token norm |
| no_distillation | 流水线跳过一步蒸馏阶段 |
| no_gan | 流水线跳过 GAN |
| oracle_maps | 编码端使用真实教师缓存，单列 oracle 成本 |

独立组件范围为 maps、allocation、entropy、field、decoder，只限制当前原本允许训练的参数，不会把冻结基础模型全部解冻。参数名保存供审核。

注意：改变开关后应重新训练匹配的 checkpoint，不能仅在完整模型推理时关一层并宣称完成训练消融。`pipeline` 会按 flags 跳过相应阶段；直接调用单阶段 `train` 则按其 stage 执行，禁用 distillation 却选择 distill 会报错。不同阶段数会改变总训练预算，严谨比较还应配置等预算对照。

## 6. Prompt、beta、域与 oracle 实验

```bash
python aigi02_scripts/make_prompt_variants.py \
  --manifest aigi02_data/splits/test.jsonl --output aigi02_data/prompt_variants
```

将 correct/empty/shuffled/caption 各作为独立 manifest 编码。错误提示实验必须由编码器与解码器一致使用错误提示；不能在正确文本条件下编码，却在解码时替换 p，熵符号分布会不匹配，格式会拒绝。

beta 实验只需同一已编码文件：

```bash
python -m aigi02 decode --checkpoint aigi02_runs/idea2/gan/last.pt \
  --input image.aig --output beta0.png --beta 0
python -m aigi02 decode --checkpoint aigi02_runs/idea2/gan/last.pt \
  --input image.aig --output beta2.png --beta 2
```

这类实验码率不变，变化的是生成强度与重建结果。报告种子、beta 与训练范围。高 beta 不保证某一指标单调提升。

oracle 编码：

```bash
python -m aigi02 roundtrip --checkpoint aigi02_runs/oracle/gan/last.pt \
  --manifest aigi02_data/splits/test.oracle.jsonl --oracle-maps \
  --output aigi02_results/oracle --device cuda
```

上述 checkpoint 应由 oracle 消融配置训练。真正离线教师耗时需单独统计；已有缓存的在线耗时不是 oracle 完整编码成本。

## 7. 可复现实验归档

至少保存：上游 commit、所有资产 revision/哈希、数据清单与分组 split、提示来源、训练配置和种子、实际 checkpoint、pip freeze、GPU/CPU 型号、每图真实码率/延时/指标、评测版本、失败任务日志。不要提交 token、私人目录内容、大模型或数据集到 Git；大文件采用独立、授权的资产存储。
