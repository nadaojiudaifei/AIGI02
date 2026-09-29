# AIGI02 — DiT-IC + 语义码率分配 + 解码器生成场

本工程在固定版 DiT-IC 上**只新增文件**，提供思路一、思路二的研究实现、训练/编码/独立解码、原版对照、消融与评测入口。原版 README 仍为 `ReadMe.md`；本文件说明新增部分。

**先读 [完整中文运行手册](aigi02_docs/QUICKSTART_ZH.md)。**
算法公式与实现约定见 [ALGORITHM_ZH.md](aigi02_docs/ALGORITHM_ZH.md)，实验方案见 [EXPERIMENTS_ZH.md](aigi02_docs/EXPERIMENTS_ZH.md)，已验证范围见 [VALIDATION_ZH.md](aigi02_docs/VALIDATION_ZH.md)。

## 这份代码交付包含什么

| 入口/目录 | 用途 |
|---|---|
| `aigi02/` | 新增模型、数值定义、熵编码、码流、数据、教师、训练、评测、命令行 |
| `aigi02_configs/idea1.yaml` / `idea2.yaml` | 两种方法的生产配置 |
| `aigi02_configs/ablations/` | 18 组配置和 5 类独立组件训练配置 |
| `aigi02_scripts/` | 数据导出、分阶段训练、实验执行、GPU 验收 |
| `aigi02_tests/` | 公式、真实字节流、训练阶段、续训与数据一致性测试 |
| `aigi02_docs/upstream_sha256.json` | 原版每个文件的 SHA-256，用于证明未修改 |
| `requirements-aigi02*.txt` | 主环境、轻量测试环境及可选旧版评测环境依赖 |

```bash
python -m aigi02 --help
python -m aigi02 verify-upstream
python -m pytest aigi02_tests -q -rs
python -m aigi02 smoke --output aigi02_runs/my_smoke
```

`smoke` 使用小型测试网络和 CPU 算术编码器，只验证工程机制；它不是 SANA 模型，也不能用于报告压缩质量、速度或与 DiT-IC 的优劣比较。正式后端使用 SANA、原 DiT-IC 变换网络、ELIC 以及 CompressAI rANS。

## 正式运行入口

配置里的 `SELECT_MERGED_CHECKPOINT.pt` 和 `SELECT_ELIC_CHECKPOINT.pth` 是必须替换的显式占位符，不能直接当作模型文件。大模型、数据集和已训练新方法权重不包含在源码包中。

```bash
# 已准备本地模型、数据和教师图后：
python -m aigi02 doctor --config aigi02_configs/idea2.yaml
python -m aigi02 pipeline --config aigi02_configs/idea2.yaml --output aigi02_runs/idea2
# 上一句只生成计划；加 --execute 才实际训练。
python -m aigi02 pipeline --config aigi02_configs/idea2.yaml \
  --output aigi02_runs/idea2 --execute --nproc 1

python -m aigi02 encode --checkpoint aigi02_runs/idea2/gan/last.pt \
  --image input.png --prompt-file prompt.txt --output image.aig --device cuda
python -m aigi02 decode --checkpoint aigi02_runs/idea2/gan/last.pt \
  --input image.aig --output reconstructed.png --device cuda
```

默认码流包含完整提示词，独立解码不需要原图、重要性图、量化尺度图、教师缓存或图像文件名。若使用 `--prompt-mode free`，解码端必须拥有完全一致的外部提示词，该实验必须单列。

## 验证边界

源码实现与科学实验结论是两件事。本交付已经进行 CPU 小模型机制测试、阶段串联、独立进程解码和双进程 DDP 测试；当前环境没有实际执行预训练 SANA 的 GPU 训练，也没有生成 Kodak/CLIC/DIV2K 上的新方法率失真曲线。测试数量、跳过原因、真实日志均见验证文档，不以占位数据填充结果表。

两种新方法的若干工程选择已经固定并公开，例如有限 log-SNR 积分、oracle 均匀有效词权重、蒸馏器可学习词权重、CPU FP32 熵路径与零初始化空间调制。这些是本实现的具体版本，不代表数学最优性证明或 CADC 官方代码的逐字复刻。

## 上游与许可

上游：`Eric-qi/DiT-IC`，固定提交 `cd43f5d9761fb34f5224622145629d3ff2b89ca1`。
原始文件、作者声明、引用及第三方许可全部保留；新增文件不替原始代码重新指定许可。请阅读 [NOTICE_AIGI02.md](NOTICE_AIGI02.md)。
