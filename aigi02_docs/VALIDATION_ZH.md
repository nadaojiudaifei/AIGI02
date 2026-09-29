# 验证报告与已知限制

## 远程正式依赖验证：已完成

GitHub Actions 在 Python 3.12、PyTorch 2.8.0 CPU、CompressAI 1.2.8、Diffusers 0.35.2 的正式依赖环境中执行了完整测试，实际输出为 **57 passed in 8.71s，0 failed，0 skipped**。`pip check`、原版 DiT-IC/ELIC 架构导入、原版文件校验也全部通过。74 个原版文件经 SHA-256 校验，修改数为 0。

已测试源码提交：`b09ef06df38f4cf480d82cb7fe305004900ce184`。
安装与测试 run：<https://github.com/nadaojiudaifei/AIGI02/actions/runs/36521214910>。
后续持续集成 run：<https://github.com/nadaojiudaifei/AIGI02/actions/runs/36521362165>，同样成功。
机器可读记录见 [CI_REPORT.json](CI_REPORT.json)。首个 run 的 `AIGI02-source-and-validation` artifact 包含原始测试日志、JUnit、依赖冻结表、源码 ZIP 和上游文件校验结果。

本次远程测试包括真正的 CompressAI rANS 编解码和真正的 Diffusers SANA 小结构前后向接口测试；后者使用随机初始化的小结构，不是下载预训练 SANA 后完成图像压缩训练。**CPU 正式依赖测试通过，不等于预训练大模型 GPU 验收或论文结果复现。**

## 本地实际执行

环境：CPU-only PyTorch 2.10.0，未创建 Conda 环境，未安装或下载 SANA/ELIC 预训练权重。实际命令及输出保存在本目录的 `LOCAL_TESTS.txt`、`DDP_TEST.txt`、`SMOKE_REPORT.json`。

当前本地测试为 **55 passed, 2 skipped**。通过项目覆盖：
公式与边界、真实 CPU 算术码流、提示完整性、损坏/错误模型拒绝、不同尺寸与重复解码；
教师缓存失效、数据连通分组、ZIP 路径穿越拒绝；
maps/codec/field_pretrain/flow_teacher/distill/gan 六阶段串联；
断点续训、组件冻结、验证集记录、实验计划；
新增 latent prompter 与固定上游 MLP 在相同权重下逐元素一致。

独立机制 smoke：两张 32×32 程序生成测试图、两步训练、实际二进制往返、新 Python 进程单独解码，PNG 字节一致。里面的 PSNR、bpp、耗时只是对该测试网络的测量，不是正式 codec 基准。双进程 CPU Gloo DDP 两步训练也已执行并保存日志。

本地跳过项为真实 CompressAI rANS 接口与真实 Diffusers SANA 小结构接口测试，因为本地没有安装这两个依赖。这份历史本地记录保持不变；两项已在上述远程正式依赖环境独立执行并通过，不能把本地 skipped 改写成 passed。

## 仍需在用户 GPU 服务器完成

1. 按 `gpu_acceptance.py` 验收真实预训练模型前向/反向、rANS 往返和重新加载后解码。
2. 在目标数据集跑足阶段和步数，确认收敛、数值稳定及资源占用。
3. 执行原版复现、多个实际码率点、新方法、消融、跨生成器与 prompt 协议。
4. 用同一指标版本重评全部模型，检查目标码率、训练预算与测试集是否匹配。

**没有已训练新方法权重，没有已验证相对 DiT-IC 的指标提升，没有真实 GPU 速度/显存比较，也没有宣称论文全部数值已经复现。**

## 实现中的明确工程约定

- 有限 log-SNR c/e 是代理，e 保留符号；oracle 的词权重固定均匀，预测器的词权重才可学习。
- 码流最小头部与完整性开销 64B，极低码率必须计算该成本与提示词成本。
- 主熵路径固定 CPU FP32、单线程，仍要求相同 checkpoint、文本权重和运行 profile；不是跨架构整数神经解码规范。
- 文本编码器为冻结 CPU FP32；会增加 CPU 内存和编码/解码成本，不宣称与原版零文本路径同成本。
- 新方法为新增生成路径；texture-only 或关闭部分开关不等于完整官方 CADC/DiT-IC 原版。
- 原始 DiT-IC 文件按 SHA-256 验证不变；兼容的 latent prompter 帮助类另写在 `upstream_compat.py`，已有相同权重输出一致性测试。
- 主环境使用 PyIQA 0.1.15 解决依赖冲突，严格旧版指标需单独环境；每次评测记录实际版本。
- 训练缓存采用确定性中心裁剪，不声称等于原文随机增强；需改增强时同时解决 map 对齐。
- 原生 gradient checkpointing 与空间 norm hook 不兼容，显式拒绝而不是隐式错误运行。
- 当前区域质量实现为 PSNR/空间 LPIPS，全局 DISTS 已提供；没有伪造区域 DISTS。
- SSL/GitHub/HF 访问、私有模型授权、数据许可及下载容量由运行环境决定；下载失败会报错，不会声称已取得资产。

## 运行后更新本报告的方法

在生产环境执行 `python -m pytest aigi02_tests -q -rs` 和 GPU 验收脚本，把真实日志、`pip freeze`、`ACCEPTANCE.json` 另存为新 run 记录。不要覆盖本次 CPU 验证记录后仍保留原始日期/环境描述，也不要将 failed/skipped 改写为 passed。
