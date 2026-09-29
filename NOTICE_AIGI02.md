# 来源、许可与资产说明

原始 DiT-IC 源码来自 https://github.com/Eric-qi/DiT-IC ，固定提交
`cd43f5d9761fb34f5224622145629d3ff2b89ca1`。完整上游文件保留，哈希见
`aigi02_docs/upstream_sha256.json`。引用应遵循原 `ReadMe.md`。本工程不对原版代码重新指定许可证，也不替第三方作者授予商用或再分发权利。

`aigi02/upstream_compat.py` 是对该固定版本 MLP latent prompter 及 VAE LoRA 目标筛选规则的 checkpoint 兼容实现，保留此来源说明。其目的仅是避免新后端导入旧版 GAN/指标的副作用；原文件没有修改，测试比较了同权重输出。

研究参考和官方资产：
- DiT-IC 论文/项目：https://arxiv.org/abs/2603.13162 ，https://njuvision.github.io/DiT-IC/
- CADC 研究参考：https://arxiv.org/abs/2602.21591
- DiT-IC 权重：https://huggingface.co/JunqiShi/DiT-IC
- SANA：https://github.com/NVlabs/Sana ，https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers
- CompressAI：https://github.com/InterDigitalInc/CompressAI/tree/v1.2.8
- Diffusers SANA 参考版本：https://github.com/huggingface/diffusers/tree/v0.35.2
- PyIQA 主环境版本：https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.15
- 原版 PyIQA 指标环境：https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.14.1

数据来源：
- Kodak 原始图像：https://r0k.us/graphics/kodak/
- Kodak HF 镜像：https://huggingface.co/datasets/danjacobellis/kodak
- LSDIR 原始数据：https://ofsoundof.github.io/lsdir-data/
- LSDIR raw 镜像：https://huggingface.co/datasets/danjacobellis/LSDIR_raw
- MLIC-Train-100K：https://huggingface.co/datasets/Whiteboat/MLIC-Train-100K
- DIV2K：https://data.vision.ee.ethz.ch/cvl/DIV2K/
- CLIC：https://archive.compression.cc/challenge/
- DiffusionDB：https://huggingface.co/datasets/poloclub/diffusiondb
- JourneyDB：https://huggingface.co/datasets/JourneyDB/JourneyDB
- AGIQA-3K 镜像：https://huggingface.co/datasets/strawhat/agiqa-3k

大模型权重、数据集及运行生成的训练 checkpoint **不包含在源码包或 Git 提交中**。使用前阅读各自 model card、dataset card、访问条款和许可证；受限仓库需自行获得合法访问授权。在线数据可能含不适宜内容，研究数据过滤与组织政策需要在导入前明确。

新增研究代码未经生产安全认证；码流完整性检查不是身份认证或对抗性防护。不要用未经审查的模型 checkpoint 或下载脚本执行任意代码。加载使用 `weights_only=True`；直接 URL 权重下载要求 SHA-256。
