# 思路一与思路二：公式、代码对应及完整编码解码链

## 0. 实现边界与符号

这是基于用户方案及固定 DiT-IC 的**可训练研究实现**，不是关于最优生成强度的数学证明，也不是 CADC 官方实现的完全复刻。所有码流依赖必须能由解码器先前已解码的信息确定。原图、教师图和量化尺度图只能在编码侧使用；它们不能成为隐形免费传输信道。

`x`：RGB 图像，数值 `[-1,1]`；`p`：原始提示词或明确标注的 caption；
`z0`：SANA VAE 潜变量；`y`：DiT-IC 主分析潜变量；`z`：超先验；
`S,c,e`：语义重要性、条件困难度、提示解释增益；
`m`：编码端空间量化尺度；`ybar=y/m`：实际进入熵模型的变量；
`mu,sigma`：各组解码时可知的条件高斯均值/标准差；
`qhat`：带均值偏置、**LRP 修正前**的量化值；
`yhat`：LRP 修正后的合成输入；`w_hat`：解码端超先验预测的重要性；
`t(i)`：本实现的生成强度/局部 flow 时间；`beta`：解码端语义偏置控制。

生产空间尺度：VAE 为输入的 1/32，`y` 为 1/64，超潜变量为 1/256。输入补齐到 256 的倍数；原始 H/W 在文件头中，最后裁回。图像如 512×768，则 VAE 为 16×24、y 为 8×12、z 为 2×3。测试后端尺度不同，不能拿测试网络的 tensor 尺寸冒充原模型。

## 1. 提示词教师：先把 RF 输出转换成噪声预测

文件：`teacher.py::oracle_maps`、`math_ops.py::information_scores`。

SANA 使用 flow 形式：
\[
z_t=(1-t)z_0+t\epsilon,\qquad v=\epsilon-z_0,\qquad
\widehat\epsilon=z_t+(1-t)\widehat v.
\]
不能直接把预测速度 `v_hat` 当成扩散噪声 `epsilon_hat`。两次条件/空条件预测使用**同一个** `z_t` 和噪声样本：
\[
d_c(\ell,i)=\mathbb E_\epsilon\|\epsilon_i-\widehat\epsilon(z_t,p)_i\|_2^2,\quad
d_\varnothing(\ell,i)=\mathbb E_\epsilon\|\epsilon_i-\widehat\epsilon(z_t,\varnothing)_i\|_2^2.
\]
代码按通道求和、按 Monte Carlo 次数平均，在有限区间用梯形积分：
\[
c_\lambda(i)=\frac12\int_{\ell_{\min}}^{\ell_{\max}(\lambda)}d_c(\ell,i)\,d\ell,\quad
e_\lambda(i)=\frac12\int_{\ell_{\min}}^{\ell_{\max}(\lambda)}
[d_\varnothing(\ell,i)-d_c(\ell,i)]\,d\ell .
\]
`ell=log(SNR)`，`t=1/(1+exp(ell/2))`。默认 8 节点、2 次噪声，区间下界 -6，上界为裁剪后的 `4+ln(lambda_rd/256)`。这是有限条件去噪困难度代理，不是精确条件熵或完整互信息估计；`e` 保留负值，负值意味着在所采样节点上真实提示词未帮助预测。

教师 `S` 取 4 个均匀分布的 SANA cross-attention 层，经头、节点、噪声平均，对每个词空间归一化，再对有效非特殊词**均匀**聚合。常数图归一化为 0。无人工主体框、face/OCR 先验或编码时手工提示权重。

## 2. 轻量图预测器与语义量化分配

文件：`math_ops.py::MapPredictor,SemanticUGAQ`、`latent.py::encode_features`。

预测器输入 `y,p,log(lambda)`，经过卷积和**两个 cross-attention 块**，输出 `S∈[0,1]`、`c>=0`、有符号 `e`。预测器的词权重来自可学习 MLP，经 softplus、mask 和归一化，不由用户逐图设置；消融 `token_weights=false` 改为均匀有效词。蒸馏损失分别对 S、`log1p(c)`、`sign(e)log1p(abs(e))` 使用 L1。

重要性：
\[
w(i)=0.1+0.9S(i)^2.
\]
`w` 只回答哪里值得保护。条件困难度 `c`、解释增益 `e` 通过可学习修正项进一步决定哪里值得传输：
\[
\log m(i)=\log[1+\mathrm{softplus}(f_{\mathrm{tex}}(y-h_{\mathrm{pre}})_i)]
-\frac12\log w(i)
+\delta_\theta(S_i,\log(1+c_i),\operatorname{sgn}(e_i)\log(1+|e_i|),\log\lambda).
\]
最后截断 `log m∈[0,ln32]`，因此 `m∈[1,32]`。修正网络最后层零初始化；初始化时退化为纹理基线与重要性缩放。高 `w` 降低相对量化尺度，但这一启发式与训练目标共同工作，不能声称是已证明的精确最优率分配。

为避免循环：
1. 编码器由未缩放 `y` 计算临时超先验并量化，得到仅编码侧的 `h_pre`；
2. 由 `h_pre` 计算 `m`；
3. **真正传输的**超先验是 `z=h_a(y/m)`，不是第一步临时超先验。

实际量化变量是 `ybar=y/m`。解码器不接收 m，也不做“反乘 m”；合成变换通过训练从编码后的 ybar 恢复图像。漏掉这点会使端到端网络与文件解码不一致。

## 3. 提示词条件熵模型与四组真实编码

文件：`latent.py::hyper,stage_params,groups`。

提示词先以 UTF-8→zlib 进入码流，解码器先恢复 p，生成冻结的 CPU FP32 文本嵌入。随后才解码超先验和主码流。超合成输出经过新增 cross-attention；四个空间/通道分组的上下文也分别接入 text cross-attention。新增 residual cross-attention 输出投影零初始化，便于从原模型开始。

每组 k：
\[
n_k=\mathrm{round}(\bar y_k-\mu_k),\qquad
\hat q_k=n_k+\mu_k,\qquad
\hat y_k=\hat q_k+\frac12\tanh[\operatorname{LRP}_k(\hat q_k,h_k)].
\]
高斯尺度统一下界 0.11；训练 likelihood 和实际编码采用同一个参数路径。四组互不重叠、共同覆盖所有主符号，后组只使用已恢复的前组及超先验，不读取未来符号。

正式后端用 CompressAI EntropyBottleneck 编码 z、GaussianConditional+rANS 编码各组 y；tiny 后端另有真正的整数算术编码器供 CPU 测试。二者是不同编解码协议，文件头会区分。训练 bpp 来自 likelihood；评测 bpp 来自实际文件长度，不能混用。

## 4. 水位分配辅助目标

文件：`math_ops.py::waterfill_target`。
\[
r_i^*=\left[\frac12\log_2w_i+\frac{c_i}{\ln2}-\tau\right]_+,
\qquad \sum_i r_i^*=\sum_i r_i^{\text{current}}.
\]
逐图二分 64 次求 tau，匹配**总比特预算**而不是均值。目标及预算 stop-gradient。`L_wf=L1(r_current,r*)` 在训练前半段线性衰减到 0。它是早期形状约束，不是实际文件的严格硬预算求解器；任何目标 bpp 都必须通过真实码流检验。

## 5. 从码流导出局部 SNR、重要性和时间场

文件：`math_ops.py::GenerationField`、`model.py::generation_field`。

四组解码后得到完整条件均值/尺度图。对被编码的单位量化变量取 `Delta=1`：
\[
\widehat{\mathrm{SNR}}(i)=\frac1C\sum_c\frac{\mu_{c,i}^2+\sigma_{c,i}^2}{1/12},\qquad
t_{\mathrm{snr}}(i)=\frac{1}{1+\sqrt{\widehat{\mathrm{SNR}}(i)}}.
\]
`w_hat=0.1+0.9 sigmoid(H_w(hyper))`，由解码端已经可知的超先验预测，用编码侧 `w.detach()` 监督。时间场：
\[
t(i)=\sigma\{\operatorname{logit}[t_{\mathrm{snr}}(i)]
-\beta(2\hat w(i)-1)+\rho_\theta(\log(1+\widehat{\mathrm{SNR}}),\hat w,\beta,\log\lambda)\}.
\]
`t` 截断到 `[1e-4,1-1e-4]`；rho 最后层零初始化。高重要性在 beta>0 时降低生成强度，低重要性允许更强生成。空间重采样把 y 网格映射到 VAE/DiT token 网格。

本实现把 t 作为 `g(i)` 的可训练工程实现；没有额外发送逐 token 的最优 g 或 m。SNR 是熵模型定义的代理，不是有原图监督的真实重建信噪比，也不保证精确等于理论最优调度。

## 6. 真正的 per-token 条件注入，而不是错误 API 参数

文件：`backbone.py::SanaBackbone`、`math_ops.py::SpatialAdaLN`。

SANA 公共 `timestep` 是每张图一个标量。代码保留 `mean(t)*1000*timestep_scale` 的合法基础输入，并在每个 transformer block 的 `norm1`、`norm2` 输出处注入：
\[
u_i'=u_i[1+a_\theta(t_i,\beta,\log\lambda)]+b_\theta(t_i,\beta,\log\lambda).
\]
a/b 最后层零初始化。这样位置间的时间差确实进入每层 token 特征，而不是把 `[B,H,W]` 强行塞进只支持 `[B]` 的接口。旧式 gradient checkpointing 重算时无法自动保留 hook 上下文，因此明确拒绝开启；不得默默产生错误梯度。

## 7. 一步解码与多步教师

主码流合成得到 `mean,logvar,res,prompt_latent`，原版 latent prompter 生成条件 token，与原始文本 token 拼接。由码流中的 seed 生成噪声：
\[
z_{t_0}=(1-t_0)\,\mathrm{mean}+t_0\epsilon,\quad
\hat z_0=z_{t_0}-t_0v_\theta(z_{t_0},p,\hat y,t_0)+res,\quad
\hat x=\mathrm{VAEdecode}(\hat z_0/s).
\]
这是新增方法的单步生成器，不把关闭部分开关后的版本冒充原版 DiT-IC。真正原版由独立 baseline 入口调用原 `Codec.compress/decompress` 和调度器。

教师先独立随机丢弃文本条件 p 和潜变量条件 y，训练 flow 预测 `epsilon-z0`。平滑随机时间场预训练后，再按码流时间场训练 flow_teacher。多步教师每步用三种条件：
\[
v=v_{\varnothing,\varnothing}
+\zeta_p(v_{p,\varnothing}-v_{\varnothing,\varnothing})
+\zeta_y(v_{p,y}-v_{p,\varnothing}),
\]
本版本选 `zeta_p=1+0.5 t(1-w_hat)`，
`zeta_y=1+0.5 w_hat*SNR/(1+SNR)`，每 token Euler 步长 `-t0/K`，默认 K=4。教师冻结，学生用潜变量 MSE 蒸馏成一步。此引导系数是公开的工程设置，不是已验证最优超参数。

## 8. 损失与梯度边界

主训练：
\[
L=R_y+R_z+R_p+\lambda D_w+
\lambda\alpha_{\rm perc}L_{\rm LPIPS,w}
+\alpha_{\rm align}L_{\rm align}
+\alpha_wL_{\hat w}
+\alpha_mL_{\rm maps}
+\alpha_{\rm cell}L_{\rm cell}
+\alpha_{\rm wf}(s)L_{\rm wf}.
\]
思路二像素权重用 `w*(1-t)`，计算误差权重时 detach，避免模型通过降低重要性/扩大时间直接逃避误差。LPIPS 使用空间输出按 w 加权。GAN 是可选阶段：冻结 DINOv2-with-registers 特征、训练 patch/text projection 判别头，生成器区域权重来自 t；没有 face/OCR 先验。

量化单元：
\[
L_{\rm cell}=\mathbb E\left[\left|\frac{g_a(E(\hat x))}{\operatorname{sg}(m)}
-\operatorname{sg}(\hat q)\right|-\frac12\right]_+.
\]
必须用 LRP 前 `qhat=n+mu`，不能用 LRP 后 yhat。重编码分析变换参数通过 functional_call detach，但梯度仍流回重建图/解码器；m 和 qhat 均 stop-gradient，编码器不能移动目标格子来作弊。

EntropyBottleneck quantiles 使用独立辅助优化器。阶段与参数范围记录在 checkpoint 和参数名清单中；教师冻结，判别器单独优化，DDP 正确同步。

## 9. 文件格式与完整因果链

AIG2 V2：固定头 50B + 6 段 varint 长度 + 6 个 payload + 8B 完整性后缀。最小非 payload 开销 **64B**，实际会随段长表略增。记录原 H/W、seed、lambda、后端/提示模式、模型与运行环境指纹、文本特征和原始提示指纹。六段依次为 p、z、y0、y1、y2、y3。没有 pickle，没有原图文件名作为解码依赖；校验和用于损坏检测，不是安全认证。

完整编码顺序：
RGB/EXIF处理 → padding → VAE与ELIC分析 → 主分析y → 教师图或蒸馏图 →
临时超先验 → m → y/m → 实际超分析z → 编码p → 编码z → 用解码z重建hyper →
四组按顺序预测参数/量化/编码/局部解码/LRP → 写文件头和各段 → 统计真实字节。

完整解码顺序：
检查边界/版本/完整性/模型指纹 → 从p段或明确外部条件恢复提示 →
冻结文本嵌入与指纹检查 → EB解码z → text-conditioned hyper →
四组上下文预测/rANS解码/均值加回/LRP → 收集mu/sigma与yhat →
合成mean/res/latent prompt → hyper重要性 → SNR → t场 →
根据seed生噪声 → 一步DiT+空间AdaLN → VAE解码 → clamp与裁剪 → PNG。

码率同时报告：文件总 bpp、视觉 payload bpp、提示 payload bpp、剔除提示 payload 但保留头的 bpp。64B 在 512×512 上约为 0.001953 bpp，极低码率目标不能忽略这一固定开销。

## 10. 与长期研究设计的明确差异

oracle 使用固定均匀非特殊词权重，预测器才有可学习词权重；有限积分不是精确互信息；rho/g 的形式和引导强度尚待验证；标准浮点熵模型仅面向固定 CPU FP32 软件/执行配置，不宣称跨硬件无条件比特可移植；训练缓存采用固定中心裁剪；区域提供 PSNR/空间 LPIPS、全局提供 DISTS，没有把裁剪图 DISTS 冒称严格加权区域 DISTS。每项差异应在实验报告中写清楚。
