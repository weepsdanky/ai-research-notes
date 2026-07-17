# Module 3：Diffusion Transformer 数学原理与代码推导

本文沿用 DDPM 讲义的记号，从“扩散模型需要一个什么函数”开始，推到 DiT 的 patch、Transformer、adaLN-Zero、训练目标与 classifier-free guidance。主实验为 CIFAR-10；原始论文设置是 ImageNet 图像经 VAE 压缩后的 latent。

配套材料：

- 学习计划：[6_dit-case-study.md](./6_dit-case-study.md)
- 一手资料：[dit-primary-source-notes.md](./dit-primary-source-notes.md)
- 教学代码：`code/dit/dit.py`、`code/dit/diffusion.py`
- 原始论文：[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- 官方实现：[facebookresearch/DiT](https://github.com/facebookresearch/DiT)

## 1. 最重要的分层：diffusion 与 denoiser

DDPM 定义 forward noising process：

$$
q(x_t\mid x_{t-1})
=
\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I),
\qquad \alpha_t=1-\beta_t.
$$

令 $\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$，可以一步采样任意噪声时刻：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

训练一个噪声预测器：

$$
\epsilon_\theta(x_t,t,c),
$$

其中 $c$ 是类别、文本或其他条件。简化目标为：

$$
L_{simple}
=
\mathbb E_{x_0,t,\epsilon,c}
\left[
\lVert \epsilon-\epsilon_\theta(x_t,t,c)\rVert_2^2
\right].
$$

从 UNet 换成 DiT 时，上面四行数学不变。改变的只是实现 $\epsilon_\theta$ 的神经网络：

$$
\text{UNet}(x_t,t,c)
\quad\longrightarrow\quad
\text{DiT}(x_t,t,c).
$$

这是理解 DiT 的第一原则：**DiT 是 denoiser backbone，不等于一套新的 diffusion 概率过程。** 同一个 noise schedule、loss 和 sampler 可以连接不同 backbone。

## 2. Pixel diffusion 与 latent diffusion

原始 DiT 不直接对 $256\times256\times3$ 像素做 Transformer。它继承 Latent Diffusion 的做法，用预训练 VAE encoder $E$ 压缩图像：

$$
z_0=E(x_0),
\qquad
z_0\in\mathbb R^{4\times32\times32}
\quad (256\times256\text{ 输入的典型形状}).
$$

扩散发生在 latent 上：

$$
z_t
=
\sqrt{\bar\alpha_t}z_0
+
\sqrt{1-\bar\alpha_t}\epsilon.
$$

DiT 预测 latent noise，最后由 decoder $D$ 还原图像：

$$
\hat x_0=D(\hat z_0).
$$

VAE 压缩的作用不只是省显存。若空间边长缩小 8 倍，二维位置数量缩小 $64$ 倍，而 self-attention 的成对交互数量近似再平方缩小。代价是生成器只能在 VAE 保留的信息空间里工作，VAE reconstruction 上限会影响最终细节。

本仓库教学版直接在 CIFAR-10 的 $32\times32\times3$ pixel space 工作，省去预训练 VAE 依赖。它用于验证架构，不用于复现 ImageNet 指标。

## 3. 从图像到 patch token

设输入为：

$$
x\in\mathbb R^{B\times C\times H\times W}.
$$

用不重叠的 $p\times p$ patch 切分。网格尺寸与 token 数是：

$$
H_p=H/p,
\qquad W_p=W/p,
\qquad N=H_pW_p.
$$

每个 patch 展平后有 $p^2C$ 个数，再线性投影到 hidden size $d$：

$$
u_i=W_p\operatorname{vec}(x^{(i)})+b_p,
\qquad u_i\in\mathbb R^d.
$$

代码通常用 kernel size 与 stride 都等于 $p$ 的 `Conv2d` 一次完成切块和线性投影：

```python
nn.Conv2d(C, d, kernel_size=p, stride=p)
```

卷积输出形状为 $[B,d,H_p,W_p]$，flatten 与 transpose 后为：

$$
U\in\mathbb R^{B\times N\times d}.
$$

### 3.1 两个具体例子

CIFAR-10 教学版：

$$
H=W=32,\ p=4
\Longrightarrow H_p=W_p=8,\ N=64.
$$

ImageNet-256 的 latent DiT-XL/2：VAE 后 $H=W=32$，$p=2$：

$$
H_p=W_p=16,\ N=256.
$$

把 patch 从 4 减到 2，token 数增加 4 倍；attention score matrix 从 $N^2$ 增加到 $16$ 倍。patch size 是 DiT 最强也最昂贵的扩展旋钮之一。

## 4. 二维位置编码

self-attention 本身对 token 排列没有二维空间概念。DiT 给每个 patch token 加固定二维 sine/cosine position embedding：

$$
h_i^{(0)}=u_i+p_i.
$$

二维编码可把行坐标与列坐标分别编码，再拼接：

$$
p_{(r,c)}=
[\operatorname{PE}(r);\operatorname{PE}(c)].
$$

单个一维坐标 $r$ 的频率特征为：

$$
\operatorname{PE}(r)_{2k}=\sin(r\omega_k),
\qquad
\operatorname{PE}(r)_{2k+1}=\cos(r\omega_k),
$$

$$
\omega_k=10000^{-k/m}.
$$

固定位置编码没有可学习参数，且不同分辨率可重新生成。它不是唯一选择；后续视频模型常用 3D/MM-RoPE 处理时间、宽、高和多模态位置。

## 5. 时间与类别条件

DiT 的每次 denoising forward 都必须知道当前噪声时刻 $t$。先得到 sinusoidal embedding：

$$
e_t=[\cos(t\omega_0),\ldots,\cos(t\omega_{m-1}),
\sin(t\omega_0),\ldots,\sin(t\omega_{m-1})],
$$

再通过两层 MLP：

$$
c_t=W_2\operatorname{SiLU}(W_1e_t+b_1)+b_2.
$$

类别 $y$ 查 embedding table 得到 $c_y$。原始 class-conditional DiT 将二者相加：

$$
c=c_t+c_y,
\qquad c\in\mathbb R^{B\times d}.
$$

这里的 $c$ 不是额外图像 token，而是生成每层 LayerNorm modulation 参数的全局条件向量。对长文本，现代 MMDiT 常保留文本 token 并通过 cross/joint attention 交互，不能只靠一个 pooled vector 替代全部语义。

## 6. 从 LayerNorm 到 adaLN-Zero

### 6.1 LayerNorm

对一个 token $h\in\mathbb R^d$：

$$
\mu(h)=\frac1d\sum_{j=1}^d h_j,
\qquad
\sigma^2(h)=\frac1d\sum_{j=1}^d(h_j-\mu)^2,
$$

$$
\operatorname{LN}(h)
=
\frac{h-\mu(h)}{\sqrt{\sigma^2(h)+\varepsilon}}.
$$

普通 LayerNorm 再使用固定可学习 $\gamma,\beta$。adaptive LayerNorm 改由条件 $c$ 生成 sample-dependent shift 和 scale：

$$
\operatorname{adaLN}(h,c)
=
\operatorname{LN}(h)\odot(1+s(c))+b(c).
$$

写成 $1+s$ 而不是只乘 $s$，让 $s=0,b=0$ 时退化为普通无 affine 的 LayerNorm。

### 6.2 一个 DiT block 的六路参数

条件投影产生六个 $d$ 维向量：

$$
(b_{msa},s_{msa},g_{msa},b_{mlp},s_{mlp},g_{mlp})
=W_c\operatorname{SiLU}(c).
$$

attention residual branch：

$$
\tilde h
=
h+g_{msa}\odot
\operatorname{MSA}
(\operatorname{adaLN}(h;b_{msa},s_{msa})).
$$

MLP residual branch：

$$
h'
=
\tilde h+g_{mlp}\odot
\operatorname{MLP}
(\operatorname{adaLN}(\tilde h;b_{mlp},s_{mlp})).
$$

$g$ 是 residual gate，$b,s$ 控制进入分支前的特征。

### 6.3 Zero 的严格含义

将输出六路参数的线性层权重与 bias 初始化为零：

$$
b=s=g=0.
$$

于是：

$$
\tilde h=h+0\cdot\operatorname{MSA}(\operatorname{LN}(h))=h,
$$

$$
h'=h+0\cdot\operatorname{MLP}(\operatorname{LN}(h))=h.
$$

因此每个 block 在初始化时是恒等映射。最终 projection 也零初始化，所以整个 DiT 初始输出为零。这个性质可以直接写成测试，不必凭训练曲线猜初始化是否正确。

注意：零输出不意味着没有梯度。第一步的 loss 会给最终线性层非零梯度；随着输出层离开零点，梯度再传入前面的 blocks。

## 7. Multi-Head Self-Attention 的计算

对 token matrix $H\in\mathbb R^{N\times d}$：

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

每个 head 维度 $d_h=d/n_h$：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

多头输出拼接后再投影：

$$
\operatorname{MSA}(H)
=
[head_1;\ldots;head_{n_h}]W_O.
$$

attention score 为 $N\times N$，因此其主要 token 交互成本近似：

$$
O(N^2d).
$$

MLP 的主要成本近似 $O(Nd^2r)$，$r$ 是 expansion ratio。小 $N$、大 $d$ 时 MLP/linear 可能主导；高分辨率或长视频导致 $N$ 变大时，attention 更快成为瓶颈。因此只说“Transformer 是平方复杂度”仍不够，必须同时记录 $N,d,depth$。

## 8. Final layer 与 unpatchify

最后一层再次用条件调制：

$$
O_i=W_o\operatorname{adaLN}(h_i,c)+b_o.
$$

每个 token 输出 $p^2C_{out}$ 个数：

$$
O\in\mathbb R^{B\times N\times(p^2C_{out})}.
$$

将 token index 拆成二维 patch 网格，将 token feature 拆成 patch 内的行、列、通道，再 permute/reshape：

$$
[B,H_p,W_p,p,p,C_{out}]
\longrightarrow
[B,C_{out},H_p p,W_p p].
$$

若只预测噪声，$C_{out}=C_{in}$。若同时学习 reverse variance，官方 DiT 默认 `learn_sigma=True`：

$$
C_{out}=2C_{in},
$$

前一半通道是 mean/noise parameterization，后一半用于 variance parameterization。教学代码支持输出通道形状，但最小训练 loss 只覆盖 fixed-variance noise prediction；不能把“shape 支持”误称为已经实现完整 mixed MSE/VLB。

## 9. 训练目标：模型如何连到 DDPM

一次训练 step：

1. 取 clean image/latent $x_0$ 与 condition $y$；
2. 均匀采样 $t\in\{0,\ldots,T-1\}$；
3. 采样 $\epsilon\sim\mathcal N(0,I)$；
4. 构造 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$；
5. 计算 $\hat\epsilon=\operatorname{DiT}(x_t,t,y)$；
6. 最小化 $\operatorname{MSE}(\hat\epsilon,\epsilon)$。

从 $\hat\epsilon$ 可恢复 $x_0$ 估计：

$$
\hat x_0
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}
{\sqrt{\bar\alpha_t}}.
$$

后验：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_tI),
$$

$$
\tilde\beta_t
=
\beta_t\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t},
$$

$$
\tilde\mu_t
=
\frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}x_0
+
\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}x_t.
$$

采样时把 $x_0$ 换为 $\hat x_0$。$t=0$ 时不再添加随机噪声，这是必须测试的边界条件。

## 10. Classifier-Free Guidance

训练时以概率 $p_{drop}$ 把类别替换为 learned null class $\varnothing$。同一模型学到：

$$
\epsilon_\theta(x_t,t,y)
\quad\text{与}\quad
\epsilon_\theta(x_t,t,\varnothing).
$$

推理时使用同一个 $x_t,t$ 计算 conditional 与 unconditional prediction：

$$
\epsilon_{cfg}
=
\epsilon_{uncond}
+s(\epsilon_{cond}-\epsilon_{uncond}).
$$

边界：

$$
s=0\Rightarrow\epsilon_{cfg}=\epsilon_{uncond},
$$

$$
s=1\Rightarrow\epsilon_{cfg}=\epsilon_{cond}.
$$

$s>1$ 外推条件方向，通常提高条件一致性但可能降低多样性、产生过饱和或伪影。CFG 不是训练 loss 的替代品，而是同时训练 cond/uncond 后的采样组合。

官方 DiT 为精确复现曾只 guidance 前三个通道；标准数学式通常对全部 noise channels 应用。复现实验要说明采用哪一种。

## 11. Scaling：参数、token 与 Gflops

原始 DiT 通过两个维度增加计算：

- S/B/L/XL：增加 depth、hidden size、heads；
- `/8`、`/4`、`/2`：减小 patch size，增加 token 数。

论文在固定训练框架下观察到更高 forward-pass Gflops 与更好 FID 的经验关系，并报告 DiT-XL/2 在 ImageNet 256 的 FID 2.27。正确表述是“论文设定内的经验 scaling 结果”，不是任何数据和训练预算下参数越大都必然更好。

公平消融至少记录：

$$
(N,d,L,h,\#params,\text{Gflops},\text{steps},\text{tokens seen}).
$$

只固定 epoch 而改变 batch 或数据增强，不是相同训练预算。只比较 parameter count 而忽略 patch 导致的 token 数变化，也不能解释计算来源。

## 12. 从图像 DiT 到视频 DiT

视频 latent 形状可写为：

$$
z\in\mathbb R^{B\times C\times F\times H\times W}.
$$

若 tube patch 为 $p_t\times p_h\times p_w$，token 数：

$$
N
=
\frac{F}{p_t}\frac{H}{p_h}\frac{W}{p_w}.
$$

full space-time attention 成本近似 $O(N^2d)$。帧数增加会迅速放大成本。

Latte 展示 factorized 思路。令每帧空间 token 数 $S=H_pW_p$，帧 token 数 $F_p$：

- full attention：$O((F_pS)^2d)$；
- spatial attention：对每帧做，$O(F_pS^2d)$；
- temporal attention：对每个空间位置做，$O(SF_p^2d)$；
- factorized 合计：$O((F_pS^2+SF_p^2)d)$。

当 $F_p,S>1$ 时，通常远小于 full attention。代价是空间与时间信息需要跨 block 逐步交换，归纳偏置更强。

后续系统的重点不只 attention：

- CogVideoX：3D VAE、expert adaptive LayerNorm、文本/视频融合；
- Open-Sora：STDiT、任意 shape、数据 curriculum；
- LTX-Video：极高 VAE 压缩、full attention、rectified flow；
- Seedance 1.0：MMDiT、3D/MM-RoPE、多任务条件与级联 refiner；
- HunyuanVideo：dual/single stream、MLLM text encoder、3D VAE、flow、并行与 FP8。

## 13. 从 epsilon prediction 到 rectified flow

现代视频模型常使用 flow/velocity。以线性 path 的一种约定：

$$
z_t=(1-t)z_0+t\epsilon,
\qquad t\in[0,1].
$$

对 $t$ 求导：

$$
\frac{dz_t}{dt}=\epsilon-z_0.
$$

训练 velocity field：

$$
L_{flow}
=
\mathbb E\left[
\lVert v_\theta(z_t,t,c)-(\epsilon-z_0)\rVert_2^2
\right].
$$

注意不同论文可能把时间方向或 target 符号反过来，必须用 path 端点自行求导。推理变为 ODE 数值积分，例如 Euler：

$$
z_{t+\Delta t}
=
z_t+\Delta t\,v_\theta(z_t,t,c).
$$

此时不再使用 DDPM 的 $q(x_{t-1}\mid x_t,x_0)$ posterior。DiT backbone 可以保持相似，但 probability path、target 和 sampler 都换了。

## 14. 代码与公式逐项映射

| 数学对象 | 教学代码 | 核心测试 |
|---|---|---|
| $q(x_t\mid x_0)$ | `DiffusionSchedule.q_sample` | 指定 noise 后等于闭式 |
| $\hat x_0(x_t,\hat\epsilon)$ | `predict_x0_from_eps` | forward 后精确反演 |
| $q(x_{t-1}\mid x_t,x_0)$ | `q_posterior` | variance 边界、$t=0$ |
| patch projection | `PatchEmbed` | token shape |
| $p_i$ | `get_2d_sincos_pos_embed` | shape、确定性、位置不同 |
| $c_t,c_y$ | `TimestepEmbedder`、`LabelEmbedder` | odd dim、null token |
| adaLN-Zero | `DiTBlock` | zero-init block 为 identity |
| unpatchify | `DiT.unpatchify` | 输出空间 shape |
| $L_{simple}$ | `diffusion_loss` | finite loss 与非零梯度 |
| CFG | `forward_with_cfg` | $s=0,1$ 与线性组合 |
| reverse loop | `p_sample`、`sample_loop` | finite、shape、mode restore |

## 15. 建议手推题

1. 从两步 forward process 推出 $q(x_t\mid x_0)$ 的闭式。
2. 对 CIFAR-10 的 $p=8,4,2$ 分别计算 token 数和 attention matrix 元素数。
3. 已知 $d=384$、heads=6，写出单 head 的 $Q,K,V$ shape。
4. 令六路 adaLN 参数为零，逐行证明 block 是 identity。
5. 从 forward 闭式解出 $x_0$，再代入 posterior mean。
6. 证明 CFG 在 $s=0,1$ 的边界，并解释 $s>1$ 是外推而非概率插值。
7. 对 $F_p=16,S=256$ 比较 full 与 factorized attention 的 score pair 数量。
8. 对 $z_t=(1-t)z_0+t\epsilon$ 求 velocity；若 path 改成 $tz_0+(1-t)\epsilon$，target 如何变化？

## 16. 证据边界与延伸阅读

本讲义的 DiT 架构、ImageNet scaling 和官方代码行为来自 [DiT 论文](https://arxiv.org/abs/2212.09748) 与 [官方仓库](https://github.com/facebookresearch/DiT)。latent diffusion 背景来自 [LDM](https://arxiv.org/abs/2112.10752)。视频扩展依次参考 [Latte](https://arxiv.org/html/2401.03048)、[CogVideoX](https://arxiv.org/abs/2408.06072)、[Open-Sora](https://arxiv.org/html/2412.20404)、[LTX-Video](https://arxiv.org/html/2501.00103)、[Seedream 2.0](https://arxiv.org/html/2503.07703)、[Seedance 1.0](https://arxiv.org/html/2506.09113v1)、[OmniWeaving](https://arxiv.org/html/2603.24458v2) 与 [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)。

[Seedance 2.0](https://arxiv.org/abs/2604.14148) 当前公开的是 model card。可以引用其明示的输入输出与能力范围，但不能据此断言未披露的数学目标或 block 细节，更不能把 Seedance 1.0 的实现自动当作 2.0 的事实。

