# DiT 原论文与官方实现逐项审计

> 审计对象：Peebles & Xie, *Scalable Diffusion Models with Transformers*（arXiv:2212.09748v2）及 `facebookresearch/DiT` 官方 PyTorch 实现。
> 当前项目：`code/image_dit/`。
> 审计日期：2026-08-13。官方源码固定到 commit [`ed81ce2`](https://github.com/facebookresearch/DiT/tree/ed81ce2229091fd4ecc9a223645f95cf379d582b)，避免把日后变化混入结论。

## 结论先行

当前完整版已经实现了大部分 **DiT backbone 核心设计**：patchify/unpatchify、官方 x→y 顺序的固定二维 sin-cos 位置编码、256 维 timestep 频率编码、类别 null token 与 label dropout、`t + y` 条件、pre-norm adaLN-Zero、零门控、GELU-tanh、官方 S/B/L/XL 尺寸、learned-range variance、AdamW、EMA 和 CFG。

此外，`paper_compatible` profile 已经把此前缺少的 **diffusion recipe 对齐项**落地：

1. learned variance 使用官方默认的 `MSE + raw VB`，并保留 `rescaled VB` 作为教学/历史选项；
2. learned-variance loss 与 sampling 都可关闭 predicted-`x0` clipping，profile 默认 no-clip；
3. 从原 1000-step schedule 选择 250 个 timestep，重建 beta 并保持所选点的 cumulative alpha，再做 ancestral DDPM；
4. CFG 可只作用于前三个 epsilon 通道，其余 epsilon/variance 保持各自分支输出；
5. 位置编码默认使用官方 x→y 半维顺序，并作为 persistent buffer 写入 state dict；
6. checkpoint 写入 `profile`、完整 `recipe`、`prediction_type` 与 `pipeline`，推理默认从 checkpoint 恢复策略；
7. profile 默认关闭 gradient clipping 与 AMP，epsilon prediction 为强制条件。

这些行为由 [`profiles.py`](../../code/image_dit/profiles.py)、[`diffusion.py`](../../code/image_dit/diffusion.py)、[`dit.py`](../../code/image_dit/dit.py)、[`train_image_dit.py`](../../code/image_dit/train_image_dit.py) 和 [`inference.py`](../../code/image_dit/inference.py) 共同实现，并有相应测试保护。

但 `paper_compatible` 的名字必须按字面理解为“**当前已实现部分与论文 diffusion recipe 兼容**”，而不是“端到端复现论文”。仍缺四个决定性环节：

1. **VAE latent 管线**：目前仍在 MNIST/FashionMNIST/CIFAR-10 像素空间训练，没有 Stable Diffusion f8 VAE、posterior sampling、`0.18215` scaling 与 decoder。
2. **ImageNet 训练管线与预算**：没有 ImageNet-256/512、ADM precise center crop、1000 类、论文 global batch 和 400K～7M step 训练实验。
3. **评估管线**：没有 50K 样本导出以及 ADM-compatible FID/sFID/IS/Precision/Recall。
4. **官方 checkpoint 直接兼容**：虽然位置编码已数值对齐，当前 state-dict key 结构、模块命名和若干数值细节仍不能直接加载官方权重，也没有逐元素官方 checkpoint parity fixture。

因此，当前代码应定义为“**像素空间 DiT 教学实现 + 可验证的论文 diffusion recipe profile**”，不能称为“ImageNet latent DiT 原论文复现”。

## 1. 如何理解“原论文 trick”

需要区分三层事实：

- **论文实验**：论文声称用于其结果的设置；这是复现实验的最高依据。
- **官方 PyTorch 仓库**：作者后来公开的简化实现。论文模型最初在 JAX/TPU 上训练，仓库权重从 JAX 转换而来；官方 README 明确说明 PyTorch 与 JAX 可能因精度产生细微差异（[README L135-L140](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/README.md#L135-L140)）。
- **官方脚本默认值**：用于方便运行，不一定等于某张论文表的评估设置。例如 `sample.py` 默认 CFG=4，而论文最优 256² FID 使用 CFG=1.5。

论文比较过 in-context、cross-attention、adaLN、adaLN-Zero 四种条件块，但随后正式模型采用 adaLN-Zero（[论文 §3.2](https://arxiv.org/html/2212.09748#S3.SS2.SSS0.Px2)、[§5](https://arxiv.org/html/2212.09748#S5.SS0.SSS0.Px1)）。前三种是**消融基线**，不是最终 DiT 必须同时开启的 trick。

状态标记：

- ✅：当前实现具有论文/官方行为，或在当前像素教学域内语义等价。
- ⚠️：能力已实现，但适用域、默认值或严格数值兼容仍有限制。
- ❌：复现论文所必需但当前缺失。
- ℹ️：当前工程增强，不属于论文 trick。

## 2. 架构与初始化：逐项核对

| 项目 | 论文实验 / 官方实现 | 当前 `code/image_dit` | 状态 |
|---|---|---|---|
| 输入域 | 256² RGB 经 f8 VAE 得到 `32×32×4` latent；512² 得到 `64×64×4`（[论文 §4 Diffusion](https://arxiv.org/html/2212.09748#S4.SS0.SSS0.Px2)） | 直接输入 28²/32²、1/3 通道像素；`DiTConfig` 可配置 4 通道，但训练器未接 VAE | ❌ 管线缺失 |
| Patchify | 无重叠 `p×p` patch 线性投影；`p∈{2,4,8}`，token 数 `(I/p)^2`（[论文 Patchify](https://arxiv.org/html/2212.09748#S3.SS2.SSS0.Px1)；[官方 `models.py` L169-L179](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L169-L179)） | stride=kernel 的 `Conv2d`，再 flatten/transpose | ✅ |
| 位置编码 | 固定、不可训练的二维 sin-cos ViT 位置编码，无 class token；官方先 x/列、后 y/行（[官方 L172-L198, L274-L321](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L172-L198)） | 默认官方 x→y 顺序，并以 persistent buffer 写入 state dict；缺失 position 的旧本地 checkpoint 会注入历史 y→x encoding（[`dit.py` L199-L269](../../code/image_dit/dit.py#L199-L269)） | ✅ 新模型；⚠️ legacy 路径 |
| Timestep embedding | 256 维 sinusoidal frequency embedding，`max_period=10000`，再 `Linear→SiLU→Linear`（[论文 Appendix A](https://arxiv.org/html/2212.09748#A1.SS0.SSS0.Px1)；[官方 L27-L64](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27-L64)） | 相同 | ✅ |
| 类别条件与 null token | class embedding；训练时随机替换为 learned null embedding。官方默认 dropout `0.1`（[官方 L67-L94, L158-L171](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L67-L94)） | CLI 默认 0.1，额外 null embedding | ✅ |
| 条件合并 | timestep 与 class embedding 相加为 `c=t+y`（[官方 L233-L248](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L248)） | 相同 | ✅ |
| Transformer 范式 | ViT 式 pre-norm；LayerNorm 无 affine，`eps=1e-6`（[官方 L101-L122](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122)） | 相同 | ✅ |
| Self-attention | 标准 multi-head self-attention，`qkv_bias=True`，无 attention dropout（[官方 L105-L112](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L105-L112)） | `nn.MultiheadAttention` 语义等价；state-dict key 与官方 timm Attention 不同 | ✅ 语义；❌ 直接权重兼容 |
| MLP | ratio=4，GELU tanh approximation，无 dropout（[官方 L110-L116](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L110-L116)） | 相同 | ✅ |
| adaLN-Zero | `SiLU→Linear(6d)` 产生两组 shift/scale/gate；零 gate 令 block 初始为 identity（[论文 adaLN-Zero](https://arxiv.org/html/2212.09748#S3.SS2.SSS0.Px2)；[官方 L101-L122](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122)） | 相同 | ✅ |
| Final layer | adaptive final LN；每 token 输出 `p²·2C`，即 epsilon 与 diagonal covariance，再 unpatchify（[论文 Transformer decoder](https://arxiv.org/html/2212.09748#S3.SS2.SSS0.Px4)） | `--learn-sigma` 时相同；训练 CLI 默认开启，但裸 `DiTConfig` API 默认关闭 | ✅ CLI；⚠️ 裸 API 默认 |
| 模型尺寸 | S=`12/384/6`，B=`12/768/12`，L=`24/1024/16`，XL=`28/1152/16`（层/宽/heads）（[论文 Table 1](https://arxiv.org/html/2212.09748#S3.T1)） | 四档 preset 数值相同；patch 显式传入 | ✅ |
| 初始化 | Linear Xavier、patch Conv 按 Linear 初始化；label/time MLP `N(0,0.02)`；adaLN 与 final output 零初始化（[官方 L182-L216](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L182-L216)） | 相同 | ✅ |

### 位置编码已修正，但不等于官方 checkpoint parity

当前默认 embedding 将 x/列编码放在前半维、y/行编码放在后半维，与官方 `grid[0]`/`grid[1]` 顺序一致，并存入 state dict。旧本地 checkpoint 在当时没有保存 position tensor；兼容加载时必须补回其训练时使用的 y→x encoding，否则会改变旧模型行为。

这只解决了位置编码一项。官方 `timm` Attention/MLP 与当前 PyTorch 模块的 state-dict key 和模块结构不同，因此不能据此宣称可以直接载入官方 `.pt`。

## 3. Diffusion、目标函数与方差学习

| 项目 | 论文实验 / 官方实现 | 当前实现 | 状态 |
|---|---|---|---|
| Forward process | Gaussian DDPM，按闭式从 `x0` 采样 `xt`（[论文 §3.1](https://arxiv.org/html/2212.09748#S3.SS1.SSS0.Px1)） | `q_sample` 相同 | ✅ |
| Timestep sampling | 每个样本在 1000 个 timestep 上均匀采样（[官方 `train.py` L201-L207](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L201-L207)） | `torch.randint` 均匀采样 | ✅ |
| Noise schedule | 1000 步 linear beta，`1e-4→2e-2`（[论文 §4](https://arxiv.org/html/2212.09748#S4.SS0.SSS0.Px2)） | 默认相同 | ✅ |
| Mean parameterization | 原论文预测 epsilon（[论文 §3.1](https://arxiv.org/html/2212.09748#S3.SS1.SSS0.Px1)） | 默认 epsilon；另有 v-parameterization 教学消融，`paper_compatible` 明确拒绝 v | ✅ 论文默认；ℹ️ v 是额外实验 |
| Learned covariance | 在 posterior clipped log variance 与 `log(beta_t)` 间做 `LEARNED_RANGE` 插值（[官方 `gaussian_diffusion.py` L285-L293](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L285-L293)） | 相同，不额外 clamp variance value | ✅ |
| Hybrid loss | epsilon simple MSE；variance VLB/KL；VB 计算时 detach mean prediction（[官方 L746-L783](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L746-L783)） | MSE、KL、`t=0` decoder NLL 与 stop-gradient 均实现（[`diffusion.py` L225-L287](../../code/image_dit/diffusion.py#L225-L287)） | ✅ |
| Loss 中的 `x0` clipping | 官方 `_vb_terms_bpd(..., clip_denoised=False)`（[官方 L756-L765](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L756-L765)） | 参数化开关；`paper_compatible` 将 `clip_denoised` 设为 false，`pixel_teaching` 设为 true | ✅ profile 对齐 |
| MSE/VB 权重 | 官方默认 `LossType.MSE` 即 `MSE + raw VB`；`RESCALED_MSE` 才乘 `T/1000`（[factory L10-L26](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/__init__.py#L10-L26)） | `vb_mode=raw|rescaled` 显式区分；paper profile 选 raw | ✅ |
| 系数精度 | 官方以 NumPy float64 预计算 schedule，抽取时转 float32（[官方 L153-L190, L861-L873](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L153-L190)） | PyTorch float32 预计算 | ⚠️ 严格数值 parity 未完成 |

`pixel_teaching` 与 `paper_compatible` 的分离很重要：像素实验做 clipping 通常更稳、DDIM 更快；latent 论文采样则明确 no-clip。把二者放进同一个隐式默认值，会让“生成方便”和“论文对齐”互相污染。

## 4. Sampling 与 CFG

### 4.1 250-step respaced ancestral DDPM 已实现

论文以 250 DDPM steps 报告 FID-50K（[论文 Evaluation](https://arxiv.org/html/2212.09748#S4.SS0.SSS0.Px3)）。官方 `create_diffusion("250")` 从原 1000 步选取 timestep，并重建 beta 以保持所选点的 cumulative alpha；它不是重新创建一个独立的 250-step linear schedule（[官方 `respace.py` L12-L87](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/respace.py#L12-L87)）。官方采样脚本使用 ancestral `p_sample_loop` 并传 `clip_denoised=False`（[`sample.py` L60-L65](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/sample.py#L60-L65)）。

当前 [`space_timesteps` / `respaced_schedule`](../../code/image_dit/diffusion.py#L150-L182) 实现同一单区间 fractional-stride 选择与 cumulative-alpha-preserving beta 重建；`sample_loop(..., sampling_steps=250)` 在压缩 schedule 上做 ancestral DDPM，并把局部 timestep 映射回原模型的 0～999 时间域。`paper_compatible` 将其命名为 `paper_ddpm`，默认 250 steps 且 no-clip。DDIM 仍保留为教学/速度对照，不冒充论文指标 sampler。

### 4.2 first-3 CFG 已实现

官方为了精确复现，在 4-channel latent 中只 guidance **前三个 epsilon 通道**；第四个 epsilon 通道和 variance 不做线性 guidance（[论文 Appendix A](https://arxiv.org/html/2212.09748#A1.SS0.SSS0.Px2)；[官方 `models.py` L250-L266](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L250-L266)）。论文也说明，对全部四通道 guidance 并重新调 scale 可以得到近似甚至略好的 FID，所以 first-3 是**复现兼容行为**，不是普适定律。

当前 [`forward_with_cfg`](../../code/image_dit/dit.py#L336-L372) 提供 `cfg_channels="first_3"|"all"`：

- `paper_compatible` 选择 `first_3`；其余 epsilon 通道和 learned variance 保持 conditional/unconditional 分支各自的值；
- `pixel_teaching` 选择 `all`；对 1/3-channel 像素实验更直观；
- CFG 基本公式仍为 `eps_uncond + s * (eps_cond - eps_uncond)`。

不要混淆 CFG scale：论文 256² 最佳 FID 2.27 使用 1.5，官方 `sample.py` 展示默认 4.0，官方 `sample_ddp.py` 评估默认 1.5；大量 scaling 实验不用 guidance。当前 UI 的教学默认值不是 paper default。

## 5. 数据与 latent VAE 管线

论文的端到端路径是：ImageNet RGB `[-1,1]` → frozen Stable Diffusion f8 VAE posterior sample → latent `×0.18215` → DiT diffusion → latent `÷0.18215` → VAE decoder（[论文 §4 Diffusion](https://arxiv.org/html/2212.09748#S4.SS0.SSS0.Px2)、[官方 `train.py` L139-L155, L198-L207](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L139-L155)、[官方 `sample.py` L60-L68](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/sample.py#L60-L68)）。

当前训练器直接在 MNIST、FashionMNIST 或 CIFAR-10 pixels 上做 diffusion。论文明确说 DiT 也可不经修改地用于 pixel space（[论文 Latent diffusion models](https://arxiv.org/html/2212.09748#S3.SS1.SSS0.Px3)），所以这是合法教学实验；但它不能验证 latent VAE、ImageNet 类条件和论文 FID。

VAE decoder 也不能混成一个默认值：scaling/无 guidance 表主要用 ft-MSE decoder，最终主结果使用 ft-EMA decoder；两者 encoder 相同（[论文 Appendix D](https://arxiv.org/html/2212.09748#A4)）。当前项目两者都未接入。

## 6. 训练 recipe 与工程增强

| 项目 | 论文 / 官方 | 当前 `paper_compatible` | 结论 |
|---|---|---|---|
| Optimizer / LR / WD | AdamW、constant `1e-4`、WD 0 | 默认相同 | ✅ |
| EMA | 每 step，decay 0.9999；论文结果全用 EMA | 默认相同 | ✅ |
| Label dropout | 官方默认 0.1 | 默认相同 | ✅ |
| Warmup / decay | 无 | CLI 默认 warmup=0、constant；仍可显式做教学消融 | ✅ 默认 |
| Gradient clipping | 无 | profile 默认关闭 | ✅ profile 对齐 |
| AMP | 官方训练脚本无 AMP | profile 默认关闭 | ✅ profile 对齐 |
| Global batch | 256 | 本地默认 128；可 accumulation，但未强制等效 256 | ⚠️ |
| DDP / TPU recipe | JAX TPU-v3；官方 PyTorch DDP | 单进程 Mac/CUDA/CPU | ℹ️ 系统差异 |
| 训练预算 | 400K～7M steps，依实验而定 | 本地教学预算 | ❌ 指标不可比 |

AMP、gradient accumulation、clipping、warmup/cosine、`torch.compile`、resume、周期 sample、DDIM 和 Web UI 都是有价值的工程/教学能力，但不是原论文 trick。profile 与 checkpoint 元数据必须继续让这些差异可见。

## 7. Checkpoint 可追溯性

每次训练写出两种用途不同的 checkpoint。可信恢复版保存：

- `profile`：`pixel_teaching` 或 `paper_compatible`；
- `recipe`：VB 模式、loss/sample clipping、CFG channel policy、sampler、sampling steps、gradient clipping、AMP；
- `prediction_type`：epsilon 或教学 v-parameterization；
- `pipeline`：当前明确写为 `space="pixel"`、`vae_id=None`、`vae_scaling=None`；
- 模型、EMA、optimizer、schedule、dataset/class names、data policy 与 RNG state。

紧凑推理版 `dit_image_inference.pt` 保存 raw/EMA state、schedule 与同一组可追溯元数据，但不含 optimizer/scaler/RNG；CLI/UI 通过 `weights_only=True` 加载它。恢复版含 pickle 对象，只应传给训练器的 `--resume` 且来源必须可信。

[`DiTInference.from_checkpoint`](../../code/image_dit/inference.py#L61-L101) 恢复这些元数据，采样默认读取 `recipe`。这可以阻止一个 pixel checkpoint 被静默描述成 latent 模型，也可以阻止 paper checkpoint 在推理时静默切回 clipping/DDIM。旧 checkpoint 缺少字段时按历史 `pixel_teaching` / epsilon 解释。

## 8. 当前实现总清单

### 已正确应用或由 `paper_compatible` 对齐

- [x] 非重叠 patchify/unpatchify、官方 x→y 固定 2D sin-cos position。
- [x] timestep/label embeddings、null token、label dropout、`t+y`。
- [x] pre-norm、affine-free LayerNorm、self-attention、ratio-4 GELU-tanh MLP。
- [x] adaLN-Zero 六路调制、identity/zero-output initialization。
- [x] 官方 S/B/L/XL presets 与 patch 2/4/8。
- [x] epsilon prediction、1000-step linear beta schedule。
- [x] learned-range covariance、MSE + VB、variance branch stop-gradient、`t=0` decoder NLL。
- [x] raw VB / rescaled VB 显式分离；paper profile 选择 raw。
- [x] loss/sampling clipping 显式分离；paper profile 选择 no-clip。
- [x] 1000→250 cumulative-alpha-preserving respaced ancestral DDPM。
- [x] CFG `first_3` / `all` 策略；paper profile 选择 first-3。
- [x] AdamW/constant LR/WD 0、EMA；paper profile 默认关闭 clipping/AMP。
- [x] profile/recipe/pipeline/prediction type 写入 checkpoint 并由 inference 读取。

### 仍缺的端到端论文复现能力

- [ ] Stable Diffusion f8 VAE encode/decode、posterior sampling、`0.18215` scaling。
- [ ] ImageNet ImageFolder、ADM precise center crop、1000 classes、256/512 latent presets。
- [ ] ft-MSE / ft-EMA decoder 与论文表格的明确绑定。
- [ ] 论文 batch、DDP/多机规模与各训练预算复现。
- [ ] 50K sample export 与 ADM-compatible FID/sFID/IS/Precision/Recall。
- [ ] 官方模型 key/Attention layout 转换、官方 checkpoint 直接加载与固定样本 parity。
- [ ] float64 schedule coefficient 等逐元素数值 parity。

### 不是原论文 trick，保留为明确消融/工程能力

- [x] v-parameterization（DDPM v，不是 rectified-flow velocity）。
- [x] AMP、gradient accumulation、gradient clipping、warmup/cosine。
- [x] `torch.compile`、resume、完整 RNG/scaler checkpoint。
- [x] 周期 sample/logging、DDIM、去噪轨迹和 Web UI。
- [x] canonical overfit / balanced batches 等本地质量诊断路径。

## 9. 下一步优先级

### P0：完成真正的 latent 纵向闭环

1. 接入 frozen f8 VAE，严格记录 VAE id、encoder posterior sampling 与 scaling。
2. 增加 ImageNet/小型 ImageFolder 共用的数据接口与 ADM crop 测试。
3. 让同一个 `paper_compatible` diffusion recipe 可在 `space=latent` 上运行；pixel 模式继续保留为教学路径。

### P1：官方权重与模块 parity

1. 为 timm Attention/MLP key 写显式转换器，而不是宽松 `strict=False`。
2. 对 position、初始化、block、unpatchify、CFG、loss terms 和 250-step timestep map 做官方 fixture 对照。
3. 固定官方 checkpoint/class/seed，与官方 sample script 做数值或至少输出级回归。

### P2：论文指标复现

1. 生成 50K 样本并保存 ADM-compatible `.npz`。
2. 对齐 ft-MSE/ft-EMA decoder、CFG 与 DDPM-250 的具体表格设置。
3. 记录 dataset、weight hash、commit、dtype、采样协议与训练预算；不把小数据实验的 loss 当作论文 FID 证据。

普通 Mac 可以完成 P0/P1 的 smoke/parity tests、小型 latent 教学训练和官方 checkpoint 推理；它无法现实地重训 675M 参数、数百万 steps 的 ImageNet DiT-XL/2。代码 recipe 对齐和生成质量声明必须分开。

## 10. MinimalDiT 的合规边界

[`minimal_dit.py`](../../code/image_dit/minimal_dit.py) 有意只保留

`patchify → fixed position + additive timestep → vanilla Transformer → unpatchify → epsilon MSE`

并省略 class conditioning、CFG、adaLN-Zero、learned variance、专门初始化与 EMA。它适合说明“Transformer 可以充当 diffusion denoiser”，但不属于“所有论文 trick 已应用”的审计对象；完整实现和 minimal 实现应始终分开描述。

## 11. 一句话判定

当前代码已经较完整复刻 **DiT 网络结构**，并为 **raw VB + no-clip + 250-step respaced DDPM + first-3 CFG + 官方位置编码 + 可追溯 checkpoint** 提供了可测试的 `paper_compatible` 路径；它仍缺 **VAE/ImageNet/FID/官方 checkpoint 直接 parity**，所以是论文 diffusion recipe 教学实现，不是端到端 ImageNet DiT 复现。

## Primary sources

- William Peebles, Saining Xie, [*Scalable Diffusion Models with Transformers*, arXiv:2212.09748v2](https://arxiv.org/html/2212.09748).
- Meta/Facebook Research, [`facebookresearch/DiT` official repository at commit `ed81ce2`](https://github.com/facebookresearch/DiT/tree/ed81ce2229091fd4ecc9a223645f95cf379d582b).
- Official architecture: [`models.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py).
- Official training: [`train.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py).
- Official diffusion factory and loss/sampler: [`diffusion/__init__.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/__init__.py), [`gaussian_diffusion.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py), [`respace.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/respace.py).
- Official sampling/evaluation: [`sample.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/sample.py), [`sample_ddp.py`](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/sample_ddp.py).
- Robin Rombach et al., [*High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv:2112.10752](https://arxiv.org/abs/2112.10752).
