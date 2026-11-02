# Video DiT moving-mosaic：论文—实现—证据审计

> 时期：Phase 3 — 根因定位前的论文与实现审计
> 日期：2026-08-23
> 状态：审计完成；后续 M0 实验结果见 [Phase 3 根因与复训记录](06-phase-3-root-cause-debug-and-retrain.md)  
> 输入证据：Phase 2 诊断、`c1-direct-train-v3`、现有 100-clip epsilon/velocity checkpoint 与 teacher-forced 报告、DiT/Latte/DDPM/Improved DDPM/LDM 一手来源
> 下一门禁：M0 partial-noise 与 sampler parity 已完成；10-clip patch2 对照已在 5k no-go，下一步是 1-clip 高噪声可达性，仍不复训 102 条数据  
> 顺序：[总索引](README.md) → [Phase 2 诊断](04-phase-2-moving-mosaic-diagnosis-and-next-actions.md) → [Phase 2 一手资料](05-phase-2-moving-mosaic-primary-source-notes.md) → 本文 → `06-phase-3-root-cause-debug-and-retrain.md`

## 审计结论

Phase 2 文档的总策略是正确的：停止盲目延长训练，先做 VAE、扩散数学、partial-noise、单样本与条件门禁。但根因排序和复训建议需要三处重要修正。

1. **最高优先级应是严格的训练—采样 contract oracle 与跨 timestep 恢复曲线，而不是先认定 `patch=3` 是主因。** 当前 `t=999` 的 `sqrt(alpha_bar)` 只有约 `0.00635`；epsilon 预测误差换算成 `x0` 时会被放大约 `157.4×`。但 Improved DDPM 也指出线性 schedule 末段几乎是纯噪声，跳过反向链开头一部分对 FID 影响很小；因此 `t=999` 单点失败本身不是 bug 证明。必须看 `t=100/500/800/900/950/999` 的连续退化位置，并核对 schedule、target 与 sampler 完全同源。
2. **epsilon 应保留为论文 faithful control，velocity 应保留为本地候选；现有 `t=999` retrieval 不能给两者排胜负。** epsilon 是 DDPM、官方 DiT 和 Latte 的主线，且与 learned-range sigma、1000-step linear training 和 250-step stochastic DDPM sampling 构成一套公开 recipe。现有本地结果中 epsilon 的 `t=999` retrieval 是 `9%`，velocity high-noise 微调后为 `94%`；但在接近纯噪声时 velocity target 约含 `-x0`，epsilon target 约为 condition-independent noise，这个指标结构性地更容易从 velocity 中读出条件。它是有价值的诊断，却不能证明 velocity 的完整 reverse chain 更好。
3. **`patch=3` 是高价值消融，不是已证实根因。** 当前每帧只有 `4×4=16` 个 attention token，远少于 Latte 常用的 latent `32×32`、patch 2（每帧 256 token）。DiT 论文支持“更多 token / 更多 Gflops 通常改善 FID”，但 patch token 并不是输出像素块：unpatchify 仍为每个 patch 预测独立的 `3×3` latent 值。因此不能仅凭“一个 token 覆盖 24×24 RGB 像素”推出马赛克必然由 patch 造成。

当前最符合证据的工作假设是：**低/中噪声 teacher-forced 任务已经学到部分训练分布与文本差异，但“解析正确的训练—采样 contract、跨 SNR 的结构恢复、完整 reverse chain”尚未被同一套实验连起来。moving mosaic 可能来自恢复误差沿 reverse chain 累积，并被粗 token、scratch 小数据、DDIM-50、短 EMA 或 CFG 加重；当前证据不能把其中任何一项单独宣布为根因。**

## 证据分层

本文严格使用三类标签：

- **论文事实**：论文或官方仓库直接说明的设计、公式或公开实验设置；
- **本地证据**：仓库代码、checkpoint metadata、训练/评估产物直接给出的观察；
- **推断**：把前两类证据连接到 moving-mosaic 的可证伪解释。

论文事实不能替代本地实验；训练 loss、CLIP 相似度和 teacher-forced retrieval 也不能替代自由生成视频的盲评。

## 逐项审计 Phase 2 根因

### 1. contract oracle / 跨 timestep 恢复 / sampler：从第 3 位提升到第 1 位

**论文事实。** DDPM 的训练与采样必须共享同一个 `q(x_t|x_0)`、累计 alpha、预测目标和反向更新；原论文 Algorithm 1–2 的主线是 uniform timestep、epsilon MSE 与 ancestral reverse process。[DDPM Algorithm 1](https://arxiv.org/html/2006.11239#A1) Improved DDPM 说明 noise schedule、reverse variance 与少步采样互相关联；它还观察到线性 schedule 末四分之一几乎只有噪声，采样时跳过 reverse chain 开头最多约 20% 仍几乎不改变 FID。因此端点 `t=999` 不是应孤立优化的神奇门槛，跨 timestep 曲线比单点更可靠。[Improved DDPM §3.2](https://arxiv.org/html/2102.09672#S3.SS2)；[官方 improved-diffusion](https://github.com/openai/improved-diffusion)

**本地证据。** 当前 schedule 是 1000 步线性 `beta=1e-4→0.02`，训练支持 epsilon/velocity，推理用本地 deterministic DDIM respacing；相关实现见 [`diffusion.py`](../../../code/video_dit/diffusion.py)。该 schedule 的关键数值为：

| t | `alpha_bar` | SNR | epsilon 误差映射到 `x0` 的放大系数 `sqrt(1-alpha_bar)/sqrt(alpha_bar)` |
|---:|---:|---:|---:|
| 100 | 0.89514 | 8.53665 | 0.342× |
| 500 | 0.07780 | 0.08436 | 3.443× |
| 900 | 0.000270 | 0.000270 | 60.82× |
| 999 | 0.0000404 | 0.0000404 | 157.41× |

已有 100-clip 实验比 Phase 2 文档记录得更深入：

| checkpoint / 诊断 | 参数化 | `t=999` teacher-forced prompt top-1 | margin |
|---|---|---:|---:|
| `commons_g0_tokens_v2_100clip_large_10k`, EMA | epsilon | 9% | -0.000240 |
| 上述 checkpoint 后 high-noise 2k，raw | epsilon | 2% | -0.000135 |
| 上述 checkpoint 后 high-noise 2k，EMA | epsilon | 13% | -0.000034 |
| epsilon checkpoint 初始化后 velocity high-noise 2k，raw | velocity | 93% | 0.28637 |
| 同上，EMA | velocity | 94% | 0.27993 |

对应原始报告位于 `code/video_dit/runs/commons_g0_tokens_v2_100clip_*/teacher-forced-*.json`。同一个 epsilon 100-clip checkpoint 在 `t=100/500/900` 聚合 teacher-forced retrieval 上 raw/EMA 都达到 `100%`，而 `t=999` 是 `9%`。这证明该指标在端点发生突变，但还不能把故障直接定位为“epsilon 高噪声没有学会”。

**证据边界。** teacher-forced retrieval 比较不同 prompt 对单步预测 target 的 MSE，且不是 target-neutral：在 `t≈999`，`v=alpha*epsilon-sigma*x0≈-x0`，而 epsilon target 主要是 noise，所以 velocity 天然保留更强的条件可辨信号。`94%` 不等于“velocity sampler 已修好”，`9%` 也不等于“epsilon sampler 必然失败”。此外，高噪声微调实验同时改变了 target 和 timestep 分布，不能把提升归因于单一变量。

**推断。** 当前最值得首先证伪的是：`c1-direct-train-v3` 是否能在一段连续 timestep 上恢复同一真实 `z0`，以及一个 oracle model 在本地 ancestral/DDIM 更新下是否能回到已知 `z0`。若 oracle 失败是实现/contract；若 oracle 通过但真实模型随噪声逐渐崩坏是学习能力；若单步 `x0_hat` 好而多步坏才是 sampler 累积。

**修正后的行动。** 先不训练：先用解析 oracle 跑 epsilon/velocity 转换和 ancestral/DDIM reverse；再对 `c1-direct-train-v3` 与现有 100-clip checkpoint 做 partial-noise recovery：真实 `z0`、固定 noise、`t=100/500/800/900/950/999`，分别保存 1-step `x0_hat` 与对应区间的完整恢复。sampler 只需先比 DDIM 50/250 与 ancestral reference；报告 latent MSE、decode 后 LPIPS/PSNR 和未挑选联系表。

### 2. `patch=3`：保留高优先级，但从“主因”降为“待证消融”

**论文事实。** DiT 将 latent 切成 patch token；论文发现增加网络 Gflops——包括增大网络或增加 token 数——持续改善 FID。[DiT 论文](https://arxiv.org/html/2212.09748#S3) 官方 DiT 和 Latte 都提供 `/2`、`/4`、`/8` 变体。Latte §4.2 报告逐帧 uniform 2D patch 优于压缩多个相邻帧的 tube embedding，后者会损失时空信号；其公开最佳配置使用 Latte-XL/2。[Latte §4.2–4.3](https://arxiv.org/html/2401.03048#S4.SS2)；[Latte 官方配置](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/configs/ffs/ffs_train.yaml#L12-L29)

**本地证据。** `c1-direct-train-v3` 是 `12×12` latent、patch 3，即每帧 16 token；模型用 Conv2d patch embed，最后线性层输出 `patch_size² × channels` 再 unpatchify，见 [`video_dit.py`](../../../code/video_dit/video_dit.py)。所以空间 attention 网格确实很粗，但输出不是把每个 token 复制成常量方块。

**推断。** 16 token/frame 很可能限制对象边界和局部结构，尤其模型还要从零学习空间先验；但它也可能只是让“高噪声恢复不足”表现得更像低频色块。只有在相同数据、target、优化步数和采样器下比较 patch 1/2/3 才能定责。

**修正后的行动。** patch 消融必须放在数学/partial-noise 门之后。单 clip 使用 patch 1/2/3；正式 102-clip 不要直接改 patch 后长训。衡量的不只是最终 MP4，还包括每个 timestep 的 `x0` recovery 和 fixed-seed trajectory。

### 3. 数据与从零训练规模：仍是高概率系统性限制，但不是当前可接受的 bug 解释

**论文事实。** DiT 的成功实证来自 ImageNet 256/512 大规模类条件训练；官方 PyTorch recipe 使用 global batch 256、AdamW `1e-4`、VAE posterior sample、latent scaling `0.18215` 和 EMA。[DiT 官方训练代码](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L165-L192) Latte 在 FaceForensics、SkyTimelapse、UCF101、Taichi-HD 等标准数据集上训练，并研究从预训练 DiT 初始化和 video-image joint training；公开配置是 16 帧 256px、Latte-XL/2、最多 1,000,000 steps，论文实验使用 8×A100。[Latte 论文](https://arxiv.org/abs/2401.03048)；[Latte 官方配置](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/configs/ffs/ffs_train.yaml#L12-L29)

**本地证据。** `c1-direct-train-v3` 只有 102 条视频、23.17M 参数、30k updates、batch 8，约 240k 样本呈现；模型和文本 cross-attention 都从零初始化。VAE 与文本编码器冻结，但没有预训练视觉 denoiser。训练内 IID 盲评 triple 为 `33.3%`，C1 triple 为 `0%`；说明模型并非完全没有吸收训练分布，但远未形成开放世界视频先验。

**推断。** 即便消除实现问题，这一数据/预训练规模也不支持期待通用 T2V 或可靠未见组合。它可以支持教学型 overfit 与受限分布实验。因此“102 条太少”是最终能力上限的高置信解释，却不能作为跳过 oracle、partial-noise 和单 clip 的理由。

**行动。** 本轮不扩数据。若单 clip 和 8–16 clip 都通过、102 clip 仍失败，再把“使用预训练空间 DiT / 冻结空间模块 / 只学时间模块”列为下一阶段路线，而不是继续堆 scratch updates。

### 4. VAE 与 latent contract：主链基本正确，但 mode/sample 需要作为低优先级单变量

**论文事实。** LDM 是两阶段系统：先用预训练 autoencoder 把图像压缩到 latent，再学习 latent 的生成分布；autoencoder 重建质量构成上限，但真实 latent 能重建不保证生成 latent 位于 decoder 熟悉的分布。[LDM](https://arxiv.org/abs/2112.10752) 官方 DiT/Latte 训练使用 posterior `sample()` 后乘 `0.18215`，采样结果解码前除以同一因子。[DiT train](https://github.com/facebookresearch/DiT/blob/main/train.py)；[DiT sample](https://github.com/facebookresearch/DiT/blob/main/sample.py)；[Latte train](https://github.com/Vchitect/Latte/blob/main/train.py)

**本地证据。** 本地 cache 使用 posterior `mode()`，然后乘 VAE scaling；解码时除回 scaling，且 checkpoint 固化 VAE ID/scaling，见 [`vae_latents.py`](../../../code/video_dit/vae_latents.py)。VAE round-trip 联系表仍能辨认猫；latent 全局约 mean `0.042`、std `0.781`；latent 推理明确 `clip_denoised=False`，见 [`inference.py`](../../../code/video_dit/inference.py)。因此“重复 scaling”或“latent 被错误 clamp 到 [-1,1]”不符合当前代码与证据。

**推断。** posterior mode 与官方 sample 是 recipe 差异，可能降低训练 latent 的随机性，但不太可能单独制造整段 moving mosaic。它应在最小基线通过后做单变量验证，不应排在高噪声恢复之前。

### 5. EMA 0.99：是 recipe 偏差与 checkpoint 选择风险，不是已证实根因

**论文事实。** 官方 DiT 的 `update_ema` 默认 decay 是 `0.9999`；Latte 也是 `0.9999`，论文报告结果来自 EMA。官方代码先用 decay 0 把初始模型同步到 EMA，再逐步更新。[DiT train](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/train.py#L165-L192)；[Latte EMA](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/utils.py#L190-L200)

**本地证据。** `c1-direct-train-v3` 使用 `0.99`，约 100-step 有效时间尺度；推理只冻结选择了最终 EMA。当前训练代码默认已改为 `0.9999`，并能保存 raw 与 EMA，但历史 checkpoint 的 EMA 无法事后重建成精确的 `.9999`，除非保留了每步 raw 权重。

**推断。** `.99` 更贴近近期 raw weights，可能抖动或过拟合，也可能在短训练中反而比 `.9999` 更快跟上；仅由有效窗口不能推出它导致马赛克。

**行动。** 先对历史 checkpoint 做 raw vs EMA 同 seed 对照。复训时保存 raw，EMA 用 `.999/.9999` 二选一的预注册单变量，不要同时改 target、patch 与 EMA 后宣称因果。

### 6. CFG 2.5：可能放大弱条件，但不是基础生成失败的首要解释

**论文事实。** DiT/Latte 通过 condition dropout 训练 classifier-free guidance；官方 DiT sampling 将同一 noise 复制成 cond/uncond pair，并对预测通道做 guidance。[DiT models](https://github.com/facebookresearch/DiT/blob/main/models.py)；[DiT sample](https://github.com/facebookresearch/DiT/blob/main/sample.py)

**本地证据。** 本地同样复制 noise、强制 cond/null 两路并在预测通道上线性组合；`c1-direct-train-v3` 训练 dropout 0.1，冻结评估用 CFG 2.5。CLIP matched 与 shuffled 都是 `15.4%`，表明最终自由生成没有稳定文本绑定。

**推断。** CFG 可能放大一个错误或极弱的 conditional residual；但 CFG=1 若仍是马赛克，就不能归因于 guidance。反之，如果 CFG=1 清晰、2.5 变坏，才可定责。

**行动。** CFG sweep 是同 checkpoint 的廉价诊断，排在 partial-noise 后、重训前；只需固定 seed 的 1.0/1.5/2.5，不要先加入 5 或更大值。

### 7. learned variance 与 DDIM-50：论文原味不等于必须一次性恢复

**论文事实。** 官方 DiT/Latte recipe 是 epsilon prediction + learned-range sigma：模型输出 `2C`，epsilon mean 使用 MSE，variance 使用 detached-mean VB；训练是 1000-step linear schedule。DiT 报告采样使用 250-step stochastic DDPM，而不是把 DDIM-50 当作默认。[DiT diffusion 配置](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/__init__.py#L8-L42)；[DiT loss](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/diffusion/gaussian_diffusion.py#L701-L737)；[Latte loss](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/diffusion/gaussian_diffusion.py#L735-L791) 官方 DiT 示例明确对 latent 设置 `clip_denoised=False`。[DiT sample](https://github.com/facebookresearch/DiT/blob/main/sample.py)

**本地证据。** 历史 latent 训练关闭 learned sigma，因为当前 VLB 分支含 pixel-specific discretized likelihood，而且 [`train.py`](../../../code/video_dit/train.py) 明确拒绝 latent + `learn_sigma`；所以这里的 `faithful` 模式并不能原样复现官方 latent DiT/Latte recipe。采样固定 DDIM、eta 0、50 steps。关闭 learned sigma 本身仍定义了合法 fixed-variance DDPM；因此它是与官方 recipe 的差异，不是数学错误的充分证据。

**行动。** 先用 fixed variance 保持训练变量少，做本地 DDIM 50/100/250 与 ancestral 1000 parity。只有 ancestral 明显好、少步 DDIM 坏，才把 solver/variance 提到修复项；不要为“更像论文”同时打开 learned sigma 并改 loss。

## 当前实现与论文的准确关系

当前模型可称为“**受 Latte Variant 1 启发的教育型 Video DiT**”，不应称为 Latte 或原始 DiT 的原样复现：

| 组件 | 论文/官方主线 | 本地实现 | 审计判断 |
|---|---|---|---|
| backbone | DiT adaLN-Zero；Latte spatial/temporal block 交替 | 保留 adaLN-Zero、固定 sin/cos 空间/时间位置、空间/时间完整 block | 架构主干一致 |
| 条件 | DiT/Latte 主实验以 class/unconditional 为主；LDM 用 cross-attention 做文本条件 | frozen CLIP token + 每个 block 的新 cross-attention + pooled adaLN | 合理扩展，但从零训练且不是 Latte 主实验原样设计 |
| latent | 官方 posterior sample × 0.18215 | posterior mode × pinned VAE scaling | contract 一致，latent 分布略有差异 |
| target | 官方 DDPM/DiT/Latte 主线 epsilon + learned-range sigma | 历史 C1 使用 velocity + fixed variance；代码 latent 路径拒绝 learned sigma | epsilon 是 faithful control；velocity 是合理消融，均须匹配 sampler |
| variance | 官方 DiT/Latte learn sigma | latent 路径 fixed variance | 合法简化；少步采样可能受影响，待 parity |
| sampler | 官方 DiT 示例 ancestral respacing 250 steps | deterministic DDIM 50 | 高价值诊断差异 |
| token grid | Latte 常见 32×32 latent、patch 2 | 12×12 latent、patch 3 | 16× 更少空间 token，待单变量消融 |
| data/pretrain | 标准数据集、大规模训练、可用预训练/联合图像训练 | 102 clips，23.17M denoiser 和文本层从零训练 | 明确能力上限 |

## 最小可执行调试矩阵

目标是用最少运行回答四个互斥问题：数学错、端点没学会、sampler 太激进，还是架构/数据容量不足。所有格固定同一真实 clip、同一 noise、同一 prompt 和 seed；禁止挑样本。

### M0：无需训练，约 8 个核心单元

在下表之前先运行 M-1：给定已知 `z0` 与 noise，用解析 epsilon/velocity oracle 经过本地 ancestral 和 DDIM 更新；若不能在浮点误差容许范围内恢复，立即停止，先修 contract。M-1 通过后才评价 checkpoint。

| 单元 | checkpoint | 起点 | reverse | weights / CFG | 判定 |
|---|---|---|---|---|---|
| 1–4 | `c1-direct-train-v3` velocity | 真实 `z0` 加噪到 100/500/900/999（另记录 800/950 单步） | 对应区间 DDIM，steps 按区间比例且至少 50 | raw，CFG 1 | 找到连续退化拐点；不以 t999 单点定 bug |
| 5 | 同上 | `t=999` | DDIM 50 | EMA，CFG 1 | raw/EMA 是否决定性 |
| 6 | 同上 | `t=999` | DDIM 250 | raw，CFG 1 | 少步误差是否主因 |
| 7 | 同上 | `t=999` | ancestral 1000 | raw，CFG 1 | DDIM 与参考 reverse parity |
| 8 | 同上 | `t=999` | 最优 reverse | raw，CFG 2.5 | CFG 是否破坏已存在结构 |

每个单元保存：输入 `z0`、加噪 `zt`、第一步 `x0_hat`、最终 latent channel mean/std/quantiles、解码联系表、PSNR/LPIPS、固定 JSON contract。M0 的 primary score 是真实 clip recovery，不是 CLIP prompt retrieval。

解释表：

| 结果模式 | 根因方向 | 下一步 |
|---|---|---|
| `t=100` 都失败 | 实现/轴次序/训练基本失败 | 停止复训，做 oracle 与 shape trace |
| 100/500 成功，之后随 t 连续恶化；各 sampler 都失败 | 模型随 SNR 降低而失去恢复能力 | 先做 1-clip；之后才比较 target / timestep sampling |
| 单步 `x0_hat` 可辨，DDIM-50 坏，250/ancestral 好 | solver/respacing | 修 sampler baseline，不改网络 |
| raw 好、EMA 坏 | EMA checkpoint | 调 EMA 或先用 raw；不改 patch |
| CFG1 好、CFG2.5 坏 | guidance 校准 | CFG1 baseline，条件稳定后再扫 |
| 所有 partial-noise recovery 好，纯噪声仍坏 | 分布外累积/容量/数据先验 | 进入 M1 单 clip 与 patch 消融 |

### M1：只有 M0 不能闭合时才训练

| 实验 | 数据 | 固定项 | 唯一变量 | 通过门 |
|---|---|---|---|---|
| A | 1 clip，无文本/CFG | velocity、fixed variance、raw、参考 sampler | 无 | 100/500/900/999 recovery 全通过，纯噪声可辨 |
| B1–B3 | 同 1 clip | A 的所有项 | patch 1/2/3 | 量化 patch 对结构的因果影响 |
| C | 8–16 clips，离散且明显不同条件 | 选择 B 的最小足够 patch | 加条件 | train prompt retrieval ≥90%，同 seed 换条件方向正确 |
| D | C | 数据、patch、steps、sampler 全固定 | velocity ↔ epsilon | 直接比较高噪声恢复与自由生成，不按论文默认预判胜者 |

M1 不应同时打开 learned sigma、改 cosine schedule、改 VAE posterior、增大模型和更换 EMA。任何单元失败都回到最近一个通过门。

## 对下一轮修复/复训的具体建议

在 M0 结果出来前，**不要 resume `c1-direct-train-v3`，也不要直接启动 Phase 2 文档中的 patch2 + epsilon 102-video baseline**。

如果 M0 显示单步正确而 DDIM-50 错，则无需重训，先修 sampler。如果 M0 显示恢复质量随 SNR 连续崩坏，先用单 clip 闭合，再用完全独立的 runs 对照 faithful epsilon 与 velocity；不能把现有 `t=999` retrieval 的 `9% vs 94%` 当成 target 选择结果。如果 M0 所有路径都正确但视频仍是低频块，才进入 patch 1/2/3 与 T=1→16 消融。

当单 clip 和小集合门通过后，推荐的教学型 102-clip baseline 才能冻结为：

- `12×12×16` latent，patch 2（patch 1 仅在 M1 明显胜出且算力允许时采用）；
- hidden 192–256、4 spatial-temporal pairs，保持模型小于当前 23M；
- target 由 M1-D 实证选择，当前不能预注册 epsilon 必胜；
- fixed variance 先保留，reference ancestral/高步 DDIM 先建立正确性；
- raw 必存，EMA `.999` 或 `.9999` 只选一个并记录有效窗口；
- CFG 1 作为固定诊断，条件门通过后再用 1.5/2.5；
- 500-step 固定诊断必须包含 timestep bucket loss、partial-noise recovery 与 fixed-seed free sample；
- 5k updates 为第一次 go/no-go，不因总 loss 下降自动延长。

## 已排除、尚未排除与停止条件

**已基本排除为单独主因：** VAE 无法显示猫、latent scaling 重复/遗漏、latent sampling 被 `[-1,1]` clamp、完全静止视频、NaN/Inf。

**尚未排除：** `c1-direct-train-v3` 高噪声结构恢复不足；DDIM-50 累积/respacing；raw 与 EMA 差异；CFG 放大；16 token/frame 限制；scratch 数据/容量上限；posterior mode 与 sample 的小幅分布差异。

**立即停止复训：** `t=100` partial-noise 失败；oracle epsilon/velocity 转换不通过；同一 oracle 目标下 ancestral 与 DDIM 高步无法对齐；生成 latent 统计出现非有限值；实验一次改变两个以上核心变量。

**允许进入 102-clip 复训：** 单 clip 从纯噪声可辨；8–16 clip 条件 overfit 通过；target、patch、sampler、EMA 各完成至少一次单变量对照；固定 seed 的 raw/EMA 趋势与人眼可辨识度一致。

## 一手来源

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), 2021；[官方实现](https://github.com/openai/improved-diffusion).
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), 2022；[官方实现](https://github.com/CompVis/latent-diffusion).
- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), 2023；[官方 DiT](https://github.com/facebookresearch/DiT).
- Ma et al., [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048), 2024；[官方 Latte](https://github.com/Vchitect/Latte).
