# DiT / 视频 Diffusion 一手资料调研笔记

> 范围：只使用论文原文、arXiv 页面及官方 GitHub 仓库；检索日期为 2026-07-17。本文是后续讲义、代码与测试设计的事实底稿，不代替完整学习计划。

## 1. 从 DDPM 到 DiT：数学接口不变，主干网络改变

DiT 并没有发明新的前向扩散过程。它沿用 latent diffusion：先由预训练 VAE 把图像 \(x\) 编码为潜变量 \(z_0=E(x)\)，再在潜空间做高斯加噪；LDM 的核心贡献正是把扩散放到感知压缩后的潜空间，并用 cross-attention 接入文本、布局等条件（[LDM 论文](https://arxiv.org/abs/2112.10752)）。

以离散 DDPM 表示：

\[
q(z_t\mid z_{t-1})=\mathcal N(\sqrt{1-\beta_t}z_{t-1},\beta_t I),\qquad
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\,\epsilon,
\quad \epsilon\sim\mathcal N(0,I).
\]

DiT 的网络接口可写作

\[
f_\theta(z_t,t,c)\rightarrow(\hat\epsilon_\theta,\hat\Sigma_\theta),
\]

其中输入是带噪 latent、时间步和条件，输出与输入 latent 同空间的噪声预测及可学习方差参数。官方实现默认 `in_channels=4`、`learn_sigma=True`，因此输出通道是输入的两倍（[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)）。训练仍是 improved-DDPM 风格的混合目标：噪声预测 MSE 加用于学习反向过程方差的 VLB 项；官方训练脚本调用仓库内的 Gaussian Diffusion 实现（[gaussian_diffusion.py](https://github.com/facebookresearch/DiT/blob/main/diffusion/gaussian_diffusion.py)、[train.py](https://github.com/facebookresearch/DiT/blob/main/train.py)）。

因此，从 DDPM 迁移到 DiT 时应牢牢记住：

- **概率模型层不变**：\(q(z_t\mid z_0)\)、反向高斯、噪声/方差参数化、采样器仍可独立于 backbone 推导。
- **表示层变化**：像素变为 VAE latent；二维 latent 再变成 patch token。
- **函数逼近器变化**：U-Net 的卷积、多尺度下采样和 skip connection 被等分辨率 Transformer token 流替换。

## 2. 原始 DiT 的核心架构

### 2.1 Token 化与主干

给定 \(z_t\in\mathbb R^{I\times I\times C}\)，以边长 \(p\) 的不重叠 patch 切分，token 数

\[
T=(I/p)^2.
\]

每个 patch 线性投影到隐藏维度 \(d\)，加入固定二维 sin-cos 位置编码，经过 \(N\) 个 Transformer block，最后线性投影并 unpatchify 回 latent 网格。官方实现的 `PatchEmbed → blocks → FinalLayer → unpatchify` 可以逐行对应这一数据流（[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)）。

论文系统比较四种条件注入方式：in-context token、cross-attention、adaptive LayerNorm、adaLN-Zero，最终采用 **adaLN-Zero**（[DiT 论文](https://arxiv.org/pdf/2212.09748)）。实现中时间嵌入与类别嵌入相加为 \(c=t_{emb}+y_{emb}\)，每块由 \(c\) 产生 attention/MLP 各自的 shift、scale、gate：

\[
\operatorname{modulate}(h;s,b)=h\odot(1+s)+b,
\]

\[
h' = h+g_{msa}\operatorname{MSA}(\operatorname{modulate}(\operatorname{LN}(h))),\quad
h''=h'+g_{mlp}\operatorname{MLP}(\operatorname{modulate}(\operatorname{LN}(h'))).
\]

调制层和最终输出层零初始化，使每个 block 初始近似恒等映射、模型初始输出为零；这是 `adaLN-Zero` 的“Zero”在代码中的具体含义（[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)）。

### 2.2 模型族与扩展旋钮

DiT 同时沿两条轴扩展：

1. S/B/L/XL 增加 Transformer 深度、宽度与 head 数；
2. `/8`、`/4`、`/2` 减小 patch size，从而增加 token 数和计算量。

论文以单次前向 Gflops 为统一尺度，观察到无论通过深/宽还是更多 token 增加 Gflops，FID 都持续改善；最大 DiT-XL/2 在 ImageNet 256×256 得到论文报告的 FID 2.27（[arXiv 摘要](https://arxiv.org/abs/2212.09748)）。这不是“参数越多必然越好”的一般定律，而是同一训练设定下 **计算量与生成质量的经验扩展关系**。官方权重表给出 XL/2 的 256 分辨率 119 Gflops、512 分辨率 525 Gflops，后者说明 token 数随分辨率增长会迅速放大 attention 成本（[官方 README](https://github.com/facebookresearch/DiT#sampling-)）。

## 3. 条件、CFG 与采样

### 3.1 类别条件与 classifier-free guidance

官方 DiT 以类别为条件。训练时以 `class_dropout_prob=0.1` 随机把标签替换为额外的 unconditional embedding，因而同一网络同时学习条件与无条件预测（[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)）。推理时：

\[
\hat\epsilon_{cfg}=\hat\epsilon_{uncond}
+s\left(\hat\epsilon_{cond}-\hat\epsilon_{uncond}\right).
\]

`forward_with_cfg` 将同一噪声复制成两份、批量计算 cond/uncond，再做上式组合。需要注意一个复现实务细节：官方代码为了精确复现，默认只对前三个输出通道施加 CFG；注释同时给出了对所有 latent channels 做标准 CFG 的改法（[models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)）。

### 3.2 采样器是可替换模块

官方仓库把模型和扩散/respacing 分开：`models.py` 只实现 \(f_\theta\)，`diffusion/` 实现 DDPM 反向均值方差、时间步重采样和循环，`sample.py` 负责 VAE 解码（[仓库结构](https://github.com/facebookresearch/DiT)、[respace.py](https://github.com/facebookresearch/DiT/blob/main/diffusion/respace.py)、[sample.py](https://github.com/facebookresearch/DiT/blob/main/sample.py)）。这为学习测试提供清晰边界：可以固定模型假输出，单测 posterior、variance、respacing、CFG，而不必真正训练大模型。

官方复现实验说明：其 400K-step 对照使用 250 个 DDPM 采样步、`mse` VAE decoder、无 guidance；多卡评估脚本生成 50K 样本和 ADM 评估所需 `.npz`（[官方训练与评估说明](https://github.com/facebookresearch/DiT#pytorch-training-results)）。

## 4. 新论文的贡献，以及它们在 DiT 学习路径中的位置

| 资料 | 可由原文确认的贡献 | 对学习 DiT 的价值 |
|---|---|---|
| [Latte (2401.03048)](https://arxiv.org/html/2401.03048) | 把 latent DiT 扩展到视频，系统比较四种时空分解：空间/时间 block 交错、late fusion、block 内串行时空 attention、按 head 并行拆分；并消融 tube patch、时间位置编码、条件注入、图像视频联合训练。 | 最直接的“图像 DiT → 视频 DiT”教学桥梁；适合实现 factorized attention 的形状与复杂度测试。 |
| [CogVideoX (2408.06072)](https://arxiv.org/abs/2408.06072) | 使用时空压缩的 3D VAE；以 expert adaptive LayerNorm 让文本、视频模态保有各自参数并深度融合；报告 10 秒、16 fps、768×1360 的生成设定。 | 研究从单一类别 embedding 到文本—视频双模态条件、3D latent 与 expert modulation。 |
| [Open-Sora (2412.20404)](https://arxiv.org/html/2412.20404) | STDiT 解耦空间与时间 attention；高压缩 3D autoencoder；支持 T2I/T2V/I2V、最长 15 秒、720p、任意宽高比，并公开描述分阶段数据训练。 | 适合学习 variable shape、mask、bucket/data curriculum，以及“开源系统”而非单网络。 |
| [LTX-Video (2501.00103)](https://arxiv.org/html/2501.00103) | 将 patchify 前移到 Video-VAE，达到 1:192、每 token 对应 32×32×8 像素；高压缩后仍使用完整时空 self-attention；denoising VAE decoder 承担最后一步像素域去噪；训练采用 rectified-flow velocity。 | 展示 token budget、VAE 压缩与 attention 选择的联动，并把学习从 DDPM ε-pred 扩展到 flow/velocity 参数化。 |
| [Seedream 2.0 (2503.07703)](https://arxiv.org/html/2503.07703) | 图像 token 与文本 token 拼接输入 Transformer；自研中英双语 LLM text encoder，额外 Glyph-Aligned ByT5 支持字符级文字渲染，Scaled RoPE 泛化分辨率；后训练包含 continuing training、SFT、RLHF。 | 说明工业 DiT 的能力瓶颈常在数据、caption/text encoder、文字渲染和偏好对齐，而非只在 backbone。 |
| [Seedance 1.0 (2506.09113)](https://arxiv.org/html/2506.09113v1) | causal Video-VAE；空间/时间层解耦；空间层使用 MMDiT 融合文图 token、时间层只处理视觉；3D/MM-RoPE；用 noisy input、clean/zero frame 与 mask 统一 T2I/T2V/I2V；480p base 后接 720/1080p diffusion refiner。 | 是从教学 Latte 走向多任务、multi-shot、级联高分辨率视频系统的关键案例。 |
| [OmniWeaving (2603.24458v2)](https://arxiv.org/html/2603.24458v2) | Qwen2.5-VL MLLM + HunyuanVideo-1.5 MMDiT + VAE；MLLM thinking 先推导增强 prompt，跨浅/中/深层 hidden states 通过 DeepStacking 注入 MMDiT；三阶段完成模态对齐、组合能力、reasoning-augmented fine-tuning，并提出 IntelligentVBench。 | 学习重点从“条件编码”升级为“理解/推理系统如何控制生成器”；应置于掌握 MMDiT 后。 |
| [Seedance 2.0 model card (2604.14148)](https://arxiv.org/abs/2604.14148) | 当前公开文本明确的是原生音视频联合、多模态输入（文/图/音/视频）、4–15 秒、480p/720p，以及 Fast 版本；arXiv 将其标为 model card。 | 用于观察 2026 产品能力边界；**公开 card 未给足数学目标和具体 DiT block 细节，不能把 Seedance 1.0 架构直接外推为 2.0 的已验证事实。** |

### 4.1 Latte 的四种时空分解为什么值得先做

完整 attention 的 token 数为 \(T=FHW/p^2\)，复杂度近似 \(O(T^2d)\)。Latte 将其分解为空间 attention（逐帧）和时间 attention（逐空间位置），提供理解视频 DiT 的最小实验台。其原文还显示 Variant 1（空间/时间 block 交错）为最佳经验选择，并尝试从 ImageNet DiT 初始化、复制空间位置编码到时间维、丢弃不匹配标签 embedding（[Latte 方法](https://arxiv.org/html/2401.03048#S3)）。这使“加载 2D 权重到 3D 模型”的缺失键、形状与初始化测试成为很好的工程练习。

### 4.2 从 DDPM 到 rectified flow 时，哪些概念要重写

LTX-Video 的 rectified flow 采用线性路径

\[
z_t=(1-t)z_0+t\epsilon,
\]

并学习 velocity，而不是原始 DDPM 的离散 ε 目标；推理成为对速度场的数值积分。论文还采用非均匀 timestep sampling 并针对分辨率调整 noise schedule（[LTX-Video §2.5](https://arxiv.org/html/2501.00103#S2.SS5)）。学习时应分别测试：路径端点、target 符号约定、solver 单步更新、schedule shift，而不要把 DDPM posterior 公式直接套用。

## 5. `facebookresearch/DiT`：可验证的代码结构与工程点

官方仓库规模很小，适合作为第一份实现阅读材料（[仓库](https://github.com/facebookresearch/DiT)）：

| 文件 | 职责 | 建议测试 |
|---|---|---|
| [`models.py`](https://github.com/facebookresearch/DiT/blob/main/models.py) | timestep/label embedding、adaLN-Zero block、patchify/unpatchify、模型规格、CFG | shape round-trip；零初始化输出；强制 label drop；CFG \(s=0,1\) 边界；各 `/p` token 数 |
| [`diffusion/gaussian_diffusion.py`](https://github.com/facebookresearch/DiT/blob/main/diffusion/gaussian_diffusion.py) | \(q\) 与 \(p_\theta\)、loss、DDPM sampling | closed-form \(q(z_t\mid z_0)\) 统计；posterior 系数；t=0 边界；learned variance 通道拆分 |
| [`diffusion/respace.py`](https://github.com/facebookresearch/DiT/blob/main/diffusion/respace.py) | 少步采样的 timestep 映射 | 原始/压缩时间步映射、端点与重复检查 |
| [`train.py`](https://github.com/facebookresearch/DiT/blob/main/train.py) | ImageNet、VAE encode、DDP、AdamW、EMA、checkpoint | 单步 smoke test；EMA 更新；随机时间步范围；latent scaling；checkpoint schema |
| [`sample.py`](https://github.com/facebookresearch/DiT/blob/main/sample.py) | 权重、CFG、扩散循环、VAE decode | 固定 seed；cond/uncond batch 配对；输出范围和尺寸 |
| [`sample_ddp.py`](https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py) | 多卡生成 FID 样本 | rank 不重号、总样本数、`.npz` shape/dtype |

训练脚本的关键事实：VAE 把图像缩小 8 倍；默认 1000-step linear noise schedule；AdamW 学习率 \(10^{-4}\)、weight decay 0；维护 decay 0.9999 的 EMA；PyTorch DDP 启动方式由 README 给出（[train.py](https://github.com/facebookresearch/DiT/blob/main/train.py)、[训练命令](https://github.com/facebookresearch/DiT#training-dit)）。

工程限制也应写进学习任务，而非误认为官方代码是完整生产训练器：README 明列尚缺周期 FID/EMA sample、resume、AMP/bfloat16；Flash Attention 和 `torch.compile` 也只是建议项（[官方 Enhancements](https://github.com/facebookresearch/DiT#enhancements)）。仓库模型与权重为 CC-BY-NC，复用前要检查用途（[LICENSE](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt)）。

## 6. `Tencent-Hunyuan/HunyuanVideo`：可验证的代码结构与工程点

HunyuanVideo 官方仓库公开的是推理代码和模型权重，不是完整训练栈（[README / open-source plan](https://github.com/Tencent-Hunyuan/HunyuanVideo)）。其系统数据流由官方说明确认：causal 3D VAE 压缩视频，decoder-only MLLM 编码文本，Transformer 在压缩 latent 上从 Gaussian noise 生成结果，再由 VAE decode（[官方架构说明](https://github.com/Tencent-Hunyuan/HunyuanVideo#hunyuanvideo-overall-architecture)）。

可读代码层次：

- [`hyvideo/modules/models.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/modules/models.py)：`HYVideoDiffusionTransformer`，包含 dual-stream block 与 single-stream block；对应 README 所述先分别处理文/视频 token，再拼接融合的结构。
- [`hyvideo/modules/token_refiner.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/modules/token_refiner.py)：对 causal MLLM text feature 做额外双向 token refinement；这是官方为改善扩散条件而加入的模块（[README](https://github.com/Tencent-Hunyuan/HunyuanVideo#mllm-text-encoder)）。
- [`hyvideo/modules/posemb_layers.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/modules/posemb_layers.py)：视频 RoPE；推理侧依据 latent 的 \(T,H,W\) 和 patch size 建频率。
- [`hyvideo/vae/autoencoder_kl_causal_3d.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/vae/autoencoder_kl_causal_3d.py)：causal 3D VAE；README 声明时间/空间/通道压缩比分别为 4/8/16（[3D VAE 说明](https://github.com/Tencent-Hunyuan/HunyuanVideo#3d-vae)）。
- [`hyvideo/diffusion/schedulers`](https://github.com/Tencent-Hunyuan/HunyuanVideo/tree/main/hyvideo/diffusion/schedulers)：flow-matching scheduler 与 solver；CLI 暴露 `flow-shift`、reverse、solver。
- [`hyvideo/inference.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/inference.py)：装配 DiT、双 text encoder、VAE、FlowMatch scheduler 和 pipeline；还实现基于 xDiT 的 sequence parallel 切分。
- [`hyvideo/modules/fp8_optimization.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/modules/fp8_optimization.py)：FP8 linear 转换；[`tests/test_attention.py`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/tests/test_attention.py) 是仓库现有的 attention 测试入口。

运行资源不可忽略：官方表格给出 batch=1 时 720×1280×129 frames 约需 60 GB peak，544×960×129 约需 45 GB，并推荐 80 GB GPU；默认示例为 50 sampling steps、embedded CFG 6.0、flow shift 7.0（[requirements](https://github.com/Tencent-Hunyuan/HunyuanVideo#requirements)、[推理参数](https://github.com/Tencent-Hunyuan/HunyuanVideo#more-configurations)）。因此学习仓库时应优先做静态 shape/unit tests 和小 tensor module tests，而不是把全模型 smoke test 设为普通笔记本电脑的必过门槛。

建议补充的验证矩阵：

1. dual-stream → single-stream 前后文/视频序列长度守恒；
2. 3D RoPE 在 \(T,H,W\) 切分下的 shape、dtype/device；
3. causal VAE 的 \(4n+1\) 帧约束与 encode/decode 尺寸；
4. flow scheduler 的端点、shift 单调性与 Euler 更新；
5. CFG/embedded CFG 分开测试，避免混淆推理时双前向 guidance 与蒸馏进模型的 guidance embedding；
6. sequence parallel 切分/聚合可逆，以及不能整除时明确报错；
7. FP8 路径只在支持设备上集成测试，CPU CI 保留模块导入和参数验证。

## 7. 建议的阅读先后与证据等级

1. **必读数学底座**：[LDM](https://arxiv.org/abs/2112.10752) → [DiT](https://arxiv.org/abs/2212.09748) → 官方 [DiT code](https://github.com/facebookresearch/DiT)。
2. **视频最小扩展**：[Latte](https://arxiv.org/html/2401.03048)，先搞懂 factorized attention、3D/tube token 和时间位置编码。
3. **开放文本与系统化视频**：[CogVideoX](https://arxiv.org/abs/2408.06072)、[Open-Sora](https://arxiv.org/html/2412.20404)、[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)。
4. **新参数化与效率**：[LTX-Video](https://arxiv.org/html/2501.00103)，集中学习 rectified flow、VAE 压缩—token budget 联动。
5. **多任务、后训练与推理控制**：[Seedream 2.0](https://arxiv.org/html/2503.07703)、[Seedance 1.0](https://arxiv.org/html/2506.09113v1)。
6. **理解/推理驱动生成**：[OmniWeaving](https://arxiv.org/html/2603.24458v2)。
7. **能力观察但证据有限**：[Seedance 2.0 model card](https://arxiv.org/abs/2604.14148)；只记录 card 明示事实，等待更完整技术报告再补数学和代码结论。

资料中“论文报告的指标”只能证明作者在其设定下的结果；“仓库当前实现”只能证明当前公开 commit 的工程路径；未公开的训练数据、配方或产品内部结构不应由同系列名称推断。
