# Video DiT 生成“移动马赛克”：原因、排障顺序与零基础阅读路线

> 时期：Phase 2 — 失败后的论文回读与假设整理（2026-08-15）  
> 状态：一手来源参考；根因仍须由 Phase 3 实验决定  
> 顺序：[总索引](README.md) → [Phase 2 诊断](04-phase-2-moving-mosaic-diagnosis-and-next-actions.md) → 本文 → [Phase 3 根因实验记录](06-phase-3-root-cause-debug-and-retrain.md)  
> 本文的证据边界：论文与官方代码能够说明一个设计为何合理、公开实验用了什么规模、哪些指标适合回答什么问题；它们不能单独证明本地失败的唯一原因。下文用“来源结论”标记原论文直接支持的事实，用“本地推断”标记针对当前实验的判断。排障实验完成前，所有本地原因都只是有优先级的假设。

## 一句话结论

当前模型不是“生成了一个比较差的猫视频”，而是尚未跨过更基础的门槛：**从纯噪声生成可辨认的自然视频**。因此现在不应优先讨论未见文本组合、C1 泛化或是否记住了训练视频；应先证明 VAE 重建、扩散目标、采样公式、单样本过拟合、空间外观、时间运动、文本控制这七层链路逐层成立。

还有一个容易混淆的预期：无条件或文本条件扩散采样是从随机噪声产生新样本，并不应该在输出中保留某一条“原视频”的可见痕迹。真正异常之处不是“看不出原视频”，而是“看不出任何稳定对象或场景”。DDPM 将生成建模表述为学习固定加噪过程的反过程；LDM 只是把这个反过程搬到预训练自编码器的潜空间中，并没有把任务改成输入视频重建。[DDPM §2–3](https://arxiv.org/abs/2006.11239)；[LDM §3.2](https://arxiv.org/abs/2112.10752)

## 已知事实与需要修正的旧结论

当前本地实验的已知配置是：102 条训练视频、每条 16 帧、96×96；冻结的 `sd-vae-ft-mse` 逐帧编码为 12×12×4 latent；约 23.17M 参数的 Latte Variant-1 风格模型从零初始化；patch size 为 3，因此每帧只有 4×4 个空间 token；训练 30,000 updates；velocity 目标；无 learned variance；EMA 0.99；DDIM 50 步、CFG 2.5。

本次新增的 VAE 往返可视检查中，训练视频经过 `decode(encode(x))` 后，猫的外形和动作仍清楚可辨。**本地推断：VAE 不是这次整团马赛克的首要嫌疑，但仍需用全数据数值统计确认；问题更可能出在去噪模型没有学成、训练与采样不一致，或模型从零训练规模不足。**

先前报告把 0.977–0.979 的 CLIP 最近邻相似度描述为“near-frame-identical reconstruction”，这个证据链不成立，应撤回或降级。CLIP 的训练目标是让配对图文的嵌入余弦相似度高于未配对图文，它度量的是共享语义嵌入中的接近程度，不是逐像素、逐帧复制。[CLIP §2.2、Figure 3](https://arxiv.org/abs/2103.00020) 原论文在数据重叠分析中也对最近邻进行人工复核并为每个数据集设阈值，而不是把一个相似度数值直接当作重复证明。[CLIP §5](https://arxiv.org/abs/2103.00020)

因此：

- 高 CLIP 相似度最多是“值得人工审计的候选”，可能来自相似主体、颜色、背景或构图。
- 用户肉眼看到的是移动马赛克，就不能同时声称存在“清晰可见的逐帧复现”。
- 若要证明复制，应先时间对齐，再报告像素误差/PSNR、LPIPS 等感知距离、帧级匹配的一致性，并展示未经挑选的并排视频；必要时再做人工盲审。LPIPS 的原始工作表明深层特征可用于感知相似性判断，但它仍不是单独的“训练样本复制判定器”。[LPIPS paper](https://arxiv.org/abs/1801.03924)

## 可能原因：按目前的优先级排序

### P0：训练/采样参数化或实现不一致

**来源结论。** DDPM 给出前向加噪、噪声预测训练目标和反向采样之间的成套推导；其中任一处的时间索引、累计 alpha、方差或预测量转换不一致，都会使逆过程失效。[DDPM §2–3](https://arxiv.org/abs/2006.11239) Improved DDPM 又说明 learned variance、混合目标和采样步数是相互关联的设计，不能只凭训练 loss 下降推断采样正确。[Improved DDPM §2–4](https://arxiv.org/abs/2102.09672)

**本地推断。** 当前代码使用 velocity prediction，而 Latte 论文公式与官方训练路径以 epsilon prediction 为主；当前实现还关闭 learned variance/VLB，却使用自定义 DDIM-50。velocity 本身不是错误，但它扩大了“训练目标转换、CFG 合并、DDIM 更新三处是否完全一致”的排障面。现有单元测试证明局部代数和形状成立，不等于证明一条 1000→0 的完整反向链能恢复已知样本。

**修复/验证。** 先建立 oracle sampler 测试：固定一个真实 latent `z0` 和噪声 `epsilon`，构造任意 `zt`，将真实 epsilon（或由二者精确算出的 velocity）喂给每个反向步，验证最终恢复 `z0`。覆盖 1000 步 ancestral、50/100/250 步 DDIM、CFG=1、epsilon/velocity 两种预测。oracle 不通过时禁止继续训练。第一轮复现实验退回论文最近的 epsilon 目标、learned sigma + hybrid/VLB（或严格复制官方实现），一次只改一个变量。

### P0：模型学会降低平均 MSE，却没学会从高噪声建立全局结构

**来源结论。** DDPM 的简化目标是在随机时间步上预测噪声；Improved DDPM 指出不同时间步的目标方差与重要性并不相同，并提出 loss-second-moment importance sampling 来降低训练噪声。[DDPM Eq. 14](https://arxiv.org/abs/2006.11239)；[Improved DDPM §3.3](https://arxiv.org/abs/2102.09672)

**本地推断。** 单个平均 loss 从约 0.57 降至 0.12，只能说明训练分布上的平均 velocity MSE 下降。它可能主要来自容易的低/中噪声时间步、重复样本或局部纹理；从 `t≈999` 纯噪声建立猫的全局轮廓仍可能完全失败。当前“loss 下降”和“自由生成不可辨认”并不矛盾。

**修复/验证。** 对 train/validation 分别按 SNR 或时间步分桶记录 MSE、预测 `z0` 误差和解码后可视图；至少使用低、中、高噪声三档。做 teacher-forced denoising：从真实 `z0` 加噪到 `t=50/200/500/800/999`，再反推并解码。若低噪声可恢复、高噪声崩溃，就是全局生成能力/高噪声学习不足，而非 VAE 问题。

### P0：线性噪声日程可能不适合 12×12 latent

**来源结论。** Improved DDPM 发现原始线性 beta schedule 在 32×32、64×64 图像上会过快破坏信息，提出 cosine schedule 让信号衰减更平缓；论文也显示少步采样质量取决于训练目标、variance 与 sampler，不能把“DDIM 50 步”视为普适正确配置。[Improved DDPM §3.2、§4](https://arxiv.org/abs/2102.09672)

**本地推断。** 当前扩散空间只有 12×12，代码使用 `beta=1e-4→0.02` 的 1000 步线性 schedule。它不一定是失败原因，但属于比“再加训练步数”更值得先检查的高优先级假设：有效 SNR 分布可能让大量训练预算落在信息几乎已消失或过于容易的区间。

**修复/验证。** 画出 linear/cosine 的 `alpha_bar`、SNR 与真实 latent 在关键时间步的可视加噪结果；oracle 与 epsilon 基线通过后，用相同小数据、相同 updates 做 linear vs cosine 单变量对照。先用完整或高步数 ancestral sampler 建正确性基线，再比较 DDIM 50/100/250。

### P1：从零训练的数据、模型与算力规模不在公开成功实验的同一量级

**来源结论。** Latte 的公开实验使用 16 帧、256×256 视频，在 FaceForensics、SkyTimelapse、UCF101、Taichi-HD 等标准数据上训练；最小公开配置 Latte-S 为 32.48M 参数，最大为 673.68M，并使用 8 张 A100 80GB。[Latte §4.1, Tables 1–2](https://arxiv.org/abs/2401.03048) DiT 的核心实证结论不是“换成 Transformer 就自然生成得好”，而是样本质量随模型 Gflops 增长而改善，并在 ImageNet 上进行大规模训练。[DiT paper](https://arxiv.org/abs/2212.09748) 官方 DiT 仓库也明确把训练命令定位为 ImageNet 256×256/512×512，并提供经过大规模训练的 checkpoint。[facebookresearch/DiT](https://github.com/facebookresearch/DiT)

**本地推断。** 102 条异质自然视频对 23.17M 参数模型从零学习“世界长什么样、猫长什么样、如何运动、文字如何绑定”几乎没有冗余。30k updates 相当于反复看同一小集合，并不等价于获得大规模数据中的 30k 个独立批次。增加训练步数更可能强化局部拟合，而未必建立可泛化的视觉先验。

**修复/验证。** 不要先盲目把 30k 加到 100k。Improved DDPM 的 CIFAR-10/ImageNet-64 实验还观察到继续训练时 FID 可能恶化，说明生成质量应按 checkpoint 独立监测，不能用训练 loss 代替 early stopping。[Improved DDPM Appendix F–G](https://arxiv.org/abs/2102.09672) 先完成 1 条和 8–16 条视频的可辨认过拟合；若成功，再选一条路线：教学路线使用低熵、视觉统一、动作清楚的小数据集，先做无条件/类别条件视频；实用路线从官方图像 DiT/PixArt 或公开视频模型初始化，只训练/微调时间模块与条件层。Latte 自己验证了从预训练 DiT 初始化和图像—视频联合训练；后者在其表 3 中显著降低四个数据集的 FVD。[Latte §3.3.4、Table 3](https://arxiv.org/abs/2401.03048)

### P1：空间 token 过粗，且“空间外观”和“时间运动”同时从零学

**来源结论。** Latte 将逐帧 latent 切成时空 token，并比较空间/时间分解的四种 Transformer 变体；Variant 1 交替使用空间块和时间块。论文还专门消融 patch embedding、时间位置编码、信息注入、预训练和图像—视频联合学习，说明这些不是无关细节。[Latte §3.2–3.3](https://arxiv.org/abs/2401.03048)

**本地推断。** 当前 12×12 latent 使用 patch=3，每帧仅 4×4=16 个空间 token。对 96×96 的猫视频，这可能促使模型只保留低频色块；5 对空间—时间 block 又要同时学自然图像统计与运动。即使参数量接近 Latte-S，输入尺度、token 网格、训练数据和预训练状态都不同，不能由参数量接近推断能力接近。

**修复/验证。** 在单/小数据过拟合中比较 patch=1、2、3，保持其他因素固定；先做 T=1 的图像生成，确认能生成清楚单帧，再冻结或初始化空间模块，逐步扩展到 T=4、8、16。若 T=1 都是马赛克，不要调时间注意力；若 T=1 清楚而 T>1 崩溃，才检查时间位置编码、reshape 轴次序和时间 attention。

### P1：潜变量落在 VAE 解码器未见过的区域

**来源结论。** LDM 把训练分成感知压缩与潜空间生成两阶段；第一阶段的重建质量构成第二阶段可达到的上限，并存在压缩率与细节保真的折中。[LDM §3.1、§4.1](https://arxiv.org/abs/2112.10752) Latte 使用预训练 VAE 逐帧压缩视频，并在潜空间学习时空分布。[Latte §3.1–3.2](https://arxiv.org/abs/2401.03048)

**本地推断。** VAE 往返图清楚，说明 decoder 对真实编码 latent 有效；但去噪器产生的 latent 可能在均值、方差、尾部范围或空间相关性上偏离真实 latent 流形，于是同一个 decoder 输出色块。这里应区分“VAE 重建失败”和“生成 latent 离分布”。

当前缓存代码还使用 posterior `mode()`，而官方 DiT 训练代码使用 `latent_dist.sample()` 后乘 `0.18215`，解码时再除以同一因子；这不证明 `mode()` 有错，但它是一个应明确记录并单变量验证的分布差异。[DiT official `train.py`](https://github.com/facebookresearch/DiT/blob/main/train.py)；[DiT official `sample.py`](https://github.com/facebookresearch/DiT/blob/main/sample.py)

**修复/验证。** 统计真实训练 latent、加噪 latent、每个采样阶段的生成 latent 的 channel-wise mean/std/quantile/max；确认 scaling factor 编解码各应用且只应用一次。把最终生成 latent 与真实 latent 混合插值后解码，定位从何处开始失真。检查逐帧 decode 前后的轴顺序 `[B,C,T,H,W] ↔ [B*T,C,H,W]`，并保存每个时间步的 latent 与解码帧。小基线上再比较 posterior mode 与 sample，其他配置保持不变。

### P2：文本条件没有真正进入生成路径

**来源结论。** 原始 DiT 是 ImageNet 类别条件模型，不是文本生成器；LDM 的文本条件依靠 cross-attention；Latte 的主体实验也以类别/无条件视频为主，其 T2V 扩展不能由一个 CLIP embedding 层自动复现。[DiT](https://arxiv.org/abs/2212.09748)；[LDM §3.3](https://arxiv.org/abs/2112.10752)；[Latte §4.5](https://arxiv.org/abs/2401.03048)

**本地推断。** 当前冻结 CLIP 的 prompt retrieval 与 shuffled control 相同，只能说明现有生成没有可测的文本绑定；它不能区分“文本层坏了”和“整个视频本来就不可辨认”。在基础视觉质量失败时调 CFG 或写更长 prompt 通常不会解决主因。

**修复/验证。** 等无条件小数据能生成清楚视频后，再加入两个视觉差异极大的标签（例如猫/瀑布），测同一 seed 改条件是否显著改变输出；记录 cross-attention gate/梯度范数，做正确文本、空文本、打乱文本三组对照。最后再恢复自由文本和 C1 组合测试。

### P2：数据质量与任务熵过高

**来源结论。** 视频生成必须同时建模单帧视觉质量、时间一致性和生成多样性；FVD 因此用整段视频特征而不是只看逐帧指标。[FVD paper §1–3](https://arxiv.org/abs/1812.01717) Latte 选择标准化的 16 帧、256×256 clip，并在标准数据集上做系统实验。[Latte §4.1](https://arxiv.org/abs/2401.03048)

**本地推断。** 102 条来自不同来源、构图、压缩、镜头运动和动作定义的自然视频，对从零模型是一个高熵分布。短文本也可能只描述片段的一部分；同一动作的视觉差异可能大于不同动作。即使 manifest 无泄漏，也不代表数据足以学习生成。

**修复/验证。** 先输出全训练集 contact sheet 和运动统计，剔除黑帧、转场、严重压缩、主体过小、镜头运动压过主体运动的 clip；检查 caption 与 16 帧窗口是否逐条一致。教学过拟合集应先追求单一背景/主体尺度/动作，再逐级增加变化，而不是一开始追求开放世界覆盖。

## 一套不会浪费四小时的排障流程

每一级都设置“通过门”，未通过就停止进入下一层。

1. **媒体门（分钟级）**：直接播放抽样后的 16 帧训练 clip；检查帧序、FPS、RGB 范围、resize/crop 和时间方向。通过标准：随机抽 20 条均可辨认，caption 符合窗口。
2. **VAE 门（分钟级）**：对同一批数据保存原始/重建并排；算每帧 PSNR、LPIPS及相邻帧变化。通过标准：主体与动作肉眼可辨；失败则先换 VAE、分辨率或预处理。本次可视样本已初步通过，但仍需全量统计。
3. **数学门（秒至分钟）**：oracle 验证 `q_sample`、epsilon/velocity↔`x0`、DDPM/DDIM 完整链和边界时间步；检查 train/inference 的 beta schedule、scaling factor、prediction type 完全同源。通过标准：浮点容差内恢复已知 `z0`。
4. **单样本门（几十分钟）**：一条视频，关闭文本与 CFG，patch 尽量细，训练到从多个 seed 能输出可辨认的该视频分布；每 100–500 步固定 seed 采样。通过标准不是 loss 数字，而是清楚可辨。
5. **小集合门（小时级）**：8–16 条视觉差异大的视频，先类别条件；建立 train reconstruction、teacher-forced denoise、free sample 三栏。若 teacher-forced 好而 free sample 差，重点查高噪声/采样；二者都差，重点查优化/容量。
6. **空间门**：先 T=1，再 T=4/8/16。只有 T=1 清楚后才调时间模块。
7. **条件门**：同 seed、两个极不同条件、空条件和打乱条件；确认输出随条件变化且 CFG=1/2.5/5 的趋势合理。
8. **规模门**：通过以上门后才扩大数据、使用预训练、延长训练，并将单变量 ablation 预先写入配置。
9. **评估门**：先报告可辨认率与失败类型；样本量足够时再做 FVD。Latte 按 2,048 个 16 帧视频计算 FVD，当前 52 个生成的 FVD 即使能算也不宜与论文数值直接比较。[Latte §4.1](https://arxiv.org/abs/2401.03048) 文本准确率和 C1 组合测试应放在基础视觉门之后。

建议每次实验都保存：git commit、数据 hash、完整配置、训练曲线、按 timestep 分桶的验证误差、固定 seed 的中间 checkpoint 样本、VAE 往返、生成过程轨迹、最终 latent 统计。一次只改一个因素，否则“变清楚了”也无法定位原因。

## 下一步行动方案

### 第一阶段：先判定是 bug 还是学习规模（优先，预计半天）

- 补齐完整 oracle sampler 测试，尤其是 velocity + CFG + DDIM 的组合。
- 对 20 条训练视频做 VAE 往返量化与并排图，记录真实 latent 的分布。
- 对现有 checkpoint 做 `t=50/200/500/800/999` teacher-forced 恢复，并按 timestep 分桶 loss。
- 用同一初始噪声比较 DDIM 50/100/250 与 ancestral 1000；这只是诊断采样器，不能替代重训。
- 画 linear/cosine schedule 的 SNR，并在最小基线上做一次严格单变量对照。
- 更正证据报告：删除“CLIP 高相似度证明逐帧复制”，保留为“语义最近邻候选，人工观察不支持可见复制”。

判定规则：oracle 失败就是实现 bug；oracle 通过但单样本不能清楚过拟合，查模型/优化/latent 尺度；单样本成功而 102 条失败，主要是数据—容量—训练规模问题。

### 第二阶段：建立最小成功基线（预计半天到一天）

- 1 条 clip，无条件，epsilon prediction，CFG=1，patch=1 或 2。
- 8–16 条 clip，类别条件；先 T=1，再 T=4、8、16。
- 对比 raw 与 EMA；EMA 衰减先使用论文/官方常见的 0.9999，同时保留 raw checkpoint。当前 0.99 的有效平均窗口很短，是值得测试的差异，但不能仅凭论文断言它导致本次失败。Latte 报告 EMA 0.9999，并只用 EMA 结果。[Latte §4.1](https://arxiv.org/abs/2401.03048)
- 只有基线成功后，再逐一恢复 velocity、DDIM-50、文本 cross-attention 和 CFG。

### 第三阶段：选择真正的训练路线

- **为了学原理**：保留小模型，从低熵数据起步；目标是每层都可解释，不追求开放世界猫狗视频。
- **为了看到可用 T2V**：不要用 102 条视频从零训练世界模型；从 Latte/DiT 图像先验或官方公开视频模型初始化，冻结大部分空间/文本模块，训练时间模块或做参数高效微调。
- **为了研究未见组合**：先建立可辨认且能响应已见条件的基线，再扩大独立 source group 与每个二元组合的覆盖；预注册 C1 split 和人工盲评。否则 0% C1 只是在重复“基础生成失败”。

## 零基础阅读路线：每篇只回答一个调试问题

不要按发表时间机械通读。以下顺序先建立能读懂代码和失败现象的最短闭环，再进入视频架构。

1. **DDPM：扩散模型到底训练什么、采样什么？**
   阅读 [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) 的 §2、§3 和 Algorithm 1–2。读完应能自己写出 `q(x_t|x_0)`、epsilon MSE、由预测 epsilon 得到反向均值，并解释为什么训练 loss 下降不等于自由采样清楚。同步在本项目实现 oracle 单步和完整链测试。
2. **Improved DDPM：loss、方差和快速采样为什么不能混着抄？**
   阅读 [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) 的 §2–4，重点是 learned variance、hybrid objective、timestep importance sampling 和较少采样步。读完应能审计本项目“无 learned sigma + velocity + DDIM-50”是一个自定义组合，而不是 Latte 原样复现。
3. **DiT：Transformer 替换了哪一部分？**
   阅读 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) 的 §2–4，配合 [facebookresearch/DiT 官方代码](https://github.com/facebookresearch/DiT) 的 `models.py`、`diffusion/`、`train.py`、`sample.py`。重点追 patchify/unpatchify、timestep/class conditioning、adaLN-Zero、learned sigma、EMA 和 checkpoint sampling。官方复现表是在 ImageNet、global batch 256、400k steps 上比较，XL/2 最终实验更长；这能校准“23M 参数、102 条视频、30k steps”与论文成功结果的距离，而不是用来简单复制超参数。[DiT README: Training](https://github.com/facebookresearch/DiT#training-dit) 读完应能区分“DiT backbone”与“完整生成系统”。
4. **LDM：为什么先测 VAE，再测完整 DiT pipeline？**
   阅读 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) 的 §1 “Departure to Latent Space”、§3.1、§3.2、§4.1。读完应理解两阶段系统、VAE 重建上限、latent scaling，以及为什么真实 latent 可解码不代表生成 latent 也在分布内。这里是遇到 `4×12×12` latent 后的定向依赖补课，不遮住 DiT backbone 主线。
5. **Latte：图像 DiT 怎样变成视频 DiT？**
   阅读 [Latte](https://arxiv.org/abs/2401.03048) 的 §3.1–3.3、§4.1–4.3，配合 [Vchitect/Latte 官方仓库](https://github.com/Vchitect/Latte)。重点比较四种空间—时间分解、patch embedding、时间位置编码、预训练、图像—视频联合训练。读完应能设计 T=1→T=16 和 patch=1/2/3 的单变量实验。
6. **评估附录：指标回答的不是同一个问题。**
   阅读 [FVD 原论文](https://arxiv.org/abs/1812.01717) 的 §1–3，理解整段视频的质量、时间一致性和多样性；阅读 [LPIPS 原论文](https://arxiv.org/abs/1801.03924)，理解感知距离；阅读 [CLIP §2.2、§5](https://arxiv.org/abs/2103.00020)，理解语义嵌入、分布外限制与最近邻人工复核。读完应停止用单一 CLIP cosine 声称像素复制。

对完全没有读过论文的学习者，第一周只需完成前三篇的指定章节和对应小测试；第二周再进入 DiT 与 Latte。判断“读懂”的标准不是做完笔记，而是能从公式写出最小实现、能用 oracle 定位一处故意注入的 schedule/prediction-type 错误。

## 主要来源索引

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), 2020.
- Nichol & Dhariwal, [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672), 2021.
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752), 2022.
- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), 2023；[官方代码](https://github.com/facebookresearch/DiT).
- Ma et al., [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048), 2024；[官方代码](https://github.com/Vchitect/Latte).
- Unterthiner et al., [Towards Accurate Generative Models of Video: A New Metric & Challenges](https://arxiv.org/abs/1812.01717), 2018.
- Zhang et al., [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924), 2018.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.
