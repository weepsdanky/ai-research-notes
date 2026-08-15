# Video DiT 生成“移动马赛克”：当前原因总结与下一步行动方案

> 状态：2026-08-15；对象：`c1-direct-train-v3`；结论：先停止扩大训练，回到可证伪的最小诊断。

## 1. 结论先行

当前模型没有生成足够可辨识的猫及动作。最准确的描述是：**视频具有时间变化，但语义结构坍缩为移动的低频色块/马赛克**。

上一份结果把 CLIP 最近邻的高相似度和缩小后联系表的 PSNR 描述成“近乎逐帧复制”，证据强度过头了。CLIP 衡量的是特征相似，缩小联系表的 PSNR 又主要受颜色、构图和低频轮廓影响；它们最多说明模型可能吸附到训练分布中的某些外观模式，**不能单独证明人眼可见的原视频复制**。用户看到的固定代表样本仍然是一团移动马赛克。因此后文把它改称为“记忆风险信号”，而不是“已证明复制”。

目前最可能的主因不是某一个坏参数，而是四件事叠加：

1. 空间 token 过粗：`12×12` latent 使用 `patch=3` 后，每帧只剩 `4×4=16` 个空间 token；一个 token 覆盖约 `24×24` 个输出像素。
2. 从零训练的文本—视频映射数据太少：只有 102 条训练视频，却要训练 23.17M 参数及新的 cross-attention。
3. 训练次数与数据规模失衡：30,000 updates × batch 8，相当于约 240,000 次样本呈现，即约 2,353 个“数据集遍历”；loss 下降不代表从纯噪声生成成功。
4. 当前 recipe 同时引入了太多未隔离变量：velocity target、EMA 0.99、DDIM-50、CFG 2.5、framewise image VAE、文本 cross-attention。任何一项失败都会在最终 MP4 中混成“马赛克”。

## 2. 我们已经知道什么

| 观察 | 证据 | 能推出什么 | 不能推出什么 |
|---|---|---|---|
| VAE 重建仍可辨识猫 | 同一 frozen SD VAE 的抽样重建约 20.14 dB，联系表中主体清楚 | VAE 不是马赛克的唯一主因；故障主要在 latent diffusion 生成之前/期间 | VAE 完全无损或适合视频时间一致性 |
| 生成视频确实在变化 | 52 段的 adjacent-frame MSE 均值 0.00954，endpoint MSE 0.03450 | 视频时间轴不是完全冻结 | 变化具有正确动作或主体 |
| 人工 C1 动作是 0/28 | 两名 prompt-hidden 评审；C1 triple 0/28，action 0/28，24/28 有分歧 | 语义可辨识度和动作绑定失败 | 单凭 loss 或 CLIP 可以替代人眼 |
| CLIP matched=shuffled=15.4% | 完整 52-video 网格 | 没有自动化文本检索证据 | CLIP 可以证明“复制”或视频动作正确 |
| 训练 loss 0.568→0.117 | first/last 500-update mean | 模型学会了训练目标的一部分 | 从标准高斯噪声出发能生成清晰视频 |
| latent 全局均值/标准差约 0.042/0.781 | 102 个缓存 latent 实算 | 没有明显 NaN、爆炸或数量级灾难 | scaling、SNR 或 sampler 一定最优 |

快速视觉对照：

- [VAE 重建联系表](artifacts/video-dit/c1-direct-train-v3/results/diagnostic-vae-reconstruction.png)：真实视频经过 `encode→decode` 后仍可辨识；
- [固定生成样本联系表](artifacts/video-dit/c1-direct-train-v3/results/diagnostic-generated-mosaic.png)：从噪声生成后只保留低频色块。

这把故障定位到：**VAE 能显示结构，但 denoiser 没有从高噪声恢复出正确结构**。

## 3. 可重复的“红灯”

当前最紧、确定、可自动运行的失败判据直接绑定两人盲评，而不是凭某一帧的主观印象：

```bash
cd code/video_dit
uv run --offline python - <<'PY'
import json
p = json.load(open('runs/c1-direct-train-v3/blind-human-evaluation.json'))
assert p['c1']['action_accuracy'] == 0.0
assert p['c1']['unanimous_triple_accuracy'] == 0.0
assert p['c1']['ambiguous_samples'] >= 20
print('RED: semantic recognizability failed — C1 action=0/28, triple=0/28, ambiguous=24/28')
PY
```

实际输出：

```text
RED: semantic recognizability failed — C1 action=0/28, triple=0/28, ambiguous=24/28
```

它不是一个快速训练测试，但它严格捕捉了用户所说的“看不出内容”。下一轮训练前还需要更快的分层门禁，见第 5 节。

## 4. 排序后的原因假设

每个假设都必须有一个会让它失败或成立的预测，避免看到 loss 后凭感觉解释。

| 排名 | 假设 | 当前依据 | 可证伪预测 | 若成立怎么修 |
|---:|---|---|---|---|
| 1 | `patch=3` 导致空间表征太粗 | `12×12→4×4` token；每帧仅 16 token | 保持数据与训练步数，改 `patch=2` 或单片过拟合用 `patch=1` 后，轮廓应显著更清楚 | 正式小模型用 `patch=2`；单片 debug 用 `patch=1` |
| 2 | 数据不足以从零学习文本到视频 | 102 videos 对 23.17M 参数；C1 action=0；CLIP matched=shuffled | teacher-forced 条件检索也低，或交换 prompt 对预测几乎无影响 | 先做 8–16 clip 平衡 overfit；之后使用预训练 image/video backbone，而不是继续盲目加参数 |
| 3 | 高噪声区域没有学好 | loss 是所有 timestep 的平均；自由生成必须从 `t=999` 开始 | 从真实 latent 加噪到 `t=100/500` 能还原，`t=900/999` 失败 | high-noise curriculum、更多有效 batch，或先用 epsilon baseline；按 timestep 报 loss |
| 4 | CFG 放大了一个不可靠的条件差值 | condition binding 弱，但推理 CFG=2.5 | 同 seed 下 CFG 1.0 比 2.5 更清楚，或不同 prompt 的 CFG residual 很小 | debug 阶段先 CFG=1；条件检索过门后再扫 1/1.5/2/2.5 |
| 5 | EMA 太短且采样 checkpoint 抖动 | decay 0.99 的有效窗口约 100 updates；DiT 官方 recipe 使用更慢 EMA | raw、EMA.99 与离线 EMA.999/.9999 质量差异大 | 同时保存 raw；实验 `.999`/`.9999`，不只看单一 EMA |
| 6 | DDIM-50 对当前欠拟合速度场过激 | 当前只锁 DDIM 50；velocity 是本地消融而非原始 DiT 主线 | 同 checkpoint 用 100/250 steps 明显改善 | 先以 epsilon + 100/250-step baseline 排除 solver；之后再恢复 velocity |
| 7 | VAE/latent 分布或 framewise VAE 限制 | VAE 重建约 20.14 dB；latent std 0.781 | VAE oracle 清楚但生成糊，说明不是主因；若相邻 VAE 重建闪烁才是时间 VAE 问题 | 保留当前 VAE做基础实验；高清阶段再换 causal 3D VAE |
| 8 | diffusion 数学/实现错误 | 337 个单测通过，finite/scaling/checkpoint 均锁定 | 单片在低噪声都无法重建，或 epsilon/velocity 代数对照不等价 | 回到 posterior、target、timestep mapping、no-clip 和 sampler 单测逐项修复 |

### 为什么“loss 很低”仍然会是马赛克

训练 loss 测的是：给定真实 `x₀` 加噪得到 `x_t` 后，模型能否预测 noise/velocity。自由生成测的是：从一个从未见过的标准高斯 `x_T` 开始，连续 50 次把小误差累积回 `x₀`。平均 loss 还会混合容易的低噪声步和决定全局结构的高噪声步。因此：

```text
平均 denoising loss 下降
  ≠ t=999 学得好
  ≠ sampler 的 50 次离散更新稳定
  ≠ 文本条件被使用
  ≠ VAE 解码后人眼可辨识
```

## 5. 一般遇到“移动马赛克”时怎么 debug

### Gate A：先拆开三条轴

分别回答，不要用一个指标代替另外两个：

1. **像素/latent 数值轴**：finite、范围、均值/方差、VAE 重建是否正常？
2. **扩散采样轴**：从低噪声能否还原？故障从哪个 timestep 开始？
3. **视频语义轴**：主体、动作、场景是否被人或独立分类器识别？

“在动”只证明第 3 条中的时间变化不为零，不证明语义正确。

### Gate B：VAE oracle（分钟级）

对 8 条真实视频做 `RGB→VAE encode→VAE decode`，保存输入/重建并排联系表，报告 PSNR/SSIM/LPIPS。若这里已经是马赛克，先修 VAE、尺寸、颜色范围或 scaling；不要训练 DiT。

当前状态：**通过基本可辨识门禁**，但仍应补齐统一的 8-clip 数值报告。

### Gate C：单视频、无文本 overfit（小时内）

只用一条视频，关掉 condition dropout 和 CFG；`patch=1`，小模型。需要同时通过：

- 从 `q(x_t|x₀)` 的 `t=100/500/900/999` 分别启动，能恢复该视频；
- 从纯噪声的固定 seed 启动，至少能生成稳定、可辨识的同一主体/场景；
- raw 与 EMA 均保存。

若单片都失败，问题在模型/扩散/sampler，不在数据多样性。

### Gate D：8–16 视频条件 overfit

选可明显区分的主体、动作、场景，冻结 prompt。要求：

- teacher-forced prompt retrieval ≥90%；
- 同 seed 交换 prompt 会产生明显、方向正确的变化；
- CFG=1 时已可辨识；
- 两名盲评的训练内 triple 达到预注册工程阈值（建议先 ≥80%）。

这里的百分比是工程放行阈值，不是论文声称的通用科学标准。

### Gate E：采样器矩阵，不重训

同一 checkpoint、prompt、seed，一次只改一个变量：

| 变量 | 值 |
|---|---|
| weights | raw / EMA |
| CFG | 1.0 / 1.5 / 2.5 |
| steps | DDIM 50 / 100 / 250；已有 paper-DDPM 时再对照 |
| start | 从真实 latent 的 100 / 500 / 900 / 999；纯噪声只用 999 |

如果所有组合都是马赛克，停止 sampler 调参，返回训练；如果只有 CFG 高时坏，修条件校准；如果只有 50-step 坏，修 solver/respacing。

### Gate F：最后才回到 102 条数据

只有 A–E 全部过门，才扩到当前数据。训练时必须按 timestep bucket 报 loss，并定期生成**固定 seed 的 raw/EMA、CFG=1**诊断，不允许看过样本后换 seed。

## 6. 下一轮具体行动方案

### P0：不训练，先把当前 checkpoint 解剖清楚

1. 运行 teacher-forced retrieval，分别报告 `t=100/500/900/999`，不要只给聚合值。
2. 对一个固定 IID prompt 生成 raw/EMA × CFG 1/1.5/2.5 × steps 50/100/250；固定 seed，不挑图。
3. 对真实 latent 做 partial-noise reconstruction，定位首次崩坏的 timestep。
4. 把 nearest-training 报告改为“特征相似/记忆风险”，不再写“已证明逐帧复制”。

### P1：最小训练实验，预计比再跑 30k 快得多

| 实验 | 数据 | 配置 | 只回答一个问题 |
|---|---|---|---|
| A | 1 clip | patch1、hidden128–192、2–3 pairs、epsilon、CFG1 | 网络与 sampler 能否生成清楚的一条视频？ |
| B | 同 1 clip | A 的配置，只把 patch1→2→3 | 马赛克是否主要来自空间 token 过粗？ |
| C | 8–16 clips | patch2、约 4–8M 参数、epsilon、EMA .999/.9999 | 小数据下能否学会条件区分？ |
| D | C | epsilon→velocity，其他不变 | velocity 是否让当前高噪声学习更难？ |
| E | C | CFG 1→1.5→2.5，仅推理变更 | guidance 从何处开始破坏清晰度？ |

不要同时改 patch、模型大小、target、EMA 和 sampler；否则即使成功也不知道原因。

### P2：推荐的下一版 102-video baseline

在 P1 通过后再运行：

- `latent 12×12×16`，`patch=2`（每帧 36 token，不再是 16）；
- hidden 192–256、4 spatial-temporal pairs，先控制在约 4–10M 参数；
- epsilon prediction 作为 DiT/Latte 主线 baseline；固定 variance；
- batch 8，保存 raw，EMA 至少 `.999`，另做 `.9999`；
- CFG=1 的固定诊断优先，条件绑定通过后才使用 CFG；
- 先 5k updates，每 500 步跑同一套固定诊断；只有指标持续改善才延长；
- 报 `t=0–199/200–399/.../800–999` bucket loss，而不是只报总 loss。

这套配置不是为了追求最好看，而是减少不确定变量。它通过后才能讨论更大模型、3D VAE、flow 或更多数据。

## 7. 从零开始的论文阅读顺序

现在不要先读 HunyuanVideo、Seedance、CogVideoX 或 flow matching。它们会一次引入太多组件，不能帮助定位当前马赛克。

### 第 1 篇：DDPM

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

只带着三个问题读：`q(x_t|x₀)` 怎么一次得到？网络预测什么？Algorithm 2 为什么能从噪声反推样本？先读图 2、公式 (4)、(11)、(14) 与 Algorithms 1–2。目标是能手写 `q_sample` 和一次 reverse step。

### 第 2 篇：Improved DDPM（为这次 debug 插入）

[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

先只读 noise schedule、learned variance、采样步数和 early stopping。它直接服务于“12×12 latent 是否被 linear schedule 太快破坏”“DDIM 50 是否太激进”和“为什么平均 loss 不代表样本质量”。暂时不实现所有 trick。

### 第 3 篇：DiT（仍先于 LDM）

[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) 与[官方代码](https://github.com/facebookresearch/DiT)

只回答：DiT 替换了扩散系统的哪一部分？重点看 patchify、timestep conditioning、adaLN-Zero、最终层和 patch size/compute 对质量的关系。然后在官方 `models.py` 追一次张量形状。此时先不深挖 VAE。

### 第 4 篇：LDM（定向补依赖）

[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

只回答：为什么不在 RGB 像素上扩散？画出 `RGB→E(x)→z→DiT→D(z)`，理解感知压缩的计算收益和重建上限。把本次“VAE oracle 清楚、自由生成马赛克”放进这张图中定位。

### 第 5 篇：Latte

[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048) 与[官方实现](https://github.com/Vchitect/Latte)

只回答：图像 DiT 如何多出视频时间轴？重点看 per-frame patch、空间/时间 attention 分解、位置编码与训练目标。把当前 `[B,C,F,H,W]` 每一步的形状写在纸上。

### 阅读时必须同步做的三个练习

1. 每篇只画一张数据流图，不做全文摘要。
2. 每遇到一个公式，在本仓库找到对应函数和一个测试。
3. 每读完一篇，用固定 seed 改一个变量并预测结果，再运行；预测错误比“看懂了”更有学习价值。

本地配套材料：

- [DDPM 数学讲义](module1-ddpm-math-derivation.md)
- [DiT 数学讲义](module3-dit-math-derivation.md)
- [DiT 论文—代码审计](dit-original-paper-implementation-audit.md)
- [Mac Video DiT 设计](video-dit-mac-design.md)
- [一手来源笔记](video-dit-moving-mosaic-primary-source-notes.md)

## 8. 下一步停止/通过条件

### 立即停止训练的条件

- VAE oracle 不可辨识；
- 单 clip 在 `t=100` 都不能还原；
- 生成出现 NaN/Inf 或 latent 尺度漂移；
- teacher-forced 条件交换几乎无差异；
- 增加 steps/降低 CFG 后仍无任何结构改善。

### 允许回到完整数据的条件

- 单 clip 从纯噪声生成可辨识；
- 8–16 clip 条件 overfit 达到预注册工程门禁；
- patch/target/EMA/sampler 各自完成单变量对照；
- 固定 seed 的 raw/EMA 结果随训练稳定改善；
- 人眼语义门禁与自动指标方向一致。

最重要的策略变化是：**下一步不是再训练更久，而是把一次四小时训练拆成数个可以在几十分钟内给出“哪一层坏了”的实验。**
