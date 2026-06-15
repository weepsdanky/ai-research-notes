# LPM 1.0 Case Study: Large Performance Model

Paper: [arXiv 2604.07823](https://arxiv.org/abs/2604.07823) | Project: [large-performance-model.github.io](https://large-performance-model.github.io/)

## 论文简介

LPM 1.0（Large Performance Model）是一个统一的视频扩散系统，目标是生成逼真的、可交互的数字人（talking avatar）。它不仅仅做嘴型同步，而是生成包括面部表情、手势、肢体语言在内的全身表演视频，支持对话中的 speaker 和 listener 双角色，能实时流式生成。

**核心技术栈：**

- 骨干网络：Video Diffusion Transformer (DiT)
- 生成框架：Flow Matching（替代 DDPM/DDIM）
- 对齐方法：GRPO + DPO（从 RL 角度优化生成质量）
- 条件输入：音频、文本、情感信号（多模态条件化）
- 推理方式：Causal/Streaming（支持实时）
- 效率优化：Flash Attention

**为什么不能直接复现：** LPM 没有开源模型权重和训练数据，数据集是私有的大规模视频集合。本 case study 的目标是**系统理解 LPM 的每个技术组件**，并通过开源替代品和玩具实验建立真实的工程直觉。

---

## 学习策略

**方向：自下而上，逐组件击破**

把 LPM 的技术栈拆解成 5 个递进模块，每个模块包含：

1. 核心数学推导（数学先行）
2. 1-2 篇精读论文
3. 一个可运行的代码实验（从小到大）

**总周期：12 周（2-3 个月）**

---

## 模块拆解

### Module 1 — Diffusion 训练侧基础（Week 1-2）

**目标：** 补全训练侧知识。你已经理解推理侧（去噪采样），这里推导训练目标的数学来源。

**核心数学：**
- ELBO 推导：$\log p_\theta(x_0) \geq \mathbb{E}_q[\log p_\theta(x_0|x_1) - \sum_t D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))]$
- 为什么简化为 $L_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$
- 三种预测目标的关系与取舍：预测 $\epsilon$、预测 $x_0$、预测 $v$（velocity）

**精读论文：**
- DDPM (Ho et al. 2020)：打通训练侧完整推导
- Improved DDPM (Nichol & Dhariwal 2021)：学习可学习的方差和余弦 schedule

**代码实验（本地 RTX 4060）：**
- 用约 200 行 PyTorch 训练一个 MNIST 上的 DDPM → [train_mnist_ddpm.py](../../code/diffusion/train_mnist_ddpm.py)
- 目标：亲手写 forward diffusion、loss 计算、采样循环
- 参考实现：[minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- 预计训练时间：30-60 分钟

---

### Module 2 — Flow Matching（Week 3-4）

**目标：** 理解 LPM 实际使用的生成框架。Flow Matching 比 DDPM 数学更干净，是当前视频生成模型的主流选择（Sora、CogVideoX、LPM 均使用）。

**核心数学：**
- 从 ODE 视角看生成：$\frac{dx}{dt} = v_\theta(x, t)$，对比 DDPM 的 SDE 视角
- Conditional Flow Matching 的推导：为什么可以绕过 marginal probability path，直接对 conditional path 回归
- 目标函数：$L_{CFM} = \mathbb{E}_{t,x_0,x_1}\[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\]$（线性插值路径）
- Flow Matching vs DDPM：训练稳定性、采样步数、理论优雅性的对比

**精读论文：**
- Flow Matching for Generative Modeling (Lipman et al. 2022)：核心理论
- Rectified Flow (Liu et al. 2022)：线性路径的直觉解释
- minRF：约 100 行的极简实现，适合对照数学公式理解代码

**代码实验（本地 RTX 4060）：**
- 用 Flow Matching 重写 Module 1 的 MNIST 实验
- 对比两者：收敛曲线、采样步数、生成质量
- 预计训练时间：30 分钟

---

### Module 3 — Diffusion Transformer (DiT)（Week 5-6）

**目标：** 理解 LPM 的骨干网络。DiT 用 Transformer 替代 UNet，是所有现代大规模扩散模型（SD3、FLUX、CogVideoX、LPM）的架构基础。

**核心数学/概念：**
- Patch embedding：图像/视频如何变成 token 序列
- Adaptive Layer Norm (adaLN)：时间步 $t$ 和类别条件 $c$ 如何注入网络
  - $\text{adaLN}(x, t) = \gamma(t) \cdot \text{LayerNorm}(x) + \beta(t)$
- Scalability：DiT-S / DiT-B / DiT-L / DiT-XL 的参数量 vs FID 的 scaling law
- 与 UNet 的对比：感受野、归纳偏置、参数效率

**精读论文：**
- DiT (Peebles & Xie 2022)：核心架构论文，必读
- Scalable Diffusion Models with Transformers：scaling 分析

**代码实验（Kaggle 免费 T4）：**
- 在 CIFAR-10 上训练 DiT-S（33M 参数）
- 目标：理解 patch embedding、adaLN、attention 在实际代码中的样子
- 参考实现：[facebookresearch/DiT](https://github.com/facebookresearch/DiT)
- 预计训练时间：3-4 小时（T4）

---

### Module 4 — Video DiT + 多模态条件化（Week 7-9）

**目标：** 把图像 DiT 扩展到视频，理解 LPM 如何处理时序信息和多模态输入。

**核心数学/概念：**
- 3D patch embedding：$(T, H, W)$ 维度如何 tokenize
- 时序注意力 vs 全注意力：计算复杂度 $O(T^2 H^2 W^2)$ 的问题和解法（factorized attention、spatial-temporal decomposition）
- 多模态条件化：cross-attention 注入音频特征，adaLN 注入情感信号
- Causal attention mask：如何实现流式（streaming）生成

**精读论文/代码：**
- CogVideoX（清华/智谱）：开源 Video DiT，代码质量高，对照 LPM 论文理解设计决策
- Open-Sora Plan：另一个开源 Video DiT 参考实现
- LPM 论文 Section 3-4：对照以上开源代码，理解 LPM 的私有改进点

**代码实验（Vast.ai / RunPod 租 RTX 4090，约 $5-10）：**
- 跑通 CogVideoX 推理（需 16GB+ VRAM）
- 修改代码：可视化每层 attention map，理解时序信息如何流动
- 在小视频数据集（UCF-101 或 WebVid-10M 子集）上做推理，不做训练

---

### Module 5 — RL 对齐用于生成（Week 10-11）

**目标：** 把你已经掌握的 GRPO/DPO 知识迁移到扩散生成领域。这是 LPM 用于质量对齐的方法，也是你的已有优势所在。

**核心数学：**
- DDPO：把扩散采样链看作 MDP，每个去噪步骤是一个 action
  - $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[r(x_0)]$，用 REINFORCE 对采样链求梯度
  - 对比 GRPO：奖励归一化、KL 约束的相似性
- Diffusion-DPO：把偏好数据 $(x^w, x^l)$ 直接用于 diffusion loss
  - $L_{DPO} = -\log \sigma(\beta \cdot (r_\theta(x^w) - r_\theta(x^l)))$
- 两者的取舍：DDPO 需要 online rollout（贵），DPO 用 offline 数据（便宜但分布偏移）

**精读论文：**
- DDPO (Black et al. 2023)：Training Diffusion Models with Reinforcement Learning
- Diffusion-DPO (Wallace et al. 2023)：Diffusion Model Alignment Using Direct Preference Optimization
- LPM 论文 Section 5：GRPO/DPO 在视频生成中的具体应用

**代码实验（本地 RTX 4060 或 Kaggle T4）：**
- 用 DDPO 微调一个小型图像扩散模型（如 SD 1.5 的 LoRA），优化简单的可计算奖励（如 CLIP score）
- 参考实现：[huggingface/trl DDPO trainer](https://huggingface.co/docs/trl/ddpo_trainer)
- 预计训练时间：2-4 小时

---

### 整合实验（Week 12）

**目标：** 用开源组件拼一个最小化 talking-head pipeline，亲手触碰每个接缝。

**Pipeline 设计：**
```
音频输入
  → Wav2Vec2（音频特征提取）
  → 小型 DiT（Flow Matching 骨干，Module 2+3 的成果）
  → 视频帧序列
  → 简单渲染（直接输出 GIF 或 MP4）
```

**目标不是超过 LPM，而是：**
- 理解为什么 LPM 要用 causal attention（流式需求）
- 理解为什么 LPM 要用 Flow Matching 而非 DDPM（采样步数）
- 理解多模态条件化的工程复杂度

**参考基线：** [SadTalker](https://github.com/OpenTalker/SadTalker)（开源 talking-head，可作为对照基线）

**算力：** Vast.ai / RunPod / Nebius RTX PRO 6000，约 $10-15

---

## 算力规划

### 本机 RTX 4060（8GB VRAM）

适合：小型实验、代码调试、数据子集验证

| 任务 | 预计时间 |
|------|---------|
| Module 1：MNIST DDPM | 30-60 分钟 |
| Module 2：MNIST Flow Matching | 30 分钟 |
| Module 5：DDPO 小实验（如果用小模型） | 2-4 小时 |

**黄金原则：上云前先在本地用 10% 的数据跑通代码，确认逻辑没问题再租卡。**

### 免费资源：Kaggle（每周 30 小时 T4，16GB VRAM）

适合：中等规模训练，不需要付费

| 任务 | 预计时间 |
|------|---------|
| Module 3：CIFAR-10 DiT-S 训练 | 3-4 小时 |
| Module 5：DDPO 完整实验 | 2-4 小时 |

使用方式：把代码整理成 Notebook，session 结束前下载 checkpoint。

### 付费租卡（按需，预计总费用 $20-40）

三个平台对比：

| 平台 | 推荐 GPU | 价格 | 优势 | 适用场景 |
|------|---------|------|------|---------|
| **Vast.ai** | RTX 4090 (24GB) | ~$0.4/hr | 最便宜，社区供应商 | Module 4 推理、一次性实验 |
| **RunPod** | RTX 4090 (24GB) | ~$0.5/hr | UX 好，有持久 Volume | 需要保存 checkpoint 的实验 |
| **Nebius** | RTX PRO 6000 (48GB) | ~$1/hr（preemptible $0.95/hr）| 稳定，延迟低，适合大批量推理 | Week 12 整合实验、较大规模任务 |

**Nebius 注意事项：** 最低充值 $25，没有 RTX 4090，最接近的消费级选项是 RTX PRO 6000（48GB VRAM）。价格比 Vast.ai 贵 2-3 倍，但基础设施更可靠，适合不想折腾环境的场景。H100 实例（$2-4/hr）仅在有明确大规模需求时考虑。

**租卡操作流程：**
1. 本地验证代码（10% 数据，确认 loss 下降）
2. 上传到云端（git push 或直接粘贴 notebook）
3. 跑完整实验，下载 checkpoint 和日志
4. 关闭实例（**租卡最大的坑：忘记关机**）

---

## 参考资源索引

### 必读论文（按学习顺序）

1. DDPM — Ho et al. 2020
2. Improved DDPM — Nichol & Dhariwal 2021
3. Flow Matching for Generative Modeling — Lipman et al. 2022
4. Rectified Flow — Liu et al. 2022
5. DiT — Peebles & Xie 2022
6. CogVideoX — Yang et al. 2024
7. DDPO — Black et al. 2023
8. Diffusion-DPO — Wallace et al. 2023
9. LPM 1.0 — arXiv 2604.07823

### 关键开源代码库

- [minDiffusion](https://github.com/cloneofsimo/minDiffusion)：200 行 DDPM
- [minRF](https://github.com/cloneofsimo/minRF)：100 行 Flow Matching（同一作者）
- [facebookresearch/DiT](https://github.com/facebookresearch/DiT)：官方 DiT 实现
- [CogVideoX](https://github.com/THUDM/CogVideo)：开源 Video DiT，LPM 的最佳可读替代品
- [huggingface/trl DDPO trainer](https://huggingface.co/docs/trl/ddpo_trainer)：DDPO 参考实现
- [SadTalker](https://github.com/OpenTalker/SadTalker)：开源 talking-head 基线

### 辅助学习资源

- [Lilian Weng's blog — Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)：最好的 DDPM 数学综述
- [Flow Matching guide](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)：Flow Matching 直觉解释
- Karpathy's nanochat（见 README）：RL 对齐侧的工程参考
