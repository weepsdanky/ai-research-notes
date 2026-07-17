# DiT 学习计划：从 CIFAR-10 教学实现到视频生成系统

> 目标：沿用 [LPM case study](./5_lpm-case-study.md) 的“数学先行、论文精读、代码实验、可验证产物”方法，系统掌握 Diffusion Transformer，而不是只会调用 pipeline。

## 1. 最终能力目标

完成本计划后，应能独立回答并通过代码验证五类问题：

1. **概率层**：DiT 改变了 DDPM 的哪一部分，哪些 forward/reverse diffusion 公式完全没有变？
2. **架构层**：latent 如何变成 token，时间与类别如何通过 adaLN-Zero 控制每个 block，输出如何 unpatchify？
3. **训练层**：$epsilon$、$x_0$、$v$ prediction 的 target 如何构造，learned variance 和 CFG 分别解决什么问题？
4. **扩展层**：patch size、latent 压缩率、帧数和分辨率如何共同决定 token 数与 attention 成本？
5. **系统层**：图像 DiT 如何演化成 Latte、CogVideoX、Open-Sora、LTX-Video、Seedance 与 HunyuanVideo？公开资料支持哪些结论，哪些不能猜？

本仓库的交付物：

- 数学讲义：[module3-dit-math-derivation.md](./module3-dit-math-derivation.md)
- 一手资料底稿：[dit-primary-source-notes.md](./dit-primary-source-notes.md)
- 可运行代码：[code/dit/README.md](../../code/dit/README.md)
- PDF：`output/pdf/module3-dit-math-derivation.pdf`

## 2. 数据与实验梯度

MNIST 不作为主实验。采用三级数据/模型梯度，避免一开始就被 ImageNet 和视频算力挡住：

| 层级 | 数据与表示 | 目的 | 预计资源 |
|---|---|---|---|
| L0 测试 | 随机 32x32 RGB tensor | 形状、公式、梯度与 CLI smoke test | CPU，1 分钟内 |
| L1 教学训练 | CIFAR-10，32x32 RGB pixel space | 看见颜色、纹理、类别条件与 CFG；做 patch/depth 消融 | 8-16 GB GPU，数小时 |
| L2 论文复现 | ImageNet-256 的 VAE latent，官方 DiT checkpoint/code | 对齐原论文的 latent、learned variance、EMA、FID 评估 | 24 GB+ 推理；多卡训练 |
| L3 视频系统 | 小视频 clip 的 toy Latte；HunyuanVideo 静态/小张量测试 | 时空 token、factorized attention、3D VAE、flow | toy 可单卡；全模型按官方资源要求 |

CIFAR-10 的结果不应和 ImageNet-256 论文 FID 横向比较。它的价值是便宜、可重复、能做受控消融；原始 DiT 结论必须回到官方 ImageNet latent setting 验证。

## 3. 12 周路线图

每周默认 6-10 小时。每个模块都有“阅读 - 推导 - 实验 - 验收”四件套。

### Module A：DDPM 接口迁移（Week 1）

**问题：DiT 到底替换了什么？**

- 复习 $q(x_t\mid x_0)$、$q(x_{t-1}\mid x_t,x_0)$、$L_{simple}$。
- 画出 `NoiseSchedule -> noisy input -> denoiser -> loss/sampler` 的接口。
- 把 DDPM UNet 和 DiT 接在同一个 `DiffusionSchedule` 上，确认输入输出 shape 相同。
- 阅读：[DDPM 讲义](./module1-ddpm-math-derivation.md)、[LDM](https://arxiv.org/abs/2112.10752)。

**验收：** 不看笔记推导 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$；运行 forward 闭式与反演测试。

### Module B：最小 DiT 架构（Week 2-3）

**问题：一张图如何穿过 DiT？**

- 推导 patch token 数 $N=(H/p)(W/p)$ 和 attention 主成本 $O(N^2d)$。
- 实现/阅读 `PatchEmbed`、二维 sin-cos position、`DiTBlock`、`FinalLayer`、`unpatchify`。
- 手算 CIFAR-10 `32/4`：$8\times8=64$ tokens；ImageNet latent `32/2`：$16\times16=256$ tokens。
- 精读：[DiT](https://arxiv.org/abs/2212.09748)、[facebookresearch/DiT models.py](https://github.com/facebookresearch/DiT/blob/main/models.py)。

**验收：** 解释为什么 DiT 没有 UNet skip connection；通过 shape、位置编码、zero-init、learned-sigma 测试。

### Module C：条件与 adaLN-Zero（Week 4）

**问题：条件为什么可以控制每一个 residual branch？**

- 从 LayerNorm 写到 shift/scale modulation，再写到六路 adaLN-Zero 参数。
- 证明 gate 为零时 block 是恒等映射。
- 实现 class dropout/null token 和 classifier-free guidance。
- 实验 CFG scale $s\in\{0,1,2,4,8\}$ 对类别一致性与多样性的影响。

**验收：** 通过强制 label drop、CFG 边界和线性组合测试；能说明 $s=0$、$s=1$ 分别是什么。

### Module D：CIFAR-10 训练与消融（Week 5-6）

**问题：算力花在哪里，改动是否真的有因果效果？**

- 先跑 synthetic `--dry-run`，再跑 CIFAR-10 3 epoch smoke run，最后跑完整训练。
- 固定 seed 与训练预算，消融 patch size `8/4/2`、depth、hidden size、CFG。
- 每次记录参数量、token 数、step/s、peak VRAM、loss、固定 seed sample grid。
- 可选评价：用统一样本数计算 FID/KID；小样本时优先 KID，并明确置信区间。

**验收：** 一页实验表，能区分“参数更多”和“token 更多”的影响；至少保存一个可复现 checkpoint。

### Module E：官方 ImageNet latent DiT（Week 7）

**问题：教学版和论文版差了什么？**

- 阅读官方 `train.py`、`sample.py`、`gaussian_diffusion.py`、`respace.py`。
- 跑官方预训练 DiT-XL/2 sample；检查 VAE latent 4 channels、8 倍压缩和 `learn_sigma=True`。
- 对照教学代码：EMA、mixed MSE/VLB、250-step respacing、VAE decode、distributed FID。
- 记录官方仓库未覆盖的 resume、AMP、周期 FID 等工程缺口。

**验收：** 形成“教学实现 vs 官方实现”差异表；固定 seed 重复生成并记录环境与权重 hash。

### Module F：从图像到视频的最小桥梁（Week 8-9）

**问题：视频 token 为什么不能简单做一次全 attention？**

- 精读 [Latte](https://arxiv.org/html/2401.03048)，比较四种时空 attention variant。
- 在 toy clip 上实现 alternating spatial/temporal attention；验证 reshape 前后 token 守恒。
- 推导 full attention 与 factorized attention 复杂度。
- 阅读 [CogVideoX](https://arxiv.org/abs/2408.06072) 和 [Open-Sora](https://arxiv.org/html/2412.20404)，对比 3D VAE、expert adaLN、STDiT 和训练 curriculum。

**验收：** 通过时空 reshape、mask、position、2D 权重迁移的 shape 测试；画出三者架构差异。

### Module G：Flow/velocity 与高压缩 Video VAE（Week 10）

**问题：现代视频模型为什么常离开 DDPM posterior？**

- 精读 [LTX-Video](https://arxiv.org/html/2501.00103)，推导线性 path 与 velocity target。
- 对比 epsilon prediction 的离散反向链与 rectified flow ODE solver。
- 实验 Euler/Heun 单步误差、timestep shift、VAE 压缩率对 token budget 的影响。

**验收：** 测试 path 两端、target 符号、solver 更新；不混用 DDPM posterior 与 flow 更新公式。

### Module H：工业多模态与后训练（Week 11）

**问题：backbone 之外，生成质量还受什么控制？**

- [Seedream 2.0](https://arxiv.org/html/2503.07703)：双语 text encoder、字形条件、后训练。
- [Seedance 1.0](https://arxiv.org/html/2506.09113v1)：MMDiT、3D RoPE、多任务条件与级联 refiner。
- [OmniWeaving](https://arxiv.org/html/2603.24458v2)：MLLM reasoning 与 DeepStacking 条件注入。
- [Seedance 2.0 model card](https://arxiv.org/abs/2604.14148)：只记录公开 card 明示能力；不把 1.0 架构外推为 2.0 事实。

**验收：** 为每篇论文写“明确证据 / 作者报告 / 我的推断 / 未公开”四栏卡片。

### Module I：HunyuanVideo 系统 case study（Week 12）

**问题：一个开源视频生成器如何把全部组件装起来？**

- 阅读 [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) 的 dual/single stream blocks、token refiner、3D RoPE、causal VAE、flow scheduler、sequence parallel 与 FP8。
- 普通设备只做 import、配置和小 tensor module test；全模型推理遵循官方显存表，不把 80 GB 级 smoke test 设为本地必过项。
- 设计七类测试：序列长度守恒、RoPE shape、$4n+1$ 帧、flow shift、CFG 模式、并行切分可逆、FP8 capability gate。

**验收：** 交付一张端到端数据流图和风险/资源清单；能定位每个数学组件在仓库中的文件。

## 4. 每次实验的固定记录模板

```text
问题：本次只想验证什么？
控制变量：除一个旋钮外，哪些设置保持一致？
数据：版本、split、预处理、样本数。
模型：patch/depth/width/heads/参数量/token 数。
扩散：prediction type、schedule、训练/采样步数、CFG。
系统：commit、seed、device、dtype、峰值显存、wall time。
结果：loss、FID/KID 或 task metric、固定 seed 样本。
结论：支持/不支持什么；不能推出什么。
下一步：最小的后续实验。
```

## 5. 测试金字塔

| 层 | 必测内容 | 是否需要训练 |
|---|---|---|
| 公式性质 | $q$ 闭式、$x_0$ 反演、posterior、flow endpoints | 否 |
| 模块单元 | patch/unpatch、position、adaLN identity、label drop、CFG | 否 |
| 组合测试 | loss backward、采样 loop、checkpoint schema、mode restore | 一步 |
| 数据 smoke | CIFAR-10 小批次、归一化、label 范围 | 一步 |
| 实验回归 | 固定 seed 样本、loss/吞吐阈值、checkpoint resume | 短训练 |
| 大模型能力 | 官方 checkpoint sample、视频输出、FID/VBench | 按需 GPU |

判断标准不是“测试数量多”，而是每个测试都守住一个数学或工程不变量。

## 6. 算力与停止规则

- **CPU**：所有 unit/property tests 与 synthetic dry-run 必须通过。
- **本地 8-16 GB GPU**：CIFAR-10；先 100 steps，再 3 epochs，再决定完整训练。
- **24 GB GPU**：官方 DiT checkpoint 推理通常比视频模型现实；先检查 VAE、dtype 与 batch。
- **HunyuanVideo**：官方公开表明标准 129 帧推理需要约 45-60 GB 峰值并推荐 80 GB GPU；本计划默认做代码阅读和小张量验证。
- 任何付费训练前必须满足：本地单步 loss 有限、梯度非零、checkpoint 可读、采样 shape 正确、预算上限明确。

## 7. 阅读索引（按依赖关系而非发布日期）

1. [LDM](https://arxiv.org/abs/2112.10752)
2. [DiT](https://arxiv.org/abs/2212.09748) + [官方代码](https://github.com/facebookresearch/DiT)
3. [Latte](https://arxiv.org/html/2401.03048)
4. [CogVideoX](https://arxiv.org/abs/2408.06072) + [Open-Sora](https://arxiv.org/html/2412.20404)
5. [LTX-Video](https://arxiv.org/html/2501.00103)
6. [Seedream 2.0](https://arxiv.org/html/2503.07703)
7. [Seedance 1.0](https://arxiv.org/html/2506.09113v1)
8. [HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)
9. [OmniWeaving](https://arxiv.org/html/2603.24458v2)
10. [Seedance 2.0 model card](https://arxiv.org/abs/2604.14148)

