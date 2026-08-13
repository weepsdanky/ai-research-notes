# Image DiT Learning Lab

这是与 DDPM 学习实验平行、可以完整训练和交互推理的 Diffusion Transformer 实现。可选 MNIST、Fashion-MNIST 或 CIFAR-10；MNIST 用于快速验证生成闭环，CIFAR-10 用于观察 RGB patch token、颜色纹理、类别条件和 classifier-free guidance。重点不是追求 SOTA FID，而是让论文中的设计、训练技巧和采样过程都能读到、测到、看到。

## 对应关系

| 论文概念 | 代码 |
|---|---|
| 图像 patchify | `PatchEmbed` |
| 固定二维位置编码 | `get_2d_sincos_pos_embed` |
| 时间与类别条件 | `TimestepEmbedder`、`LabelEmbedder` |
| adaLN-Zero | `DiTBlock`、`FinalLayer` |
| token 还原为图像 | `DiT.unpatchify` |
| $q(x_t\mid x_0)$ 与 $L_{simple}$ | `DiffusionSchedule.q_sample`、`diffusion_loss` |
| classifier-free guidance | `DiT.forward_with_cfg` |
| DDPM 反向采样 | `p_sample`、`sample_loop` |
| Paper 250-step respaced DDPM | `space_timesteps`、`respaced_schedule` |
| learned variance / hybrid loss | `model_variance`、`diffusion_loss` |
| EMA 与断点恢复 | `update_ema`、`save_checkpoint`、`--resume` |
| DDIM 少步采样 | `ddim_sample_loop` |
| 推理轨迹 | `DiTInference.generate` |
| 交互 Web UI | `web_ui.py` |

## 原版设计与工程增强

尽量保留的 DiT / improved-DDPM 原味设计：官方 x→y 顺序的固定 2D sin-cos 位置编码、timestep MLP、class embedding + 10% label dropout、adaLN-Zero、GELU-tanh、零初始化输出层、AdamW、learned-range variance、raw VB、EMA、仅对 epsilon 前三通道做 CFG，以及保留原始 cumulative alpha 的 respaced ancestral DDPM。

两个 profile 把论文策略和教学便利明确分开：`pixel_teaching`（默认）沿用像素 clipping、DDIM、全 epsilon 通道 CFG、梯度裁剪和可选 AMP；`paper_compatible` 使用 raw VB、loss/sample no-clip、first-3 CFG、250-step respaced DDPM，并关闭梯度裁剪与 AMP。checkpoint 会写入这些策略以及 `space=pixel`、`vae_id=None`，因此 `paper_compatible` 只表示 diffusion recipe 对齐，**不表示已经实现 ImageNet latent VAE/FID 复现**。

默认训练目标是原始 DiT 使用的 epsilon prediction。`--prediction-type velocity` 提供明确的质量消融，训练、DDPM/DDIM 采样、checkpoint、resume 和推理会一致采用 v-parameterization；`paper_compatible` 固定为 epsilon，避免把消融误称为论文复现。旧 checkpoint 没有该字段时按 `epsilon` 加载。

为了单机学习增加的工程能力：CIFAR-10 像素空间输入、AMP、梯度累积、梯度裁剪、可选 warmup/cosine LR、周期 checkpoint、恢复训练、固定 seed 样本、DDIM、去噪轨迹和 Web UI。默认仍是官方式 constant LR。官方 ImageNet DiT 使用预训练 VAE 的 latent；这里用像素空间是为了让普通电脑可以从零训练，网络结构本身仍保持 DiT 的关键接缝。

本实验刻意没有伪装成大规模复现：多机 DDP、ImageNet latent 预提取、VAE 缩放常数、TF32 和 FID-50K 属于下一阶段的官方复现工程，应在读懂本目录后转到 `facebookresearch/DiT`。它们不改变 DiT block，却会决定吞吐量和基准指标是否可比。

## 运行

```bash
cd code/image_dit
uv sync --dev

# 不下载数据；只做一次合成数据参数更新
uv run python train_image_dit.py --dry-run --hidden-size 16 --depth 1

# 先学没有 trick 的核心闭环：patch → time → Transformer → epsilon → DDPM
uv run python train_minimal.py --dry-run --device cpu \
  --output-dir runs/minimal-dry-run

# MinimalDiT 的真实无条件 MNIST 训练（没有类别/CFG/adaLN-Zero/VB/EMA）
uv run python train_minimal.py --device mps --epochs 20 --timesteps 100 \
  --max-steps 5000 --output-dir runs/minimal-mnist

# 验证论文兼容 diffusion recipe（仍是本地像素数据，不会伪装成 latent 复现）
uv run python train_image_dit.py --profile paper_compatible --dry-run \
  --hidden-size 16 --depth 1

# CIFAR-10 正式实验（Apple Silicon；CUDA 可提高 batch size）
uv run python train_image_dit.py \
  --epochs 200 --batch-size 128 --hidden-size 256 --depth 8 \
  --num-heads 8 --patch-size 2 --learning-rate 1e-4 \
  --warmup-steps 1000 --sample-every 2000 --checkpoint-every 2000 \
  --output-dir runs/dit-cifar10

# v-prediction 质量消融（其余配置应与 epsilon 对照实验一致）
uv run python train_image_dit.py \
  --dataset mnist --prediction-type velocity --epochs 100 \
  --batch-size 128 --hidden-size 128 --depth 6 --patch-size 2 \
  --output-dir runs/mnist-velocity

# 快速质量闭环：每类固定一个 MNIST 原型，明确标注为 overfit 验收
uv run python train_image_dit.py \
  --dataset mnist --canonical-overfit --prediction-type velocity \
  --epochs 100 --max-steps 3000 \
  --batch-size 100 --hidden-size 64 --depth 3 --patch-size 2 \
  --no-learn-sigma --class-dropout-prob 0.2 --cfg-scale 5 \
  --learning-rate 2e-4 --ema-decay 0.99 \
  --sample-every 1000 --checkpoint-every 1000 \
  --output-dir runs/mnist-canonical-velocity

# 质量优先档：256 tokens（patch 2）+ 更大容量；建议至少 50k steps
uv run python train_image_dit.py \
  --epochs 200 --max-steps 50000 --batch-size 128 \
  --hidden-size 128 --depth 6 --num-heads 4 --patch-size 2 \
  --timesteps 1000 --learning-rate 1e-4 --warmup-steps 0 \
  --ema-decay 0.9999 --sample-every 5000 --checkpoint-every 5000 \
  --output-dir runs/dit-cifar10-quality

# epoch 级恢复；其余模型参数必须与 checkpoint 一致
uv run python train_image_dit.py \
  --epochs 100 --batch-size 128 --hidden-size 256 --depth 8 \
  --num-heads 8 --patch-size 2 --resume runs/dit-cifar10/dit_image.pt \
  --output-dir runs/dit-cifar10-resumed

# CLI：仓库自带的 MNIST 演示权重，输出图片与完整去噪 GIF
uv run python inference.py demo_artifacts/image_dit_inference.pt \
  --class-id 8 --cfg-scale 5 --seed 42 --sampler ddpm

# 自己训练后，推理请加载自动导出的安全 checkpoint
uv run python inference.py runs/dit-cifar10/dit_image_inference.pt \
  --class-id 3 --cfg-scale 2 --seed 42 --sampler ddim --sampling-steps 50

# checkpoint 若是 paper_compatible，这些默认自动读取；也可以显式覆盖
uv run python inference.py runs/paper/dit_image_inference.pt \
  --sampler paper_ddpm --sampling-steps 250 --no-clip-denoised

# 对比 EMA 与 raw 权重（理解 EMA 有效窗口）
uv run python inference.py runs/dit-cifar10/dit_image_inference.pt \
  --class-id 3 --no-use-ema

# 本机已训练的 canonical v-pred 演示：http://127.0.0.1:7860
uv run python web_ui.py --device mps

# 绘制 loss、VB、LR 和吞吐曲线
uv run python plot_training.py runs/dit-cifar10/step_log.csv \
  --output runs/dit-cifar10/training_curves.png

# 全部测试
uv run python -m pytest -q
```

训练会同时写出两个用途不同的文件：`dit_image.pt` 含 optimizer、scaler 与 RNG，仅用于从可信训练恢复；`dit_image_inference.pt` 只含张量与基础元数据，CLI/UI 以 `weights_only=True` 安全加载。不要将未知来源的 `.pt` 传给恢复训练。周期 checkpoint 在下次启动时从该 epoch 开头恢复，并非逐 batch 精确恢复。

仓库内已附 `demo_artifacts/image_dit_inference.pt` 及代表性样本，重新 clone 后可以直接推理；hash 见根目录 `ARTIFACTS.md`。它来自 254,276 参数、3,000 optimizer steps 的 MNIST canonical overfit + v-prediction 实验。完整 1,000-step DDPM 已从纯噪声生成多种清楚的数字形状，不再只是马赛克，但类别服从仍不均匀（例如某些 seed 下请求 8 会更像 7）。它是“模型学会图像流形”的闭环验收，不是完整 MNIST 泛化或高精度 class-conditional benchmark。完整恢复 checkpoint、summary 和日志仍位于被 Git 忽略的 `runs/mnist-canonical-velocity/`；新训练的 `summary.json` 与 checkpoint 会明确记录 `data_policy`。

历史 CIFAR demo 是 `hidden=64, depth=3, patch=4` 的 263k 参数教学模型，共 10,001 optimizer steps。它只能生成低频纹理，保留为失败诊断。短步 DDIM 对上述小型 MNIST checkpoint 也仍会留下高频网格；Web UI 因而默认选择实测更清晰的完整 DDPM，并保留 DDIM 供采样速度/质量消融。要学习“质量随 token/容量/训练量扩展”，再运行上面的完整数据质量档。

如果不使用 `uv`，在已安装 PyTorch、torchvision 和 pytest 的环境里直接运行相同 Python 命令即可。

## 测试分层

- 架构单元测试：patch 数量、输出形状、二维位置编码、learned-sigma 通道。
- 数学性质测试：forward diffusion 闭式、$x_0$ 反演、后验方差边界、$t=0$ 不加噪。
- 条件机制测试：null class、强制 label dropout、CFG 线性组合。
- 初始化测试：adaLN-Zero block 初始为恒等映射，整个 DiT 初始为零函数。
- 训练/采样测试：loss 反向传播、采样有限值、训练模式恢复、CLI dry-run 产物。

## MinimalDiT 与完整 DiT

| 能力 | `minimal_dit.py` + `train_minimal.py` | `dit.py` + `train_image_dit.py` |
|---|---:|---:|
| patch token + fixed 2D position + Transformer | ✓ | ✓ |
| timestep conditioning | 简单相加 | adaLN modulation |
| 类别条件 / classifier-free guidance | - | ✓ |
| adaLN-Zero residual gate | - | ✓ |
| learned-range variance + VB | - | ✓ |
| EMA | - | ✓ |
| 论文兼容 profile / respaced sampler | - | ✓ |

Minimal 版本只有无条件 epsilon simple loss 和固定 posterior variance；训练入口把完整 ancestral reverse loop 直接写开，适合逐行对公式。先确保它能反传和采样，再逐项迁移到完整版本，能避免把性能 trick 错当作 DiT 的定义。

## 建议实验顺序

1. 先运行 `pytest -q`，逐个阅读测试所表达的不变量。
2. 运行 `--dry-run`，检查 checkpoint、CSV 和 JSON 产物。
3. 用 `--timesteps 100 --epochs 3` 做快速 CIFAR-10 smoke run。
4. 完整训练后改变 `--patch-size`、`--depth`、`--hidden-size`，记录参数量、速度和样本。
5. 将 denoiser 从 DiT 换回仓库里的 UNet，保持 diffusion schedule 相同，做公平消融。

## 建议亲手做的 trick 消融

| 实验 | 改动 | 观察 |
|---|---|---|
| adaLN-Zero | 移除 modulation/gate 的零初始化 | 初始输出、早期 loss 稳定性 |
| Patch size | `2 → 4 → 8` | token 数、step/s、细节质量 |
| CFG | UI 中 `1 → 2 → 4 → 8` | 类别一致性与多样性/过饱和 |
| Label dropout | `0 → 0.1 → 0.2` | CFG 是否有效 |
| Sigma | `--learn-sigma` / `--no-learn-sigma` | MSE、VB、采样质量 |
| EMA | 推理加载 raw/EMA 权重 | 稳定性与清晰度 |
| DDPM/DDIM | UI 切换 sampler | 随机性、速度、轨迹差异 |
| Prediction target | `--prediction-type epsilon` / `velocity` | 不同噪声强度下的目标尺度与生成质量 |
| Model scale | hidden/depth/head 成比例增加 | 参数量、step/s 与质量趋势 |

`dit.py` 还提供 `dit_preset("DiT-S" | "DiT-B" | "DiT-L" | "DiT-XL", patch_size=2|4|8)`，对应论文的宽度、深度和 head 数。它不会隐式替你接 VAE；输入分辨率和通道仍需按 pixel/latent 实验显式设置。

正式复现 ImageNet-256 latent DiT 时，请直接使用 [facebookresearch/DiT](https://github.com/facebookresearch/DiT)。本目录是 pixel-space CIFAR-10 教学实现，不承诺与官方 checkpoint 的键名或数值完全兼容。
