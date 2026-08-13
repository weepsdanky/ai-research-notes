# DiT Learning Lab

这是与 DDPM 学习实验平行、可以完整训练和交互推理的 Diffusion Transformer 实现。主实验使用 CIFAR-10（32x32 RGB、10 类），比 MNIST 更适合观察 patch token、颜色纹理、类别条件和 classifier-free guidance。重点不是追求 SOTA FID，而是让论文中的设计、训练技巧和采样过程都能读到、测到、看到。

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
| learned variance / hybrid loss | `model_variance`、`diffusion_loss` |
| EMA 与断点恢复 | `update_ema`、`save_checkpoint`、`--resume` |
| DDIM 少步采样 | `ddim_sample_loop` |
| 推理轨迹 | `DiTInference.generate` |
| 交互 Web UI | `web_ui.py` |

## 原版设计与工程增强

尽量保留的 DiT / improved-DDPM 原味设计：固定 2D sin-cos 位置编码、timestep MLP、class embedding + 10% label dropout、adaLN-Zero、GELU-tanh、零初始化输出层、AdamW、learned-range variance、`RESCALED_MSE`、EMA 和仅对 epsilon 做 CFG。

为了单机学习增加的工程能力：CIFAR-10 像素空间输入、AMP、梯度累积、梯度裁剪、可选 warmup/cosine LR、周期 checkpoint、恢复训练、固定 seed 样本、DDIM、去噪轨迹和 Web UI。默认仍是官方式 constant LR。官方 ImageNet DiT 使用预训练 VAE 的 latent；这里用像素空间是为了让普通电脑可以从零训练，网络结构本身仍保持 DiT 的关键接缝。

本实验刻意没有伪装成大规模复现：多机 DDP、ImageNet latent 预提取、VAE 缩放常数、TF32 和 FID-50K 属于下一阶段的官方复现工程，应在读懂本目录后转到 `facebookresearch/DiT`。它们不改变 DiT block，却会决定吞吐量和基准指标是否可比。

## 运行

```bash
cd code/dit
uv sync --dev

# 不下载数据；只做一次合成数据参数更新
uv run python train_cifar10_dit.py --dry-run --hidden-size 16 --depth 1

# CIFAR-10 正式实验（Apple Silicon；CUDA 可提高 batch size）
uv run python train_cifar10_dit.py \
  --epochs 200 --batch-size 128 --hidden-size 256 --depth 8 \
  --num-heads 8 --patch-size 2 --learning-rate 1e-4 \
  --warmup-steps 1000 --sample-every 2000 --checkpoint-every 2000 \
  --output-dir runs/dit-cifar10

# 质量优先档：256 tokens（patch 2）+ 更大容量；建议至少 50k steps
uv run python train_cifar10_dit.py \
  --epochs 200 --max-steps 50000 --batch-size 128 \
  --hidden-size 128 --depth 6 --num-heads 4 --patch-size 2 \
  --timesteps 1000 --learning-rate 1e-4 --warmup-steps 0 \
  --ema-decay 0.9999 --sample-every 5000 --checkpoint-every 5000 \
  --output-dir runs/dit-cifar10-quality

# epoch 级恢复；其余模型参数必须与 checkpoint 一致
uv run python train_cifar10_dit.py \
  --epochs 100 --batch-size 128 --hidden-size 256 --depth 8 \
  --num-heads 8 --patch-size 2 --resume runs/dit-cifar10/dit_cifar10.pt \
  --output-dir runs/dit-cifar10-resumed

# CLI：输出 sample.png 与完整去噪 GIF
uv run python inference.py runs/dit-cifar10/dit_cifar10.pt \
  --class-id 3 --cfg-scale 2 --seed 42 --sampler ddim --sampling-steps 50

# 对比 EMA 与 raw 权重（理解 EMA 有效窗口）
uv run python inference.py runs/dit-cifar10/dit_cifar10.pt --class-id 3 --no-use-ema

# Web UI：http://127.0.0.1:7860
uv run python web_ui.py runs/dit-cifar10/dit_cifar10.pt

# 绘制 loss、VB、LR 和吞吐曲线
uv run python plot_training.py runs/dit-cifar10/step_log.csv \
  --output runs/dit-cifar10/training_curves.png

# 全部测试
uv run pytest -q
```

checkpoint 使用 PyTorch pickle 格式，只加载自己训练或可信来源的 `.pt` 文件。周期 checkpoint 在下次启动时从该 epoch 开头恢复，并非逐 batch 精确恢复。

训练产物位于 `runs/` 且默认被 Git 忽略。本机演示模型的最终路径是 `runs/cifar10-demo/dit_cifar10.pt`；若要在另一台机器使用，请显式复制该 checkpoint 和对应的 `summary.json`/日志，不要指望代码提交包含模型权重。

本机随交付运行的 demo 是 `hidden=64, depth=3, patch=4` 的 263k 参数教学模型，共 10,001 optimizer steps。它能显示 loss 收敛、EMA/raw 差异和从噪声到低频 CIFAR 纹理的过程，但类别对象仍不可辨识；要学习“质量随 token/容量/训练量扩展”的 trick，请运行上面的质量优先档。

如果不使用 `uv`，在已安装 PyTorch、torchvision 和 pytest 的环境里直接运行相同 Python 命令即可。

## 测试分层

- 架构单元测试：patch 数量、输出形状、二维位置编码、learned-sigma 通道。
- 数学性质测试：forward diffusion 闭式、$x_0$ 反演、后验方差边界、$t=0$ 不加噪。
- 条件机制测试：null class、强制 label dropout、CFG 线性组合。
- 初始化测试：adaLN-Zero block 初始为恒等映射，整个 DiT 初始为零函数。
- 训练/采样测试：loss 反向传播、采样有限值、训练模式恢复、CLI dry-run 产物。

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
| Model scale | hidden/depth/head 成比例增加 | 参数量、step/s 与质量趋势 |

`dit.py` 还提供 `dit_preset("DiT-S" | "DiT-B" | "DiT-L" | "DiT-XL", patch_size=2|4|8)`，对应论文的宽度、深度和 head 数。它不会隐式替你接 VAE；输入分辨率和通道仍需按 pixel/latent 实验显式设置。

正式复现 ImageNet-256 latent DiT 时，请直接使用 [facebookresearch/DiT](https://github.com/facebookresearch/DiT)。本目录是 pixel-space CIFAR-10 教学实现，不承诺与官方 checkpoint 的键名或数值完全兼容。
