# DiT Learning Lab

这是与 DDPM 学习实验平行的最小 Diffusion Transformer 实现。主实验使用 CIFAR-10（32x32 RGB、10 类），比 MNIST 更适合观察 patch token、颜色纹理、类别条件和 classifier-free guidance。重点不是追求 SOTA FID，而是让论文中的每个接缝都能被单独阅读和测试。

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

## 运行

```bash
cd code/dit
uv sync --dev

# 不下载数据；只做一次合成数据参数更新
uv run python train_cifar10_dit.py --dry-run --hidden-size 16 --depth 1

# CIFAR-10 正式实验
uv run python train_cifar10_dit.py --epochs 100 --batch-size 128

# 全部测试
uv run pytest -q
```

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

正式复现 ImageNet-256 latent DiT 时，请直接使用 [facebookresearch/DiT](https://github.com/facebookresearch/DiT)。本目录是 pixel-space CIFAR-10 教学实现，不承诺与官方 checkpoint 的键名或数值完全兼容。
