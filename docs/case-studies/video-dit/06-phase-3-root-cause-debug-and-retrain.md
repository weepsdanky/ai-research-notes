# Video DiT moving-mosaic：根因实验、修复与复训记录

> 时期：Phase 3 — 根因定位与最小复训（始于 2026-08-23）  
> 状态：M0 已完成；M1 在 5k 门禁停止（no-go）  
> 输入证据：[Phase 2 失败诊断](04-phase-2-moving-mosaic-diagnosis-and-next-actions.md)、`c1-direct-train-v3` safe inference checkpoint、固定训练 manifest  
> 下一门禁：1-clip 可达性与 3-prompt 可辨识基线已完成；后续结果见 [Phase 4 恢复记录](08-phase-4-meaningful-generation-recovery.md)  
> 顺序：[总索引](README.md) → [Phase 2 诊断](04-phase-2-moving-mosaic-diagnosis-and-next-actions.md) → [论文审计](07-phase-3-paper-debug-audit.md) → 本文

## 1. 本轮先修正了一个重要判断

“当前模型只会生成移动马赛克”对整个 checkpoint 来说过于绝对。固定 checkpoint、seed 和 prompt 后重新检查发现：

- 训练分布内的 IID prompt `a cat sitting by a window indoors`，EMA 在 CFG 1/2.5、raw 在 CFG 1 的固定对照中都能生成肉眼清楚的白猫；
- C1 `cat|playing|indoors` 的盲评仍是 action `0/28`、triple `0/28`，多数样本是低频块或语义不确定；
- 因此故障不是“VAE 或 sampler 对所有输入都坏了”，而是**模型可以记住/重现部分训练分布外观，却没有学会稳定动作和未见组合；高噪声反向轨迹对真实样本的恢复也很脆弱**。

这与用户看到的马赛克并不矛盾：用户查看的是失败的生成结果；新增固定 IID 对照说明问题具有条件与分布依赖性，缩小了根因范围。

## 2. 固定红灯

用户症状仍由两名 prompt-hidden 评审的结果定义，而不是由训练 loss 定义：

```bash
cd code/video_dit
uv run --offline python - <<'PY'
import json
p = json.load(open('runs/c1-direct-train-v3/blind-human-evaluation.json'))
assert p['c1']['action_accuracy'] == 0.0
assert p['c1']['unanimous_triple_accuracy'] == 0.0
assert p['c1']['ambiguous_samples'] >= 20
print('RED: semantic recognizability failed — C1 action=0/28, triple=0/28')
PY
```

该门禁只能由一轮新 checkpoint 的完整、预注册、双人盲评转绿。调试期间使用下面更快的 latent gate 判断是否值得进入昂贵盲评。

## 3. 新增的快速诊断工具

[`mosaic_debug.py`](../../../code/video_dit/mosaic_debug.py) 绑定 checkpoint 与 manifest 的 SHA-256，固定真实 latent、noise 和 timestep，同时报告：

1. 单次模型预测换算出的 `x0_mse`；
2. 从 `q(x_t|x0)` 启动 DDIM 后的 `reconstruction_mse`；
3. raw/EMA、latent 均值/标准差、token grid 和采样步数。

复现命令：

```bash
cd code/video_dit
uv run --offline python mosaic_debug.py \
  runs/c1-direct-train-v3/train/video_dit_inference.pt \
  data/formal_c1_expanded_peertube_v3/c1-direct-train-v3-latent.jsonl \
  --device mps --samples 8 \
  --timesteps 0,100,500,900,999 \
  --partial-timesteps 100,500,900,999 \
  --sampling-steps 50 \
  --output runs/c1-direct-train-v3/moving-mosaic-probe.json
```

扩散 oracle 回归还覆盖 epsilon/velocity × DDIM 10/100 steps，均在约 `2e-5` 误差内恢复精确 `x0`。因此核心 target 转换和 DDIM 更新没有显示足以解释马赛克的代数错误。

## 4. M0：不训练的根因定位结果

### 4.1 当前 30k velocity checkpoint 的恢复曲线

8 个训练样本的 direct `x0_mse`：

| timestep | raw | EMA |
|---:|---:|---:|
| 0 | 0.000097 | 0.000097 |
| 100 | 0.01053 | 0.00609 |
| 500 | 0.02449 | 0.01376 |
| 900 | 0.22773 | 0.24782 |
| 999 | 0.28545 | 0.30371 |

固定第一条训练视频、DDIM-50 partial denoising：

| start timestep | raw reconstruction MSE | EMA reconstruction MSE |
|---:|---:|---:|
| 100 | 0.02519 | 0.01740 |
| 500 | 0.05028 | 0.02735 |
| 900 | 0.93512 | 0.72154 |
| 999 | 0.93085 | 0.94141 |

解释：低/中噪声恢复成立，但接近纯噪声时反向轨迹离开真实样本流形。总 loss 下降没有覆盖这个失败。

### 4.2 sampler 不是首要根因

同一 raw checkpoint、同一 `q(x999|x0)`：

| reverse | reconstruction MSE |
|---|---:|
| DDIM 50 | 0.9027 |
| DDIM 100 | 0.9054 |
| DDIM 250 | 0.9066 |
| ancestral DDPM 1000 | 1.1370 |

增加 function evaluations 或切到 ancestral 并未恢复结构。因此不能只把 UI 的 50 改成 250 就称为修复。

### 4.3 历史小集合对照说明什么

使用同一 probe 重算历史 checkpoint：

| 数据/配置 | t999 partial MSE（EMA） | 解释边界 |
|---|---:|---|
| 3 clips，patch2，epsilon，10k | 0.0801 | 小集合能明显过拟合；同时改变了 pooled condition，不能单独归因 patch |
| 10 clips，patch3，epsilon，10k | 0.1906 | 比约 100 clips 容易，仍不能证明自由生成语义正确 |
| 100 clips，patch3，velocity，10k | 0.7900 | 扩大数据后高噪声恢复显著恶化 |
| 102 clips，patch3，velocity，30k | 0.9414 | 更大模型/更久训练没有解决，排除“只要继续训”的假设 |

这里最强的结论是：**当前 scratch 模型的训练容量没有随数据分布复杂度一起增长为有效的视频先验；继续增加 updates 会降低平均 loss，却不会自动修复高噪声端点或未见组合。** `patch=3` 仍需同数据单变量实验。

## 5. 当前根因结论

按证据强度排序：

1. **已确认的直接故障：** 高噪声进入反向链时，训练内真实 latent 的恢复从 `t≈900` 开始急剧崩坏；多步误差累积后形成低频外观。
2. **已确认的任务故障：** 模型对某些 IID 外观能清晰生成/记忆，但 C1 action 与 triple 为零；它没有学会可靠动作和 `cat + playing + indoors` 组合。
3. **最可能的系统根因：** 约 102 clips 不足以从零训练 23.17M 空间、时间和文本 cross-attention 参数形成可组合视频先验；30k updates 主要增加重复暴露和记忆。
4. **高概率放大项：** 每帧 16 token 的 patch3 空间网格、EMA `.99`、CFG 对弱条件 residual 的放大。
5. **当前未支持为主因：** VAE 完全损坏、latent scaling/clipping 错、DDIM 50 公式错、单纯采样步数不足。

## 6. 已实施的修复与单变量训练

代码侧先加入了两类不会改变生成算法的安全修复：

- 完整 DDIM oracle 回归，防止后续修改破坏 epsilon/velocity contract；
- checkpoint-bound `mosaic_debug.py`，以后每个 checkpoint 都必须画出 timestep recovery curve。

M1 第一项严格复用历史 10-clip epsilon run 的数据、hidden、pairs、optimizer 与 EMA，计划最多 10k steps，只改 `patch=3→2`：

```bash
cd code/video_dit
uv run --offline python train.py \
  --dataset real --manifest data/commons_g0/train_tokens_v2_latent_10clip.jsonl \
  --text-conditioning tokens --text-embedding-size 512 \
  --space latent --no-learn-sigma --mode faithful --device mps \
  --max-steps 10000 --batch-size 5 \
  --frames 16 --image-size 12 --patch-size 2 \
  --hidden-size 256 --num-heads 8 --num-pairs 4 \
  --learning-rate 1e-4 --weight-decay 0 --max-grad-norm 1 \
  --prediction-type epsilon --ema-decay 0.999 \
  --horizontal-flip-prob 0 --sample-every 0 \
  --checkpoint-every 2500 --seed 42 \
  --output-dir runs/mosaic-debug-patch2-epsilon-10clip
```

选择 epsilon 不是因为已经证明它优于 velocity，而是因为它与现存 patch3 epsilon checkpoint 构成真正的 patch 单变量对照，并接近 DDPM/DiT/Latte 主线 target。target 对照必须在 patch 结果之后另做。

### 6.1 5k 门禁结果

训练在写出 step 5000 checkpoint 后主动停止；日志因人工中断记到 step 5066。前 500 步 loss 均值为 `0.24494`，最后 500 步为 `0.08112`，说明优化在进行，但 loss 下降不等于生成可辨识。

DDIM-50 partial recovery：

| start timestep | raw MSE | EMA MSE |
|---:|---:|---:|
| 100 | 0.01981 | 0.02759 |
| 500 | 0.19966 | 0.26340 |
| 900 | 0.40506 | 0.46384 |
| 999 | 0.47489 | 0.52765 |

与历史 patch3 epsilon 10k EMA `0.1906` 不能做严格的同 step 数值比较，但 patch2 并没有在 5k 显示出足以跨过门禁的改善。固定 `a cat sitting indoors` / seed 17 / CFG 1 的 raw 与 EMA 视频仍是低频块状纹理，只有模糊的猫形轮廓。因此 `patch=3` 不是单一根因，按预注册 5k gate 停止而不继续消耗到 10k。

训练内 prompt retrieval 进一步定位了噪声端的条件信号消失：

| timestep | raw top-1 | EMA top-1 |
|---:|---:|---:|
| 100 | 10/10 | 10/10 |
| 500 | 10/10 | 10/10 |
| 900 | 6/10 | 6/10 |
| 999 | 2/10 | 3/10 |

`t=999` 的 epsilon retrieval 存在结构性不利（目标接近纯 noise），不能单独用来宣判 target 优劣；但 `t=900` 已明显从 100% 降到 60%，与 partial reverse 的崩坏方向一致。

## 7. go / no-go 结论

本轮结论是 **no-go：不将 patch2 直接扩到完数据，也不继续训到 10k 来追逐已失败的可视门禁**。下一步按以下顺序：

1. 用 1 clip 、无 CFG 训到从 `t=900/999` 可恢复，作为 optimizer/model/schedule 的可达性门禁；
2. 在 1 clip 仍失败时，只做 schedule/target 单变量实验，不增加模型或数据；
3. 若 1 clip 可恢复而 10 clips 失败，将主路线转为预训图像/视频空间先验，再学时间与文本层；
4. 任何新 checkpoint 都必须同时通过 recovery curve、固定 prompt/seed 可视性和动作识别，不再用平均 loss 代替结果。

即使小集合转绿，也不能声称 unseen composition 已解决。正式 C1 仍需满足数据门禁，并重新执行预注册完整 grid 和两名独立盲评。
