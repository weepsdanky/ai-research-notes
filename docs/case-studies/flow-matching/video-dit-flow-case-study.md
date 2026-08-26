# Case study：同一 Video DiT 的 DDPM vs Flow Matching

## 1. 研究问题

在 backbone、数据、条件、VAE、优化预算与评测噪声相同的情况下，把 DDPM epsilon objective + DDIM 替换成 linear CFM + ODE solver，是否能：

1. 更快达到 canonical 可辨识生成；
2. 用 10–30 NFE 保持主体、方向和时间连续性；
3. 在相同 wall time / NFE 下获得更好的视频质量。

这不是模型规模消融，也不用于声称真实视频泛化。

## 2. 必须锁死的控制变量

| 类别 | 固定内容 |
|---|---|
| 数据 | 同一 dataset/manifest hash、split、clip、预处理、batch 顺序 |
| 表示 | pixel 或同一份 VAE latent；相同 VAE ID/scaling |
| 条件 | 同一 structured IDs 或缓存 text embeddings；相同 dropout/CFG |
| 模型 | 同一 `VideoDiTConfig`、参数量、初始化 seed；两者都 `learn_sigma=False` |
| 优化 | batch、updates、AdamW、LR、weight decay、grad clip、EMA |
| 评测 | 同一 prompts、initial-noise tensors、seeds、输出 FPS 与指标代码 |
| 系统 | commit、device、dtype；分别记录训练 wall time 与 peak memory |

只允许 generative-process 列不同：

| DDPM baseline | Flow baseline |
|---|---|
| discrete linear-beta schedule | continuous linear interpolation |
| epsilon target | $\epsilon-z_0$ velocity target |
| uniform discrete timestep | uniform continuous $t$ |
| deterministic DDIM | deterministic Euler；Heun 为后续 solver 消融 |

## 3. 预注册实验矩阵

### Stage A：公式与 smoke gate

| ID | 内容 | 通过条件 |
|---|---|---|
| A0 | Flow property tests | endpoints、反演、方向、solver 全通过 |
| A1 | CPU dry-run | 1 update、finite loss、非零梯度、safe checkpoint |
| A2 | MPS 100 steps | 无 NaN/OOM；time mean/std 合理 |

### Stage B：canonical teacher

| ID | Objective | train time | sampler | eval budgets |
|---|---|---|---|---|
| B0 | DDPM epsilon | uniform discrete | DDIM | 10/20/30/50 NFE |
| B1 | linear CFM | uniform, shift=1 | Euler | 10/20/30/50 NFE |
| B2 | B1 checkpoint | — | Heun | 10/20/30/50 NFE 对应 5/10/15/25 steps |

不要比较 B0/B1 的 loss 数值。比较首次达到 gate 的 optimizer updates、samples seen、wall time，以及固定 checkpoint 的生成结果。

### Stage C：只在 B1 通过后做

| ID | time sampling | shift | 目的 |
|---|---:|---:|---|
| C0 | uniform | 1 | baseline |
| C1 | logit-normal $(0,1)$ | 1 | sampling 分布单变量 |
| C2 | uniform | 2 | noisy-time shift 单变量 |
| C3 | logit-normal $(0,1)$ | 2 | 只在 C1/C2 有正信号时组合 |

## 4. 指标与盲评

每个 checkpoint 固定至少 3 prompts × 5 seeds：

- **条件**：subject/color/scene 命中率，左右方向位移符号；
- **时间**：相邻帧 MSE、首尾 MSE、质心轨迹平滑度、chunk 前暂不测；
- **视觉**：无标签 A/B 盲评，记录 artifact、形状完整性与偏好；
- **效率**：NFE、实际 forward batch（CFG）、seconds/sample、peak memory；
- **复现**：checkpoint hash、配置、commit、eval request JSON。

小型 canonical overfit 不报告 FVD 为主结论；真实 held-out 数据阶段才加入分布指标，并保持相同样本数和 feature extractor。

## 5. 诊断顺序

生成全噪声或方向反转时，按顺序检查：

1. endpoints 是否是 data@0/noise@1；
2. target 是否为 $\epsilon-z_0$；
3. solver 的 $\Delta t$ 是否为负，网络输出是否被错误取负；
4. train/sample 的 `model_time_scale` 是否一致；
5. latent scaling/VAE contract 是否一致；
6. condition dropout 与 CFG 是否匹配；
7. 再看 time sampling、LR、EMA 和训练预算。

前六项没排除前，不扩大 backbone。

## 6. 结论模板

```text
问题：在相同 Video DiT 与预算下，Flow 是否优于 DDPM？
控制变量：列出 config/data/VAE/optimizer/eval hashes。
唯一主变量：DDPM+DDIM vs linear CFM+Euler。
结果：updates、wall time、NFE、条件/时间指标、盲评。
支持：本次证据直接支持什么。
不支持：不能外推到真实人物、音频条件、长视频或 LPM 17B。
下一步：只选一个——time sampling、shift、solver 或进入 audio condition。
```
