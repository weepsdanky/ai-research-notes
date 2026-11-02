# Video DiT moving-mosaic：可辨识生成恢复记录

> 时期：Phase 4 — 最小可辨识视频模型（2026-08-23）  
> 状态：最低工程门禁通过；仅是 3-prompt training-distribution overfit，不是开放词表或 unseen-composition 成功  
> 输入证据：[Phase 3 根因实验](06-phase-3-root-cause-debug-and-retrain.md)、[Phase 3 论文审计](07-phase-3-paper-debug-audit.md)  
> 下一门禁：在不丢失可辨识性的前提下扩到 8–10 个 prompt；若再次退化，停止 scratch 扩展并转向预训空间先验  
> 顺序：[总索引](README.md) → [Phase 3 根因实验](06-phase-3-root-cause-debug-and-retrain.md) → 本文

## 1. 结论先行

moving-mosaic 不是一个单独的 sampler 或 VAE bug。当每个视觉模式获得足够暴露，并使用 `patch=1` 与高噪声→uniform 分阶段课程时，同一个 1.60M 参数的文本条件 Video DiT 能生成：

- 可辨识的室内猫；
- 可区分的海滩和池塘场景；
- 池塘中随时间移动的狗；
- 同 seed 下不同 prompt 对应的三种不同生成分布。

海滩狗仍较小，“奔跑”动作不如“游泳”清楚。因此本轮证明了本地训练栈能产生有意义画面，没有证明泛化或动作语义已充分解决。

## 2. 最小数据与模型

只使用了既有 10-clip manifest 中的三条，没有增加新数据：

1. `a cat sitting indoors`；
2. `a dog running on a beach`；
3. `a dog swimming in a pond`。

manifest SHA-256：`d32093da75b6547c803575f7667c7c0cc446f85603fb6d04a40e88e9a1109de4`。

模型固定为：16 frames、12×12 SD-VAE latent、patch1、hidden128、4 heads、2 组 spatial/temporal blocks、CLIP token cross-attention、epsilon target、fixed variance、CFG1、约 1.60M 参数。

## 3. 诊断链

| 阶段 | 数据/训练 | 关键结果 |
|---|---|---|
| A | 1 clip，patch1，uniform 5k | raw `t900=0.102`、`t999=0.286`；自由生成出现可辨识室内猫 |
| B | 10 clips，同 recipe 5k | `t900=1.535`；t900 retrieval raw `1/10`；画面再次块状化 |
| C | B 后 uniform 5k control | raw `t999=2.085`；改善缓慢 |
| D | B 后 high-noise 5k | raw `t999=0.983`，但 t900 retrieval 仍 `1/10` |
| E | 3 clips，batch3，uniform 5k | 三场景可区分；EMA t900 retrieval `3/3` |
| F | E 的 EMA → high-noise 2.5k | raw `t999=0.469`；EMA 全 timestep retrieval `3/3`，但运动趋静态 |
| G | F 的 EMA → uniform 2.5k | raw `t999=0.265`；场景与运动均改善 |
| H | G 的 EMA → uniform 2.5k | raw `t999=0.141`；当前最佳 checkpoint |

这个链条支持的根因分类是：

1. **工程栈可学：** 1 clip oracle 和自由生成可转绿，排除核心代数、VAE 或 optimizer 完全损坏。
2. **高噪声信号不足：** high-noise 课程明显改善纯噪声恢复，但单独使用会削弱动态细节。
3. **scratch 先验/有效暴露是主瓶颈：** 1 clip 通过而 10 clips 在同 updates 下失败；每个 clip 匹配约 5k 次暴露后，3-clip 模型转为可辨识。
4. **patch3 是放大项而非唯一根因：** patch1 有助于细节，但 10 clips patch1 仍会失败。
5. **1.6M 并非小到无法生成：** 它能记住并区分三种分布，但还不具备扩展到宽数据分布的先验。

## 4. 当前最佳本地产物

safe inference checkpoint：

```text
code/video_dit/runs/meaningful-3clip-patch1-final-uniform-ft2500/video_dit_inference.pt
SHA-256: 7e3c8d4ad3558c56879217ac963411ee2f9c0bfb9375cbdd20b443394152b675
```

固定 seed 17、CFG1、DDIM50 的 raw/EMA MP4 和 contact sheet 位于同目录的 `inference-{raw,ema}-p{1,2,3}/`。`runs/` 是本地 ignored 训练产物；新 checkout 需要按本文 recipe 重训或另行发布 checkpoint。

DDIM-50 partial recovery：

| timestep | raw MSE | EMA MSE |
|---:|---:|---:|
| 100 | 0.0134 | 0.0152 |
| 500 | 0.0675 | 0.0950 |
| 800 | 0.1110 | 0.1610 |
| 900 | 0.1086 | 0.1532 |
| 950 | 0.1152 | 0.1665 |
| 999 | 0.1413 | 0.2063 |

EMA teacher-forced retrieval 在 `100/500/800/900/950/999` 全部为 `3/3`。raw 除 `t999=2/3` 外均为 `3/3`。

## 5. 运动证据与边界

固定 raw 生成的 adjacent-frame MSE / endpoint MSE / 水平 motion-centroid displacement：

| prompt | adjacent MSE | endpoint MSE | displacement |
|---|---:|---:|---:|
| cat sitting indoors | 0.0381 | 0.1282 | 11.72 px |
| dog running on a beach | 0.0149 | 0.0036 | 1.99 px |
| dog swimming in a pond | 0.0121 | 0.0336 | 7.26 px |

这些数值只证明帧间变化，不单独证明动作语义。肉眼检查中，池塘狗的位移最清楚，室内猫有姿态变化，海滩狗的“奔跑”仍是三者中最弱的一项。

## 6. 下一步

1. 将本 checkpoint 作为可辨识性 regression baseline，不再用 moving-mosaic checkpoint 作默认 UI 模型。
2. 下一次只扩到 8–10 clips，匹配每 clip 暴露，并重复 high-noise → uniform 课程。
3. 若扩展后再次变成马赛克，不再增加 scratch 参数或 updates；转为冻结预训图像空间模块，只学 temporal blocks 与 text adapter。
4. 要宣称 unseen composition，仍须回到 Phase 1 的 held-triple 数据门禁和独立盲评；本文不支持该声称。
