# Video DiT case study：时期、顺序与当前入口

> 当前时期：Phase 4 — 最小可辨识视频模型（2026-08-23）  
> 当前入口：[Phase 4 可辨识生成恢复记录](08-phase-4-meaningful-generation-recovery.md)  
> 规则：先通过分层 debug gate，再扩大数据或训练步数。

这个目录把原先散落在 `docs/case-studies/` 根目录的 Video DiT 文档按实验时期集中起来。编号表示建议阅读和执行顺序；日期表示当时的证据快照，不应把历史数字当成当前结果。

| 顺序 | 时期 | 文档 | 状态 | 作用 |
|---:|---|---|---|---|
| 1 | Phase 0 · 2026-08-13 | [Mac 架构与教育基线](01-phase-0-mac-design.md) | 历史基线 | 理解 Latte 风格空间/时间 block、Mac 预算和演示模型边界 |
| 2 | Phase 0 · 2026-08-13 | [视频训练数据版图](02-phase-0-training-data-landscape.md) | 背景参考 | 区分开放代码、开放权重与可审计训练数据 |
| 3 | Phase 1 · 2026-08-13—15 | [真实数据与组合泛化计划](03-phase-1-real-data-and-compositional-plan.md) | 历史执行日志 | 记录数据许可、去重、split、C1/C2 和多轮来源审计 |
| 4 | Phase 2 · 2026-08-15 | [moving-mosaic 诊断与行动方案](04-phase-2-moving-mosaic-diagnosis-and-next-actions.md) | 已确认失败 | 固化用户看到的“移动马赛克”、红灯判据和调试阶梯 |
| 5 | Phase 2 · 2026-08-15 | [moving-mosaic 一手资料笔记](05-phase-2-moving-mosaic-primary-source-notes.md) | 参考 | 区分论文事实、本地证据和待证伪假设 |
| 6 | Phase 3 · 2026-08-23— | [根因实验、修复与复训记录](06-phase-3-root-cause-debug-and-retrain.md) | M1 5k no-go | 记录 oracle、恢复曲线、patch2 单变量复训与停止门禁 |
| 7 | Phase 3 · 2026-08-23— | [原论文与实现审计](07-phase-3-paper-debug-audit.md) | 已完成当前轮 | 对照 DiT、Latte、DDPM/Improved DDPM 区分论文事实与本地推断 |
| 8 | Phase 4 · 2026-08-23 | [可辨识生成恢复记录](08-phase-4-meaningful-generation-recovery.md) | 3-prompt 最低门禁通过 | 记录 1-clip→10-clip→3-clip 诊断、高噪声→uniform recipe 与当前最佳模型 |

## 当前实验状态

- `c1-direct-train-v3` 的盲评已经确认失败：C1 action `0/28`、triple `0/28`，输出是移动低频色块，不是可辨识视频。
- VAE 往返样本仍可辨识，因此当前优先排查 denoiser、噪声日程、训练/采样一致性与 token 分辨率。
- patch2 单变量复训已在 5k 可视门禁 no-go：空间 token 增多不是单独解法。
- 1-clip 可达性已通过；3-prompt 模型已能生成可辨识的室内猫、海滩和池塘，并有可见时间变化。
- 当前结果是训练分布 overfit，不代表 unseen composition；扩到 8–10 clips 后若再次退化，应转向预训空间先验。
- 数据来源与正式 C1 数据门禁仍保留，但当前 Phase 3 不再扩数据。

## 文档维护约定

每一份新的中间文档都必须在标题下写明：`时期`、`日期`、`状态`、`输入证据`、`下一门禁`。若一份长文保留多个历史快照，应显式使用“当时”“随后”“当前”区分，避免后来的数字覆盖早期决策语境。
