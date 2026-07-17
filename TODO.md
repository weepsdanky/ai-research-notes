# TODO

## Milestones 

- [x] Proximal Policy Optimization 理解原理
- [x] DeepSeek Math 理解
- [x] DeepSeek R1 理解
- [ ] 自己实现使用 RL Train 一个小模型（Goal/reward 待定）
- [ ] On Policy Distillation
- [ ] https://github.com/Infrasys-AI/AIInfra
- [ ] https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook

## 领域

- GAN 
- VAE 

## DDPM 

- [x] EPOCHS=20 偏少；minDiffusion 默认跑到 100 epoch，而且 README 也提到更多参数/训练会改善 MNIST。
- [x] LR=1e-3 对 diffusion 可能偏激，minDiffusion 用 2e-4；更稳的 AdamW/EMA 通常会明显改善采样。
- [x] 采样用 sqrt(beta_t) 当方差，简单但粗；可改成 posterior variance tilde_beta_t。
- [x] 没有 EMA 权重。DDPM/UNet 采样常用 EMA，视觉质量会比最后一步原始权重稳定。
- [x] 这个 UNet 很简化：没有 GroupNorm/attention/residual block，MNIST 能学，但样本会更“抖”。
