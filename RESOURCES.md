# DiT Learning Resources

## Knowledge

- [Paper: _Denoising Diffusion Probabilistic Models_](https://arxiv.org/abs/2006.11239)
  扩散基础入口。用于理解一次加噪、noise prediction、训练循环与从高斯噪声反向采样。
- [Paper: _Improved Denoising Diffusion Probabilistic Models_](https://arxiv.org/abs/2102.09672)
  用于在掌握 DDPM 后理解 learned variance、noise schedule、采样步数和质量/似然指标差异。
- [Paper: _Scalable Diffusion Models with Transformers_](https://arxiv.org/abs/2212.09748)
  Image DiT 的核心论文。用于理解 patchify、adaLN-Zero、模型缩放、latent diffusion 与 CFG。
- [Official facebookresearch/DiT](https://github.com/facebookresearch/DiT)
  原作者 PyTorch 实现。用于核对初始化、learned variance、训练 recipe 与采样细节。
- [Paper: _Latte: Latent Diffusion Transformer for Video Generation_](https://arxiv.org/abs/2401.03048)
  Video DiT 的主要架构依据。用于学习空间/时间 attention 分解及 Variant 1。
- [Official Vchitect/Latte](https://github.com/Vchitect/Latte)
  用于核对 Video DiT 的 per-frame patch、时空位置编码、S-AdaLN 与 CFG 实现。
- [Paper: _HunyuanVideo: A Systematic Framework For Large Video Generative Models_](https://arxiv.org/abs/2412.03603)
  用于学习 3D VAE、92M–6.6B scaling sweep、图像到视频的渐进课程、混合分辨率 token bucket 与大模型训练设计。
- [Paper: _LPM 1.0: Video-based Character Performance Model_](https://arxiv.org/abs/2604.07823)
  最终任务的一手系统资料。用于学习 17B Base LPM 的音频/文本/多参考条件、Flow Matching、长时训练，以及 Online LPM 的因果流式蒸馏、rollout shift 和分布式基础设施；不作为当前 1.6M 模型的直接复现目标。
- [Paper: _Flow Matching for Generative Modeling_](https://arxiv.org/abs/2210.02747)
  Flow Matching 主论文。用于理解 CNF、conditional probability path、simulation-free vector-field regression 与 OT path。
- [Paper + official code: _Flow Matching Guide and Code_](https://arxiv.org/abs/2412.06264) · [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)
  用于核对 path、scheduler、loss、Euler/Heun solver 的接口和符号约定。
- [Local learning path: Flow Matching for Video DiT](docs/case-studies/flow-matching/README.md)
  线性 path 推导、10–30 NFE 验收、time sampling/shift 和 DDPM 公平 A/B 协议。
- [Paper: _High-Resolution Image Synthesis with Latent Diffusion Models_](https://arxiv.org/abs/2112.10752)
  用于理解 DiT 论文为何在 VAE latent 而非高分辨率像素上扩散；它是依赖背景，不是学习 DiT block 的第一篇。
- [Local audit: DiT original-paper implementation](docs/case-studies/dit-original-paper-implementation-audit.md)
  论文、官方代码和本地实现的逐项差异矩阵。
- [Video DiT case-study timeline and current entry](docs/case-studies/video-dit/README.md)
- [Local design: Video DiT on Mac](docs/case-studies/video-dit/01-phase-0-mac-design.md)
  Latte 依据、计算预算、合成课程与能力边界。
- [Local diagnosis: Video DiT moving mosaic](docs/case-studies/video-dit/04-phase-2-moving-mosaic-diagnosis-and-next-actions.md)
  将当前失败拆为 VAE、noise level、condition、sampler 与规模门禁；用于下一轮单变量实验。
- [Primary-source notes: moving-mosaic diagnosis](docs/case-studies/video-dit/05-phase-2-moving-mosaic-primary-source-notes.md)
  DDPM、Improved DDPM、LDM、DiT、Latte、CLIP/FVD/LPIPS 的一手证据与本地推断边界。

## Wisdom (Communities)

- [Hugging Face Diffusers discussions](https://github.com/huggingface/diffusers/discussions)
  适合验证 scheduler、latent scaling 和视频 pipeline 的实际兼容问题。
- [PyTorch Forums](https://discuss.pytorch.org/)
  适合排查 MPS、attention 性能与数值稳定性问题。
