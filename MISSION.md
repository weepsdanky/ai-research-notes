# Mission: 从原理到可运行的 DiT 图像与视频生成

## Why
通过亲手训练、诊断和交互推理掌握 DiT，而不是只会调用预训练模型；最终能够区分图像扩散的采样过程与真正的视频时间轴，并理解论文 trick 如何改变训练行为。

## Success looks like
- 能从数学公式追到 Image DiT 与 Video DiT 的张量实现
- 能在当前 Apple Silicon Mac 上训练并生成可辨识图片
- 能输入受支持的猫、狗或红衣人物 prompt，得到时间连续的 MP4
- 能用 minimal / paper-compatible 对照实验解释每个 trick 的作用

## Constraints
- 24 GB Apple Silicon Mac，可实际训练与推理
- 小数据、小模型用于学习；质量声明必须与训练规模一致
- 图像生成与视频生成保存在两个独立项目

## Out of scope
- 在本机从零复现 ImageNet DiT-XL/2 或十亿参数开放词表视频模型
- 把合成数据预览或 diffusion 去噪轨迹冒充训练模型生成的视频
