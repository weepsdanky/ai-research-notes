# Setup the environment 

There are two types of code here mainly: **tutorial**, or **engineering**. Previous code mainly aims to show the mathematical principles of processes in training large language model, the later mainly aim to build reusable components for training production-grade LLMs. 

For engineering part, we mainly use [verl](https://github.com/volcengine/verl/) framework developed by ByteDance team for training LLMs, and [ai-toolkit](https://github.com/ostris/ai-toolkit) to train/finetune diffusion models.

## DiT learning projects

- [`image_dit/`](image_dit/)：Image DiT，含无 trick 的 MinimalDiT、paper-compatible 完整版、训练、采样、测试与 7860 Web UI。
- [`video_dit/`](video_dit/)：Video DiT / Latte Variant 1，含时空 Transformer、Mac 可训练合成视频课程、MP4 推理与 7861 Web UI。

图像去噪 GIF 沿 diffusion sampling time 前进；Video DiT 输出的 MP4 才沿 video time 前进。
