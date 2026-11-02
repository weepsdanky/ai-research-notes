# 视频生成模型的训练数据从哪里来？

> 时期：Phase 0 — 数据版图与来源边界（2026-08-13）  
> 状态：背景参考；不是当前训练清单  
> 顺序：[总索引](README.md) → [Phase 0 架构基线](01-phase-0-mac-design.md) → 本文 → [Phase 1 真实数据计划](03-phase-1-real-data-and-compositional-plan.md)  
> 调研时间：2026-08-13。本文只引用论文、官方仓库、项目页和官方数据集卡。这里的“开放”拆成三件事：代码开放、权重开放、训练数据开放；三者不能互相替代。

## 结论先行

当前高质量视频基础模型的数据，通常不是来自某一个干净数据集，而是下面几层混合而成：

1. 从公开互联网或商业授权源收集长视频；
2. 用镜头检测切成 2–10 秒左右的单镜头片段；
3. 去重，去损坏、转场、静态、抖动、低清、字幕、水印、黑边和 NSFW 内容；
4. 计算美学、清晰度、运动量和文视频相关性分数，按训练阶段逐步提高门槛；
5. 用 VLM 重新生成长而细的字幕，必要时再由 LLM 汇总；
6. 混入大量图像保持空间质量，最后用少量人工筛选的高质量视频做 SFT。

真正的瓶颈并不是“找到很多 MP4”，而是取得合法、稳定、可追溯的视频，并完成切镜、过滤、重标注和分桶。主流所谓“开源视频模型”绝大多数只开放代码和权重，**没有开放训练它的逐字节精确语料快照**。

## 主流开放权重模型披露了什么

| 模型 | 官方披露的数据来源与规模 | 过滤、字幕方式 | 精确训练集能否取得 |
|---|---|---|---|
| Open-Sora 1.2 | 阶段 1：WebVid-10M，约 4 万小时；阶段 2：Panda-70M 官方 30M 子集过滤至 20M/约 4.1 万小时；阶段 3：约 2M/5,000 小时，含 Pexels、Pixabay、Mixkit、MiraData、Vript 等 | Panda 以美学分数 4.5 过滤；后期数据由 PLLaVA 重标注，并加入美学、运动和镜头运动条件 | **部分可复现**。命名数据和流程很具体，但“其他数据”、下载时点及完整媒体快照没有统一发布。([官方训练报告](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_03.md)) |
| Open-Sora Plan | 披露 10M SAM 图像、5M 内部图像、6M HQ Panda70M，后期 1M HQ Panda70M + 约 100K 自采 HQ 视频 | 静态/抖动/切镜、字幕 OCR/裁剪、美学过滤和重写字幕 | **不能完整复现**，因为包含内部图像和自采数据；项目公开了部分过滤后标注，而非全部源视频的可再分发快照。([v1.2 报告](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.2.0.md), [v1.3 报告](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.3.0.md)) |
| HunyuanVideo 13B | “互联网规模”的图像和视频，未列网站或视频总数；最终约 1M 人工筛选 SFT clips | PySceneDetect 切镜；VideoCLIP 去重、约 10K 概念聚类；DOVER 美学/技术质量、清晰度、光流运动、OCR、字幕/水印/边框/Logo 过滤；内部 VLM 结构化重标注 | **不能**。代码和权重开放，但原始池、最终训练列表及媒体未发布。([技术报告 §3](https://arxiv.org/html/2412.03603#S3)) |
| CogVideoX | 过滤后约 35M 单镜头 clips，平均约 6 秒；原始网站未披露；另使用来自 LAION-5B、COYO-700M 的约 2B 美学筛选图像 | 用 20K 人工标签训练 Video-LLaMA 过滤器，去编辑、断裂运动、低质、讲课、文字主导和录屏；光流/美学阈值；Panda caption→逐帧 CogVLM→GPT-4 汇总，再蒸馏 CogVLM2-Caption | **不能**。Caption 模型开放，35M 视频语料没有开放。([论文 §3.4](https://arxiv.org/html/2408.06072#S3.SS4), [官方仓库](https://github.com/THUDM/CogVideo)) |
| Wan2.1 | 数十亿图像和视频；官方只描述“大规模去重候选池”，不列来源和明细 | 基础属性、视觉质量、运动质量、文视频相关性四级清洗 | **不能**。代码和模型开放，训练数据未发布。([官方报告](https://arxiv.org/abs/2503.20314), [官方仓库](https://github.com/Wan-Video/Wan2.1)) |
| LTX-Video | 论文称使用公开可得数据和授权素材的图像/视频混合，但未给数据集名和数量 | 数万人工偏好对训练美学排序器；去低美学/弱运动、裁黑边；内部图像/视频 captioner 全量重标注，再做高美学微调 | **不能**。官方 Trainer 能处理用户自己的视频，但它不是 base model 语料。([论文 §3](https://arxiv.org/html/2501.00103#S3), [数据准备工具](https://github.com/Lightricks/LTX-Video-Trainer/blob/main/docs/dataset-preparation.md)) |
| Mochi 1 | 官方只说从零训练，并承认模型会继承训练数据偏差；未披露来源、数量或处理过程 | 未公开 base-model 数据管线；LoRA 工具要求用户自备 MP4/MOV 和段落字幕 | **不能，且透明度最低之一**。Apache-2.0 权重/代码不代表语料开放。([官方仓库](https://github.com/genmoai/mochi), [LoRA 数据格式](https://github.com/genmoai/mochi/blob/main/demos/fine_tuner/README.md)) |
| Latte | 类条件实验用 FaceForensics、SkyTimelapse、UCF101、Taichi-HD；T2V 使用 Vimeo25M，并从 PixArt-α 初始化 | 官方提供 loader/config，但不托管一个精确处理后的训练快照 | **可按公开数据近似复现研究实验**，但仍受各上游数据条款、失效链接和预处理差异影响。([论文](https://arxiv.org/html/2401.03048#S4), [官方仓库](https://github.com/Vchitect/Latte)) |
| VideoCrafter2 | WebVid-10M 视频 + LAION-COCO 图像做联合训练，再用 JourneyDB 约 4M Midjourney 图像/提示词做高质量调优 | 以图像数据补偿公开视频低清问题；空间模块初始化自 SD 2.1 | **来源较透明，仍非自包含快照**；媒体是否还能下载及能否再分发取决于上游。([论文](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_VideoCrafter2_Overcoming_Data_Limitations_for_High-Quality_Video_Diffusion_Models_CVPR_2024_paper.html)) |

因此，“全开源模型”更准确的说法通常是 **open code + open weights**，而不是 open-data reproducible model。Open-Sora、Open-Sora Plan 和 VideoCrafter2 的数据说明相对可操作；Hunyuan、CogVideoX、Wan、LTX、Mochi 的 base corpus 均不完整开放。

## 可以获取的开放视频数据集

| 数据集 | 规模和来源 | 如何取得 | 字幕/过滤 | 许可与现实边界 |
|---|---|---|---|---|
| WebVid-10M | 约 10.7M 短视频、约 5.2 万小时，来自公开视频/stock-footage 网页，原始文本多为页面 alt-text/标题 | 官方项目发布 CSV/URL；通常需从原站下载，因此会遇到链接失效 | 字幕短，视频多为低于 360p且带水印；适合低清预训练，不适合作为唯一高质数据 | 元数据开放不等于视频版权转让；必须遵守源站及单条内容条款。([官方项目](https://m-bain.github.io/webvid-dataset/), [论文](https://arxiv.org/abs/2104.00650)) |
| HD-VILA-100M | 100M clips、3.3M YouTube 长视频、约 371.5K 小时、主要 720p | 官方发布 YouTube URL、时间戳和切片脚本；用户自行下载原视频 | 文本主要来自 ASR 转录，不一定是视觉描述 | 发布的主要是索引，不是作者拥有版权的视频包；可用性随 YouTube 删除和地域变化。([官方说明](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m)) |
| Panda-70M | 70.7M clips、约 167K 小时、约 36TB；从 HD-VILA-100M 的 3.8M 长视频切镜得到 | 下载官方元数据，用工具重新抓取源视频；也有 10M/2M 子集索引 | 多个跨模态 teacher 生成字幕，后续增加 desirability 和 shot-boundary 标注 | Snap 明确限定**非商业研究**；原视频仍需服从其各自许可，不能把“数据集可下载”理解为统一商业授权。([官方项目与条款](https://snap-research.github.io/Panda-70M/)) |
| InternVid | 官方完整标注称 230M 文视频对，并提供 10M-FLT 等过滤子集 | Hugging Face/OpenDataLab 发布 annotation；大量样本仍依赖源 URL 下载 | 自动切分、字幕和 ViCLIP 相似度过滤；10M-FLT 是更常见的生成训练入口 | 仓库 Apache-2.0 是**代码许可**，不能自动覆盖第三方视频版权；逐条数据条款仍需核验。([官方仓库](https://github.com/OpenGVLab/InternVideo), [官方 HF 集合](https://huggingface.co/collections/OpenGVLab/internvid)) |
| OpenVid-1M | 1M 高质量文视频对，其中约 0.4M 为 1080p；OpenVidHD 约需 4.5TB | Hugging Face 提供打包视频、CSV 和官方下载脚本 | 从公开视频候选池按美学、清晰度、运动等筛选，并生成表达性字幕 | 它是少数可直接取得媒体包的大型集合；但官方仓库未把所有底层视频权利简化成一份无条件再分发/商用许可，使用前应核对 HF 当前条款和来源。([官方仓库](https://github.com/NJU-PCALab/OpenVid-1M)) |
| MiraData | 330K/93K/42K/9K 多版本；来自人工选定 YouTube 频道、HD-VILA、Videvo、Pixabay、Pexels | 下载 HF/Google Drive 元数据，再按 URL、时间戳运行官方脚本 | PySceneDetect 切镜，多模型合并/质量过滤；8 帧网格交给 GPT-4V 生成主体、背景、风格、运镜、短/密集字幕 | 官方 README 标 GPL-v3 并写支持商业使用，同时又提示商业许可可联系作者；这两句话存在解释空间，而且源视频权利不因元数据 GPL 自动改变。([官方仓库](https://github.com/mira-space/MiraData)) |
| FineVideo | 43,751 条、约 3,425 小时、约 600GB，来自 YouTube Commons 的 CC-BY 视频 | HF gated dataset；接受条款后可整包、streaming 或 snapshot 下载，视频 bytes 在数据中 | 从 1.8M 候选筛选；Gemini 生成时间级标注，GPT-4o 结构化；含场景、人物、叙事、音视关系和 QA | 这是许可链最清楚的选择之一：源视频为 CC-BY，并保留 provenance；仍须逐条署名、跟随删除版本，若再托管必须传递官方 Terms。([官方数据集卡](https://huggingface.co/datasets/HuggingFaceFV/finevideo)) |

## “怎么获取”在工程上意味着什么

公开数据大致有三种交付模式：

- **URL 索引型**：WebVid、HD-VILA、Panda、InternVid、MiraData。下载的是 CSV/JSON、源 URL 和切片时间戳，再由 `yt-dlp`/官方脚本抓取。优点是元数据小；缺点是链接衰减、地域限制、平台 ToS 和版权责任落在使用者身上。
- **媒体打包型**：OpenVid-1M、FineVideo。可从 Hugging Face 下载/stream 视频 bytes，更容易得到确定快照，但体积从数百 GB 到数 TB，仍要遵守数据卡、源许可、删除请求和署名条件。
- **模型私有混合型**：Hunyuan、CogVideoX、Wan、LTX、Mochi。论文只披露规模、管线或来源类别；外部用户无法重建完全相同的 base corpus。

以可重复、相对合规的教学实验为目标，建议按以下顺序：

1. **先用 FineVideo 小子集**：许可和 provenance 最明确，可 streaming，不需要立即下载 600GB；但其长视频与叙事标注更偏理解任务，训练生成模型前仍要切单镜头短 clips。
2. **再用 OpenVid-1M 的 1K–10K 子集**：更贴近 T2V 生成分布和高画质，但先固定数据集 revision，并保存每条样本的来源、哈希和许可快照。
3. **研究复现用 WebVid/Panda/InternVid metadata**：可以学习大规模下载、切镜和过滤，但不要假定 URL 对应媒体可自由商用或再分发。
4. **商业用途不要从“模型/仓库开源”推导数据权利**：逐数据源做 legal review；尤其 Panda 明确非商业，YouTube 索引型数据还涉及平台条款和每条视频许可。

一个稳健的数据清单至少要保存：`sample_id`、源 URL、下载时间、内容哈希、源许可/作者、切片时间戳、删除状态、过滤分数、caption 模型与版本、最终 train/val/test split。这样训练结果才可审计，也能响应删除请求。

## 尚未解决的歧义

- 论文中的“publicly available”“internet-scale”只描述可见性，不等于作者取得了可再分发或商业训练授权。
- GitHub 仓库的 Apache/GPL 许可证通常只覆盖代码；不能据此推断视频媒体的许可。
- URL 数据集会随时间改变。即使两个人拿到同一 CSV，也可能下载到不同数量或不同版本的视频。
- OpenVid-1M 和 MiraData 虽提供便利下载/脚本，其统一数据集许可证与底层第三方视频权利之间仍需逐项确认；本文不把仓库标签当作法律结论。
- HunyuanVideo、CogVideoX、Wan、LTX-Video、Mochi 目前都无法从公开资料恢复精确训练语料，因而不能声称数据级可复现。
