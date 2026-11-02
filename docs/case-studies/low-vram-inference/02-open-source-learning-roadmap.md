# 开源优化路线与八周学习计划

更新：2026-10-22。[硬件盘点](00-hardware-inventory.md) · [已完成实验与数据](01-experiments-and-results.md)。本项目聚焦8GB显存、32GB RAM条件下的大模型本地推理与优化。Computer-use 的计划与实现归独立的 `Documents/Github/cua` 项目。

**主线：理解一次token生成 → 读懂llama.cpp/ggml数据流 → 观测MoE专家访问 → 在预算内验证一种优化 → 形成可复现贡献。** 八周完成一个小研究闭环，不把学习所有框架当作目标。

## 研究目标与下一步

已有结果：固定155GB DeepSeek GGUF可运行，历史最好单次decode约0.92token/s。[2026-10-22时间线诊断](01-experiments-and-results.md#2026-10-22cpu硬缺页与gpu时间线诊断)进一步区分了硬缺页关联阻塞、OpenMP运行库活动和CUDA同步。[三对KMP_BLOCKTIME=0实验](01-experiments-and-results.md#2026-10-22kmp_blocktime0三组ab)已完成：逐对中位数请求CPU占用下降62.46%，decode仅提高2.22%，TTFT变化不一致，不能宣称稳定提速。**下一步推进专家访问/分页观测；线程数/P-E核分配另做单变量实验，GPU算子重写后置。** 真实专家命中率尚未测清，短输入诊断不能外推长上下文。随后只研究：

> **在固定工作负载和内存预算下，专家访问有多少时间复用？减少重复搬运或读取，是否足以改善TTFT与decode？**

第一步先整理现有日志，固定prompt、输出长度、模型SHA、后端commit、缓存条件和时间预算，画SSD→RAM→CPU/GPU的数据路径。区分三种量：路由选择的逻辑权重字节、后端实际拷贝字节、操作系统物理读取字节。先检查已有trace/回调是否够用，再决定补计数；不预设GPU专家缓存RFC适用于当前CPU专家执行路径。

首个交付物是放在实验记录中的一页benchmark约定与观测缺口，随后才采样真实路由。先回答是否值得缓存，再写缓存算法；路由平坦、收益不足也是有价值的结果。

## 学习顺序与框架取舍

| 优先级 | 学什么 | 需要达到的理解 | 本机安排 |
| --- | --- | --- | --- |
| 主线 | [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | GGUF/量化块、mmap/repack、计算图与调度、`MUL_MAT_ID`、CPU/GPU分工 | 深入实际热点；保留现有Windows基线，源码构建另设目录 |
| 方法参照 | [KTransformers](https://github.com/kvcache-ai/ktransformers) | CPU专家kernel、GPU prefill、异构调度与同步 | 读论文和局部代码；[V4完整教程](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md)的5090 32GB/≥200GB RAM配置超出本机预算 |
| 第二阶段 | [Unsloth](https://github.com/unslothai/unsloth) | 量化质量、GGUF导出、训练与推理链路的区别 | 先学量化，不把QLoRA安排为当前加速任务；其Studio也包含本地推理，不能只称训练框架 |
| 服务补课 | [Mini-SGLang](https://github.com/sgl-project/mini-sglang) → [SGLang](https://github.com/sgl-project/sglang) / [vLLM](https://github.com/vllm-project/vllm) | prefill/decode调度、批处理、prefix cache、KV分页；吞吐与延迟的区别 | 先选一个紧凑实现，用能放入显存的小模型；不同时装全套服务栈 |
| 对应问题出现时 | [LMCache](https://github.com/LMCache/LMCache) | KV分层与跨请求复用 | KV缓存与专家权重缓存不同，不假设可直接插入现有llama-server |
| 后续专题 | [Megatron-LM/Core](https://github.com/NVIDIA/Megatron-LM)、[NeMo RL](https://github.com/NVIDIA-NeMo/RL)、[verl](https://github.com/verl-project/verl) | TP/PP/DP/EP/CP、训练与rollout/reward/update边界 | 八周后按需要选读；Megatron主要是训练并行基础设施，也有RL入口，不是本机SSD推理的前置课程 |

所以：**应该研究llama.cpp底层；Unsloth围绕量化问题按需深入；Megatron/RL暂后置。** 不要求先通读任何仓库。C/C++补指针、RAII、内存布局、线程与计时；CUDA补显存层次、stream、同步与带宽。每项知识都对应正在追踪的一段数据路径。

当前运行依赖是NVIDIA驱动、llama.cpp发行包的CUDA/ggml运行库、GGUF；Python只是编排和监测，核心推理不需要PyTorch。源码编译另需MSVC、CMake、CUDA Toolkit/nvcc，可选Ninja，按[官方说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。Mini-SGLang官方面向Linux，Windows建议WSL2，另有sgl-kernel/FlashInfer依赖；服务实验使用独立环境，先查版本及GPU支持，再安装。

## 八周安排

默认每周6–8小时：约2小时读论文/源码、3–4小时测量或实现、1小时整理、30分钟社区跟踪。每周只有一个主要产物；任务超过预算就延长该周，不跳过正确性检查以追日历。

| 周 | 学习与实践 | 交付物与通过标准 |
| --- | --- | --- |
| 1：建立成本模型 | 重算已有TTFT、decode、每token逻辑/物理读量与条件roofline；读LLM in a flash的问题设定 | 一页benchmark约定、数据路径图；能解释为何0.92、1.8、2.8token/s不是同一种结论 |
| 2：追一次token | 从固定commit的模型图→专家索引→ggml调度→CPU/CUDA算子读源码；必要时小模型定位调用 | 标注函数与buffer归属的调用图；分清实际CPU执行、H2D拷贝、OS缺页，不能只看`-ngl`猜路径 |
| 3：补齐观测 | 用现有接口或最小计数，将专家ID/访问字节、SSE时间、工作集与I/O对齐；先验证计数和开销 | 小范围trace与口径说明；数据缺失标明，不用整盘计数冒充PID或专家命中 |
| 4：评估复用价值 | 读KT论文；在同一trace上比较无缓存、LRU及理想离线策略；按真实余量算预算 | 命中率、节省字节、重用距离曲线；说明CPU页缓存与GPU专家池各能解决哪段开销 |
| 5：只做一个改动 | 根据第4周证据，在缓存、预取/读合并、CPU算子或卸载调度中只选一种；无收益窗口则完善诊断 | 小原型A/B：固定工作量、正确性、额外内存和观测开销，首轮至少3次重复；保留原始值与负结果 |
| 6：验证上下文边界 | 先小模型验证长输入测量，再对目标做有时限的2k→4k→8k递增；通过才继续16k/32k/64k | 实际输入长度、KV/工作区峰值、TTFT/decode、检索/跨段质量表；分配成功不算通过 |
| 7：量化与服务补课 | 阅读Unsloth量化/导出路径；若已有可用小模型对照文件，比较质量—内存—速度；用同一兼容客户端测试共享/新前缀 | 一张权重格式与KV格式分开的表，一组prefix reuse对照；服务实现只选llama-server或一个小型框架 |
| 8：复现与贡献 | 整理最佳改动/负结果，去掉观测扰动后复测；对照相关issue/PR与贡献要求 | 可重跑benchmark、完整版本/命令/输出、限制和最小diff；有具体问题再讨论issue/PR，不为交付强造bug |

八周完成标准：能解释主要成本、复现一个单变量对照，并让别人检查数据与结论。无需承诺将155GB模型加速到交互水平；若学习后确认预算不足，也应给出证据及下一步资源需求。

## 源码从哪里开始

2026-10-22补充：[DeepSeek推理开源栈专题](../../chapter-8-model-efficiency/8_deepseek-inference-open-source.md)已核查官方45个公开仓库，克隆39个相关源码和2个TileLang生态仓库，附commit清单、阅读路径与4060适配判断。DeepInfer整栈及多数现成DeepSeek kernel不支持当前SM89/GGUF组合；TileLang本体可以作为后续独立小算子验证入口。第1–2周先读其调度/benchmark并对照下方llama.cpp路径，第3周之后有热点证据再考虑移植，不替换现有基线。

先读已测commit `d81235049384534c167caea52b85a694f6103d14`，避免最新代码和历史二进制混在一起：

1. [deepseek4.cpp](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/deepseek4.cpp)：模型图、路由与专家运算。
2. [llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-model.cpp) / [llama-mmap.cpp](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-mmap.cpp)：权重buffer与文件映射。
3. [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-backend.cpp)：图切分与跨后端数据流；再沿实际热点进入[CPU](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-cpu/ggml-cpu.c)或[CUDA量化矩阵运算](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-cuda/mmq.cuh)。
4. [llama-kv-cache-dsv4.cpp](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-kv-cache-dsv4.cpp)：第6周再读SWA/CSA/HCA/indexer及分配策略。

每读一个模块只回答四件事：输入输出是什么、字节在哪里、何时移动/计算、哪个测量能验证理解。若后端升级，先作为独立实验确认输出与资源变化，再决定是否替代基线。

## 论文：必读两篇，其他按问题选择

| 顺序 | 一手入口 | 阅读重点与边界 |
| --- | --- | --- |
| 必读1 | [LLM in a flash，ACL2024](https://arxiv.org/abs/2312.11514) · [Apple介绍](https://machinelearning.apple.com/research/efficient-large-language) | 有限DRAM下减少闪存读量、窗口复用、连续布局；不是当前GGUF可开启的插件 |
| 必读2 | [KTransformers，SOSP2025作者PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) · [代码](https://github.com/kvcache-ai/ktransformers) | CPU/GPU异构MoE、kernel与调度；大RAM实验不能等同本机SSD分页 |
| 专家/激活热点 | [PowerInfer，SOSP2024](https://arxiv.org/abs/2312.12456) · [官方代码](https://github.com/Tiiny-AI/PowerInfer) | 热冷激活分工；神经元稀疏与MoE专家路由不是同一假设，不能照搬倍速 |
| 长上下文 | [SolidAttention，FAST2026论文与演讲](https://www.usenix.org/conference/fast26/presentation/zheng) | SSD KV、稀疏attention和预取；优化KV不等于优化专家权重，需核对DSV4架构 |
| 吞吐调度 | [FlexGen](https://arxiv.org/abs/2303.06865) · [代码FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | GPU/CPU/磁盘放置、大batch吞吐；aggregate token/s不能与单请求TTFT混比 |

每篇留一张短卡，直接写入实验记录或对应代码说明：瓶颈、关键假设、搬运/计算路径、正确性代价、自己的证伪实验。额外课程可选 [Mini-SGLang](https://github.com/sgl-project/mini-sglang) 与 [Datawhale课程](https://github.com/datawhalechina/zero-to-sglang)，不新增并行学习主线。

## 社区正在做什么，如何学习

| 工作 | 证据状态与学习价值 |
| --- | --- |
| [专家池RFC #28248](https://github.com/ggml-org/llama.cpp/discussions/28248)、[两级缓存提案 #20757](https://github.com/ggml-org/llama.cpp/issues/20757) | RFC/fork/提案不等于默认发布功能。不同模型变体、路由与显存预算的命中报告不一致，先测本机trace；作者4090上的标题倍速不能当4060预期收益 |
| [DSV4支持 #24162](https://github.com/ggml-org/llama.cpp/pull/24162)、[DSpark #25784](https://github.com/ggml-org/llama.cpp/pull/25784) | 先前已合并的架构/推测解码工作；当前preview权重与后续0731/draft须匹配，不能自动套用。运行前按[当前指南](https://unsloth.ai/docs/models/deepseek-v4)重新核算额外内存 |
| [indexer工作区问题 #25468](https://github.com/ggml-org/llama.cpp/issues/25468) | 已修复旧问题，适合学习KV小但临时工作区大的原因；不能写成当前版本仍必现 |
| KV复用与调度 | SGLang/vLLM、LMCache处理共享前缀、批处理与KV移动；首次独特长输入仍需计算，不能替代权重访问优化 |

重点跟踪三个研究来源：**Apple该论文作者**（Keivan Alizadeh、Iman Mirzadeh、Mehrdad Farajtabar等；按论文作者而非推断当前任职）看[研究页](https://machinelearning.apple.com/research/efficient-large-language)；**交大IPADS**（Haibo Chen、Zeyu Mi、Yixin Song及相关作者）看[论文目录](https://ipads.sjtu.edu.cn/pub/publication)的PowerInfer/SolidAttention；**清华MADSys与KT合作团队**（Mingxing Zhang、Yongwei Wu、Hongtao Chen等）看[实验室](https://madsys.cs.tsinghua.edu.cn/)与KT代码。工程上跟实际热点文件维护者和PR评审，不必按大厂逐一订阅。

每周30分钟看一个相关issue/PR：问题→最小复现→diff→review→验证。论文队列最多3篇，只加入能改变当前实验的材料；每月看一次实验室页面。已发布功能、论文结果、社区自报和待测假设分开标记，不根据模型名字猜后端支持。

上游路径：先准备包含硬件、commit、SHA、命令、原始输出、重复统计的私有证据包，再查重复并按[贡献指南](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)沟通。当前llama.cpp要求AI代码披露与本人审查，并限制AI代写公开issue/PR描述；对外内容需本人理解后依照届时规则撰写。优先贡献可复现反例、诊断或小修复，保留原作者归属。

## 长上下文、视频与后续专题

当前只验证ctx512；300k/1M的KV计算与测量限制已并入[实验记录](01-experiments-and-results.md#300k与1m上下文只有预算推算)。第6周只探索预算内可用范围，不能承诺八周内达到1M交互。KV压缩、专家缓存与RAG解决不同问题，比较时同时看质量和时间。

八周后再选一个扩展：更深入的量化/Triton、小模型服务调度，或视频生成的分阶段卸载/attention/VAE tiling。GLM与MiniMax H3等原始目标保留，但开始前重新核查组件、许可证、checkpoint与磁盘预算；视频还要固定帧数、分辨率、步数和质量。Megatron/verl则用于另一个训练/RL专题，避免把全部方向同时开工。

维护约定：硬件变化更新00；每周实验和论文卡追加01并链接artifacts；方向、阅读优先级与后续计划更新本文。三份项目文档持续维护，不再按进度新增编号文件；跨项目可复用的源码专题资料放在 `docs/chapter-8-model-efficiency/` 并从本文链接。
