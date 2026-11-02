# 低显存推理：实验过程、数据与结论

实验日期：2026-10-09；合并整理：2026-10-19。本文合并原01–12的过程与数据，以及原14的上下文计算，删除重复进度快照；原始SSE、命令、采样与校验数据仍在 [artifacts](../artifacts/low-vram-inference/)。[硬件环境](00-hardware-inventory.md) · [下一步与学习计划](02-open-source-learning-roadmap.md)。

## 已经验证什么

Windows原生llama.cpp使用155.095GB DeepSeek-V4-Flash GGUF，在32GB RAM、RTX4060 Laptop 8GB机器上完成本地生成。五种短题配置共40次输出正确；另完成18次128token输出、6轮SSD与9轮RAM读取测试。已测最好decode **0.920token/s**；27/150token提示最快TTFT分别 **26.938/106.545秒**，三个最好值不来自同一请求。

这些结果证明指定配置可运行，但不代表流畅交互、普遍质量或300k/1M上下文可用。现有数据没有证明全系统绝对理论最快值；约1.77–1.85及2.72–2.85token/s分别是附带假设的CPU缩放和SSD读取参考。

## 过程与关键决策

| 阶段 | 做了什么 | 结果与证据 |
| --- | --- | --- |
| 后端选型 | 比较llama.cpp、KT/SGLang、Transformers/Accelerate；核算权重、RAM与磁盘 | 选择GGUF+mmap+CPU专家；KT完整V4示例的≥200GB RAM超出本机预算，通用disk offload不保证只读激活专家 |
| 环境验证 | 将WSL上限改24GB；实际推理选择Windows官方预编译CUDA包 | Qwen0.5B CUDA生成通过，未把WSL GPU可见当作CUDA计算通过 |
| 固定下载 | 只下载DeepSeek Q4五片，一份权重；预算保留40GB | 最终发布者SHA全部通过，GLM与视频权重未下载 |
| 断流恢复 | WSL curl提前EOF、代理连接失败后，受控切换原生Python，两个worker、512MiB有界Range | 每次追加核对206/Content-Range/总大小；不因观察超时重复启动写入者 |
| 损坏修复 | 第二片全哈希失败；独立复算确认，再按64MiB检查原有前8GiB | 128块中72块原位修复，之后整片及全部五片发布哈希通过；没有证明损坏原因 |
| 测量验证 | 小模型验证SSE、输出门禁、PID GPU计数、磁盘积分、下载与PID保护 | 小模型4次及增强版8次工具烟雾测试通过；与DeepSeek正式实验分开 |
| 正式实验 | CPU→GPU卸载→线程数→128token长输出→CPU/I/O诊断与带宽探针 | 详见下表；实验结束后停止本轮server，没有部署常驻服务 |

源码依据：[固定版server参数](https://github.com/ggml-org/llama.cpp/blob/b11429/tools/server/README.md)、[模型加载](https://github.com/ggml-org/llama.cpp/blob/b11429/src/llama-model.cpp)、[mmap](https://github.com/ggml-org/llama.cpp/blob/b11429/src/llama-mmap.cpp)、[Accelerate卸载机制](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)。关闭repack是为避免额外CPU实体buffer；lazy mode不是显式SSD专家调度，mmap也不保证没有页缓存压力。

下载证据：[原生过程快照](../artifacts/low-vram-inference/download-observations-native.jsonl)、[完整哈希诊断](../artifacts/low-vram-inference/shard2-hash-diagnosis.json)、[修复结果](../artifacts/low-vram-inference/shard2-repair.result.json)、[恢复后日志](../artifacts/low-vram-inference/deepseek-download-after-repair.log)。门禁使用结构化verified.json，避免PowerShell混合编码日志造成错误判断；文件大小/mtime改变时拒绝沿用旧证明。

## 模型身份与下载预算

- 后端：llama.cpp b11429，commit `d81235049384534c167caea52b85a694f6103d14`；[官方CUDA12.4发行包](https://github.com/ggml-org/llama.cpp/releases/tag/b11429)。
- 权重：`unsloth/DeepSeek-V4-Flash-GGUF`，revision `e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0`，`UD-Q4_K_XL`，5片共 **155,095,240,320 bytes**。
- 固定的是GGUF发布revision；发布模型卡未给出基础checkpoint转换时的精确SHA。它不是0731版本，不能直接套用后续同名模型或draft。
- 分片大小和全部SHA见 [下载清单](../../../code/low_vram_inference/deepseek-q4-manifest.json)，完整通过证明见 [deepseek-verified.json](../artifacts/low-vram-inference/deepseek-verified.json)。第一片仅约5MB元数据，不能单独推理。
- 权重保存在 `D:\ai-research\low-vram-inference\models\deepseek-q4`，不保存在Git仓库或Mac副本中。

历史候选预算（GB，未下载者只读发布元数据）：

| 候选 | UD-Q4_K_XL | UD-IQ4_XS | UD-IQ2_XXS |
| --- | ---:| ---:| ---:|
| DeepSeek-V4-Flash | 155.095（5片，已下载） | 137.904（4片） | 90.861（3片） |
| GLM-5.3-Flash | 199.707（6片） | 156.822（5片） | 101.845（4片） |

来源：[DeepSeek固定文件树](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0)、[GLM固定文件树](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/a38483c8cd5df544f53d70fb281afe97369d5ab6)。当时基线部署后D盘可用277.406GB，DeepSeek单份后122.311GB；两套Q4合计354.803GB，超出该预算。下载完成后观测到约187.46GB可用，期间盘上其他占用变化不能算成本项目释放空间。混合量化标签不等同每个tensor都是4bit，容量与质量须分开。

## CUDA小模型基线

Qwen2.5-0.5B-Instruct Q4_K_M，491,400,032 bytes，revision `9217f5db79a29953eb74d5343926648285ec7e67`，SHA256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`。算术输出 `2 + 2 equals 4.`，CUDA日志确认实际卸载。

`llama-bench`使用threads6、prefill128、decode32、3次重复，`-nopo 1`隔离默认运算卸载，保留warmup。

| 路径 | Prefill中位数token/s（范围） | Decode中位数token/s（范围） |
| --- | --- | --- |
| CPU，ngl0 | 318.87（291.66–319.11） | 83.63（82.59–84.04） |
| CUDA，ngl99 | 13570.70（9710.73–14862.80） | 329.79（311.09–335.56） |

[原始结果](../artifacts/low-vram-inference/baseline-20261009/) · [baseline.ps1](../../../code/low_vram_inference/baseline.ps1)。这是热运行环境微基准，不是DeepSeek性能或冷启动结果。

## 正式测量协议

共用参数：mmap、no-repack、no-host、lazy mode、cpu-moe、no-op-offload、no-kv-offload；ctx512、batch=ubatch32、单请求、temperature0、reasoning off，`cache_prompt=false`并核对实际cache_n=0。

- 加载时间：新server启动到health ready。TTFT：请求到首个非空SSE content，排除加载时间。
- 短题：数学`2+3`与猫句翻译，分别首次一次、同server再三次；每配置8次。输出仅2/5token，decode采用后端timings，不能代表长输出。
- 长输出：短提示27token、长提示150token，要求写SSD说明，实际每次截断于128输出token；客户端decode=`127/(总时长−TTFT)`。输出可读不等于完整300词回答或知识质量评测。
- 资源：约1秒采样PSAPI RSS/private/page faults、PDH物理磁盘读取、PID GPU dedicated/shared、整卡nvidia-smi；日志另存CUDA模型buffer。后续加入GetProcessTimes与GetProcessIoCounters。

重要边界：新进程不是严格冷磁盘，文件缓存未清空；磁盘积分属于整机，显存峰值属于整卡；PID GPU也是WDDM代理计数，不能直接当精确物理归属。缺失/无效counter保留null，不能补0；PageFaultCount含软缺页；shared GPU与private RAM不可直接相加。接口依据：[PSAPI](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo)、[PDH](https://learn.microsoft.com/en-us/windows/win32/api/pdh/nf-pdh-pdhaddenglishcounterw)、[WDDM统计限制](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/gpu-process-memory-counters-report-wrong-value)。

工具验证证据：[基础4次](../artifacts/low-vram-inference/instrumentation-check-20261009/)、[增强8次](../artifacts/low-vram-inference/instrumentation-gpu-pid-20261009/)。增强版PID dedicated峰值506,261,504bytes，日志模型buffer373.73MiB、compute2.22MiB，整卡4217MiB；口径不同，差额不能自动解释为泄漏。

## 短题40次结果

每题三次重复中位数，数学/翻译同序；40次均答对固定题意。

| GPU层/线程 | 加载秒 | TTFT秒 | Prefill token/s | Decode token/s | RSS GiB | 整卡MiB | 整机读GB |
| --- | ---:| --- | --- | --- | ---:| ---:| ---:|
| 0/8 | 22.63 | 28.16/29.57 | .641/.577 | .827/.449 | 24.32 | 992 | 292.35 |
| 1/8 | 23.63 | 27.20/30.11 | .663/.566 | .789/.735 | 24.55 | 1564 | 280.29 |
| 8/8 | 28.68 | 28.33/30.62 | .637/.556 | .811/.761 | 24.42 | 2654 | 279.20 |
| 32/8 | 30.70 | 25.60/25.10 | .705/.679 | .866/.472 | 24.64 | 6265 | 239.93 |
| 32/4 | 36.74 | 36.00/33.35 | .501/.511 | .730/.394 | 24.69 | 6286 | 245.94 |

CUDA模型buffer：ngl1/8/32分别536.83/1600.38/5221.29MiB。32层8线程的两题TTFT比CPU约缩短9%/15%，4线程更慢，更多GPU层不保证所有指标更快。配置顺序未随机化，缓存及其他活动未完全控制，不能宣称统计显著或纯因果效应。

各配置完整首次/重复请求、内存细项和采样：[CPU](../artifacts/low-vram-inference/deepseek-cpu-20261009/deepseek-ngl0-20261009-203906/)、[ngl1](../artifacts/low-vram-inference/deepseek-gpu1-20261009/deepseek-ngl1-20261009-204425/)、[ngl8](../artifacts/low-vram-inference/deepseek-gpu8-20261009/deepseek-ngl8-20261009-204950/)、[ngl32](../artifacts/low-vram-inference/deepseek-gpu32-20261009/deepseek-ngl32-20261009-205515/)、[t4](../artifacts/low-vram-inference/deepseek-t4-20261009/deepseek-t4-ngl32-20261009-210018/)。

## 长输出18次结果

下表8/16线程各每题3次，其余各每题1次；另有16线程诊断每题1次，总计18次128token请求。

| GPU层/线程 | 短TTFT秒 | 短decode token/s | 长TTFT秒 | 长decode token/s | 统计 |
| --- | ---:| ---:| ---:| ---:| --- |
| 32/8 | 53.172 | .622 | 169.862 | .751 | 各3次中位数 |
| 32/16 | 29.663 | .865 | 115.290 | .824 | 各3次中位数 |
| 32/16诊断 | 34.220 | .834 | 114.160 | .823 | 各1次 |
| 32/20 | 26.938 | .892 | 111.404 | .904 | 各1次 |
| 40/20 | 27.656 | .920 | 106.545 | .845 | 各1次 |

40层CUDA模型buffer6428.26MiB，整卡峰值7829MiB、RSS约24.37GiB。它的长提示decode反而慢于32层20线程；显存已接近预算，不宜直接加context或缓存。单次最好值不替代重复统计。

为保留计算中间步骤，下表列出8/16线程逐次数据；读量是生成阶段整机积分÷127，读取参考速度=`2.615286/读量`，不是全系统极限。

| 线程/请求 | 输入token | TTFT秒 | Decode | 读GB/token | 读取参考token/s | Prefill读取参考秒 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| 8/1 | 27 | 53.243 | .622 | 1.075 | 2.43 | 15.24 |
| 8/2 | 150 | 237.743 | .638 | 1.052 | 2.49 | 63.18 |
| 8/3 | 27 | 53.172 | .588 | 1.040 | 2.51 | 14.95 |
| 8/4 | 150 | 169.862 | .760 | .926 | 2.82 | 51.55 |
| 8/5 | 27 | 40.730 | .771 | .935 | 2.80 | 13.41 |
| 8/6 | 150 | 153.626 | .751 | .927 | 2.82 | 48.93 |
| 16/1 | 27 | 29.358 | .855 | .918 | 2.85 | 14.14 |
| 16/2 | 150 | 113.530 | .831 | .960 | 2.72 | 50.86 |
| 16/3 | 27 | 30.445 | .871 | .916 | 2.85 | 13.52 |
| 16/4 | 150 | 117.372 | .824 | .952 | 2.75 | 50.18 |
| 16/5 | 27 | 29.663 | .865 | .923 | 2.83 | 13.73 |
| 16/6 | 150 | 115.290 | .821 | .961 | 2.72 | 49.01 |

证据：[8线程与phase-analysis](../artifacts/low-vram-inference/limits-t8-20261009/limits-t8-ngl32-20261009-213406/)、[16线程与phase-analysis](../artifacts/low-vram-inference/limits-t16-20261009/limits-t16-ngl32-20261009-220850/)、[CPU/I/O诊断](../artifacts/low-vram-inference/cpu-io-profile-20261009/cpu-io-profile-ngl32-20261009-223547/)、[20线程](../artifacts/low-vram-inference/limits-t20-20261009/limits-t20-ngl32-20261009-224821/)、[40层](../artifacts/low-vram-inference/limits-gpu40-20261009/limits-gpu40-ngl40-20261009-225745/)。

## 带宽测量与条件上限计算

### 独立探针

SSD在现有第四分片只读：NO_BUFFERING、对齐buffer；每轮16MiB块顺序读4GiB、4MiB块固定随机读1GiB，各3轮，共15GiB。绕过Windows文件缓存，不控制设备缓存，也不等同真实4KiB缺页访问。RAM用8×128MiB相等buffer，memcmp每轮累计两输入共8GiB，尾字节不等自检通过；包含比较与线程开销。

| 探针 | 三次GB/s | 中位数GB/s |
| --- | --- | ---:|
| SSD顺序16MiB | 3.102/2.942/2.805 | 2.942 |
| SSD随机4MiB | 2.847/2.478/2.615 | 2.615 |
| RAM，1线程 | 21.635/21.520/21.255 | 21.520 |
| RAM，2线程 | 30.805/30.988/30.102 | 30.805 |
| RAM，4线程 | 35.785/35.259/22.938 | 35.259 |

[SSD数据](../artifacts/low-vram-inference/disk-read-bandwidth.json) · [RAM数据](../artifacts/low-vram-inference/ram-read-bandwidth.json)。单轮波动保留，不把中位数当物理峰值。

### 从量化大小到每token访问量

1. 284B参数纯4bit理想权重=`284e9×4/8=142GB`。实际155.095GB对应粗略平均4.369bit/参数，含高精度tensor及元数据。
2. 日志中专家张量：`43层×3组×1088MiB=147.170GB`。每层256专家选6，若选中权重每token读一次，逻辑专家访问=`147.170×6/256=3.449GB/token`。
3. 逻辑访问不等于SSD读取。设字节命中率h、对应访问模式有效带宽B，则读取情景 `t≥3.449×(1−h)/B`；本轮没有直接测到h。用整盘读取比例反推不能冒充专家命中率。

### 读取参考：保留当前读量，忽略计算

以16线程第5次为例：生成读取约`.923GB/token`，用随机4MiB探针校准，`t_read=.923/2.615≈.353秒`、参考速度约`2.83token/s`；实测`.865token/s`。首token前读35.920GB，读取参考时间=`35.920/2.615≈13.735秒`，实测TTFT29.663秒。长提示第6次读取128.171GB，对应约49.008秒，实测115.290秒。

全组读取参考2.722–2.855token/s，短/长TTFT读取参考13.516–14.145/49.008–50.861秒。它依赖整盘归因、约1秒积分边界与访问粒度假设，不可直接套给40层或另一缓存策略；实测只达到约30%不能解释为其余70%都可消除。

### CPU约束与硬件规格情景

16线程诊断的decode平均CPU活动约9.034/9.321逻辑核；短提示用户态1093.109、内核态281.845CPU秒，覆盖墙钟152.206秒。长提示末尾0.665秒无采样覆盖，分析已截去。进程读字节增量为0，但整盘大量读取：GetProcessIoCounters不能据此排除mmap硬缺页。

假设CPU工作量不变、20逻辑核等速且完美并行：短decode=`.8344×20/9.0335≈1.847`，长=`.8232×20/9.3207≈1.766token/s`。P/E、SMT、频率、自旋和算法工作量都会破坏这些假设；它只是缩放情景。完美重叠的通用参考是 `t≥max(Ddisk/Bdisk, Dram/Bram, Dpcie/Bpcie, C/F)`，无法重叠时可能相加。

| 参考情景 | 计算 | 不可外推的条件 |
| --- | --- | --- |
| 专家已全在RAM，探针校准 | `35.259/3.449=10.22token/s` | 忽略解量化、运算、其他访问，不是当前SSD路径 |
| 双通道DDR4规格 | `51.2/3.449=14.84token/s` | 最多双通道、忽略计算与片上复用 |
| 理想激活权重已全在GPU | 13B纯4bit为6.5GB，`256.032/6.5=39.39token/s` | 混合精度、动态专家、工作区和CPU/SSD路径均未计入 |

51.2GB/s=`3200e6×8×2`；256.032GB/s按GDDR6时钟的有效数据率假设换算；PCIe Gen4×8编码后=`16×128/130×8/8=15.754GB/s`。硬件规格参考、SSD校准与CPU缩放是不同场景，不能拼成一个“本机必达上限”。

**为何还没到参考值：** 实际还需CPU专家运算/解量化、串行层依赖、页管理、同步、其他内存访问；真实SSD请求也不同于4MiB探针。线程数变化有实测影响，已经否定“只有SSD带宽决定速度”的简化模型；尚未隔离各因素的准确占比。

## 2026-10-22：CPU、硬缺页与GPU时间线诊断

**判断：prefill优先减少专家权重分页；decode同时存在硬缺页、CPU运行库等待/调度和GPU同步成本。先做低成本的OpenMP等待策略对照，再推进减少专家分页；GPU算子重写后置。** 本轮只测量，没有应用加速改动。

沿用同一155.095GB GGUF和后端commit，Windows原生、`-ngl32 -t16 -c512 -b32 -ub32`、CPU专家、CPU KV、mmap、`-v`，端口18080。固定27-token短提示，temperature=0、禁用prompt cache，主要请求生成64token；不清空OS文件缓存。完整命令、失败尝试及来源见[运行索引](../artifacts/low-vram-inference/phase-profile-20261022/run-index.json)。

| 请求 | TTFT秒 | 客户端decode token/s |
| --- | ---: | ---: |
| 无profiler，前对照 | 30.006 | .8741 |
| 精简ETW | 29.669 | .8062 |
| 无profiler，后对照 | 29.162 | .7930 |
| Nsight 2026，仅CUDA | 28.998 | .7883 |

速度仍按`63/(请求总时长−首内容时间)`计算。后对照速率比前对照低约9.3%，ETW落在其中；不能把每次差异都归因于profiler，也不能据此宣布优化收益。

### CPU与硬缺页：线程状态证据

使用现成Windows Performance Toolkit/xperf，采集`PROC_THREAD+LOADER+CSWITCH+DISPATCHER+PROFILE+HARD_FAULTS+DISK_IO+DISK_IO_INIT`；有效ETL约2.33GB，报告丢失事件/缓冲均为0。为控制解析成本，事先固定四个1秒窗：请求开始后5–6秒，以及首内容之后5–6、25–26、45–46秒。[目标事件CSV压缩包](../artifacts/low-vram-inference/phase-profile-20261022/etw/target-events.csv.gz) · [分析JSON](../artifacts/low-vram-inference/phase-profile-20261022/etw/etw-analysis.json)。

| 1秒采样窗 | 平均正在CPU上运行的逻辑核数 | 硬缺页关联的离CPU等待并集，秒 | OpenMP占该窗CPU采样 |
| --- | ---: | ---: | ---: |
| prefill +5秒 | 3.96 | .952 | 4.8% |
| decode +5秒 | 10.50 | .549 | 46.9% |
| decode +25秒 | 11.74 | .426 | 55.0% |
| decode +45秒 | 11.45 | .492 | 47.0% |

计算过程：`CSwitch`重建每线程上CPU区间；`ReadyThread`区分阻塞和就绪排队。对每条目标PID的`HardFault`，I/O区间为`[TimeStamp−ElapsedTime, TimeStamp]`，与**同一TID**的离CPU等待相交，再跨线程取区间并集。因此表中的.549表示“这一秒中，至少一个目标线程有约.549秒在与硬缺页重合的阻塞中”，不是“整台机器54.9%的时间只等SSD”。不能只数`WrPageIn`：decode中大部分关联阻塞标为`Executive`，所以使用同TID与时间相交识别。窗起点已有的离CPU状态不一定能恢复，边界碎片可能少记。

**CPU占用不等于有效计算。** decode只有14.5–22.6%的CPU采样落在`ggml-cpu-alderlake.dll`，46.9–55.0%落在`libomp.dll`，另有内核/驱动工作；每秒还看到约2.9–3.1万次带`WrYieldExecution`原因的目标线程切出。它支持优先检查运行库等待与调度，但没有证明全部OpenMP时间都能消除，也不能把这些CPU采样比例当墙钟比例。旧的“20逻辑核完美缩放≈1.8token/s”情景更不能被解释为已测得的有效计算上限。

最初的默认WPR试采过重，归档超时；原始轨迹已从临时目录保存在Windows实验目录。其有效硬缺页汇总直接归因到目标GGUF：2,073,247次、I/O大小合计94.352GB，平均约44.44KiB/次，涵盖该次完整请求。[目标进程汇总](../artifacts/low-vram-inference/phase-profile-20261022/etw-broad/target-hardfault-summary.csv)。这是硬缺页请求字节口径，不能冒充不重叠的物理介质字节。它也解释了为什么4MiB SSD探针推得的约2.8token/s只能作条件参照：实际请求粒度差了约两个数量级。

### GPU与同步：单独采集CUDA

旧Nsight 2024.5不支持当前驱动暴露的CUDA 13.2接口，其数据只保留作失败/探索记录。已从[NVIDIA官方](https://developer.nvidia.com/nsight-systems/get-started)下载2026.5.1，验证NVIDIA数字签名，仅解包至`D:\nsys26`，不修改显卡驱动；安装包SHA与解包记录见[工具来源](../artifacts/low-vram-inference/phase-profile-20261022/cuda-2026/provenance.json)。

新版只记录CUDA，不重复采CPU切换；请求中途约60秒停止采样，生成仍继续至64token。以Nsight UTC起点对齐SSE，剔除采集首尾250ms；将kernel、CUDA graph整体跨度、memcpy/memset区间合并去重。graph跨度可能包含内部空隙，因此这是活动覆盖，不是SM利用率。

| 采样阶段 | 统计墙钟秒 | GPU活动并集秒/覆盖率 | CUDA同步API区间并集秒/覆盖率 |
| --- | ---: | ---: | ---: |
| 首token前 | 28.767 | 2.089 / 7.3% | 2.118 / 7.4% |
| decode片段 | 30.984 | 6.159 / 19.9% | 6.741 / 21.8% |

同步API主要为`cudaStreamSynchronize`，时长也包含主机/驱动开销，不能全解释为GPU在算。内部区间的2,146个GraphLaunch和26,234个Synchronize关联ID均找到对应活动；现有路径确实已使用CUDA Graph。采集结束仍报告1条未完成事件及完整性告警，裁边和关联检查不能证明绝无其他漏项，**这些比例用于诊断估计，不是严格理论上限**。本机使用软件插桩式CUDA trace；graph内部没有逐kernel展开。

[CUDA分析及告警](../artifacts/low-vram-inference/phase-profile-20261022/cuda-2026/cuda-analysis.json) · [SQLite压缩包](../artifacts/low-vram-inference/phase-profile-20261022/cuda-2026/cuda.sqlite.gz) · [Nsight原始报告](../artifacts/low-vram-inference/phase-profile-20261022/cuda-2026/cuda.nsys-rep)。ETW与CUDA来自不同请求，CPU核秒、离CPU等待、GPU活动还会重叠，不能拼成相加100%的饼图。

### 诊断后的验证顺序

1. **CPU等待策略A/B已完成：** 下节记录三对`KMP_BLOCKTIME=0`对照，明显节省CPU，但未证明稳定提速。下一次可在该等待策略下单独研究线程数/P-E核分配，不把几个改动混在一起。
2. **结构优化主线仍是减少专家分页：** 测专家重用和页访问，评估RAM驻留、缓存、预取/读合并。prefill采样尤其支持此方向。D盘现已确认是Samsung 970 EVO Plus 1TB；比起仅追求SSD顺序读规格，先减少缺页次数和等待更贴近证据。
3. **最后选GPU热点：** 先定位graph内具体算子再写CUDA/TileLang。仅作理想情景，若20%全在关键路径、其他成本不变，GPU段2倍速对应整体约`1/(.8+.2/2)=1.11倍`；活动覆盖率并非已测得的关键路径比例，不能把此式当收益承诺。

复现脚本：[采集器](../../../code/low_vram_inference/profile_phases.py)、[ETW导出](../../../code/low_vram_inference/export_phase_etw.py)、[CPU/缺页分析](../../../code/low_vram_inference/analyze_phase_trace.py)、[CUDA分析](../../../code/low_vram_inference/analyze_cuda_trace.py)。分析自检覆盖区间并集/裁边、稀疏窗口边界、CPU运行/就绪/硬缺页的合成时间线，以及Nsight PID解码；四组主要请求的64token输出文本SHA一致；真实请求检查输出长度和采样完成。结束检查确认18080已关闭、无遗留llama-server/内核录制/Nsight session，原8080仍为PID26160，见[清理快照](../artifacts/low-vram-inference/phase-profile-20261022/final-host-check.json)。所有结论仅适用于当前短输入、ctx512、单请求配置，不能直接外推到200k/1M上下文。

## 2026-10-22：KMP_BLOCKTIME=0三组A/B

**结论：显著节省CPU；decode仅小幅改善，TTFT改善不一致，尚不能宣称稳定提速。** 按逐对`(B/A−1)×100%`取中位数，请求CPU占用下降**62.46%**、decode CPU占用下降**67.66%**；decode token/s提高**2.22%**，TTFT缩短**5.06%**。这是三个配对的描述统计，不是显著性或长期收益保证。

A为子进程不设置`KMP_BLOCKTIME`，B仅对子进程设置`KMP_BLOCKTIME=0`。继续使用上述155.095GB GGUF、固定b11429后端、Windows原生`-ngl32 -t16 -c512 -b32 -ub32`及全部其他相同参数。每次启动新server，同一27-token提示，temperature=0、禁用prompt cache、生成64token；不开ETW/Nsight。预先固定顺序**A→B、B→A、A→B**，每次完成后间隔5秒再启动下一进程；OS文件缓存不清空。其他已记录的OpenMP环境变量均未设置，全局环境和原8080服务不变。

| 组 / 顺序 | 条件 | Decode token/s | TTFT秒 | 请求平均CPU | Decode平均CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / A→B | A 默认 | .8156 | 28.186 | 42.10% | 47.57% |
| 1 / A→B | B =0 | .8337 | 28.309 | 16.31% | 16.11% |
| 2 / B→A | B =0 | .8242 | 25.848 | 15.79% | 15.54% |
| 2 / B→A | A 默认 | .8028 | 29.862 | 43.37% | 48.98% |
| 3 / A→B | A 默认 | .7906 | 29.606 | 41.63% | 46.61% |
| 3 / A→B | B =0 | .7914 | 28.108 | 15.63% | 15.08% |
| 各列中位数 | A 默认 | .8028 | 29.606 | 42.10% | 47.57% |
| 各列中位数 | B =0 | .8242 | 28.108 | 15.79% | 15.54% |

中位数行的各指标不一定来自同一请求；逐对收益中位数也不等于两个条件中位数之比。三组decode变化分别**+2.22%、+2.67%、+0.09%**，TTFT分别**+0.43%、−13.44%、−5.06%**，请求CPU占用分别**−61.26%、−63.59%、−62.46%**。CPU节省在反向运行时也重复出现；速度差异较小，不能排除缓存、温度/频率与后台活动的影响。

### 计算口径与检查

- `TTFT`：发送HTTP请求到第一个非空内容；不包含启动server和加载模型。
- `decode token/s = (64−1)/(请求总时长−TTFT)`。第一组A：`63/(105.4267775−28.1860373)=.8156317`；B：`63/(103.8712930−28.3086380)=.8337452`，相对变化`(.8337452/.8156317−1)×100%=2.2208%`。
- `CPU秒 = Δ(GetProcessTimes.user + kernel)`；`平均逻辑核当量 = CPU秒/阶段墙钟秒`；`全机归一化CPU% = 平均逻辑核当量/20×100%`。只统计目标server进程，包含运行库活动和内核工作，不能等同有效算术。约1秒采样的累计计数在请求开始、首内容和结束边界做线性插值；结束后继续采样2秒，确保完整夹住边界。
- 第一组A请求CPU为887.6434秒，`887.6434/105.4267775/20×100%=42.0976%`；B为338.8079秒、16.3090%。六次运行的最大采样间隔为1.120秒以内，调用方UTC锚点与内部计时的结束偏差绝对值小于4ms。
- decode每输出token的CPU秒，中位数从**11.792降到3.810**；逐对降幅中位数**67.69%**。这说明节省不只是墙钟分母变化；但CPU秒减少不是同等比例的延迟或能耗收益。

六次请求均完整输出64token，文本SHA256全部为`ad52a847d0b7bc042fd232b703030115563cb2ccb2c27d9b4ab33f44e2a5eca4`；server命令、其余记录的OpenMP环境一致，采样与资源门禁通过。仅使用包含软缺页的普通进程计数，**本轮没有重新测量硬缺页等待、GPU时间线或整机功耗**，不能据此声称SSD/GPU成本降低。

[LLVM文档](https://openmp.llvm.org/design/Runtimes.html#kmp-blocktime)说明该变量控制并行区结束后进入休眠前的等待时间，文档默认值200ms；本实验A的定义是“未设置”，没有额外打印运行库最终生效值。CPU大幅下降与减少主动等待的解释相符，具体等待点仍需源码/更细采样验证。当前适合作为**节省CPU的子进程启动选项**保留；速度目标仍应推进专家访问/分页观测，或另开单变量线程数实验，GPU算子重写继续后置。结果只适用于本轮短输入、ctx512、单请求，不外推到200k/1M。

复现：[三组编排与CPU统计脚本](../../../code/low_vram_inference/run_kmp_ab.py)调用[采集器](../../../code/low_vram_inference/profile_phases.py)，在原Windows Python环境执行`python run_kmp_ab.py --root D:\ai-research\low-vram-inference`。脚本先检查既有权重证明和空闲18080，逐次启动/清理server，不改全局环境；CPU插值与边界覆盖自检为`python run_kmp_ab.py --self-check`。

证据：[批次汇总及每组变化](../artifacts/low-vram-inference/kmp-ab-20261022/batch.json) · [原始SSE、采样、server日志与源码快照](../artifacts/low-vram-inference/kmp-ab-20261022/) · [验证记录](../artifacts/low-vram-inference/kmp-ab-20261022/validation.json) · [结束环境](../artifacts/low-vram-inference/kmp-ab-20261022/host-after.json)。结束检查：18080关闭、无llama-server遗留、WPR未录制，8080仍为PID26160，进程/用户/系统级`KMP_BLOCKTIME`均未设置。

## 300k与1M上下文：只有预算推算

已测ctx512、最长输入150token，不能据此宣称最大context可用。模型/固定GGUF元数据为1,048,576；模型上限不等于本机容量与延迟。[官方config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json)。

固定commit的[DSV4缓存源码](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-kv-cache-dsv4.cpp)使用K-only raw/SWA、CSA、HCA与indexer，不能套普通完整K/V公式。现有单序列、F16缓存、ubatch32、raw256cells配置：

```text
CSA_cells = pad256(ceil(N/4)); HCA_cells = pad256(ceil(N/128))
raw = 43 × 256 × 512 × 2 bytes
CSA = 21 × CSA_cells × 512 × 2
HCA = 20 × HCA_cells × 512 × 2
indexer = 21 × CSA_cells × 128 × 2
compressor state ≈ (1.31 + 10 + .33) MiB
```

| 容量N | KV与状态GiB（估算） |
| --- | ---:|
| 512 | .033 |
| 32768 | .232 |
| 65536 | .442 |
| 300000 | 1.948 |
| 1048576 | 6.741 |

ctx512日志与公式一致；[计算数据](../artifacts/low-vram-inference/long-context-kv-estimate.json)。未计attention/indexer工作区、激活、checkpoint/缓存副本、权重页、输出、系统及多会话。4bit是权重标签，不表示KV为4bit；CPU KV增长会挤占专家页RAM。以`.5–1.35输入token/s`简单线性演示，300k约2.57–6.94天、1M约8.99–24.27天；**这不是ETA**，batch、缓存和attention会改变吞吐。

实际最大可用上下文尚未测出。下一阶段应逐级2k→4k→8k→16k→32k→64k，填入真实token并同时验证质量、内存和延迟；先经小模型验证工具，不直接跑300k。RAG/分层摘要可处理大型资料库，但不等于模型同时读取全部上下文。

## 2026-10-22：双Spark与双PRO的200k／1M条件成本模型

**这里没有新增GPU实测，也没有已经校准的采购级速度预测。** 下表是为制定试跑预算而计算的情景；效率、访存代理和其他开销尚未测量，实际可在这些情景之外。200k指实际输入200,000token，1M指1,000,000token，不是只把context容量设大。模型已加载，无prefix复用、无MTP，单请求；保持同一155.095GB文件，但全部权重、KV与计算需放到加速器，不能沿用现有`--cpu-moe --no-kv-offload`路径。

先比较**按层切分**：同一token依次经过两设备，所以带宽用单设备273／1792GB/s，不乘二。Spark通过本地/RPC后端，PRO在同一Linux主机内；张量并行、跨microbatch流水线或专家并行应另测，不属于下表。两设备仍可增加容量与多请求吞吐。按本模型的ubatch512，F16 K-only缓存约1.333／6.456GiB，权重加缓存约145.777／150.899GiB；继承原ubatch32时分别为145.756／150.878GiB。均未计工作区、系统、临时副本与单设备分配不均。

### Decode：先算字节，再代入效率

本地旧日志已确认专家129个张量为MXFP4，512个张量为Q8_0，另有BF16/F32等；文件名`UD-Q4_K_XL`不能解释为所有活跃参数都按纯4bit读取。保守的一次权重访问代理为：

```text
专家全部 = 43 × 3 × 1088 MiB = 147.169739 GB
每token激活专家 = 147.169739 × 6/256 = 3.449291 GB
其余文件字节 = 155.095240 − 147.169739 = 7.925502 GB
D_weight_proxy = 3.449291 + 7.925502 = 11.374792 GB/token
```

非专家整读包含不必全读的embedding与元数据，因而是保守代理；重复加载和临时量又可能增加流量。之前13B“纯4bit=6.5GB”只是假设参照，不适用于这份混合格式的实际活跃读量。

从固定[模型图源码](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/deepseek4.cpp#L590)可见：21个CSA层扫描indexer，拼接raw/压缩KV；20个HCA层也拼接并读压缩KV。因此另设：`D_KV_proxy = 2×(raw+CSA+HCA) + LID + 2×(raw+HCA+21×512×512×2)`。第一项模拟concat读写，第二项为一次LID扫描，第三项模拟K/V读取。它不是实测DRAM字节：融合、缓存会减少，重复加载、未融合score和mask/top-k会增加。

| 实际输入 | D_weight代理GB/token | D_KV代理GB/token | 总代理GB/token | Spark带宽除法参考token/s | PRO参考token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 200k | 11.375 | 2.732 | 14.107 | 19.35 | 127.03 |
| 1M | 11.375 | 12.909 | 24.284 | 11.24 | 73.79 |

最后两列仅为`标称带宽/D_proxy`，忽略其他开销；**不是全系统绝对上限**。中间情景人为取有效带宽40%、每token剩余开销Spark20ms／PRO5ms：

```text
t_decode = D_proxy/(η_B × B_spec) + δ
200k Spark: 14.106595/(.4×273) + .020 = .14918秒 → 6.70 token/s
200k PRO:   14.106595/(.4×1792) + .005 = .02468秒 → 40.52 token/s
1M同式代入24.283615GB → Spark 4.13、PRO 25.72 token/s
```

η与δ没有在目标硬件上标定，PRO/Spark的实际η也可能不同。敏感性输入仅为三个例子，不是预测上下界或置信区间：

| 情景 | η_B | δ Spark／PRO | Spark 200k／1M token/s | PRO 200k／1M token/s |
| --- | ---: | --- | --- | --- |
| 较低效率 | 25% | 40／10ms | 4.05／2.53 | 24.10／15.58 |
| 中间假设 | 40% | 20／5ms | 6.70／4.13 | 40.52／25.72 |
| 较高效率 | 60% | 10／2.5ms | 10.40／6.32 | 64.02／39.86 |

### Prefill与TTFT：不能用decode速度除输入长度

选ubatch512作基准。若各token的6/256专家选择独立、均匀，512token触及的专家比例为`1−(1−6/256)^512≈99.9995%`。在每块权重只从设备内存读取一次的代理下，200k有391块、1M有1954块，约读60.642／303.055TB权重。路由集中、跨块缓存、kernel分块重复读取都会改变它。

矩阵运算采用由架构舍入的13B参数/input-token代理：`C_mat≈2×13e9×N`。长上下文还要计算attention/indexer。按因果平均历史长度N/2、64头、head512、indexer head128推导：

```text
LID = 21 × 2 × 64 × 128 × N²/(2×4)   = 43,008 N² FLOP
HCA = 20 × 4 × 64 × 512 × N²/(2×128) = 10,240 N² FLOP
SWA = 43 × 4 × 64 × 512 × 128 × N
CSA理想只算选中的512条 = 21 × 4 × 64 × 512 × 512 × N
CSA扫描全部压缩位置   = 21 × 4 × 64 × 512 × N²/(2×4) = 344,064 N²
```

乘4包括QK与AV各一次乘加；分母2为因果三角。两种CSA情景表示不同有效工作量；dense mask、FlashAttention分块跳过和稀疏kernel会落在不同位置，不保证被这两种理想模型夹住。`A`为上面LID+HCA+SWA+所选CSA情景：

| 输入 | C_mat PFLOP | A理想选中 PFLOP | A全压缩扫描 PFLOP |
| --- | ---: | ---: | ---: |
| 200k | 5.200 | 2.556 | 16.037 |
| 1M | 26.000 | 55.379 | 398.033 |

中间情景再人为取等效操作率Spark50TFLOP/s、PRO200TFLOP/s。它们仅用dense FP4参照500／2000TFLOP/s的10%作为灵敏度标尺，**不是本GGUF的实测利用率，也不表示Q8、MXFP4、F16 attention能共享同一峰值/效率**。低/高情景分别用参照的5%／20%。广告1PFLOP／4000TOPS有结构化稀疏条件，不能直接当这份GGUF的有效算力。

```text
T_mat_stage = max(权重读取代理/(η_B×B_spec), C_mat/F_eff)
TTFT ≈ T_mat_stage + A/F_eff + T_extra
200k Spark，理想选中：max(60.642TB/109.2GB/s,5.2PF/50TF) + 2.556PF/50TF
                       ≈ 555.3 + 51.1 = 606.4秒 + T_extra
```

`T_extra`尚未知，可能很大：包含tokenizer、上传/排队、输出head/采样、softmax/top-k、launch/RPC、额外KV/indexer/temp访存、各种非矩阵操作。分阶段算法与精度不同时，有效操作率也需要分开标定。下表是**TTFT主体资源时间，所有格子都要加T_extra**：

| 输入 | 配置 | 理想选中CSA，分钟 | 全压缩CSA扫描，分钟 |
| --- | --- | ---: | ---: |
| 200k | 双Spark | 10.11 | 14.60 |
| 200k | 双PRO | 1.62 | 2.75 |
| 1M | 双Spark | 64.71 | 178.93 |
| 1M | 双PRO | 11.66 | 40.22 |

这些不是实际TTFT预测区间。仅将中间情景ubatch增到2048，理想选中主体就变为Spark200k **3.17分钟**／PRO **.65分钟**、Spark1M **30.03分钟**／PRO **6.78分钟**，尚未验证能否分配工作区及保持假定效率。反之，原ubatch32会显著减少块内专家重用。不能将小batch旧实验的prefill速度简单放大到新机器。

**缓存容量小不等于prefill工作区小。** 未融合LID若显式生成`64 heads × (N/4) × 512 queries × F32`，200k单个score张量就是6.554GB，1M达32.768GB；后续中间量还会增加。它足以超过双PRO中某一张卡扣除权重后的余量。因此融合indexer/attention、chunk、每卡权重分配和峰值必须实测，不能只凭约6.5GiB KV宣称1M可运行。

### 多花的钱换来什么：先设验收门槛

沿用上一轮**预算假设**双Spark11万元、双PRO50万元，采购价约4.55倍，差39万元，尚非确认报价。中间情景得到decode约6.04／6.23倍，TTFT主体约4.45–6.23倍；这些优势大部分已经由4倍算力参照、6.56倍带宽和共同效率写进输入，**不是独立的性能证据**。不能据此证明整机值得买。

只按采购价/吞吐算，双PRO要比双Spark快**4.55倍以上**才更划算；未计电费、售后、噪声与研究效率。仅看带宽模型，若PRO实际效率只有Spark的一半，优势约`1792/273×.5=3.28倍`，对应单位吞吐采购价反而约`4.55/3.28=1.39倍`。真实结论可能翻转。更大batch、张量并行或MTP也可能改变比值，必须以同量化、同质量、同输入、同解码策略另作对照。

购买前需要两套候选分别提交：2k→32k→200k→1M实际token输入的SSE与timings；无prefix命中、单请求生成至少128token；记录模型SHA、backend commit、CUDA/精度/融合选项、batch、split方式、每卡/每节点显存与临时峰值、CPU回退、RPC真实字节和设备吞吐。先使三次重复与质量门禁通过，再以TTFT和decode验收；长输入需检查跨段检索/文档问答正确性，不能只看输出长度。仅配大context、分配成功，或另一量化/推测解码的快结果都不能替代。

复算：[estimate_hardware.py](../../../code/low_vram_inference/estimate_hardware.py)（`--self-check`及`--output`）· [全部情景JSON](../artifacts/low-vram-inference/hardware-cost-model-20261022/scenarios.json) · [固定源码与来源快照](../artifacts/low-vram-inference/hardware-cost-model-20261022/)。本轮只读既有记录/源码和做本地计算，没有启动现有电脑或候选GPU的推理实验。

### 来源与可比性边界

- [Spark官方硬件手册](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)给出单机273GB/s与FP4 **结构化稀疏**1PFLOP；[PRO工作站规格](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf)给出1792GB/s，4000TOPS同样带FP4稀疏条件。未找到这两页对本模型GGUF的有效BF16/FP16吞吐保证，不能直接拿广告TOPS计算TTFT。
- 当前155GB文件的专家是MXFP4；固定commit的[mmq.cu](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-cuda/mmq.cu#L116)在合适Blackwell编译目标下，MXFP4的MMQ默认可选4bit激活与原生W4A4；`GGML_CUDA_MMQ_PREC=q8`可覆盖。小batch专家算子则先走[MMVQ选择](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/ggml/src/ggml-cuda/ggml-cuda.cu#L1965)。不能因此认为decode也达到FP4 Tensor峰值；跨硬件比较需同时验证激活精度及输出质量。
- [ConnectX-7官方手册](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#the-qsfp-ports-and-cables)明确每端口最高200Gb/s，即25GB/s线速。固定commit的[RPC文档](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/rpc/README.md)支持可选RoCE RDMA，否则回退TCP；容量切分不等于两台设备同时计算同一个token的全部层。200Gb/s也不等于跨机读取模型全部权重的带宽。
- 未找到同时匹配**本155.095GB固定revision、无推测解码、无prefix cache命中、实际200k／1M完整输入**的双Spark或双PRO公开测量。作者[ailone双Spark报告](https://ailone.io/en/models/deepseek-v4-flash/)用另一checkpoint、vLLM、TP2及DSpark，192k输入TTFT报告81.8秒；所谓900k是context配置，不足以证明实际900k输入decode性能。[gisenberg单PRO记录](https://github.com/gisenberg/local-model-eval/blob/main/results/DEEPSEEK_V4_FLASH_RTXPRO6000.md)主要是0731 IQ2／EXL3及旧preview IQ2，均不能充当本GGUF的直接标定。它们只能证明后端、量化、输入长度与推测解码选择足以显著改变结果。

## 复现与维护

入口：[run_deepseek.py](../../../code/low_vram_inference/run_deepseek.py) · [下载与证明检查](../../../code/low_vram_inference/download_deepseek.py) · [SSD探针](../../../code/low_vram_inference/measure_disk_read.py) · [RAM探针](../../../code/low_vram_inference/measure_ram_read.py)。以下在现有Windows根目录执行；不是本轮新启动的任务：

```powershell
$py = 'C:\Users\marks\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\python.exe'
& $py D:\ai-research\low-vram-inference\download_deepseek.py --check-verified
& $py D:\ai-research\low-vram-inference\run_deepseek.py --gpu-layers 32 --threads 16 --workload limits --label limits-t16
```

`--workload fixed`复现短题；profile用于两题诊断；20线程与40层对比只改变相应参数。启动flag在各目录command.json中；运行器拒绝占用已有实验端口。独立带宽探针应在推理停止后运行，避免互相干扰。每次保存模型/后端版本、命令、prompt/输出、SSE、timings、资源采样、首次/重复顺序；扩大测试应重新核算RAM、VRAM、磁盘与时间。

GLM、视频生成、其他量化、专家缓存/预取、长上下文和常驻兼容API服务尚未做正式目标实验。llama-server可作为兼容API研究入口，常驻服务的端口、启动和质量仍需另行验证。以后实验直接追加到本文并链接新artifacts；只修改硬件时更新00，研究假设与学习安排更新02，避免再按每次进度新建一篇文档。
