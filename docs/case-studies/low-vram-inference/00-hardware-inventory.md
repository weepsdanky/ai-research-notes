# 低显存推理：硬件与运行环境盘点

整理日期：2026-10-22；主体实机观测来自2026-10-09；本轮补查D盘映射并复核驱动595.97，其他历史余量不代表当前状态。研究对象是本机大模型推理、卸载与加速；computer-use 在独立的 `Documents/Github/cua` 项目进行。

本项目只维护三份文档：**本机有什么**（本文）→ [做过什么、测到什么](01-experiments-and-results.md) → [下一步研究与八周学习计划](02-open-source-learning-roadmap.md)。实验脚本在 [code/low_vram_inference](../../../code/low_vram_inference/)，原始日志与JSON在 [artifacts](../artifacts/low-vram-inference/)。

## 硬件配置

| 项目 | 已确认结果 |
| --- | --- |
| SSH | **`ssh mark`**；用户称计算机为 mark-1，Windows `hostname` 返回 `Mark` |
| CPU | Intel Core i9-13900H，6P+8E、14物理核、20逻辑线程 |
| Windows可见物理RAM | 33,960,431,616 bytes，约31.63GiB |
| 内存条 | 2×16GiB，DDR4-3200；SMBIOS类型26、配置速率3200 |
| GPU | NVIDIA RTX 4060 Laptop GPU，8188MiB显存；128bit GDDR6 |
| NVIDIA驱动 | 595.97 |
| GPU最大值查询 | memory clock 8001MHz、SM clock 3105MHz、PCIe Gen4×8；不是持续运行实测 |
| 物理SSD | WD PC SN560 1TB、Samsung 970 EVO Plus 1TB，均为NVMe；2026-10-22补查确认D:对应Samsung（Disk 1），C:未在本轮关联 |
| C: | 初查总容量994.47GB、空闲278.04GB |
| D: | 总容量998.88GB；目标权重完成后的最后快照空闲约187.46GB，后续操作前需重查 |

2026-10-22补查：[D盘、驱动与实验清理快照](../artifacts/low-vram-inference/phase-profile-20261022/final-host-check.json)。

内存/GPU查询证据：[hardware-ceiling-inventory.json](../artifacts/low-vram-inference/hardware-ceiling-inventory.json)。规格依据：[Intel i9-13900H](https://www.intel.com/content/www/us/en/products/sku/232135/intel-core-i913900h-processor-24m-cache-up-to-5-40-ghz/specifications.html)、[NVIDIA笔记本GPU对照](https://www.nvidia.com/en-us/geforce/laptops/compare/)。

## Windows与WSL环境

| 项目 | 已确认结果 |
| --- | --- |
| 用户目录 | `C:\Users\marks` |
| 研究根目录 | `D:\ai-research\low-vram-inference`；下有 `bin`、`models`、`results` |
| 已验证推理环境 | **Windows原生** llama.cpp b11429 / `0.6.0-dev`，官方CUDA 12.4 x64发行包 |
| 固定后端commit | `d81235049384534c167caea52b85a694f6103d14` |
| 原生Python | `C:\Users\marks\AppData\Roaming\uv\python\cpython-3.12.9-windows-x86_64-none\python.exe`，用于下载、请求与监测 |
| WSL | Ubuntu-20.04、WSL2 Linux 6.18.33.2；GPU可见，不等于已经验证WSL CUDA计算 |
| WSL内存变更 | 初始约15GiB；设置 `.wslconfig` 的 `memory=24GB` 并重启后，总量约23GiB、available22GiB、swap6GiB |
| 配置备份 | `C:\Users\marks\.wslconfig.before-low-vram-20261009`；保留nestedVirtualization与mirrored networking |
| 隔离边界 | 实验使用loopback端口18080；原有8080服务保持独立 |

最初WSL约15GiB是当时的虚拟机内存预算，不是Windows少了16GB。24GB是WSL上限，不表示预先占满宿主RAM；本轮未显式配置swap，6GiB为重启后的观测。实际性能数据全部来自Windows原生路径，不能混称为WSL结果。

初查WSL系统Python为3.8.10，未找到torch/transformers/accelerate/ktransformers/numpy，未遍历其他虚拟环境；git/cmake/gcc可见，nvcc不在PATH。此历史检查不代表当前整机所有环境。运行已下载的llama.cpp二进制需要驱动和发行包运行库；源码编译另需MSVC、CMake与CUDA Toolkit/nvcc，见[官方构建说明](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)。

## 带宽与容量预算

| 量 | 数值与口径 |
| --- | --- |
| SSD顺序读 | 16MiB块，中位数2.942GB/s |
| SSD随机读 | 4MiB块，中位数2.615GB/s；不是实际缺页粒度的带宽 |
| RAM探针 | memcmp两输入读取，4线程中位数35.259GB/s；含比较与调度开销 |
| RAM规格参考 | 假设双64bit通道，`3200e6×8×2=51.2GB/s`；通道配置与持续带宽未独立验证 |
| GPU/PCIe规格参考 | GDDR6时钟换算假设约256.032GB/s；PCIe编码后单向15.754GB/s；均不是实测传输 |

测量方法、原始轮次和计算边界见 [实验记录](01-experiments-and-results.md#带宽测量与条件上限计算)。GB为十进制、GiB为二进制。WSL虚拟磁盘显示的逻辑空闲不能替代宿主物理盘预算。

已存一份155.095GB DeepSeek GGUF，远大于RAM与VRAM，依靠mmap分页运行。后续保留至少40GB磁盘空间，下载前重查可用bytes；避免下载缓存与最终目录各保存一份实体权重。显存空闲、RAM available和磁盘余量都会变化，不能直接将表中的历史值当作下一次实验预算。
