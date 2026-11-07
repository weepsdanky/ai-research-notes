# Qwen Next Lab：在 16GB Mac mini 上学习混合语言模型

目标：用短小的 PyTorch 代码，从数据、forward、loss 到 optimizer 更新，走通
**GDN + global attention → indexer 蒸馏 → QSA**，而不是下载官方权重做推理。

这是受 [Qwen3.8-Next 架构论文](https://arxiv.org/html/2608.30320v1) 启发的缩小学习实现，
不是官方 Qwen 模型，也不兼容官方 checkpoint。不期待它获得 Qwen 的对话、中文或推理能力。

**本端只编写、检查和上传源码。模型执行、依赖安装、测试、下载语料与训练都交给 Mac mini 上的 ChatGPT。**
把 [MAC_MINI_GOAL.md](MAC_MINI_GOAL.md) 中的任务交给它执行。

## 1. 先做什么

| 顺序 | 实验 | 要回答的问题 | 完成证据 |
|---|---|---|---|
| 0 | CPU / MPS 单元测试 | 公式、因果性、梯度是否正确？ | 测试日志、环境版本 |
| 1 | 54 万参数，Shakespeare，20–100 steps | 整个训练链能否跑通？ | 有限 loss、checkpoint、续跑一致性 |
| 2 | 801 万参数，20 steps 测速 | 16GB 上速度与内存是否合适？ | sec/step、tokens/s、RSS、MPS 内存、swap |
| 3 | 选择 205 万或 801 万，Shakespeare，1,000 steps | 能否学到角色名、换行和局部语言模式？ | 固定验证集 loss + 真实生成文本 |
| 4 | 相同模型，OWT 子集 + 10% Shakespeare，1,000 steps | 能否从戏剧文本扩展到网页文本？ | 两份验证集分别报告 |
| 5 | 冻结 backbone，dense distillation，100 steps | indexer 能否模仿 attention 的重要性？ | KL、冻结参数检查 |
| 6 | sparse training，500 steps | 选块后 LM loss 与输出有什么变化？ | sparse / dense 同权重验证对照 |

这些步数是有界学习实验预算，不是收敛保证。短跑正常后才继续；阶段 3–6 可跨多个会话续跑。
先完成一次流程，再做相同初始化、数据、token 预算下的 AdamW / Muon 对照。
不要同时改模型宽度、数据、优化器再把 loss 差异归因于一个组件。

## 2. 参数量与资源

| `--size` | width | 层数 | heads | expert hidden | routed experts / top-k | shared expert | 总参数量* |
|---|---:|---:|---:|---:|---|---:|---:|
| micro | 64 | 4 | 4 | 96 | 4 / 2 | 1 | 536,648 |
| mini | 128 | 4 | 4 | 192 | 4 / 2 | 1 | 2,050,168 |
| small（主配置） | 256 | 4 | 4 | 384 | 4 / 2 | 1 | 8,007,896 |

\* 从当前模块尺寸静态核算；实际值由 `train.py` 的 `parameters` 打印确认。
四层的 mixer 为 `GDN → GDN → GDN → Attention`，每层还有一个 MoE；每个子层单独使用 GR。
QSA indexer 从建模起就存在，但 `pretrain` 不调用、不更新它，因此阶段切换不改变 checkpoint 结构。

主配置 FP32 权重约 32MB；用全部 AdamW 估算权重、梯度、两份动量约 128MB。
这**不是训练总内存**：GDN 反传图、attention 矩阵、激活、MPS 缓存、Python 和系统都要占内存。
保守起步：FP32、microbatch=1、seq=64；验证后用 seq=128、accum=4（每步 512 byte tokens）。
没有 AMP、compile、分布式、CUDA/Triton 依赖。16GB 主要约束也包括共享内存与系统余量。

目标是进程训练工作集保持在约 8GB 以内并避免持续 swap；这是验收目标，尚非实测。
RSS 与 MPS driver memory 是不同统计口径，统一内存环境下不能简单相加。
若速度不合适，先选 mini，再降低 seq / accum；不要关闭 MPS 内存保护强行运行。
显式 GDN scan 与小矩阵调度可能比参数存储更早成为瓶颈，**不承诺训练时长**。

磁盘先预留约 5GB：Python/PyTorch 环境、数据缓存、若干 checkpoint。OWT 的默认编码数据
约 42MB + 文档结束符；网络读取和 Parquet row-group 缓冲不等于这个数字。
不需要先买硬盘，也不下载完整 OpenWebText 或官方模型权重。扩大语料前再量磁盘。

## 3. 如何读代码

| 文件 | 阅读重点 |
|---|---|
| [model.py](model.py) | `RMSNorm` → `partial_rope` → `delta_scan` → `GDN` → `GatedResidual` → `MoE` → `Indexer` → `Attention` |
| [optim.py](optim.py) | 参数分组 → 动量 → Newton–Schulz → 更新；二维参数不等于一律用 Muon |
| [prepare.py](prepare.py) | byte tokenizer、固定来源、文档级划分、数据上限、hash |
| [train.py](train.py) | 读 batch → forward → 三种 loss → backward → clip → step → checkpoint |
| [sample.py](sample.py) | 按训练窗口重算 prefix，逐 byte 采样 |
| [test_lab.py](test_lab.py) | 用具体张量检查核心语义，而不是只看代码能 import |

建议边读边打印一个 `[B=1,T=8,D=32]` 的中间张量。不要一开始改整套架构。

### 与论文的对应和差异

| 部分 | 本项目 |
|---|---|
| §2.1.1，式 1–11 | GDN 衰减、误差写入、单位 q/k、causal short conv、sigmoid 输出门 |
| §2.2，式 30–34 | 四分支 GR；分支独立 RMSNorm、逐通道读门、逐分支写门；rank=d/8 |
| §2.1.2，式 12–20 | MQA indexer、先平均池化再 partial RoPE、完整块筛选、保留未完成尾块、两阶段 KL；教学版 student 分数采用固定温度缩放 |
| §3.1 | 学习版 FP32 Muon，8 次 Newton–Schulz；embedding/head/router/GR/indexer/norm 用 AdamW |
| MoE | 教学选择：4 routed experts、top-2 + 1 shared expert；显式 token dispatch + 0.01 balance loss |
| tokenizer | UTF-8 byte + EOD（257 词表），不是 Qwen tokenizer；token 数与论文不可直接比较 |
| 输出 | 不共享 embedding/head 权重；最后对 GR 分支取平均，是教学选择 |
| QSA core | 用 dense 矩阵 + sparse mask 表达选块语义；**并未降低 core 的 O(T²) 计算/存储** |
| indexer | 4 个小 query heads；head dimension 随模型缩小，旋转一半维度；r=4，K=32 |
| 暂不实现 | 巨型 n-gram 表、MTP、KV/recurrent 推理缓存、FP8、FlashQLA、fused sparse kernel、分布式 Muon |

这里的 GR 不等于 Attention Residual（跨历史层做 attention），也不等于 mixer 输出处的 sigmoid 门。
GDN 和 QSA 分布在不同层；只有每第四层的 global attention 切到 QSA，GDN 层保留。
zero-centered RMSNorm 的 gain 是 `1 + delta`，delta 初始化为 0；它不减去激活均值。

QSA 的 mask 允许每个 query 看见最多 K 个完整块内 tokens，另加 0–r-1 个尾部 tokens。
短 prefix 不足 K 时会退化为 dense。训练阶段 2 只在已选择的完整块上算 KL；
没有完整块的 early query 不参与 KL，尾部仍参与 LM attention。

Mac mini 实测发现，未经缩放的 indexer head 分数之和配合原蒸馏学习率会出现
ReLU 分数全零、梯度消失。当前使用 `mean_heads(relu(q @ k.T)) / sqrt(head_dim)`
作为 student logits，并将蒸馏学习率设为 `0.0001`。这是教学版的稳定性调整，
正比例缩放不改变相同权重下的 Top-K 排序；不声称逐项复现论文配方。
修正前后的真实结果见 [RESULTS.md](RESULTS.md)。

## 4. Mac mini 安装与检查

在 Mac mini 的工作目录中 clone；若已存在，则先检查自己的改动，再 `git pull --ff-only`。

```bash
git clone https://github.com/weepsdanky/ai-research-notes.git
cd ai-research-notes/code/qwen_next_lab
uv sync --python 3.11
uv run python -c 'import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_available())'
uv run python -m unittest -v
LAB_TEST_DEVICE=mps uv run python -m unittest -v
uv pip freeze > environment-mac-mini.txt
```

依赖当前按兼容区间声明；首次安装后的 `uv.lock` 和环境清单由 Mac mini 保留/提交。
本端没有安装或运行环境，因此尚无经 MPS 验证的锁文件。测试失败先修复，不跳过后直接长训。
显式 `--device mps` 不可用时会报错；`auto` 会选 MPS 或 CPU。

## 5. 数据与第一轮训练

两个语料使用**同一个 byte tokenizer**，因此阶段之间无需更换 embedding。
Shakespeare 前 90% / 后 10% 连续切分，各自采样，不跨边界。
OWT 在固定 Parquet revision 的第一个分片中，按文档 SHA-256 划分 95/5、精确去重，
达到 train / val byte 预算即停。不是全语料随机样本，也不保证清除近重复。
文档末加入 EOD，训练使用普通 packing，不额外隔断文档间 attention。

```bash
uv run python prepare.py --source shakespeare --out data/shakespeare
# 先跑 micro；这里只验证链路，不判断语言质量。
uv run python train.py --size micro --data data/shakespeare --out runs/smoke --device mps --steps 20 --seq 64 --accum 1 --eval-every 10
# 测试主配置；测到太慢或明显换页就改为 --size mini，用另一个输出目录。
uv run python train.py --size small --data data/shakespeare --out runs/probe-small --device mps --steps 20 --seq 64 --accum 1 --eval-every 10
# 选定 small 后的正式学习起点；若测量选择 mini，将 --size 改为 mini。
uv run python train.py --size small --data data/shakespeare --out runs/shakespeare --device mps --steps 1000 --seq 128 --accum 4
uv run python sample.py runs/shakespeare/last.pt --device mps --prompt 'ROMEO:'
```

每次训练调用默认在 1,200 秒后于完整 optimizer step 边界保存并退出。
日志 `stopped=time_budget` 表示**阶段尚未完成**；使用相同选项，加
`--resume runs/shakespeare/last.pt` 续跑，`--out` 保持原值。`--steps 1000` 是阶段总步数，
不是额外 1,000 步；resume 恢复 optimizer、学习率步数和数据 RNG。
首次运行拒绝复用已存在输出目录，避免覆盖。不同实验用新目录。

数据输出同样拒绝覆盖。若网络中断，检查未完成目录并换新目录重试；不把没有 `meta.json` 的数据当成功。

## 6. OWT 继续预训练 → Dense Distillation → Sparse Training

先确认前一个 checkpoint 达到计划步数。以下命令按顺序执行；每一步遇到 time budget
先用相同参数 `--resume` 完成，不能直接跳到下一阶段。

```bash
uv run python prepare.py --source openwebtext --out data/owt --train-bytes 20000000 --val-bytes 1000000
uv run python train.py --stage pretrain --init runs/shakespeare/last.pt --data data/owt --mix-data data/shakespeare --out runs/owt-dense --device mps --steps 1000 --seq 128 --accum 4 --lr 0.0003
uv run python train.py --stage distill --init runs/owt-dense/last.pt --data data/owt --mix-data data/shakespeare --out runs/indexer --device mps --steps 100 --seq 128 --accum 4 --lr 0.0001
uv run python train.py --stage sparse --init runs/indexer/last.pt --data data/owt --mix-data data/shakespeare --out runs/qsa --device mps --steps 500 --seq 128 --accum 4 --lr 0.0001
uv run python sample.py runs/qsa/last.pt --device mps --prompt 'The meaning of life is'
```

`--init` 继承**实际 checkpoint 的架构与权重**，重建优化器和 schedule；此时 `--size` 不生效。
`--mix-data` 以 10% 概率选择第二语料的整条 microbatch，近似 90/10 token 混合。
每次 eval 两份验证集分别记录为 `val0` / `val1`，不能只报告混合 loss。

### 一步更新究竟发生什么

1. **普通 LM 训练（pretrain）**：输入 bytes `[t0…t127]`，标签为 `[t1…t128]`。
   GDN 与 full attention 产生 logits。优化 `LM cross entropy + 0.01 × MoE balance`。
   indexer 不参与 forward 且冻结；其余参数用 Muon/AdamW 分组更新。
2. **Dense distillation**：backbone 冻结，仍跑 full attention；把多头 attention 平均，
   在每个完整块内取最大值并重归一化，作为 `teacher`。训练
   `KL(teacher.detach() || softmax(indexer_scores))`；只有 indexer 参数由 AdamW 更新。
   此时仍记录 LM loss，但它不进入 backward。
3. **Sparse training**：indexer Top-K 产生 mask；core 在这个 mask 下算 attention。
   teacher 来自当前 sparse core attention 的 detached 分布，不另复制一台 dense teacher。
   KL 只在选中块内重归一化。优化 `LM + 0.01 × balance + KL`。
   indexer 输入也 detach，因此 KL 不反向拉动 backbone；Top-K 索引本身不求导。
   backbone 从 LM/balance 学习，indexer 从 KL 学习。

每个 microbatch 的 loss 除以 accum 再 backward；累积后检查梯度、clip 到 1、分别 step。
学习率采用短 warmup + cosine；grad clip、辅助 loss、阶段间重置 optimizer 都是此教学 recipe，
不声称逐项复现论文的大规模训练配方。Muon 使用 FP32 矩阵迭代，便于 MPS 排错；
不是作者的分布式优化器实现，也没有声称与官方数值逐位一致。

数据形状、loss 与参数流应先理解，再考虑增大语料。1,000 步 × 512 tokens = 512,000
byte tokens，并不等于遍历 20MB OWT；此预算旨在观察机制，不是训练成熟通用语言模型。

## 7. 如何判断结果

- 先看测试、loss/梯度是否有限、训练和验证数据 hash、参数量及冻结状态。
- 保存 phase 起点和终点的同一组 held-out loss，不拿不同 tokenizer 的 loss 比较。
- sparse 阶段自动用**同一个 checkpoint、同一批验证 tokens**报告 sparse LM 与 dense LM。
  这衡量移除 mask 的即时影响，不等于独立 dense-training 对照实验。
- 看真实生成文本和失败样例，不把 loss 下降等同于学会知识或指令跟随。
- 查看 expert 使用率可作为下一次诊断；当前仅报告 balance，不声称已验证负载均衡。
- 先保留代码可读性。短序列 Mac 上 QSA/MoE 未必更快，本实现尤其不用于证明稀疏 kernel 加速。

产物：各 run 的 `config.json`、`metrics.jsonl`、`last.pt`；Mac mini 另写
`RESULTS.md` 记录环境、实际参数、速度/内存、阶段状态、loss、样本和局限。
`data/`、`runs/`、环境目录已 gitignore；不把数据、权重或 HF token 推到 GitHub。

## 8. 参考

- [原论文](https://arxiv.org/html/2608.30320v1)：§2.1.1 GDN、§2.1.2 QSA 训练、§2.2 GR、§3.1 Muon。
- [PyTorch Muon 算法说明](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)：二维隐藏层权重与其他参数的区别、矩阵形状缩放。
- [Tiny Shakespeare 原始来源](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)：下载代码固定 commit。
- [OpenWebText 数据卡](https://huggingface.co/datasets/Skylion007/openwebtext)：准备脚本使用固定的 Parquet 转换 revision。

Mac mini 执行状态与实测结果见 [RESULTS.md](RESULTS.md)；其中区分完成阶段、失败实验与实现局限。
