# 交给 Mac mini 上 ChatGPT 的执行任务

将下面内容作为 Mac mini 上具备终端执行权限的 ChatGPT / Codex 的新任务。
本文件是交接指令，写入 GitHub 不会自动启动远程任务。

---

请创建一个 goal，在这台 **16GB Mac mini** 上验证并训练
`https://github.com/weepsdanky/ai-research-notes` 的 `code/qwen_next_lab`。
目标是走通一个可解释的 GDN + GR + MoE 小模型和 QSA 两阶段转换，保留真实证据。
先读目录 README 和代码，遵守仓库指令；不要假定编写端已经运行过测试。

## 工作边界

- 所有 Python、安装、数据准备、测试和训练在本机执行。
- 若工具支持 `create_goal`，创建上述具体目标；如果不支持，说明限制后用普通执行任务推进。
  不假称已创建 goal，也不要用定时自动化替代一次性训练任务。
- 确认 Apple Silicon、16GB 内存、磁盘、macOS、arm64 Python 和 MPS 状态；记录实测。
- clone 仓库；已有 checkout 先 `git status`，不覆盖用户修改，只使用 `pull --ff-only`。
- 默认最多花 **2 小时主动执行本轮**（含安装/下载/测试/训练），单次 train 预算 20 分钟。
  这是分批执行预算，不是达标时间承诺。到时保存可续跑产物与下一条命令；不能把未完成 goal 标记完成。
- 不租 GPU、不下载官方权重、不下载完整 OWT，不修改系统 allocator 上限。
- 不做与实验无关的仓库改动；数据、checkpoint、环境与密钥不上传 GitHub。

## 执行顺序

1. `uv sync --python 3.11`，记录版本和 lock；跑 CPU 与 MPS `test_lab.py`。
   修复发现的问题后重新验证。重点检查未来信息泄漏、尾块、Muon finite update、
   indexer 蒸馏阶段 backbone 冻结。测试没过不长训。
2. 用 README 的 micro Shakespeare 流程跑 20 steps。额外验证 checkpoint 续跑：
   一个小 run 设置 `--seconds 1 --steps 4` 让它提前保存，再以相同参数加 `--resume`
   完成总共 4 步；与同 seed 的不中断 4 步 run 比较权重/loss（容许 MPS 数值误差）。
   不只验证“文件存在”。
3. 对 small 做 20-step probe：先 seq64、batch1、accum1。记录实际参数量、sec/step、
   tokens/s、process RSS、MPS driver memory、系统 memory pressure 和 swap 变化。
   RSS 和 MPS 指标不能简单相加。若实测明显不适合交互学习，降为 mini，并写出原因。
   若两个都太慢，保留 micro 完成流程，不能把 micro 的结果标成 801 万参数。
4. 用选定配置跑 Shakespeare 1,000-step 学习实验，固定验证批次，保存前后 loss 和真实生成文本。
   初始 seq128；若不适合可改 seq64、accum1，但必须用新 run 并记录 token 预算变化。
   不声称小语料短训具备通用对话能力。
5. 完成上阶段后，准备默认 20MB train + 1MB val 的固定 OWT 子集；
   继续 dense 训练 1,000 steps（混入 10% Shakespeare）。不随意增加下载预算。
6. 按 README 从已完成的 checkpoint 做 distill 100 steps 和 sparse 500 steps。
   各阶段能在时间预算内做多少就做多少，逐次恢复；不为了宣布完成跳过阶段或悄悄删减验收。
7. 比较 sparse 与同权重 dense 验证 loss，保存两个语料的生成样本；
   不把 dense-mask 参考实现称为有实际 sparse-kernel 加速的实现。

## 产出与完成条件

写 `code/qwen_next_lab/RESULTS.md`：

- commit、硬件、软件版本、测试结果，实际模型配置与参数量；
- 数据 revision/hash/数量、所有实际执行命令；
- 每阶段已完成/计划步数和 tokens，验证 LM/KL，时间与内存；
- 固定 prompts 的真实生成文本、失败案例与解释；
- checkpoint 本机路径与下一条准确 resume 命令；
- 若有修复，说明修改与验证，提交源码/报告/lock，推送本 repo 的 `codex/qwen-next-mac-mini` 分支。

只有测试通过、选定模型完成上述三个训练阶段（含 dense 数据扩展）、实际生成样本并写出报告，
才把 goal 标为完成。若只是时间预算到、安装失败或中间阶段完成，要准确报告当前进度。
学习过程中一次解释一个组件的张量形状、loss 来源与参数更新，不额外堆框架。
