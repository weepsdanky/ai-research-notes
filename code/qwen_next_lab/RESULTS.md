# Mac mini 实测结果

记录时间：2026-11-07 16:51:21 CST。本轮从 2026-11-07 16:00（Asia/Shanghai）开始，已完成全部计划训练、测试与真实生成；没有缩减正式步数或 token 预算，未触及 2 小时主动执行上限。GitHub认证与分支发布于 2026-11-08 00:37:43 CST 完成，训练未重跑；此前等待认证的时间不属于原训练执行预算。

**发布状态：已登录 GitHub 并成功推送 `codex/qwen-next-mac-mini`。**

2026-11-08 00:37:43 CST，用户要求登录GitHub后，安装官方GitHub CLI 2.103.3 arm64（官方release zip SHA-256核验通过），通过浏览器设备授权登录 `weepsdanky`，凭据保存在macOS keyring，HTTPS Git认证配置成功。随后 `git push -u origin codex/qwen-next-mac-mini` 创建远端分支；初次远端HEAD确认为 `b26fbf2932272b05e5f619e3a860142faa285d775`。此前HTTPS无凭据、SSH未授权和连接Git接口403均为历史阻碍，已通过本机CLI登录解决；未修改插件权限或系统known_hosts。

登录后重新检查了最终源码hash、CPU/MPS测试记录、真实4步续跑权重/RNG、两个语料hash、全部阶段完成步数、backbone冻结、KL梯度、mask及六份生成样本，均通过。审计结果保存在本机 `runs/evidence/final-audit-after-login.json`。没有重新训练或扩大数据预算。发布范围只有 `model.py`、`README.md`、`RESULTS.md`、`uv.lock`、`environment-mac-mini.txt`；数据、checkpoint、虚拟环境和认证凭据未上传。

```bash
# 用户在浏览器完成设备授权；不在聊天或仓库中保存token
/Users/mark/.local/bin/gh auth login --hostname github.com --git-protocol https --web
/Users/mark/.local/bin/gh auth status
/Users/mark/.local/bin/gh api user --jq .login
cd /Users/mark/Documents/Github/ai-research-notes
git push -u origin codex/qwen-next-mac-mini
git ls-remote --heads origin codex/qwen-next-mac-mini
```

## 环境、版本与边界

- 本地根目录：`/Users/mark/Documents/Github/ai-research-notes/code/qwen_next_lab`。
- 基线：`df96716f227e6cdd741d058fd627f38b4d838125`。初始工作区干净；fetch 后 HEAD 与 origin/main 均无领先/落后，无需 pull，未覆盖已有修改。
- 修正及依赖版本 commit：`83e4ad64175ca3ddd524f9941a144dcd2fd1e439`；分支 `codex/qwen-next-mac-mini`。
- 最终 `model.py` SHA-256：`f42f2ce4c8372832bfe8e26b1217b54de514c5da42e853f6ea32040baa4826ac`。
- Apple M4 Mac mini（Mac16,10），10 CPU cores（4P+6E），16GB 统一内存，arm64；macOS 26.6.2（25G83），Darwin 25.6.0。
- CPython 3.11.18 arm64，PyTorch 2.15.0，NumPy 2.4.6，datasets 4.8.5，PyArrow 26.0.1，uv 0.13.7（316c827e0，aarch64-apple-darwin）；MPS built/available 均 true。完整依赖见 `uv.lock`、`environment-mac-mini.txt`。
- MPS 推荐内存上限 12,713,115,648 bytes；未设置 MPS allocator 上限、未关闭内存保护、未启用 CPU fallback。
- 磁盘起始可用约146GiB，训练后144GiB。起始/最终 swap 均0MB，监测中未见 swap；系统压力级别1（正常），最终 memory free percentage 65%。未在其他主机或 GPU 上执行。
- 仓库及祖先目录未发现 AGENTS.md；已读取 README 与 MAC_MINI_GOAL。

## 测试与实际修正

CPU、MPS 首轮各9项全部通过（5.359s / 5.989s），修正后再次各9项全部通过（0.483s / 1.193s）。覆盖未来信息泄漏、prefix 一致性、未完成尾块、短 prefix sparse=dense、有限梯度、Muon 零梯度、优化器分组和蒸馏冻结。

micro（536,648参数）20/20步通过，seq64/accum1，1,280 tokens；验证 LM 5.524969 → 5.483573。其意义是链路通过，不能据此判断语言质量。

Checkpoint 实测：首次 micro seq64 的4步在1秒内全部跑完，因此未把它当成中断证据。另建 micro seq128/accum4 的4步 run，`--seconds 1` 在step1保存，恢复到step3，再恢复到step4；与同seed连续4步对照，最终训练loss均为5.5322446823，最大参数绝对差1.67638e-8，数据RNG完全一致。每次调用仍用相同1秒上限。见 `runs/evidence/resume-comparison.json` 及相关日志。

实际发现并修正了 indexer 退化：

| 蒸馏实验 | 分数温度 / lr | 零梯度步数 | 固定OWT批次有效分数非零比例 | 最终 OWT / Shakespeare KL |
|---|---|---:|---:|---|
| `runs/indexer`（失败，保留） | 原始head求和 / 0.001 | 75/100 | 0% | 0.206924 / 0.300139 |
| `runs/indexer-lr1e-4`（失败，保留） | 原始head求和 / 0.0001 | 1/100 | 0% | 0.207404 / 0.353712 |
| `runs/indexer-temperature`（采用） | head平均后除以√head_dim / 0.0001 | 0/100 | 75.595% | 0.113521 / 0.164272 |

原实现初始KL过大，单看KL从12.98下降会掩盖 ReLU 分数全零、softmax 退化的问题。仅降lr仍在固定验证批次上失效。最终只对 student 分数作固定正比例缩放，保持相同权重下Top-K排序不变；这是明确的教学实现稳定性调整，不声称逐项复现论文。两个失败实验均未用于QSA初始化。预训练不调用indexer，已有Shakespeare/OWT backbone不受该修正影响。

最终蒸馏前后 backbone 逐张量完全相同，只有 `q.weight`、`k.weight`、`qnorm.delta`、`knorm.delta` 四个 indexer 张量改变；预训练结束的 indexer 与seed42初始化逐位相同。最终QSA改变4个indexer和145个backbone张量。单独反传QSA KL只有indexer得到梯度，梯度norm为0.315357，且全部有限。见 `runs/evidence/distill-freeze-check.json`、`final-audit.json`。

## 数据

同一 UTF-8 byte + EOD256 tokenizer，uint16 存储，257词表。Shakespeare按连续前90%/后10%切分；OWT固定首分片，文档截断到32,000字符后按SHA-256分95/5并精确去重，达到byte预算停止；普通packing保留跨文档attention。没有下载完整OWT或官方权重。

Shakespeare固定commit：`6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`；原文件hash：`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`。
OWT固定revision：`433fe0f44ed7894fea29c08b3202aa348ccc6369`，路径 `plain_text/train/0000.parquet`。预算train 20,000,000 bytes、val 1,000,000 bytes，分别4,198/221篇（含被预算截断的末篇）。以下token数量包含各文档EOD，磁盘字节约为token数的两倍。

| 数据 | split | tokens（含EOD） | 编码文件 SHA-256 |
|---|---|---:|---|
| shakespeare | train | 1,003,855 | `6035e1c8b46ec844d5faa4c295595dfa2ebd51f4f0fa30c77317530a1eb2144a` |
| shakespeare | val | 111,541 | `66ca613268960af5e5e943118a5b129be2a1e152e35ba5e0d5f7363e65c1cd19` |
| owt | train | 20,004,198 | `d679fc4a5c7eac547c3add8f4ae012e7d6f54b6bc7dd357d9aa500584eead5eb` |
| owt | val | 1,000,221 | `9c48d9290a87ef73c4c6425469b49460591d978bed33d73c1ce8a6e9cd8c05a2` |

数据准备异常也保留：OWT 已写完预算内数据、关闭输出文件并打印完整meta，但进程未及时退出；在确认meta和产物齐全后发SIGINT结束，exit130（含退出等待总计432.48s，进程峰值RSS629,555,200 bytes）。没有将该命令记为正常退出。随后每阶段训练的Corpus入口重新校验两份hash均成功。单独datasets导入/退出检查正常；退出等待原因尚未定位，不能保证其他环境无同类问题。

依赖下载约17分28秒，曾自动重试；提前排队的uv命令等待锁超时，安装完成后已串行重跑成功。备用镜像wheel下载在官方安装完成后取消，未使用未校验的镜像文件。官方OWT Range读取返回206、8bytes，确认流式读取可用。

## 配置、进度与速度

选用 **small，8,007,896参数**：width256、4层、4 heads、expert hidden384、4 routed experts/top2+1 shared、4 GR分支；mixer为GDN→GDN→GDN→Attention。FP32、microbatch1、seq128、accum4，即512 byte tokens/optimizer step；Muon+AdamW，clip1，warmup+cosine。pretrain可训练7,925,848参数，distill仅82,048参数，sparse全部8,007,896参数。QSA block4、K32（最多8完整块+0–3尾tokens）。

small seq64/accum1 probe20步：平均0.141545s/step，中位0.110945s，排除前3步后中位0.110674s（约578 tokens/s）；日志峰值RSS614,694,912 bytes、MPS145,522,688 bytes，外部time峰值RSS619,544,576 bytes。probe前后swap0、free percentage67%。因此不降为mini，正式配置未缩减。

| 阶段 | 完成/计划步数 | 本阶段消费tokens | 平均/中位s/step | tokens/s* | 调用总秒数 | 进程峰值RSS MiB** | MPS driver峰值MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shakespeare | 1000/1000 | 512,000 | 0.394892/0.390659 | 1296.6 | 401.27 | 891.34 | 275.78 |
| OWT dense + 10% Shakespeare | 1000/1000 | 512,000 | 0.393903/0.392031 | 1299.8 | 401.83 | 872.16 | 266.78 |
| Dense distillation（修正后） | 100/100 | 51,200 | 0.127789/0.126841 | 4006.6 | 14.66 | 579.47 | 50.72 |
| QSA sparse | 500/500 | 256,000 | 0.394285/0.391658 | 1298.6 | 203.08 | 842.23 | 235.78 |

*tokens/s由512除以平均optimizer step时间计算，step时间不含eval和checkpoint写盘；调用总时间包含启动、eval和保存。**RSS来自 `/usr/bin/time -l`，MPS来自逐步日志采样；不同口径不能相加，也不是连续硬件峰值监测。四个正式训练调用均小于1,200秒，无需因20分钟预算续跑。共消费1,331,200 byte tokens，其中51,200用于蒸馏而非LM更新；两个失败蒸馏另各51,200。10%混入是整条microbatch的随机概率，实际token比例不是强制精确90/10。

## 固定验证结果

每次验证重置CPU数据generator seed1234，batch1、seq128、2批，每个语料仅256个目标tokens；不是完整验证集平均，结论有明显采样局限。LM是byte cross-entropy（自然对数），不包含balance/KL。阶段间对比保持相同数据hash、窗口和评估参数。

| 阶段 | OWT LM 起点→终点 | Shakespeare LM 起点→终点 | OWT KL 起点→终点 | Shakespeare KL 起点→终点 |
|---|---|---|---|---|
| Shakespeare | — | 5.584014 → 1.792435 | — | — |
| OWT dense | 2.160966 → 1.488915 | 1.792435 → 1.998742 | — | — |
| Dense distillation | 1.488915 → 1.488915 | 1.998742 → 1.998742 | 0.228969 → 0.113521 | 0.365529 → 0.164272 |
| QSA sparse | 1.491616 → 1.462569 | 1.996386 → 2.021752 | 0.096265 → 0.075568 | 0.127700 → 0.083988 |

QSA终点同权重、同一验证tokens对照：

| 语料 | sparse LM | dense LM（移除mask） | sparse−dense |
|---|---:|---:|---:|
| OWT | 1.462569 | 1.459024 | +0.003545 |
| Shakespeare | 2.021752 | 2.019439 | +0.002313 |

固定OWT首批128tokens最终审计：最大8个完整块，无未来位置泄漏，每个query至少保留一个位置；93/128个query裁剪了历史，mask保留45.3488%的因果位置。最终有效indexer分数83.631%非零。QSA训练消费256,000 tokens后OWT loss略改善，Shakespeare较dense阶段终点略退化；没有匹配token预算的独立dense继续训练对照，不能归因于QSA本身。

## 真实生成样本

以下均由本机sample.py真实生成，seed42、temperature0.8、最多新增200 bytes、窗口128；不修正文法或拼写。蒸馏不改backbone且采样走dense，因此其输出与OWT dense一致。完整原始文件位于 `runs/evidence/sample-*.txt`。

### Shakespeare终点，prompt `ROMEO:`

```text
ROMEO:
I would so you bemorror your body breath,
These had peven a, for the truth, and every
beat meet you leave me wroht! King England,
I beseech you but deputy drunken'd and not so,
If you from for your b
```

### Shakespeare终点，prompt `The meaning of life is`

```text
The meaning of life is not strange
I'll my servantThat blood access of his peuch
and so the truth, and every bead meet my cleave me
My hands with lordship better noW abstide with him,
And all not so, noble to the wrongdles
```

### OWT dense终点 / 蒸馏后dense，prompt `ROMEO:`

```text
ROMEO:
I not be not be mmodrer secret.

This accept:
Why, she result of the truth, and even made of eavy.

Fear is a regarding to lo of the medican Wills just and running Arm:

Republicible to the 1rink of
```

### OWT dense终点 / 蒸馏后dense，prompt `The meaning of life is`

```text
The meaning of life is proveth some partment research our her consistent dependency for the time to die story.

Commeady to easi more, he was dengle of the medican Wills just and running Army lords of it. Drard Brord (doll
```

### QSA终点，prompt `ROMEO:`

```text
ROMEO:
If you will have my serviceT and job years,
Or long-term is, you to have been case.

FRIAR LAUa 12 is a posses,
Speak with looks groups on no parable property and that another hand. District replaces
```

### QSA终点，prompt `The meaning of life is`

```text
The meaning of life is not feels to partment research bull exactly completely response to computer and even reduce beaving leaving a response denied odder been continues to property and that another hands can be for the pr
```

可以看见换行、角色标签和网页词汇，但也有如`bemorror`、`FRIAR LAUa`等错误、戏剧与网页语体混杂、长句语义不连贯。没有证据表明模型具备通用对话、知识或推理能力。

## 实际执行命令与本地产物

除Git与硬件检查外，命令工作目录均为上面的本地根目录。训练采用默认seed42、batch1、eval-every50、eval-batches2、seconds1200（特别注明者除外）。训练及数据准备命令的完整stdout/stderr保存在 `runs/evidence/`；长训用 `/usr/bin/time -l` 包裹。下面保留实际输出目录名，不能再次以首次运行方式复用已有目录。

```bash
# Git / 硬件 / 安装
git status --short
git rev-parse HEAD
git fetch origin
git rev-list --left-right --count HEAD...origin/main
git switch -c codex/qwen-next-mac-mini
uname -a
sw_vers
system_profiler SPHardwareDataType
df -h .
sysctl vm.swapusage kern.memorystatus_vm_pressure_level
memory_pressure
uv sync --python 3.11
uv --version
uv pip freeze > environment-mac-mini.txt
# arm64、torch版本、MPS built/available/recommended_max_memory记录于runs/evidence/environment.json
uv run python -m unittest -v
LAB_TEST_DEVICE=mps uv run python -m unittest -v
# 修改model.py后，上面两条测试命令各再次执行一次

# 数据与smoke
uv run python prepare.py --source shakespeare --out data/shakespeare
uv run python train.py --size micro --data data/shakespeare --out runs/smoke --device mps --steps 20 --seq 64 --accum 1 --eval-every 10
# 未触发提前退出的初次检查（不作为续跑通过证据）
uv run python train.py --size micro --data data/shakespeare --out runs/resume-check --device mps --steps 4 --seq 64 --accum 1 --eval-every 4 --seconds 1
# 真正的中断恢复，第二条实际执行两次（step1→3→4）
uv run python train.py --size micro --data data/shakespeare --out runs/resume-check-128 --device mps --steps 4 --seq 128 --accum 4 --eval-every 4 --seconds 1
uv run python train.py --size micro --data data/shakespeare --out runs/resume-check-128 --device mps --steps 4 --seq 128 --accum 4 --eval-every 4 --seconds 1 --resume runs/resume-check-128/last.pt
uv run python train.py --size micro --data data/shakespeare --out runs/resume-reference-128 --device mps --steps 4 --seq 128 --accum 4 --eval-every 4
# 权重、loss、RNG比较脚本：runs/evidence/validate-resume-probe.sh
uv run python train.py --size small --data data/shakespeare --out runs/probe-small --device mps --steps 20 --seq 64 --accum 1 --eval-every 10

# 四个正式阶段（按顺序）
uv run python train.py --size small --data data/shakespeare --out runs/shakespeare --device mps --steps 1000 --seq 128 --accum 4
uv run python prepare.py --source openwebtext --out data/owt --train-bytes 20000000 --val-bytes 1000000
uv run python train.py --stage pretrain --init runs/shakespeare/last.pt --data data/owt --mix-data data/shakespeare --out runs/owt-dense --device mps --steps 1000 --seq 128 --accum 4 --lr 0.0003
# 下两次为修正前的失败蒸馏，未用于后续阶段
uv run python train.py --stage distill --init runs/owt-dense/last.pt --data data/owt --mix-data data/shakespeare --out runs/indexer --device mps --steps 100 --seq 128 --accum 4 --lr 0.001
uv run python train.py --stage distill --init runs/owt-dense/last.pt --data data/owt --mix-data data/shakespeare --out runs/indexer-lr1e-4 --device mps --steps 100 --seq 128 --accum 4 --lr 0.0001
# 固定温度修正并重测后，采用以下蒸馏checkpoint
uv run python train.py --stage distill --init runs/owt-dense/last.pt --data data/owt --mix-data data/shakespeare --out runs/indexer-temperature --device mps --steps 100 --seq 128 --accum 4 --lr 0.0001
uv run python train.py --stage sparse --init runs/indexer-temperature/last.pt --data data/owt --mix-data data/shakespeare --out runs/qsa --device mps --steps 500 --seq 128 --accum 4 --lr 0.0001

# 三个阶段各使用相同的两个prompt，保存六份原始输出
for run in shakespeare owt-dense qsa; do
  uv run python sample.py runs/$run/last.pt --device mps --prompt 'ROMEO:'
  uv run python sample.py runs/$run/last.pt --device mps --prompt 'The meaning of life is'
done
# 本地审计脚本（不上传数据/权重）
uv run python runs/evidence/check-indexer.py runs/indexer runs/indexer-lr1e-4
uv run python runs/evidence/check-indexer.py runs/indexer-temperature
uv run python runs/evidence/check-indexer.py runs/qsa
uv run python runs/evidence/final-audit.py
```

监测还使用了`ps`、`lsof`、`nettop -P -L 1 -J bytes_in,bytes_out -p 13521`、`du`及阶段前后的`sysctl`/`memory_pressure`快照。网络诊断包括官方URL的HEAD、`curl --range -8 --max-filesize 1024`，以及PyPI镜像HEAD与取消的wheel下载。退出诊断命令为 `uv run python -c 'import faulthandler; faulthandler.dump_traceback_later(15, repeat=True); import datasets; print("datasets import completed", flush=True)'`。早期审计脚本曾因CPU/MPS比较设备不一致失败，改为所有checkpoint以`map_location="cpu"`加载后重跑通过。

主要checkpoint均在本机，包含model、optimizer、schedule step、数据RNG及数据meta；没有上传GitHub：

| 阶段 | 绝对路径 | SHA-256 |
|---|---|---|
| shakespeare | `/Users/mark/Documents/Github/ai-research-notes/code/qwen_next_lab/runs/shakespeare/last.pt` | `5f98df592236fd946d7ee08bd86842d9985d987d444b5a6e601896cc0762bc7d` |
| owt-dense | `/Users/mark/Documents/Github/ai-research-notes/code/qwen_next_lab/runs/owt-dense/last.pt` | `5a8dc2fc4cb1378e987cd51ac654aae87ddf9483872eba9ad2728c5188a64bb8` |
| indexer-temperature | `/Users/mark/Documents/Github/ai-research-notes/code/qwen_next_lab/runs/indexer-temperature/last.pt` | `fffa163e2576c6c1174ac66f651a5eaf97b4f0371ff108e8ba126df26b6db4bc` |
| qsa | `/Users/mark/Documents/Github/ai-research-notes/code/qwen_next_lab/runs/qsa/last.pt` | `f3a950a38b56989a28a55b4347e3ee0a2685373e181444c84df35fa7bfabed7e` |

所有正式阶段都已完成，没有未完成阶段需要续跑。已实际核对原run恢复入口，下列精确命令成功加载已完成的500步checkpoint，未新增训练步数；不要把`--steps`改大来冒充原schedule续跑：

```bash
uv run python train.py --stage sparse --resume runs/qsa/last.pt --data data/owt --mix-data data/shakespeare --out runs/qsa --device mps --steps 500 --seq 128 --accum 4 --lr 0.0001
```

后续新实验应另建输出目录并明确新token预算；本轮没有开始额外训练。`runs/evidence/training-summary.json`、`final-audit.json`、各run的`config.json`和`metrics.jsonl`保留更完整指标。提交仅包含源码、README、lock、依赖清单及本报告；`data/`、`runs/`、`.venv/`保持gitignore。

## 局限

- 这是机制学习实验，不是官方Qwen权重或论文训练规模复现；byte-token loss不可直接比较其他tokenizer。
- **QSA core仍先构造dense T×T矩阵再加mask，计算/存储仍为O(T²)，没有真正的sparse kernel加速。** dense与sparse阶段步时接近不能作为kernel比较。
- 验证只有每语料固定256tokens，没有多seed置信区间、完整验证集评估或匹配token预算的独立dense/Muon/AdamW对照。mask保留比例只针对一个固定批次。
- OWT为固定首分片的有界样本，非全语料随机抽样；仅精确去重，不保证近重复清除或与Shakespeare跨源去重。512k的OWT dense tokens也不等于遍历20MB数据。
- balance仅作为辅助loss记录，未验证所有专家长期负载均衡。生成文本仍明显不可靠。
- 温度修正在本轮100步蒸馏上有效，不保证其他数据/初始化不会产生dead ReLU；两个失败run和退出等待异常均如实保留。
