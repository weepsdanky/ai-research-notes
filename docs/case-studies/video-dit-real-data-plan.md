# 真实视频上的 Text-to-Video DiT：数据选择与组合泛化实验计划

> 调研日期：2026-08-13。本文只把数据集官网、官方仓库、原论文和数据许可证作为事实来源。下载可用性会随托管方和原视频删除而变化，因此开始实验时仍须重新检查链接与条款。本文不是法律意见。

## 结论先行

可以做，而且应当做；但不建议“随便从网上搜猫狗视频然后下载”。对于当前 24 GB Apple Silicon Mac，最稳妥的路线是：

1. **方法学基准：VATEX 子集**。VATEX 有 41,250 个视频、825K/826K 量级的英中人工描述，每个视频各有 10 条英文和 10 条中文描述，caption 质量远好于 ASR；annotation 采用 CC BY 4.0。先取 4K--8K 个可获得 clip，验证真实视频的数据管线、冻结文本编码器和严格 split。[VATEX 官网](https://eric-xw.github.io/vatex-website/) · [ICCV 论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_VaTeX_A_Large-Scale_High-Quality_Multilingual_Dataset_for_Video-and-Language_Research_ICCV_2019_paper.html) · [官方下载页](https://eric-xw.github.io/vatex-website/download.html)
2. **自然场景猫狗主实验：VideoUFO 的许可过滤子集，必要时辅以 FineVideo**。VideoUFO 官方 Hugging Face 数据卡声明 1,091,712 clips、1,291 topics、brief/detailed captions，视频为 CC BY 4.0，并直接提供约 800 GB、200 个 tar 分片；不要全量下载，只取包含 cat/dog/person 和目标动作、场景的 8K--20K clips。FineVideo 可作为较小但长视频丰富的补充源：43,751 个视频、约 600 GB，视频 bytes 直接托管并支持 streaming，但属于 gated 数据，必须接受其 Creative Commons、署名、删除请求和版本更新条款。[VideoUFO 数据卡](https://huggingface.co/datasets/WenhaoWang/VideoUFO) · [VideoUFO 论文](https://arxiv.org/abs/2503.01739) · [FineVideo 数据卡与条款](https://huggingface.co/datasets/HuggingFaceFV/finevideo) · [FineVideo 官方说明](https://huggingface.co/blog/fine-video)
3. **不要把 WebVid-10M 当新实验首选**。官方仓库明确写明自 2024-02-23 起因 cease-and-desist 已不能发布 URL 和 captions；其条款也只授予研究者 12 个月、不可转让的严格非商业研究许可，并要求期满删除。第三方镜像不能自动恢复合法来源或可复现性。[官方仓库](https://github.com/m-bain/webvid) · [官方条款](https://github.com/m-bain/webvid/blob/main/TERMS.md)

如果只做一轮实验，推荐 **VATEX 6K 做管线基准 + VideoUFO 12K 做猫/狗/人物组合泛化**。这比全量 WebVid/HD-VILA 更标准、更可审计，也更符合单机预算。

## 候选数据集逐项核查

| 数据集 | 官方规模与文本 | 视频实际在哪里 | 使用条件与稳定性 | 对本项目的判断 |
|---|---|---|---|---|
| **WebVid-10M** | 约 10M 个 stock-footage video-text pairs | 原作者不拥有视频；现在连 URL/caption 都不再提供 | 严格非商业研究、许可 12 个月、不可转让；官方明确撤下 | **排除**。历史 T2V 常用，但今天无法从官方入口构造可复现实验 |
| **MSR-VTT** | 10K clips、41.2 小时、200K clip-sentence pairs；每 clip 约 20 条人工句子 | 微软论文页目前提供论文信息，未发现同页直接托管 raw videos 的稳定官方下载入口 | 原论文能证明规模与标注，不能据此推断当前 raw-video 许可和镜像合法性 | **可作评估参考，不作首选训练源**；只有在官方授权下载入口重新确认后使用 |
| **VATEX** | 41.25K clips、825K+ 英中人工 captions；覆盖 600 个细粒度人类活动 | 官网发布 annotations；官方明确因法律/隐私不能分发 YouTube clips，需按 ID 自行取得，部分会失效 | 官网声明数据集 CC BY 4.0；但视频本体仍受原来源权利约束 | **首选标准 caption 基准**，但要固化成功下载清单和 hash，不能假定 41K 全部可用 |
| **YouCook2** | 2,000 个长视频、89 种菜谱，步骤有时间边界和祈使句描述；约 176 小时 | YouTube 来源；官网称可下载视频/特征，但可获取率会随原视频变化 | 官方页面未给出足以支持“视频可自由再分发”的清晰通用许可证 | **不适合猫狗/自然场景**；适合另做 person × cooking-action × kitchen-object 组合实验 |
| **Kinetics-400/600/700** | 最多约 650K、400/600/700 类人类动作；每 clip 约 10 秒，只有单一动作类标签 | CVDF 目前明确托管 K400 和 K700-2020 archives；官方也提醒不同版本之间有 split 重叠 | 比 YouTube URL-only 稳定；标签不是自然 caption，版本交叉会造成评估泄漏 | **动作预训练/对照很好，T2V 文本主集不够**。可把标签模板化，但不能称人工自然描述。[官方仓库](https://github.com/cvdfoundation/kinetics-dataset)
| **Something-Something V2** | 220,847 clips、174 类、19.4 GB；train/val/test 为 168,913/24,777/27,157；另有对象实例描述 | Qualcomm 直接提供分卷 archive，VP9 webm，240p | Research Use agreement；研究用途，不应公开再分发派生数据包 | **Mac 上最稳定的动作/组合辅助集**，但主体是手和日常物体，不是自然猫狗，也不是开放 caption。[官网](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
| **ActivityNet Captions** | 20K videos、849 小时、100K 带起止时间的描述 | 官方需填表获得 Drive 的 7 天访问；历史上视频来自外部来源，缺失需单独处理 | 可获得性优于纯 URL 清单但仍需授权流程；长视频预处理成本高 | **可做人物活动 dense-caption 扩展**，不适合第一轮猫狗实验。[官方任务页](https://activity-net.org/challenges/2021/tasks/anet_captioning.html) · [官方下载](https://activity-net.org/download.html)
| **HD-VILA-100M** | 3.3M source videos、100M clips、371.5K 小时、720p；文本是 ASR，平均 13.4 秒/32.5 words | 官方只发 YouTube URLs、timestamps、metadata，要求用户自己下载/裁剪 | R-UDA：只允许 computational、非商业研究；不保证数据权利；外链会漂移。官方材料也承认 ASR 有噪声且没有 validation split | **不适合 Mac 首轮，也不适合视觉 caption 精确监督**；仅在需要大规模预训练且能承担 URL 失效与清洗时考虑。[官方仓库](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) · [R-UDA](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/LICENSE) · [CVPR 论文](https://arxiv.org/abs/2111.10337)
| **FineVideo（HF 官方团队）** | 43,751 个长视频、约 3.4K 小时；有结构化场景、活动和描述 | HF Parquet 内含 `mp4` binary，约 600 GB，支持 streaming | gated；Creative Commons 原许可逐条适用，要求保留 provenance、响应删除请求并跟随可用版本 | **可执行的小流式替代/补充**，但要从 scene timestamps 切短 clip，不能把语音转写当视觉描述 |
| **VideoUFO** | 1.09M clips、1,291 topics，含 brief/detailed caption 和质量分数 | HF 直接托管约 800 GB 的 200 个 tar 分片 | 官方数据卡声明视频 CC BY 4.0；仍应随样本保留 attribution/provenance | **自然场景主实验首选**，但必须做按 metadata 选样、按分片下载，而非全量镜像 |

### 关于“下载稳定性”的事实边界

- **直接托管**：Something-Something V2、CVDF 的 K400/K700-2020、FineVideo、VideoUFO。这里的“稳定”仅指当前存在官方文件托管，不承诺永久可用。
- **annotation/URL 为主**：VATEX、HD-VILA-100M、ActivityNet 系列的部分入口及 YouCook2 的原始 YouTube 资产。原视频删除、地区限制和平台条款都会改变成功率。
- **已被官方撤回**：WebVid-10M 的 URL/caption 发布。
- **MSR-VTT**：官方论文页确认数据定义，但本次没有在该页核实到稳定 raw-video 下载与清晰视频本体许可。因此报告不采用来源不明的 Google Drive/HF 镜像来“补齐”这个事实。

## 24 GB Mac 的实际子集

### Phase A：VATEX-6K，先证明训练链路

- 从官方 train/val annotation 出发，记录 `source_id`、全部 captions、下载状态、原 URL、下载时间和 SHA-256。
- 成功视频中抽取 6,000 个：train 4,800、validation 600、test 600；**同一 source video 只能在一个 split**。
- 每个样本取单镜头 2 秒、8 fps、16 帧，先中心裁剪到 96×96；保留一条短 caption，其他人工 captions 用于语义等价评估而不是复制成 10 个视频样本。
- 原始下载预计几十 GB；规范化后的 H.264 clips 目标控制在约 5--12 GB。该数值是工程预算估计，不是 VATEX 官方统计。

### Phase B：VideoUFO-12K，自然猫狗/人物主实验

先只下载 metadata，用 topic、brief/detailed caption 和质量指标筛选。目标因子：

```text
subject = cat | dog | person
action  = walking | running | jumping | sitting/standing
scene   = grass/field | forest | beach | snow | indoor/home
```

人工复核候选的 10% 和全部 test；排除多主体冲突、幻灯片、强剪辑、水印遮挡、caption 与画面不一致、动物主体过小的 clip。目标 12K clips，最低每个保留组合 120 个；不足的组合宁可缩小矩阵，也不要靠重复采样伪装平衡。

预处理建议：2 秒、8 fps、16 帧、96×96 或 128×128，去音轨，H.264；保存原 caption、规范化三元组、source ID、许可、attribution、时间段、SHA-256、感知 hash 和预处理版本。12K 个这种短 clip 通常可以控制在约 10--25 GB；同样，这是需由 pilot 实测的预算估计。

### 为什么不是直接训练 800 GB

现有 248 万参数、32×32 的模型只足够过拟合几段教学动画。真实视频的背景、镜头、纹理和文本变化大得多。全量数据既不能装进本机，也不会自动带来可解释的组合泛化；先用平衡、可审计的 6K/12K 子集，才能知道失败来自模型、数据还是 split。

## 严格 compositional split

随机切分不能回答“没见过的组合能不能生成”，因为相同来源、近重复画面甚至同一个拍摄事件会同时进入训练和测试。

### 1. 先建立不可跨越的 group

以下任一相同就分到同一 group，再按 group 切分：

- 原始 source video ID；
- 同一 URL、同一频道连续片段或重叠时间段；
- 感知 hash/视频 embedding 判定的近重复；
- 同一人物/动物身份或同一连续拍摄事件（人工审计 test）。

先 group，后平衡标签；不能反过来。

### 2. 三层测试，而不是一个 test 数字

1. **ID-test**：三元组 `subject × action × scene` 在训练中出现过，但 source group 全新。衡量普通泛化。
2. **C1 组合测试**：完整三元组从训练中拿掉；每个单因子和每个二因子组合仍至少在训练中出现。例如 test 是 `cat × running × snow`，训练仍分别包含 `cat × running × grass`、`cat × walking × snow`、`dog × running × snow`。这才是严格的三元组合泛化。
3. **C2 强组合测试**：把一个二元组合整体拿掉，例如所有 `dog × snow`，但 dog、snow 以及每个 action 分别仍在训练出现。它比 C1 更难，应单独报告，不能与 C1 混成一个分数。

可用确定性的 Latin-square/整数规划选择约 15% 三元组给 C1，要求：

- 每个 subject/action/scene 在 train 和 test 都出现；
- C1 中每个二元投影在 train 至少有 `N_min` 个 source groups；
- 每个测试三元组至少 50 个独立 source groups；
- split manifest 固化为 JSONL，并记录规则、随机种子、数据版本和所有 hash。

若某些现实组合本来就不存在（例如 caption 中没有“猫在室内跳跃”），不能用别的分布硬凑；应从评估矩阵删除并公开 coverage table。

### 3. 防止 caption 泄漏

- 同一视频的 10 条 VATEX caption 必须跟随视频 group，不能分散进 train/test。
- 规范化标签只能由 train-independent 规则或冻结标注器产生；不可人工看 test 结果后反复改 taxonomy。
- caption paraphrase 不等于新样本。训练采样时可随机选择同一 clip 的 caption，但统计样本数仍按 clip/source group。
- 测试 prompt 同时使用规范化模板和原始自然句；前者测组合，后者测语言鲁棒性。

## 模型需要怎样升级

仅替换数据集还不够。当前 `StructuredConditionEmbedder` 把关键词映射为少量离散 ID，无法从 VATEX/VideoUFO 自然 caption 学习开放文本语义。建议保留 Video DiT 的空间块、时间块、adaLN-Zero、扩散和 EMA，但改成：

- 冻结的小型 text encoder（优先选择许可兼容、Mac 可运行的 CLIP text encoder），用 token sequence cross-attention 或 pooled embedding 调制 adaLN；
- 先使用冻结的 per-frame VAE latent，避免在像素空间学习所有自然纹理；VAE 的许可和预处理版本必须写入 manifest；
- 教学档建议约 10M--30M DiT、latent 12×12/16×16、8--16 帧，并用 gradient accumulation；不要假定当前 2.47M 模型在 12K 真实 clips 上会产生清晰动物。

上述尺寸和训练时长是**工程建议/待测假设**，不是论文结论。先跑 100 clips overfit、再跑 1K pilot，确认解码、caption alignment、时序和 loss 后，才启动 6K/12K。

## 评估与验收

每个 C1/C2 prompt 至少生成多个 seed，并分别报告：

- **条件正确性**：冻结 video-text encoder 的相似度，加独立 subject/action/scene 分类器；这些是代理指标，不能单独证明质量。
- **时序性**：光流幅度、静态帧复制率、temporal LPIPS/embedding consistency。
- **视觉质量与多样性**：小样本 FVD 对估计很敏感，必须同时给置信区间和真实/生成联系表。
- **人工盲评**：不展示 split 类型，分别判断主体、动作、场景、自然度与时间一致性。
- **记忆审计**：生成视频与 train clips 做最近邻检索；报告最近邻及距离，排除直接复现。

最重要的对照：同一架构分别使用随机 split 与 C1 split。随机 split 成功而 C1 失败，说明模型学会了数据内插，并没有学会组合。

## 合法与可复现的数据获取原则

任意网络抓取不理想，不只是“下载脚本容易坏”：

1. 搜索结果中的“可观看”不等于获得训练、复制或再分发权；平台条款、上传者版权和数据集 annotation 许可彼此不同。
2. 搜索排序会造成主题、国家、语言、受欢迎度和内容审核偏差，无法精确复现当时结果。
3. URL 会删除、地区屏蔽或内容替换；没有 source ID、时间段和 hash 就无法审计。
4. 人脸、儿童、隐私、暴力和错误 caption 需要专门过滤；公开模型样例还可能泄露训练内容。
5. 第三方 Hugging Face 镜像只证明某人上传了文件，不证明其拥有视频再分发权。除非 dataset card 能回溯到官方发布及清晰条款，否则不采用。

因此，只从官方清单和明确许可的数据仓库进入；保存 attribution/许可快照但不把受限原视频提交 Git；在仓库中只提交下载 manifest、处理代码、split 和统计，发布 checkpoint 前再检查相应数据条款是否允许发布训练结果。

## 建议的执行关卡

| 关卡 | 数据 | 成功标准 | 失败时停止点 |
|---|---|---|---|
| G0 | 100 clips | 单 batch/小集合可过拟合，MP4 有真实运动 | 解码、VAE、text condition 未闭环就不扩容 |
| G1 | 1K VATEX | source-group split 无泄漏；prompt 与输出主体/动作相关 | caption alignment 无提升则先修文本条件 |
| G2 | VATEX-6K | ID-test 明显优于无条件基线；日志与 manifest 完整 | 普通泛化失败则不做组合宣称 |
| G3 | VideoUFO-12K | C1 三因子正确率显著高于打乱 prompt 对照；最近邻非复制 | 仅随机 split 好看则报告内插，不叫组合泛化 |
| G4 | C2 pair holdout | 独立报告更强的二元留出结果和失败案例 | 不用扩大数据掩盖 split 缺陷 |

最终目标不是生成一个“看起来像猫”的 cherry-picked MP4，而是能回答：模型是否在没有见过某个 `subject × action × scene` 三元组、也没有见过同源和近重复视频的情况下，仍能按照文本同时满足三个条件。
