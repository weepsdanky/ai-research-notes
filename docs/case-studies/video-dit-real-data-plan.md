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

## 2026-08-13 执行记录

- 已取得并解析 VATEX 官方 `vatex_training_v1.0.json`：25,991 records，SHA-256
  `9a3b5f08e354d9543ef4f1ab004f9db9bc4e5da49d9692f0c8c6aa3bef9751c4`。
  caption 粗筛得到 cat 1,415、dog 600、cat+snow 25、dog+snow 71；这些数字只是
  metadata 候选量，不能代替视频级人工确认。
- 当前机器没有 Hugging Face 登录令牌；VideoUFO 大文件连接仍超时，FineVideo 又需
  显式接受 gated 条款，所以暂未把二者的视频写进 manifest。
- Wikimedia Commons 搜索与分类树检索当前共取得 799 个候选。已通过本地审核 UI 播放
  官方低码率 derivative，并用多帧时间联系表辅助检查；101 条自然视频已人工标为
  `approved`，统一转为 96×96、16 帧、2 秒 H.264 clip。另有 651 条 pending、47 条
  rejected，不以标题或搜索命中直接代替视频审核。approved 为 cat 59、dog 41、person 1；
  这条 person 是 `walking|beach`，其 3--5 秒单主体区间经完整播放和逐秒帧核验后进入
  manifest。person 仍有 111 条 pending，不能计入主体覆盖。
- G0 的 100 条记录已全部写入可追溯 manifest，并按 source、显式 duplicate group 与
  感知 hash 的传递闭包完成分组；冻结 OpenAI CLIP ViT-B/32 文本向量已缓存 100/100。
  MPS 真实 batch 冒烟训练完成 1 次更新（loss 1.0128），验证了解码、自然文本条件、
  扩散损失、反向传播、EMA、日志与 checkpoint 链路。
- split validator 现在按 source/duplicate/近重复感知 hash 的传递闭包统计独立来源，正式 CLI 默认每个
  C1 triple 至少 50 个独立来源；同时检查每个 C1 pair 在训练中的来源覆盖、IID triple
  的训练覆盖，并输出完整 coverage table。

因此当前状态是：G0 的真实数据、冻结文本、VAE latent、训练和推理解码链路已经闭合。
在下述全时间 velocity 运行后，“小集合可学习、MP4 有真实运动”的基础 gate 有条件通过，
但文本三因子门禁仍未通过；只开始 G1 数据准备，不开始 G1 正式训练。G0 不用于回答组合泛化：其中有少量多动物画面，
且每个组合远未达到 C1 所需的 50 个独立来源。G1--G4 仍需更大规模、严格单主体/同一
身份与拍摄事件人工审计的数据；在 C1/C2 实验完成前，不能把合成动画、单 clip smoke
test 或 G0 结果称为“训练期未见组合泛化”。

### G0 训练结果（2026-08-13）

- 像素基线：1.41M 参数、96×96×16、3,000 updates。loss 从约 1.0 降至约 0.35，
  但 EMA 与 raw 权重的 DDIM 输出均仍是高频噪声，因此明确判为失败。
- VAE 闭环：固定 `stabilityai/sd-vae-ft-mse@c01e028...`，posterior mode 编码后乘
  `0.18215`；100/100 clips 缓存为 4×16×12×12 latent。抽样重建 PSNR 20.14 dB，
  主体和真实跨帧运动可辨认。
- 100-clip latent 基线：9.87M 参数、10,000 updates、batch 4、EMA 0.999。输出已离开
  纯噪声并有跨帧结构，但相同 seed 下不同 prompt 几乎相同，主体/动作/场景不可辨；
  判定为条件坍缩，未通过 G0。
- 3-clip pooled 诊断对照：同一架构和 10,000 updates，在固定相同 seed 下三个训练
  prompt 生成可区分的雪地、池塘、林地纹理；但后来加入的冻结 CLIP 视频--文本检索
  只得 1/3（随机水平），平均正确 prompt margin 为 -0.0116。此前“可区分”只能证明
  条件会改变输出，不能证明文本语义正确。
- token 条件消融：新增冻结 CLIP 77-token sequence、每个 spatial/temporal block 的
  gated cross-attention，以及明确清零 padding 的 `tokens-v2` 缓存契约。1.60M 参数的
  3-clip/2k 与 10-clip/4k 实验仍分别只有 1/3 和 1/10 检索准确率。
- capacity 对照：同一 10 clips 使用 11.99M 参数、10,000 updates、batch 5，最近
  500-step loss 降至约 0.05。固定 seed 的 10-way 检索由小模型随机水平提高到 EMA
  3/10、raw 5/10，但平均 margin 仍为负；所有生成视频的 CLIP 最近训练邻居仍为同一
  clip。中帧已是自然场景低频结构而非纯噪声，但动物轮廓模糊，未通过条件门禁。
- latent 采样审计发现旧实现把预测 clean latent 错误裁剪到 `[-1,1]`；真实缓存范围约为
  `[-4.93, 5.77]`，约 18.4% 元素绝对值超过 1。修复为 latent no-clip 后，10-clip raw
  的单 seed 检索达到 6/10，固定三 seed 汇总为 18/30（60%），视频中猫狗及雪地、水面、
  室内等结构明显可辨；这仍是训练集记忆诊断，不能算泛化。
- 新 100-clip token 模型：11.99M 参数、10,000 updates、batch 8、EMA 0.99。训练视频
  teacher-forced 条件检索在跨 t=100/500/900 聚合后 raw/EMA 均为 100/100，证明 cross-attention
  已绑定文本；但 t=999 的 EMA 仅 9/100。默认 t=999 自由生成的 10 prompts × 3 seeds
  只有 10/30（33.3%，Wilson 95% `[19.2%, 51.2%]`），mean margin -0.0060；最近训练
  邻居多数坍缩到同一 forest-dog clip，因此 G0 仍失败。
- 从 t=900 启动的配对诊断只有 11/30（36.7%，Wilson 95% `[21.9%, 54.5%]`），mean
  margin -0.0048，且最近邻相似度没有下降；这排除了“仅修改 sampler 起点”作为修复。
  随后从上述 EMA 初始化、重置优化器，完成 2,000-step 高噪声 curriculum（75% 概率
  强制取 t=900--999，其余仍均匀采样）。t=999 teacher-forced 的 raw/EMA 分别为 2/100
  与 13/100，margin 仍为负；该 epsilon 模型的尾部条件问题没有解决，因此没有把低 loss
  误报为成功，也没有继续浪费计算生成 30 条失败视频。下一项应测试 velocity target：
  epsilon target 在 t≈999 几乎等于输入噪声，文本相关的 x0 项被极小的 sqrt(alpha_bar)
  压制，而 velocity target 在高噪声处近似 -x0。对应的 2,000-step velocity 微调把
  t=999 teacher-forced raw/EMA 提升到 93/100 与 94/100，margin 约 +0.28，支持这一机制；
  但默认从 t=999 的自由生成仍只有 10/30（33.3%，Wilson 95% `[19.2%, 51.2%]`），
  mean margin -0.0053，最近训练邻居均值 0.8843。目视联系表能分出室内、雪地、水面、
  森林、草地等低频场景，但动物主体模糊且 prompt 排名没有提升。因此单步高噪声条件
  梯度已修复，完整反向轨迹仍失败；G0 继续判失败，下一步应查多步误差累积、训练目标
  全时间权重与容量，而不是启动 1K/6K 数据或宣称 unseen composition。两次微调是在
  initialization SHA 契约加入前运行，旧产物只能以父路径、配置和运行日志旁证来源；
  当前代码已把父 checkpoint SHA、所选权重、step 与 prediction type 写入新 checkpoint。
- 全时间 velocity 验证：从同一个 100-clip epsilon EMA 初始化，以 uniform t=0--999
  再训练 10,000 updates（11.99M 参数、batch 8、EMA 0.99），父 checkpoint SHA 和初始化
  权重已写入产物。最后 500-step loss 均值 0.2126；跨 t=100/500/900/999 聚合的
  teacher-forced raw/EMA 均为 99/100，margin 为 +0.385/+0.392。默认 CFG=1 的
  10 prompts × 3 seeds 自由生成达到 17/30（56.7%，Wilson 95% `[39.2%, 72.6%]`），
  三个 seed 分别为 6/10、5/10、6/10，mean margin 首次转正为 +0.00268。CFG=2.5/5
  的 seed-42 均为 6/10，但 CFG=5 margin 转负，因此不采用更强 guidance 掩盖失败。
- 目视联系表能辨认猫狗主体以及室内、雪地、水面、森林、草地等场景，30 条视频的
  adjacent-frame MSE 均值 0.0345、endpoint MSE 0.1358，证明不是静态帧复制。它仍常把
  harbor/snow、beach/harbor、forest/pond 和 small-dog/forest 混淆；最近训练邻居 CLIP
  相似度均值 0.917、最大 0.953，符合 G0 训练内拟合但不提供泛化证据。
- 新增逐阶段轨迹 MP4/JSON。固定 seed-42 的 CLIP top-1 在初始 x999 和第一次更新后的
  x989 均为 1/10，x827--x161 只有 0--2/10，到 final clean 才升到 5/10；语义主要在最后采样阶段形成，而非
  中途形成后丢失。G0 因此只按“小集合可学习、可生成有真实运动 MP4”有条件通过，
  文本三因子仍不稳定。可以开始 G1 数据准备，但不得把它称为 C1 或 unseen composition。
- G1 数据准备已开始。Commons 搜索现已按 API page 原子写入候选与 continuation token，
  限流或进程中断后可从该页继续；恢复此前中断搜索并加入 person 查询后，本地可复核的
  再从 `Videos of people walking` 分类树加入 88 条、可恢复地检查 `People running` 树后，
  本地权威候选数为 799。现有集合为 101 approved、651 pending、47 rejected，其中 626 条
  pending 仍是 action/scene placeholder；approved 为 cat 59、dog 41、person 1，person
  pending 111。分类遍历现逐 batch 原子保存 category queue、continuation、未处理 titles 与
  visited 集合，绑定 category/subject/depth/limit 契约；429 或中断后不会丢掉页内剩余文件。
  真实运行检查 411 个 `People running` 文件仅新增 1 个视频（跑步技术访谈），两个明确的
  first-person/背影视角分支均为 0，说明 Commons 分类树无法单独补足 running 覆盖。审核 UI 支持
  subject/status 队列和保存后自动前进。首批六条人物视频中，单人沿海滩行走的 3--5 秒
  区间通过；多人接力、几乎不可见主体、主体离开画面、裁切多人腿部等五条被拒绝。批准
  样本已规范化并写入 101-record manifest，核验为 H.264 96×96、8 fps、16 帧、2 秒，
  SHA-256 `62326cb042fb8baa5c8930105a639273b53ecbe9c90252820614a2bbdeaa15d0`
  与记录一致。source/duplicate/pHash 分组验证通过；先以 91 train / 10 validation / 0 IID
  生成数据完整性 split。尝试 10% IID 会因多个只有 1--3 个来源的稀疏 triple 被 validator
  拒绝，正说明当前集合连可靠 IID 都不足，更不能构造每个 triple ≥50 来源的 C1。
  集合仍远未达到 1K 条已审核训练样本。VATEX annotation 虽完整，
  官方不分发视频本体，因此当前只能称“许可明确的 Commons G1 替代 pilot”，不能冒充
  VATEX-1K。后续必须扩充 person 候选并逐条人工审核，任何 pending 都不能进入训练。

所有训练 run 均保留逐 step `step_log.csv` 与 checkpoint；需要自由生成比较的 run 另存
推理 MP4、去噪诊断 GIF 和 `condition-alignment.json`。10-clip 对照有 EMA/raw，最新
velocity 实验只生成了 EMA 三 seed 面板；失败的 epsilon 高噪声微调只做 teacher-forced
门禁，没有生成 MP4。训练内诊断固定
同一 seed；正式多 seed 评估要求每个 seed 使用完全相同的唯一 prompt 集合，先在 seed
内做正确 prompt 检索与循环打乱对照，再跨 seed 汇总准确率、margin 和 Wilson 95% 区间。
评估目录还锁定 checkpoint hash、权重、训练/采样 step、CFG 和 encoder。CLIP 及最近邻
只是代理，必须同时目视审核主体、动作与时间一致性。当前只通过 G0 的基础运动/拟合 gate，
未通过文本三因子门禁，更不构成 C1 未见组合泛化证据。下一步继续准备并审核 G1 数据，
同时修复多步语义形成过晚的问题；文本门禁通过前不启动 1K/6K 正式训练。
