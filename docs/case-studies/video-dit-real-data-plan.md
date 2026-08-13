# 真实视频上的 Text-to-Video DiT：数据选择与组合泛化实验计划

> 调研日期：2026-08-13。本文只把数据集官网、官方仓库、原论文和数据许可证作为事实来源。下载可用性会随托管方和原视频删除而变化，因此开始实验时仍须重新检查链接与条款。本文不是法律意见。

## 结论先行

可以做，而且应当做；但不建议“随便从网上搜猫狗视频然后下载”。对于当前 24 GB Apple Silicon Mac，最稳妥的路线是：

1. **方法学参考：VATEX annotation；实际训练媒体先用逐文件可审计的 Commons**。VATEX 有 41,250 个视频、825K/826K 量级的英中人工描述，每个视频各有 10 条英文和 10 条中文描述，caption 质量远好于 ASR；annotation 采用 CC BY 4.0，但官方不分发视频本体。它适合定义 caption 和 split 基准，不应把第三方取得的 YouTube 视频自动视为同一许可。[VATEX 官网](https://eric-xw.github.io/vatex-website/) · [ICCV 论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_VaTeX_A_Large-Scale_High-Quality_Multilingual_Dataset_for_Video-and-Language_Research_ICCV_2019_paper.html) · [官方下载页](https://eric-xw.github.io/vatex-website/download.html)
2. **自然场景猫狗/人物主实验：继续扩充 Wikimedia Commons 审核集**。Commons 接受自由许可或公有领域媒体；API 能返回直接文件 URL、作者、署名和逐文件 `LicenseUrl`。这不保证搜索结果天然适合训练，但能让每个进入 manifest 的 clip 都有可核验权利链。[Commons 收录范围](https://commons.wikimedia.org/wiki/Commons:Project_scope) · [CommonsMetadata 官方文档](https://www.mediawiki.org/wiki/Extension:CommonsMetadata)
3. **VideoUFO 暂停扩容，不以 OpenVid/Panda/InternVid 替代**。VideoUFO 三分片 pilot 证明随机 shard 对动物覆盖极差；OpenVid 虽直接托管视频但混合上游来源许可，Panda 与 InternVid 则主要发布外部网页视频 metadata 且有非商业限制。下文 2026-08-14 的补充审计给出精确边界。
4. **不要把 WebVid-10M 当新实验首选**。官方仓库明确写明自 2024-02-23 起因 cease-and-desist 已不能发布 URL 和 captions；其条款也只授予研究者 12 个月、不可转让的严格非商业研究许可，并要求期满删除。第三方镜像不能自动恢复合法来源或可复现性。[官方仓库](https://github.com/m-bain/webvid) · [官方条款](https://github.com/m-bain/webvid/blob/main/TERMS.md)

如果只做一轮实验，推荐先把 **Commons-1K 审核集**做实，再决定是否扩大到 6K/12K。当前任何来源都不足以支持“已经完成严格组合泛化数据集”的说法。

## 2026-08-14 官方访问审计：VideoUFO 与 FineVideo

本节核查的是 Hugging Face 官方数据仓库在该日期公开的文件树、API 元数据和 dataset card。为保证后续可复现，下载时不要写 `main`，应固定下列 full revision；文件下载后再计算本地 SHA-256 并写入实验 manifest。

### VideoUFO：首选，但先下载 CSV 再决定 tar

- 官方 HF API 显示仓库**非 gated**，固定 revision 为 `683425832ad5b35d57d213963ecec18d5b689652`；匿名访问文件成立。[文件树](https://huggingface.co/datasets/WenhaoWang/VideoUFO/tree/main) · [提交历史](https://huggingface.co/datasets/WenhaoWang/VideoUFO/commits/main)
- 最小的完整 metadata-only 入口是根目录 `VideoUFO.csv`：1,234,455,730 bytes，SHA-256 `a456192ca88a320080f840c5ec007b0ce6c1852a98e27d0dfad61e2b1f357b52`。它包含 ID、topic、brief/detailed caption、起止时间和六个质量分数，不含视频 bytes。[CSV 文件页](https://huggingface.co/datasets/WenhaoWang/VideoUFO/blob/main/VideoUFO.csv)
- `data/Full-00000-of-00093.parquet` 到 `Full-00092-of-00093.parquet` 共 93 片、合计 45,817,998,470 bytes；dataset card 的 schema 包含 `Middle_Frame`，不是 MP4 视频列。因此 `load_dataset(..., streaming=True)` 可以流式筛 metadata/中间帧，却**不能代替下载视频 tar**。例如 `Full-00000-of-00093.parquet` 为 496,124,145 bytes，SHA-256 `2f8aa7efb3642d81fb7c4a7eed8b72772d8251838564ad714b4a55e45eb7b4a2`。[Parquet 文件树](https://huggingface.co/datasets/WenhaoWang/VideoUFO/tree/main/data) · [dataset card schema](https://huggingface.co/datasets/WenhaoWang/VideoUFO/blob/main/README.md)
- 真正视频在 `VideoUFO_tar/VideoUFO_1.tar` 到 `VideoUFO_200.tar`，共 200 片、合计 864,111,237,120 bytes。多数约 4.1--4.5 GB；最小尾片 `VideoUFO_200.tar` 为 166,451,200 bytes。`VideoUFO_1.tar` 为 4,091,576,320 bytes，SHA-256 `19d0871cd5307e73b71152d2f8197eb86044781b0162c4292402ba1c50c137b8`。[视频 tar 文件树](https://huggingface.co/datasets/WenhaoWang/VideoUFO/tree/main/VideoUFO_tar)

```bash
# metadata；先筛 ID，不要先下载 864 GB
hf download WenhaoWang/VideoUFO VideoUFO.csv \
  --repo-type dataset \
  --revision 683425832ad5b35d57d213963ecec18d5b689652 \
  --local-dir data/raw/videoufo

# 仅下载经 pilot 确认包含所需 ID 的 tar 分片
hf download WenhaoWang/VideoUFO VideoUFO_tar/VideoUFO_1.tar \
  --repo-type dataset \
  --revision 683425832ad5b35d57d213963ecec18d5b689652 \
  --local-dir data/raw/videoufo

sha256sum data/raw/videoufo/VideoUFO.csv \
  data/raw/videoufo/VideoUFO_tar/VideoUFO_1.tar
```

官方 card 没有发布“ID → tar shard”的独立小索引；这是当前可执行性缺口。**推断**：在不知道目标 ID 所处分片时，先下载最小尾片只适合验证解包/解码，不能保证含猫狗。正式 12K 选样前，应先向维护者确认映射，或逐片读取远端 tar header 建立索引；不要盲下 200 片。

许可方面，官方 card 明确把 VideoUFO 视频标为 CC BY 4.0。该许可证允许分享和改编（包括商业使用），条件是适当署名、提供许可证链接并标明修改；因此从该数据训练并发布 checkpoint **原则上属于可进行的改编利用**。这是对 CC BY 4.0 与 card 声明的解释，而不是作者对 checkpoint 的单独保证；发布时仍应附数据引用、revision、attribution/修改说明，并进行记忆泄漏检查。[官方 license 声明](https://huggingface.co/datasets/WenhaoWang/VideoUFO/blob/main/README.md#license) · [CC BY 4.0 正文](https://creativecommons.org/licenses/by/4.0/legalcode)

### FineVideo：访问更方便地流式取视频，但权利管理更严格

- 官方 HF API 显示 `gated: auto`，固定 revision 为 `84c74091e1c6ee7a5dffabfafb5c9033e4718883`。仓库页面可匿名列文件，但必须登录 HF、在网页接受条款并同意把用户名/邮箱提供给维护者，随后用有读取权限的 token 才能访问内容。[仓库与门禁条款](https://huggingface.co/datasets/HuggingFaceFV/finevideo)
- 数据只有 `data/train-00000-of-01357.parquet` 到 `train-01356-of-01357.parquet` 共 1,357 片，合计 673,393,341,968 bytes；每行的 `mp4` 是 binary，因此这些 Parquet 同时承载 metadata 和视频，**不存在单独的小型完整 metadata 文件**。当前最小分片 `train-00797-of-01357.parquet` 仍有 274,313,133 bytes；`train-00000-of-01357.parquet` 为 493,919,884 bytes，SHA-256 `607bc6f45f6c4dcf27950abaf81eea58d7d50e2defd077efd0740647cf88baed`。[数据文件树](https://huggingface.co/datasets/HuggingFaceFV/finevideo/tree/main/data) · [字段与流式示例](https://huggingface.co/datasets/HuggingFaceFV/finevideo/blob/main/README.md#how-to-download-and-use-finevideo)

```bash
# 1. 先在网页接受 FineVideo 条款；2. 本机登录
hf auth login

# 真流式读取：只按需拉取含 mp4 bytes 的 row，不做 673 GB snapshot
python - <<'PY'
from datasets import load_dataset

revision = "84c74091e1c6ee7a5dffabfafb5c9033e4718883"
ds = load_dataset(
    "HuggingFaceFV/finevideo",
    split="train",
    streaming=True,
    revision=revision,
    token=True,
)
for sample in ds.take(10):
    print(sample["json"]["original_video_filename"], len(sample["mp4"]))
PY
```

FineVideo 条款要求遵守每个原视频的 Creative Commons 许可证及署名条件、跟随维护者指定的最新可用版本处理删除请求；若重新托管/分享 FineVideo，还必须附带相同条款并要求接收者同意。仓库的笼统 `license: cc` **不是统一的 CC BY 授权**。官方条款没有明确禁止发布训练 checkpoint，也没有明确给予覆盖全部原视频的统一 checkpoint 授权。因此“checkpoint 通常不等同于重发视频”只能作为工程/法律**推断**，不能当作官方许可；公开 checkpoint 前应保留逐样本 provenance、执行删除清单、做训练样本提取/记忆测试，并向维护者确认允许的发布方式。[FineVideo Terms of Use](https://huggingface.co/datasets/HuggingFaceFV/finevideo#terms-of-use-for-finevideo) · [维护者更新线程](https://huggingface.co/datasets/HuggingFaceFV/finevideo/discussions/2)

### 当时的 VideoUFO pilot 选择（已被后续审计取代）

这一阶段曾选择**先用 VideoUFO，不把 FineVideo 混入第一版训练**。理由不是 VideoUFO 更小，而是它有可离线筛选的 1.23 GB CSV、短 clip、明确 CC BY 4.0 和独立 tar；FineVideo 的每个 metadata row 都和长视频 bytes 绑定，流式扫描 43K 长视频仍会产生大量下载，而且需要持续响应版本/删除要求。这个选择后来被三分片覆盖实验和替代源审计否决；当前执行路线见下文“Commons-first”。

当时的两步假设是：下载并校验 VideoUFO CSV，生成不含媒体的候选/组合 coverage 报告；确认 ID→tar 映射后只取约 3--8 个相关 tar 做 100-clip 与 1K-clip pilot。未经索引确认，不启动原计划的 12K 下载。后续证据表明随机 tar 无法形成平衡动物覆盖，且没有官方 ID→shard sidecar，因此本项目不再执行这条扩容路线，也不再以 FineVideo 补样。

这两项前置验证已实际完成：固定 revision 的 CSV 已完整下载，本地 SHA-256 与上述官方值一致；`prepare_videoufo.py scan` 以约 30--40 MB 常驻内存扫描全部 1,091,712 行，按明确词边界、2 秒时长和六项质量阈值生成 4,234 条**待审核**候选。当前三元组候选中，人物组合普遍充足；狗的 `running|grass=111`、`walking|grass=104`、`running|snow=27`；猫最稀疏，只有 `walking|grass=54` 达到 50，其他猫组合均低于 50。这里的数字仍是 caption 规则命中，不是独立 source group 的最终数量。

最小尾片 `VideoUFO_200.tar` 也已实际下载并校验：166,451,200 bytes，SHA-256 `8dbadeedfd0489638bb3bddeafcb9788af7e4a54547d356aca77085d60682b15`，包含 197 个 MP4。tar member 形如 `VideoUFO/<CSV ID>.mp4`，因此 CSV ID 可以精确连接实际视频；4,234 个候选中该尾片命中 1 条 `person|running|forest`。代码已支持只枚举 regular MP4、不做路径式解包，并要求 resolve 使用官方 shard checksum manifest；CSV 候选文件的 hash/count 也与 coverage report 绑定。用于 split 的 `source_id` 去掉末尾 `.clip-index`，确保同一原视频的 sibling clips 始终属于一个 source group。当时据此提出下载 3--8 个固定分片的前置假设；下述三分片 pilot 已证明随机扩片无法形成平衡动物覆盖，因此该方案已否决，当前只推进 Commons-1K。**当前不满足 C1 门禁，也没有启动 G1 训练**。

随后完成了三个固定分片的实际 pilot：`VideoUFO_1.tar`、`VideoUFO_2.tar`、
`VideoUFO_200.tar` 均与官方 SHA-256 一致，共 8.46 GB（7.88 GiB），映射出 49 条候选、46 个原始
视频 source groups。分布为 person 43、dog 5、cat 1；三元组也高度稀疏，没有任何
动物 triple 接近 50。49 个 member 已在每个 archive 只校验一次 hash 后安全提取为约
30 MB 审核媒体，逐文件 SHA-256 全部一致，状态仍为 pending。`VideoUFO_3.tar` 曾下载到
约 3.92 GB 后因 TLS handshake timeout 失败，上游 Hub 客户端删除 partial；它不计入
任何覆盖数字。这个 pilot 否定了“随机取 3--8 个 tar 即可获得平衡 C1 数据”的假设；
下一步应取得维护者的 ID→shard 索引或构建全量远端 tar-header 索引，定向下载动物候选，
而不是继续随机拉分片。

## 2026-08-14 替代源审计：VideoUFO shard-index 失败之后

本节只回答“能否在 24 GB Mac 上定向取得猫/狗/人物 × 动作 × 场景视频，并合法用于训练/发布”。“数据集卡有 license 标签”“视频能下载”和“底层视频权利可审计”是三件不同的事。以下版本、大小和字段是官方仓库在审计日公开的事实；适用到 checkpoint 的结论均明确标为推断。

| 来源 | 官方访问与最小入口（事实） | 视频与字段（事实） | 许可与 checkpoint 边界 | 本项目决定 |
|---|---|---|---|---|
| **OpenVid-1M / OpenVidHD** | 官方 HF 仓库非 gated；固定 revision `d8a63bd22989c80b5734ec2bb989f4e1b61a5807`。完整仓库约 12.40 TB；最小完整 metadata 为 `data/train/OpenVid-1M.csv`（856,907,992 B，SHA-256 `b7d3176c32fac18989c05a428fc5e466ee5cb511d4afd916b06fd995930f7c6e`），OpenVidHD metadata 为 `OpenVidHD.csv`（299,852,057 B，`c635b5c05a1a21d0435425d7333b18586b9f095fae3426af67447bdddc193dfb`），另有 15,284,572 B 的 `OpenVidHD/OpenVidHD.json` 分片映射（`2292ffafc71ae6937998cd03e8d8bfab5dba1132afdf52515972bb466cd49db6`）。[官方数据卡](https://huggingface.co/datasets/nkp37/OpenVid-1M) · [官方项目仓库](https://github.com/NJU-PCALab/OpenVid-1M) | HF 直接托管 `OpenVid_part*.zip` / `OpenVidHD*.zip`；OpenVid 主集最小 ZIP 也约 31.57 GB，常见单片为 20--50 GB。表中有 `video`、`caption`、aesthetic/motion/temporal scores、camera motion、frames、fps、seconds；项目 README 给出文件名到 ZIP 的 mapping。 | card 顶部标 CC BY 4.0，但正文同时要求遵守 Panda、ChronoMagic、Open-Sora-Plan、CelebV-HQ 等原始来源许可，并把 CelebV-HQ 标作 unknown。**推断**：顶层标签不能清除底层权利，也不能据此保证可公开或商用的派生 checkpoint。 | 工程上可“先筛 caption、再按 mapping 下 ZIP”，但不通过本项目逐 clip 权利门禁；**不采用**。 |
| **Panda-70M** | 官方仓库提供 CSV 与下载代码；2M/10M/full metadata 分别约 118 MB/504 MB/2.73 GB，对应约 1.6/8/36 TB 预估视频量。[官方仓库](https://github.com/snap-research/Panda-70M) · [论文](https://arxiv.org/abs/2402.19479) | metadata 的核心字段为 `videoID`、`url`、`timestamp`、`caption`、`matching_score`；视频不在 metadata 内，须按 YouTube URL 和时间段抓取。 | Snap 官方 LICENSE 明确只许非商业研究；README 还要求视频样本遵守 HD-VILA 的 Microsoft R-UDA。后者不保证提供者拥有底层视频权利，并限制 Results 用于商业产品。[Panda LICENSE](https://github.com/snap-research/Panda-70M/blob/main/LICENSE) · [HD-VILA R-UDA](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/LICENSE) **推断**：研究 checkpoint 也不能被写成权利无负担，更不能商用。 | metadata 定向筛选便宜，但 URL 漂移且权利不可逐条从清单证明；**不采用**。 |
| **InternVid** | 官方 HF 为 auto-gated，须登录、提供联系信息并接受“不伤害人类受试者”的条件；固定 revision `8f8e0138bd89103a914e8c97ae124ab0c4a78fcf`，约 13.39 GB metadata。最小发布文件 `InternVid-10M-DIV.jsonl` 仍为 1,691,991,951 B。[官方数据卡](https://huggingface.co/datasets/OpenGVLab/InternVid) · [官方代码](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) · [论文](https://arxiv.org/abs/2307.06942) | 发布的是 `videoID`、`timestamps`、generated caption、UMT similarity 等 JSONL；card 描述来源为公开网页视频，没有直接托管 MP4 bytes。 | card 标 CC BY-NC-SA 4.0；这约束 annotation 层，并不证明每个网页视频可训练/再分发。**推断**：即便非商业研究，也应另审底层视频；公开 checkpoint 不可仅凭 card 放行。 | gate、NC-SA、13.39 GB 元数据和不透明取视频链均不利于本机 pilot；**不采用**。 |

### 可执行选择：Commons-first，而不是“大数据集 first”

当前唯一进入训练主线的替代源是 [Wikimedia Commons](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)。这是因为 Commons 的项目范围要求自由许可/公有领域媒体，MediaWiki `imageinfo` + `extmetadata` API 能对每个文件返回直接下载 URL、作者/credit、license short name 与 `LicenseUrl`；是否满足署名、相同方式共享或公有领域标识仍须逐文件执行，不能只记一个站点级标签。

24 GB Mac 的下一步保持小而可审计：先把现有 119 approved 扩到 1K，优先补 `cat|playing|indoors` 及其训练侧 pair support；每条分别保存 Commons 描述页 revision、原媒体 timestamp/SHA-1、原文件 URL、作者、license URL、下载时间、实际下载对象（原文件或审核 derivative）的 URL/type/SHA-256、规范化 clip SHA-256、片段起点、预处理参数、人工标签和 source/duplicate group。只允许 `approved` 且这些字段完整的 clip 进入 manifest。1K 达标后再按 coverage table 决定是否扩到 6K；达不到每个 C1 triple 50 个独立 source groups，或训练侧任一 pair 达不到预注册的 10 个独立 groups，就缩小组合矩阵并公开缺口，不用来源不明的视频补数。

这里的代价也要明说：Commons caption 常是文件描述而非精细动作句，搜索分布不均，人物/跑步素材尤其稀疏，因此需要人工裁切与结构化 caption；它是**权利可审计的教学子集**，不是“大规模标准 T2V benchmark”。OpenVid 只在未来能得到逐 row 原始来源及兼容许可后重审；Panda/InternVid 只可作为非商业研究的 metadata coverage 参考，不下载视频、不混入 checkpoint。

## 候选数据集逐项核查

| 数据集 | 官方规模与文本 | 视频实际在哪里 | 使用条件与稳定性 | 对本项目的判断 |
|---|---|---|---|---|
| **WebVid-10M** | 约 10M 个 stock-footage video-text pairs | 原作者不拥有视频；现在连 URL/caption 都不再提供 | 严格非商业研究、许可 12 个月、不可转让；官方明确撤下 | **排除**。历史 T2V 常用，但今天无法从官方入口构造可复现实验 |
| **MSR-VTT** | 10K clips、41.2 小时、200K clip-sentence pairs；每 clip 约 20 条人工句子 | 微软论文页目前提供论文信息，未发现同页直接托管 raw videos 的稳定官方下载入口 | 原论文能证明规模与标注，不能据此推断当前 raw-video 许可和镜像合法性 | **可作评估参考，不作首选训练源**；只有在官方授权下载入口重新确认后使用 |
| **VATEX** | 41.25K clips、825K+ 英中人工 captions；覆盖 600 个细粒度人类活动 | 官网发布 annotations；官方明确因法律/隐私不能分发 YouTube clips，需按 ID 自行取得，部分会失效 | 官网声明数据集 CC BY 4.0；但视频本体仍受原来源权利约束 | **caption/split 方法学参考**；视频本体不进入当前 Commons-first 训练线 |
| **YouCook2** | 2,000 个长视频、89 种菜谱，步骤有时间边界和祈使句描述；约 176 小时 | YouTube 来源；官网称可下载视频/特征，但可获取率会随原视频变化 | 官方页面未给出足以支持“视频可自由再分发”的清晰通用许可证 | **不适合猫狗/自然场景**；适合另做 person × cooking-action × kitchen-object 组合实验 |
| **Kinetics-400/600/700** | 最多约 650K、400/600/700 类人类动作；每 clip 约 10 秒，只有单一动作类标签 | CVDF 目前明确托管 K400 和 K700-2020 archives；官方也提醒不同版本之间有 split 重叠 | 比 YouTube URL-only 稳定；标签不是自然 caption，版本交叉会造成评估泄漏 | **动作预训练/对照很好，T2V 文本主集不够**。可把标签模板化，但不能称人工自然描述。[官方仓库](https://github.com/cvdfoundation/kinetics-dataset)
| **Something-Something V2** | 220,847 clips、174 类、19.4 GB；train/val/test 为 168,913/24,777/27,157；另有对象实例描述 | Qualcomm 直接提供分卷 archive，VP9 webm，240p | Research Use agreement；研究用途，不应公开再分发派生数据包 | **Mac 上最稳定的动作/组合辅助集**，但主体是手和日常物体，不是自然猫狗，也不是开放 caption。[官网](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
| **ActivityNet Captions** | 20K videos、849 小时、100K 带起止时间的描述 | 官方需填表获得 Drive 的 7 天访问；历史上视频来自外部来源，缺失需单独处理 | 可获得性优于纯 URL 清单但仍需授权流程；长视频预处理成本高 | **可做人物活动 dense-caption 扩展**，不适合第一轮猫狗实验。[官方任务页](https://activity-net.org/challenges/2021/tasks/anet_captioning.html) · [官方下载](https://activity-net.org/download.html)
| **HD-VILA-100M** | 3.3M source videos、100M clips、371.5K 小时、720p；文本是 ASR，平均 13.4 秒/32.5 words | 官方只发 YouTube URLs、timestamps、metadata，要求用户自己下载/裁剪 | R-UDA：只允许 computational、非商业研究；不保证数据权利；外链会漂移。官方材料也承认 ASR 有噪声且没有 validation split | **不适合 Mac 首轮，也不适合视觉 caption 精确监督**；仅在需要大规模预训练且能承担 URL 失效与清洗时考虑。[官方仓库](https://github.com/microsoft/XPretrain/tree/main/hd-vila-100m) · [R-UDA](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/LICENSE) · [CVPR 论文](https://arxiv.org/abs/2111.10337)
| **FineVideo（HF 官方团队）** | 43,751 个长视频、约 3.4K 小时；有结构化场景、活动和描述 | HF Parquet 内含 `mp4` binary，约 600 GB，支持 streaming | gated；Creative Commons 原许可逐条适用，要求保留 provenance、响应删除请求并跟随可用版本 | **本阶段不采用**；没有逐 clip 许可审核前不混入 Commons checkpoint |
| **VideoUFO** | 1.09M clips、1,291 topics，含 brief/detailed caption 和质量分数 | HF 直接托管约 800 GB 的 200 个 tar 分片 | 官方数据卡声明视频 CC BY 4.0；仍应随样本保留 attribution/provenance | **metadata coverage 参考，媒体扩容暂停**；随机三分片 pilot 已证明动物覆盖不足 |

### 关于“下载稳定性”的事实边界

- **直接托管**：Something-Something V2、CVDF 的 K400/K700-2020、FineVideo、VideoUFO。这里的“稳定”仅指当前存在官方文件托管，不承诺永久可用。
- **annotation/URL 为主**：VATEX、HD-VILA-100M、ActivityNet 系列的部分入口及 YouCook2 的原始 YouTube 资产。原视频删除、地区限制和平台条款都会改变成功率。
- **已被官方撤回**：WebVid-10M 的 URL/caption 发布。
- **MSR-VTT**：官方论文页确认数据定义，但本次没有在该页核实到稳定 raw-video 下载与清晰视频本体许可。因此报告不采用来源不明的 Google Drive/HF 镜像来“补齐”这个事实。

## 24 GB Mac 的实际子集

### 历史候选 Phase A：VATEX-6K（不再执行）

- 从官方 train/val annotation 出发，记录 `source_id`、全部 captions、下载状态、原 URL、下载时间和 SHA-256。
- 成功视频中抽取 6,000 个：train 4,800、validation 600、test 600；**同一 source video 只能在一个 split**。
- 每个样本取单镜头 2 秒、8 fps、16 帧，先中心裁剪到 96×96；保留一条短 caption，其他人工 captions 用于语义等价评估而不是复制成 10 个视频样本。
- 原始下载预计几十 GB；规范化后的 H.264 clips 目标控制在约 5--12 GB。该数值是工程预算估计，不是 VATEX 官方统计。

### 历史候选 Phase B：VideoUFO-12K（不再执行）

先只下载 metadata，用 topic、brief/detailed caption 和质量指标筛选。目标因子：

```text
subject = cat | dog | person
action  = walking | running | jumping | sitting/standing
scene   = grass/field | forest | beach | snow | indoor/home
```

人工复核候选的 10% 和全部 test；排除多主体冲突、幻灯片、强剪辑、水印遮挡、caption 与画面不一致、动物主体过小的 clip。目标 12K clips，最低每个保留组合 120 个；不足的组合宁可缩小矩阵，也不要靠重复采样伪装平衡。

预处理建议曾是：2 秒、8 fps、16 帧、96×96 或 128×128，去音轨，H.264；保存原 caption、规范化三元组、source ID、许可、attribution、时间段、SHA-256、感知 hash 和预处理版本。12K 个这种短 clip 通常可以控制在约 10--25 GB；同样，这是未执行旧方案的预算估计，不是当前数据规模。

### 当前 Phase：Commons-1K → coverage gate

从现有 119 个 approved clips 出发，优先审核/补搜 `cat|playing|indoors` 及稀疏训练侧组合；只有逐文件许可、署名、来源、原始/裁切 SHA-256、人工三因子标签和 source/duplicate group 完整的 `approved` 样本才能进入 manifest。达到约 1K 后重新计算 source-group/pHash 闭包下的三元组覆盖：每个拟留出 C1 triple 至少 50 个独立 groups，且其每个训练侧 pair 至少 10 个独立 groups，才进入正式训练；否则缩小组合矩阵并公开缺口。当前 1,097 个候选中的 831 个仍为 pending，不能计入训练规模或覆盖。

### 2026-08-14 Commons 结构化分类审计

此前 8 个 `woman/man × walking/running × forest/snow/field` 全文窄查询只有 1 个新候选，人工核验后仍为 0 可用。官方分类树比继续堆关键词更可复现，但 Commons 并没有覆盖完整八格的 `person × {walking,running} × {forest,snow,field,beach}` 视频交叉分类；动作类、场景类和普通图片/混合媒体类必须分开处理。以下计数来自审计日的 Commons 官方分类页/Action API；`files` 与 `subcats` 都是**直属**数量，不含递归后代，分类会继续变化。

| 精确 Commons 分类 | 直属 files / subcats | 用途与事实边界 |
|---|---:|---|
| [`Category:Videos of people walking`](https://commons.wikimedia.org/wiki/Category:Videos_of_people_walking) | 20 / 3 | walking 的第一入口；直属文件已包含冬季森林和沙地行走素材，但仍需逐片确认是否有持续可见的单人主体。 |
| [`Category:Videos of people jogging`](https://commons.wikimedia.org/wiki/Category:Videos_of_people_jogging) | 4 / 1 | running 的最小高精度入口；Commons 在此主要使用 `jogging` 命名，不能假定存在同样丰富的 `Videos of people running`。 |
| [`Category:Videos of people jogging in sport events`](https://commons.wikimedia.org/wiki/Category:Videos_of_people_jogging_in_sport_events) | 0 / 2 | 两个直属子类通向 half-marathon/marathon；可递归扩召回，但预期有人群、道路和赛事偏差。 |
| [`Category:Videos of snow`](https://commons.wikimedia.org/wiki/Category:Videos_of_snow) | 170 / 8 | snow 场景池；数量大但多数不含目标人物动作，只适合反查和人工验真。 |
| [`Category:Videos of forests`](https://commons.wikimedia.org/wiki/Category:Videos_of_forests) | 以 API 实时计数 | forest 场景池；当前页能定位已经批准的 `Forest walking and outdoor cooking slovenia.webm`。审计时 API 限流，未将未锁定的计数写成事实。 |
| [`Category:Videos of beaches`](https://commons.wikimedia.org/wiki/Category:Videos_of_beaches) | 以 API 实时计数 | beach 场景池；适合反查 walking，但不能凭分类成员身份自动赋予动作或单主体标签。 |
| [`Category:Videos of agricultural fields`](https://commons.wikimedia.org/wiki/Category:Videos_of_agricultural_fields) | 以 API 实时计数 | `field` 的精确视频类只覆盖农田，不等于 grass/meadow/pasture；需要分别补查草地/牧场体系。其父 [`Category:Fields`](https://commons.wikimedia.org/wiki/Category:Fields) 在审计时将该子类显示为 4 subcategories / 3 files。 |

这些名字看似匹配、但不能直接作为视频训练池：[`Category:People walking in forests`](https://commons.wikimedia.org/wiki/Category:People_walking_in_forests) 为 11 files / 6 subcats，审计时直属 11 个均为图片；[`Category:People walking on beaches`](https://commons.wikimedia.org/wiki/Category:People_walking_on_beaches) 为 96 / 8，主要是图片；[`Category:People running in the snow`](https://commons.wikimedia.org/wiki/Category:People_running_in_the_snow) 为 2 / 0，两个都是 JPEG；[`Category:People running outdoors`](https://commons.wikimedia.org/wiki/Category:People_running_outdoors) 为 61 / 11，是混合媒体类。它们只能用于 category-membership 扩召回，并必须按媒体类型限定为 `video/*`；Commons 将 Ogg Theora 的 `.ogv` 报为 `application/ogg`，因此还只对 `.ogv + application/ogg` 开例外，普通 `.ogg` 音频继续拒绝。

推荐导入顺序是：① 全量审核 walking 的 20 个直属文件；② 全量审核 jogging 的 4 个直属文件；③ 递归 walking/jogging 子类并保存完整 category path、continuation 和 visited 集合；④ 按 snow → forests → beaches → agricultural fields 的顺序从场景池反查人物动作；⑤ 最后才使用上述普通 People 分类扩召回。每个候选仍为 `pending`，分类名和标题只能排序，不能生成 `subject/action/scene` 标签。赛事、人群、第一视角、time-lapse、动作主体过小和同一 source 的 sibling clips 仍按既有门禁拒绝或合组。

可复现查询使用 Commons 官方 [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Main_page)：

```text
# 一次锁定多个分类的直属计数；读取 categoryinfo.files/subcats
action=query&prop=categoryinfo&format=json&formatversion=2&titles=Category:...

# 枚举直属文件；跟随 continue，并按 video/* 或 .ogv+application/ogg 过滤
action=query&generator=categorymembers&gcmtitle=Category:...&gcmtype=file&gcmlimit=max&
prop=imageinfo&iiprop=url|mime|size|extmetadata&format=json&formatversion=2

# 枚举直属子类；递归时保存路径并防环
action=query&list=categorymembers&cmtitle=Category:...&cmtype=subcat&cmlimit=max&
format=json&formatversion=2
```

许可边界没有因采用分类树而变化。Commons 分类页明确文件适用各自描述页所列许可证；File namespace 的 structured data 与页面文本许可也不能替代视频本体的逐文件条款。导入时仍须保存 `LicenseShortName`、`LicenseUrl`、`Artist/Credit`、`AttributionRequired`、`UsageTerms`、页面 revision 与实际媒体 hash。**推断**：自由版权许可不等于画面中人物同意，也不自动解决隐私、人格权、公开权、敏感属性推断或商业代言问题；人物视频进入公开 checkpoint 前仍需单独风险复核。[Commons reuse 指南](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia) · [Commons 人物照片指引](https://commons.wikimedia.org/wiki/Commons:Photographs_of_identifiable_people)

### 2026-08-14 `cat × playing × indoors` 分类审计

这一组合有一个精确动作视频入口，但没有同样精确的 indoor 交叉类。审计日的官方 Action API 显示：

| 精确 Commons 分类 | 直属 files / subcats | 实际视频边界 |
|---|---:|---|
| [`Category:Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats) | 44 / 0 | 主入口。逐文件 `imageinfo` 为 39 个 `video/webm` 加 5 个 `.ogv + application/ogg`，按本项目的窄 OGV 例外共 44 个视频媒体；分类命中仍全部保持 `pending`。 |
| [`Category:Playing cats`](https://commons.wikimedia.org/wiki/Category:Playing_cats) | 258 / 8 | 宽泛父类；其子类包含上述 44-file 视频类，直属媒体主要是图片/GIF。只沿分类树进入精确视频子类，不把 258 files 当视频。 |
| [`Category:Cats with toys`](https://commons.wikimedia.org/wiki/Category:Cats_with_toys) | 3 / 4 | 直属 3 个均为 JPEG，视频数 0；四个子类是 balls、catnip toys、carrying toys in mouth、stuffed animals，也不能由物体共现自动推断 playing。 |
| [`Category:Cats playing with their prey`](https://commons.wikimedia.org/wiki/Category:Cats_playing_with_their_prey) | 12 / 1 | 直属只有 2 个视频（`Cat play 02.webm`、`Cat playing with a lizard.webm`）；捕食/活体猎物语义与“玩玩具”不同，只能作为显式允许该语义时的补充。 |
| [`Category:Videos of cats and mice`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats_and_mice) | 9 / 0 | 9 个直属视频；其中 7 个也在主 play 类，必须按 Commons page ID、原媒体 SHA-1 和 source group 去重，不能把交叉分类重复计样。 |
| [`Category:Videos of playing animals`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_animals) | 6 / 1 | 直属 6 个没有猫；唯一相关子类就是主 play 类，不导入直属文件。 |
| [`Category:Indoor cats`](https://commons.wikimedia.org/wiki/Category:Indoor_cats) | 149 / 6 | 直属列表为静态图片，且页面带有可能重分类提示；不是 indoor-cat 视频集。文件落入该类只能作为 indoor 的正线索，缺失也不能作为 outdoor 证据。 |
| [`Category:Cat behaviour`](https://commons.wikimedia.org/wiki/Category:Cat_behaviour) | 33 / 24 | 宽泛召回类；直属至少有与主 play 类重复的视频，首轮不整类导入。Commons 没有精确的 `Play behaviour in cats` 视频类。 |

主类的 44 个文件也绝不是 44 个独立来源。文件名已直接显示至少 25/44 属于六个高度相关系列：`Finchen playing 01--04`（4）、`Finchen+Lotti playing 01--11`（11）、`Kluse - Lotti ... 01--02`（2）、`Radevormwald - Kahli ... 01--03`（3）、`Radevormwald - Tiger ... 01--03`（3）、`Radevormwald - Lotti outdoor ... 01--02`（2）。最后一组还明确是 outdoor，应从当前 indoor 目标排除。**推断**：规范化文件名前缀、上传者、拍摄日期/地点、原媒体 SHA-1、pHash/视频指纹应共同形成系列 group；整组只能进入同一 split，且 pilot 可每组只留 1--2 条，避免同一猫与背景泄漏。

推荐导入顺序：① 先对主类 44 条只取 metadata/MIME/provenance，**既不下载训练媒体，也不批准**；② 优先人工审核名字明确指向室内或玩具的独立候选，例如 `... in a room`、laser pointer、round track toy、straw、string toy、playing in a box；③ 再审核其余非系列文件；④ 六个系列按 group 抽样或整组同 split；⑤ 仅在 taxonomy 接受捕食语义时补充 prey 类的 2 个视频；⑥ cats-and-mice 只导入与主类 page ID/SHA-1 的差集；⑦ 最后才宽扫 Cat behaviour/Indoor cats，强制媒体类型和逐片 `playing + indoors` 视觉确认。

复核使用同一官方 API 契约：先以 `prop=categoryinfo` 固定直属计数，再以 `generator=categorymembers` + `prop=imageinfo|categories` 读取 page ID、canonical URL、MIME、SHA-1、timestamp、categories 与 `extmetadata`，并跟随 `continue`。Indoor 只能由画面人工核验或已有文件分类提供正证据，不能从标题、玩具共现或 `Indoor cats` 分类缺失自动产生标签。所有新记录先进入 `pending`，直到完整播放、选择稳定 2 秒片段、许可快照和系列分组均通过。

许可仍逐文件执行，不存在分类级统一授权：保存 `LicenseShortName/LicenseUrl/Artist/Credit/AttributionRequired/UsageTerms`、页面 revision 与实际下载媒体 hash。猫本身通常没有人物肖像权问题，但室内视频可能包含可识别人物、私人空间、电视/艺术品、音乐或其他第三方内容；这些仍须逐片拒绝或单独复核。[Commons 分类页的逐文件许可说明](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)

### `cat × sitting × indoors` 是否应取代 playing

结论是**不切换主目标**。作出路线选择时的 102-record 历史快照中，Sitting 表面有 13 个来源、playing 有 10 个；在后续显式系列/uploader 保守分组后的当前 111-record v3 中，两者分别只有 11 和 14 个独立 groups。无论采用哪一快照，几条样本的表面差异都不代表 Commons 后续供给更好：官方分类体系给 sitting 的是静态姿态大类，而 playing 有精确的 44-file 视频类。

| 精确 Commons 分类 | 直属 files / subcats | 视频供给事实 |
|---|---:|---|
| [`Category:Sitting cats`](https://commons.wikimedia.org/wiki/Category:Sitting_cats) | 1312 / 12 | 静态图片主导。Action API 分页的首 500 个直属文件为 493 JPEG、6 PNG、1 WebM；唯一视频 `Cat playing with a lizard.webm` 又属于 playing/prey 语义，并非新增的稳定 sitting 来源。API 限流前未完成全部 1312 个 MIME 枚举，因此不把“首 500 只有 1 个”误写为全类视频总数。 |
| Sitting 的 12 个子类 | 见官方分类树 | 包括 breed/color/quantity、art、kittens、catloaf、crouching、sitting on objects/in pots、side views、Philippines 等姿态或图像组织类；官方树中没有 `Videos of sitting cats` 或 `Videos of cats sitting`。 |
| [`Category:Indoor cats`](https://commons.wikimedia.org/wiki/Category:Indoor_cats) | 149 / 6 | 静态图片主导且有重分类讨论；没有独立 sitting-video 子类。 |
| [`Category:Perched indoor cats`](https://commons.wikimedia.org/wiki/Category:Perched_indoor_cats) | 33 / 0 | 最接近 sitting × indoors 的交叉类，但直属成员是静态图片，视频 MIME 为 0。 |
| [`Category:Videos of cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats) | 78 / 26 | Sitting 的现实召回入口：官方 26 个动作视频子类含 playing/eating/drinking/kneading/grooming/jumping/yawning 等，但没有 sitting；必须从 78 个直属视频及非-playing 子类逐片找持续坐姿。 |
| [`Category:Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats) | 44 / 0 | 44 个均为 WebM/OGV 视频，提供现成的高精度 playing 行为候选；仍需 indoor 人工验真与系列分组。 |

两类也不能简单做 `1312 vs 44` 或 `78 + 44` 的加法。Sitting 的大量文件包含 crop/original 对和同一拍摄连拍；`Videos of cats` 的直属成员与动作子类可能双重分类；playing 已知有 Finchen、Finchen+Lotti、Kahli、Tiger 等系列。差集先按 Commons page ID，再按原媒体 SHA-1，最后按 uploader + 标题前缀 + 日期/地点 + pHash/视频指纹合并。同猫、同房间、同一拍摄事件必须留在同一 group。

Sitting 还有一个标签定义问题：多数猫视频会短暂出现坐姿。若未来把它作为补充对照，必须预先冻结可复现规则，例如目标 2 秒 clip 内坐姿占比达到约定阈值、主体持续可见，且没有 playing/grooming/eating 等更显著主动作；阈值要在查看 C1 test 结果前确定。这个规则是本项目的**工程定义**，不是 Commons 分类提供的事实。

因此推荐保持 `cat × playing × indoors` 为主扩容目标：先审核其 44 个精确视频成员、排除明确 outdoor，并按系列 group。Sitting 只作为补充探索：metadata-first 扫描 `Videos of cats` 的 78 个直属视频，再递归非-playing 动作子类；所有命中保持 `pending`，只有人工确认 indoor、持续坐姿、许可和独立 group 后才计入 coverage。只有该审计获得足够独立来源并达到同一 C1 门槛，才重新考虑切换。

许可边界与 playing 完全相同：分类页不是统一媒体许可证，仍逐文件保存许可证、作者/署名、页面 revision 和媒体 hash，并审查室内人物、私人空间、音乐、电视和艺术品等第三方内容。

### `cat × playing × indoors` 的训练侧 support 与下一批高精度入口

以下表格固定使用扩入 dog-playing 之前的 111-record v3 快照，并复用 coverage 报告的 `source_id + duplicate_group + pHash（Hamming ≤ 4）` 传递闭包。先把拟留出的 `cat|playing|indoors` 18 条记录（14 个独立 groups）全部移出训练，再统计其余 93 条；因此这些数字是 **holdout 后的训练侧 support 历史基线**，不是全 manifest 的边际计数。当前 114-record 结果见后文执行记录。

| support | 独立 groups / records | 训练侧来源 | 判断 |
|---|---:|---|---|
| atom `cat` | 40 / 50 | 17 个非 held triples | 已有基础，不是首要缺口。 |
| atom `playing` | 5 / 6 | cat-outdoors、dog-indoors、dog-outdoors | 最弱原子；继续只补 held triple 不会改善训练侧 playing。 |
| atom `indoors` | 34 / 41 | 18 个非 held triples | 已有基础。 |
| pair `cat × playing` | 3 / 4 | 全部是 `cat|playing|outdoors` | 很薄；要从其他真实场景补，不能把 indoor held 样本留在 train。 |
| pair `playing × indoors` | 1 / 1 | 仅 `dog|playing|indoors` | 最严重 pair 缺口；应优先补不同主体的室内玩耍。 |
| pair `cat × indoors` | 26 / 32 | drinking/eating/grooming/hissing/jumping/kneading/lying/opening-door/scratching/sitting | 相对最强，但动作分布长尾；可作为次优先扩容。 |

形式上每个 pair 只要 1 group 就能证明“见过”，但这不足以支撑可信的组合实验。**预注册工程门禁**：coverage 和正式 split 均要求三个训练侧 pair 各至少 **10 个独立 groups**，而不是只把 held triple 补到 50；当前排序是 `playing×indoors` → `cat×playing` → `cat×indoors`。该阈值已在看 C1 生成结果之前固化进 CLI 报告。

2026-08-14 的 Commons 官方分类页/Action API 审计得到以下高精度入口。表中 `files/subcats` 都是直属计数；“视频数”按逐文件 `imageinfo.mime` 核验。它们只定义**候选池**，不产生自动标签或 approved 状态。

| 官方 Commons 分类 | 直属 files / subcats；视频边界 | 对 support 的价值与风险 |
|---|---|---|
| [`Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats) | 44 / 0；44 个视频 | `cat×playing` 的最高精度入口，但至少 25/44 属于 Finchen、Finchen+Lotti、Lotti box、Kahli、Tiger、Lotti outdoor 六个系列；明确 outdoor 的两条不能进 held indoor，系列整组同 split。 |
| [`Videos of cats and mice`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats_and_mice) | 9 / 0；9 个视频 | 至少 7/9 与上述主类同名重叠；page ID 差集上限只有 2，且仍须核验 playing/scene。 |
| [`Cats playing with their prey`](https://commons.wikimedia.org/wiki/Category:Cats_playing_with_their_prey) | 12 / 1；直属仅 2 个视频 | 与主类重叠后净新增上限 1；捕食/活体猎物不是默认“玩玩具”语义，后置审核。 |
| [`Videos of cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats) | 78 / 26；78 个直属视频 | 可补 `cat×indoors` 的不同动作，但没有 indoor 视频子类，必须逐片判断。其动作子类包括 eating 23、grooming 21、drinking 10、kneading 9、looking-at-viewer 7（实际 6 视频 + 1 GIF）、jumping 5、yawning 2、fighting 2；这些数量都不保证 indoors。Finchen/Lotti 即使 page ID 不同也可能是同猫同环境。 |
| [`Videos of dogs`](https://commons.wikimedia.org/wiki/Category:Videos_of_dogs) | 141 / 7；141 个直属视频 | 当前最值得审的 `playing×indoors` 入口。标题可先排序 `puppy playing`、`play fighting`、`Two poodles playing` 等，但动作和室内场景必须看片；同一 Goldendoodle/puppy 命名系列合并。 |
| [`Playing dogs`](https://commons.wikimedia.org/wiki/Category:Playing_dogs) | 107 / 9；直属实际仅 2 个视频 | 两个可见视频是 `Two poodles playing.ogv` 与 `Пузыри собака.ogv`，其余主要是图片；不能把 107 当视频规模，且两条仍需 indoor 验真。Commons 没有精确的 `Videos of playing dogs` 类。 |
| [`Videos of playing animals`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_animals) | 6 / 1；6 个直属视频，另有 playing-cats 子类 | 可补 playing 原子，但直属多为野生/动物园动物且不保证 indoors；不能当 domestic indoor 的替代分布，只作末位召回。 |
| [`Indoor cats`](https://commons.wikimedia.org/wiki/Category:Indoor_cats)、[`Indoor animals`](https://commons.wikimedia.org/wiki/Category:Indoor_animals) | 149 / 6 与 7 / 2；两者直属视频均为 0 | 静态图片池，只能把 category membership 当 indoor 的正向辅助线索，不能作为视频入口或把缺失当 outdoor。 |
| [`Cat toys`](https://commons.wikimedia.org/wiki/Category:Cat_toys)、[`Cats with toys`](https://commons.wikimedia.org/wiki/Category:Cats_with_toys) | 87 / 4 与 3 / 4；直属为静态媒体，后者视频为 0 | Commons 没有 boxes/laser/string/toys 的精确猫视频类；只能在 playing-cats 的标题、description、depicts 中召回，再逐片审核。 |

可复现 API 分三步，所有请求都固定保存响应时间和 continuation：

```text
# 1. 同一次读取直属计数
action=query&prop=categoryinfo&
titles=Category:Videos_of_playing_cats|Category:Videos_of_cats|
Category:Videos_of_cats_and_mice|Category:Videos_of_playing_animals|
Category:Playing_dogs|Category:Videos_of_dogs|Category:Indoor_cats|
Category:Cat_toys&format=json&formatversion=2

# 2. 枚举文件并数真实 MIME；替换 gcmtitle，始终跟随 continue
action=query&generator=categorymembers&gcmtitle=Category:Playing_dogs&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2

# 3. dog playing 只作排序检索，命中仍回到 page ID/imageinfo 并人工审核
action=query&list=search&srnamespace=6&
srsearch=incategory:"Videos of dogs" (playing OR play OR toy)&
srlimit=max&format=json&formatversion=2
```

导入前先做 `candidate page IDs − current candidate page IDs`，再按原媒体 SHA-1 去内容重复，最后用 uploader、规范化标题前缀、拍摄日期/地点、关键帧 pHash/视频指纹和音频指纹合并系列。推荐顺序是：① 主类 44 中尚未审完的非系列、标题明确 room/toy/box/laser/string 候选；② `Videos of dogs` 的 playing 标题候选及 `Playing dogs` 的 2 个视频，优先建立不同主体的 `playing×indoors`；③ 主类中确认为 outdoor 的独立猫视频，补 `cat×playing` 训练 pair；④ `Videos of cats` 直属与 eating/grooming/drinking/kneading 等子类，补不同动作的 `cat×indoors`；⑤ cats-and-mice/prey 只取 page ID/SHA-1 差集；⑥ 最后才看其他 playing animals。任何一步都只写 pending；完整播放、稳定 2 秒 clip、逐文件许可/署名和显式 series/uploader group 全部通过后才可 approved。

### `cat × playing × outdoors` 精确入口与 mixed-held 身份审计

2026-08-14 的官方 Commons 分类/API 审计没有发现 `Videos of cats playing outdoors` 或 `Cats playing outdoors` 这种精确视频交叉类。高精度方法只能是：以 [`Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats) 的 44 个直属视频为动作全集，再与 [`Outdoor cats`](https://commons.wikimedia.org/wiki/Category:Outdoor_cats) 及其后代分类、MediaInfo structured depicts 的 `outside` 声明做文件级交叉，最后人工确认 clip scene。分类缺失不能反推 indoors。

| 官方入口 | 直属 files / subcats；视频边界 | 本轮可用事实 |
|---|---|---|
| `Videos of playing cats` | 44 / 0；44 个视频 | 仍是唯一高精度动作池。官方 API 对 44 个 page IDs 的 categories 扫描明确找到 `Cat playing with a lizard.webm` 属于 `Outdoor cats`；另外两条 Lotti 文件名直接写明 outdoor。三条均已在 44 中，因而相对该池的净新增为 0。 |
| `Outdoor cats` | 248 / 13；直属页面为图片，审计时视频 MIME 为 0 | 13 个后代为 Germany、Taiwan、beaches、dry leaves、forests、gardens、grass、rocky background、roofs、sand、sleeping-outdoor、snow、walls。它们是 location 正信号，不是视频集。 |
| [`Cats in forests`](https://commons.wikimedia.org/wiki/Category:Cats_in_forests) / [`Cats on sand`](https://commons.wikimedia.org/wiki/Category:Cats_on_sand) | 34 / 0 与 6 / 0；均为图片 | 不能用场景类的 files 数补视频 coverage。`Cats in snow` 也没有对应视频专类。 |
| [`Videos of cats in Germany`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats_in_Germany) | country 视频类，非 outdoor/playing 交叉类 | 列出 Lotti `outdoor v 01/02`，也列出两个不带 `v` 的 `outdoor 01/02 (1)`。后两条是相对 44 的潜在 page-ID 差集 2，但标题只证明 outdoor、不证明 playing，而且极可能是同一场景的版本或相邻片段。 |

三个有一手 outdoor 证据的主类成员是：

- `Cat playing with a lizard.webm`：page ID `68960767`，`video/webm`，原媒体 SHA-1 `705009486b5da6bb2caac9da2016fcd5ee615e10`；文件 categories 同时含 `Outdoor cats` 和 `Videos of playing cats`，structured depicts 含 outside。它涉及活体猎物/捕食语义；当前 manifest 却标为 `cat|playing|indoors`、group `commons-uploader:feb641b2afeb35c3`、署名 `Basile Morin`，必须重新查看完整视频和已选 2 秒窗口，确认究竟是户外、半户外还是错误分类，不能由 API 自动改标签。
- `Radevormwald - Lotti outdoor v 01 ies.webm`：page ID `44435788`，SHA-1 `c741faceb49d44915f775e10a6cf65733ce77067`；以及 `... v 02 ...`：page ID `44435861`，SHA-1 `e3f5504b0b9625d8b2362292d04cba120259bf41`。两条当前均 approved 为 `cat|playing|outdoors`，署名 `Frank Vincentz`，且已正确共享 `commons-series:radevormwald-lotti-outdoor`，只能贡献 1 个独立 group。

可复现查询仍先取动作池，再构建 outdoor 子类闭包：

```text
action=query&generator=categorymembers&gcmtitle=Category:Videos_of_playing_cats&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2

action=query&list=categorymembers&cmtitle=Category:Outdoor_cats&
cmtype=subcat&cmlimit=max&format=json&formatversion=2

# 补召回，不代表 playing：枚举 country 视频类后与 44 的 page IDs 作差
action=query&generator=categorymembers&gcmtitle=Category:Videos_of_cats_in_Germany&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2
```

对两个非 `v` Lotti outdoor 文件，须依次比较 page ID、SHA-1、标题前缀/上传者/拍摄时间地点、关键帧和音轨指纹；即使 SHA-1 不同，也不能当成独立来源。推荐 pending 顺序：① 重审 lizard 当前 indoor 标签；② 审核两个非 `v` Lotti 文件是否真 playing，并与现有 `v 01/02` 归为同一系列或择一；③ 对 44 全量做 Outdoor-categories/structured-outside 扫描；④ 最后才从 `Videos of cats` 和 country 视频类用 outdoor/garden/grass/snow/beach/forest + play 关键词低优先级召回。普通 Outdoor cats/grass/garden/snow/beach 静态类不进入视频候选。

复核前的 114-record v3 历史快照曾显示 held `cat|playing|indoors` 的组与 kneading、scratching、sitting 三个普通 triples 混合。展开后实际是两个 uploader 保守组，而不是 pHash 碰撞；本表记录当时的审核输入，最终消解结果见后文执行记录：

| 当时 group / attribution | held 成员 | 非 held 成员 | 必须复核的问题 |
|---|---|---|---|
| `commons-uploader:e3a2c458f1a504c3` / FASTILY | `Cat Playing with Straw 1 2017-06-08`、`2 year old cat playing with stick 1 2016-10-27`、`Cat Playing with String Toy 1 2014-02-04` | `Cat Using Scratcher 1 2016-06-09`、`Siamese Tabby Cat 1 2016-10-11`、`Cat Kneading 1 2017-07-07` | 六条跨 2014--2017；仅同 uploader 不足以证明同猫，也不足以证明独立。逐条比对猫的毛色/花纹、脸部与体态、房间/家具、拍摄元数据和音轨。能证明不同身份/事件才拆组；否则整组不能同时跨 held/train。 |
| `commons-uploader:6449c9e8d7c13f5e` / Nesnad | `Kitten playing - Tokyo - Jan 7 2020` | `Catblinking-tokyo-area-2018-4-9` | 相隔约 21 个月且标题分别为 kitten/cat，但同 uploader/Tokyo 不能排除同一动物或同一室内环境。重点核验年龄是否相容、毛色标记、房间背景和拍摄位置；未形成可审计结论前保持一组。 |

这一步不是为了最大化 group 数。若视觉身份无法可靠确认，保守选择是维持 uploader group，并从 train 或 held 一侧移除冲突成员；只有形成逐文件身份/场景审核记录后才可拆组。任何拆组决策必须在查看 C1 生成结果之前冻结。

### `Videos of playing cats` 44-member 状态闭环

2026-08-14 重新通过官方 Action API 枚举 [`Category:Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats) 的全部 44 个直属文件，并以 Commons title 与当前 1,093-row candidates 做严格一对一连接：44/44 均唯一映射，没有缺失。API 的媒体边界仍是 39 个 `video/webm` 加 5 个 `.ogv + application/ogg`。**终审前快照**为 **13 approved / 12 rejected / 19 pending**；这是分类池当时的状态，不等于 44 个独立来源，最终结果见本节后文。

官方枚举可复现为：

```text
action=query&generator=categorymembers&gcmtitle=Category:Videos_of_playing_cats&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2
```

以下是仍 pending 的完整 page ID/SHA-1 状态表；标题是目前唯一的 series 证据，scene 必须看片。所有 19 条都来自 User:Ies 的德国猫系列，并非 19 个新 groups。

| pending 文件 | page ID | 原媒体 SHA-1 | 系列/场景先验与独立组上限 |
|---|---:|---|---|
| `Finchen playing 02 (1) ies.webm` | 30905636 | `d856a7ce8f53afe7851dc4d106149c6c6000727f` | 与已 approved `Finchen playing 01` 同一 Finchen 系列；02--04 合计最多仍是 **0 个新增 group**。官方 categories 不给 indoor/outdoor，需看片，但不能拆系列。 |
| `Finchen playing 03 (1) ies.webm` | 30905748 | `b2cd48292c8594ec0d2bdb36160e48a822a44f6e` | 同上。 |
| `Finchen playing 04 (1) ies.webm` | 30905731 | `44f47dba5ff35f32823ad632a36ec1f45b2ec0ed` | 同上。 |
| `Finchen+Lotti playing 01 (1) ies.webm` | 30743100 | `2bea6b2f7f5ffbd7a20d31f592937e7995afe183` | 01--11 标题明确同一双猫系列；违反当前单主体 held 目标的先验很强。即使某个 2 秒窗口只见一猫，也须证明另一猫未影响动作/场景。该系列又以具名身份连接已有 Finchen 与 Lotti 系列，按 identity 传递闭包对当前 coverage 最多是 **0 个新增 group**。 |
| `Finchen+Lotti playing 02 (1) ies.webm` | 30743383 | `3a46780aafb8f5733d250e71c74fb8b2bb9ee934` | 同上。 |
| `Finchen+Lotti playing 03 (1) ies.webm` | 30743386 | `79a64e38d067bf265d799ced57f3eb136e0060e9` | 同上。 |
| `Finchen+Lotti playing 04 (1) ies.webm` | 30743523 | `a09e814fb335e5709c214bcad0d12288c26f160b` | 同上。 |
| `Finchen+Lotti playing 05 (1) ies.webm` | 30743446 | `7c1a77784922edc6f795ccc1ce6531f076eccfad` | 同上。 |
| `Finchen+Lotti playing 06 (1) ies.webm` | 30743701 | `5bafc65c77f8f17ebda42d8a3a1c71278a1a62d6` | 同上。 |
| `Finchen+Lotti playing 07 (1) ies.webm` | 30743655 | `ba8bad824a9e15b80e34c88a9914c4d8c0707b47` | 同上。 |
| `Finchen+Lotti playing 08 (1) ies.webm` | 30743839 | `be6a1b31523cff4245640066b8a593a4c740ccf3` | 同上。 |
| `Finchen+Lotti playing 09 (1) ies.webm` | 30743851 | `9febbd2405f05a127a4f7d794bbe14daa8f63bd8` | 同上。 |
| `Finchen+Lotti playing 10 (1) ies.webm` | 30743904 | `b8e772837fcae679280802b726eaa288a2ff10ee` | 同上。 |
| `Finchen+Lotti playing 11 (1) ies.webm` | 30905442 | `7dc7380cdb36bb846efbcb1f2160fa27f9eec0dc` | 同上。 |
| `Kluse - Lotti playing in a box v 02 ies.webm` | 157840583 | `2c194bc074ed2688a45f95d2d1b0a7652baf0a08` | 与已 approved v01 同一 Lotti-box indoor 系列，最多 **0 个新增 group**。 |
| `Radevormwald - Kahli playing with mouse 02 ies.webm` | 37818567 | `fbae222305ca29b010154ce794f254919f173df7` | 与已 approved Kahli 01 同一 toy/mouse 系列，最多 **0 个新增 group**；categories 不证明 indoors。 |
| `Radevormwald - Kahli playing with mouse 03 ies.webm` | 37818707 | `bbd477f151e885777ce90b1d7bfa1760c70fceb2` | 同上。 |
| `Radevormwald - Tiger playing with mouse 02 ies.webm` | 37818837 | `fe8ed68721edc502063a63f4b13e5804652602cf` | 与已 rejected Tiger 01 同一系列；01 因 96×96 主体过小/动作不明确拒绝。02--03 是唯一尚可能带来**至多 1 个新单猫 group**的 pending 系列，但须全长与密集窗口复核，且 categories 不证明 indoor/outdoor。 |
| `Radevormwald - Tiger playing with mouse 03 ies.webm` | 37818892 | `d2167f5151ab3f48d70a6116809b94feecd1e610` | 同上。 |

因此 19 个 pending 的标题级 series 上限只有 5 组：Finchen、Finchen+Lotti、Lotti-box、Kahli、Tiger。但按动物身份做传递闭包后，Finchen+Lotti 会把已有 Finchen 与 Lotti 系列连在一起，不能贡献新独立组；Finchen、Lotti-box、Kahli 也已有 approved 成员。对当前独立 coverage 的**净新增上限只有 1 group**，即尚无 approved 成员的 Tiger 系列。审核优先级因此是 Tiger 02/03；其余只用于同组多样性或标签复核，不用于补 group deficit。若任何 Finchen+Lotti 片段被保留，还必须使用连接相关身份系列的共同 group，不能为了增加计数另建新组。

已终结的 25 条按官方 page ID/SHA-1 留档如下，便于未来验证 candidates 没有状态漂移：

| status | 文件（page ID；SHA-1） |
|---|---|
| approved | `Cat Play` (99821145; `cee70c5e77926c734d4beccde941ad409331c67d`); `Cat playing with a laser pointer` (23819120; `3aeda29031bde0727db50dc638f457ea1ee9b632`); `Cat playing with a lizard` (68960767; `705009486b5da6bb2caac9da2016fcd5ee615e10`); `Cat playing with a round track toy` (22864110; `b75cecad2f79a52703e9898e6f884f1a516d1d42`); `Cat Playing with Straw 1` (59747422; `886c2ff1b50a4160348570ab0792727cfb68cd3c`); `Cat Playing with String Toy 1` (30986610; `dba8defef95b5f466eb62467fb1f547f081535c8`); `Cat Plays` (37239427; `9d3f2a0bd25a03b8938543eca96e9ce4a8d735d2`); `Finchen playing 01` (30905392; `6e16ebd7fa59c329bede27592f8614153ed30447`); `Kluse - Lotti ... box v 01` (157840541; `8e8c7214561bf19afc46f37e82eb82a0d4f80d68`); `Moomin1` (99462733; `b8cc558859ad4acfc423ed62d0837e9b1f29576f`); `Kahli ... mouse 01` (37818555; `a0f6d4a0f55db3907ea19748f836e02908232c84`); `Lotti outdoor v 01` (44435788; `c741faceb49d44915f775e10a6cf65733ce77067`); `Lotti outdoor v 02` (44435861; `e3f5504b0b9625d8b2362292d04cba120259bf41`). |
| rejected | `ASC Leiden ... Three cats` (83284675; `8aaa620bdf6e88d0873fc9f8502f3d6370272786`); `Black and tuxedo kittens` (67463578; `34ce7e66f87425ff63ec8e7381db7335f8b59d89`); `Cat chasing mouse in a room` (136356464; `d22a7655201d77e0ce240b6cd532dd13241a745b`); `Cat Play 03` (99821544; `97959bbda516aae8057e97fa3ed24e4d0b12552f`); `Cats Endulging in Domestic Violence` (47110441; `27e92b792e223c6c2824aa1a398c3addf093be26`); `Chat domestique` (50254718; `b515f7319bb4b89834c02ea2bf020b551eaa471d`); `Felis ... Monarca` (20074545; `29e8f3c491fbf1938489aa7b886cfbcfe3895be3`); `Larry ... Chloé` (97411267; `7d8ad97a93c959d18f433998b9659d6211276c6d`); `Muslim playing with kitten ...` (65579958; `a9d363667547359f307e981ab9a12c42af37b4c1`); `Tiger ... mouse 01` (37818689; `0eefe48259e5076d7a21745071a4ff14ff732314`); `Spelende poesjes` (34815492; `67405ff46c4271f37732133ff008703187d2aa4e`); `Whisky-Iswed` (20301970; `b9891f7c6ad6d871442af1b8aebe50015af3cee6`). |

13 个 approved 文件在当前人工 grouping 下只有 **11 groups**：laser-pointer 与 round-track-toy 共用一个 uploader 保守组；两条 Lotti outdoor 共用一个 series；其余各一组。这里的 11 只是该 44-member 分类池内部的 groups，不能与整个 held triple 的 15 groups 混为一谈。对任何 pending 的批准仍要求完整播放、稳定 2 秒窗口、单主体、scene、许可和显式 identity/series group；API category 命中本身不改变状态。

随后按上述冻结规则完成 19 条 pending 的视觉终审，而不是按标题自动通过。11 条
`Finchen+Lotti` 的全时长联系表均出现两只具名猫，违反单主体契约；Finchen、Kahli、
Lotti-box 的其余片段以及 Tiger 02 都是已有或本轮所选代表的同身份、同玩具、同房间系列
冗余，因此拒绝。Tiger 03 的全长 12 帧联系表显示一只燕尾服猫在室内木地板上持续拍打、
追逐玩具鼠，批准为该系列唯一代表，group 固定为
`commons-series:radevormwald-tiger`。批次结果是 **1 approved / 18 rejected**；审计报告绑定
候选快照 SHA-256 `ca8f025e537216bf342cf416927a5f90aedf10f77f541daf3532d4bc027e5084`
以及每条 Commons 原媒体 SHA-1，任何 pending 或缺失 title 都会 fail-closed。

重建后的权威候选仍是 1,093 条，但状态变为 **117 approved / 861 pending / 115 rejected**；
v3 manifest 为 **117 records / 89 independent groups**，held
`cat|playing|indoors=16/50`。如预期，这 19 条只增加 Tiger 的 1 个 group；训练侧 pairs 完全
不变，仍为 `cat×indoors=28/10`、`cat×playing=4/10`、
`playing×indoors=4/10`，因此仍是 0 eligible C1。manifest SHA-256 为
`d9f0b52761260df6eb5c4466a6854699552549e0ee2ad90b20603d4ae27f45ef`，coverage SHA-256 为
`20129e86a829a40725afc4411c84de3f1f489ab3c05f44e4110816ac77832ff5`。这一步把精确 44-member
池从“有待审核”变成了闭合负结果：它无法继续提供新的训练 pair support，下一轮必须转向
`dog × playing × indoors` 与 `cat × playing × outdoors` 等训练侧组合，而不能继续重复抓同一猫系列。

### 训练侧两个 pair 的官方分类差集审计

2026-08-14 以当前 1,093-row candidates 的 Commons title 与已保存原媒体 SHA-1 为集合，重新枚举官方分类并做 `(page ID/title/SHA-1) − candidates`。结论是：常见的高精度 dog-playing 分类已经基本穷尽，cat-playing 主类已经完全闭合；官方分类树目前不能直接提供足够的新独立 groups。

#### `dog × playing × indoors`

[`Videos of dogs`](https://commons.wikimedia.org/wiki/Category:Videos_of_dogs) 当前有 141 个直属 files / 7 个 subcats；逐文件 API 得到 140 个视频容器，另有一条非视频/异常媒体边界。与 candidates 做 title/SHA-1 差后有 18 个未覆盖视频，但标题主要是历史影片、工作犬、犬舍、警犬或误分类（例如 `The Butcher Boy`、`Cat birth 2`、`Dog Factory`、警犬学校与军犬新闻），没有新的高精度 indoor-playing 标题候选。不能为了补数把这 18 条自动归为 playing。

Commons 没有 `Videos of playing dogs` 精确类。[`Playing dogs`](https://commons.wikimedia.org/wiki/Category:Playing_dogs) 为 107 files / 9 subcats；严格接受 `video/*` 或 `.ogv + application/ogg` 后，直属实际有 4 个视频，且已全部在 candidates：

| 文件 | page ID / SHA-1 | 当前状态与独立性 |
|---|---|---|
| `Puppy playing.webm` | 27014028 / `733210c6cc9a1c2d54b93011b164cbbc1f2e10d8` | 官方 description 明确为 “A puppy playing in house”，是最强 indoor 正线索；但已在 candidates 且当前 approved，不是新增差集。仍应以最终 2 秒窗口和 provenance 为准。 |
| `Пузыри собака.ogv` | 15446135 / `b6c1fd97f257c36d04941cdaeed6e3d34d23fe8f` | 已 approved：单狗在室内追逐泡泡；已有独立 uploader group，不是新增。 |
| `Two poodles playing.ogv` | 10794007 / `72dd708bd5433482aadbf194b48af015ab04a8df` | 已 rejected：两狗在草地玩耍，既非单主体也非 indoor。 |
| `Goldendoodle ... puppies play fighting ...` / `... puppies playing ...` | 30032853 / `6e19ec10962101fd59ba0c00f9a7c34a935f8db5`；30032854 / `b2cc96a6c046b56452eb8e797239c8a748427f3f` | 同日、同两犬、同 grassy yard；均已在 candidates，且最多一个 outdoor series group，不补 indoor pair。 |

九个子类是 `Dogs with balls`、`Dogs carrying toys`、`Dogs chewing`、`Dogs rolling`、`Dogs fetching`、`Puppies playing`、`Social play of dogs`、`Dogs playing tug-of-war`、`Dogs playing in water`。逐类直属 MIME 差集显示：balls/rolling/fetching 等视频已在 candidates，puppies/social/tug-of-war 的直属成员是图片；唯一新的视频差集是：

- `PerroCallejeroJugandoLLuvia20240828 ohs01.webm`，page ID `153480181`，SHA-1 `0e6dbe3395601cfb0e929189b37e70d7dc1b1788`，位于 `Dogs playing in water`。标题指向流浪狗在雨水中玩耍，是 outdoor/water 的强先验，不能补 `playing×indoors`；若未来需要 dog-playing-outdoors，可作为 pending，并先核验单狗、持续动作、许可和 uploader/事件指纹。

`FB-PLAY 0.ogv`（22110824；`d37df0362271d3ef507d155508fbcb7d88165e06`）和 `Throw the Stick!.ogv`（37854436；`29f0e62245460dd44d6a7b1b5f5c2f3a45696989`）虽仍在本地 pending，但前者是 flyball、后者仅有投树枝标题；metadata 不证明 indoor，审核优先级低于寻找新的官方来源。`Indoor dogs` 本身是 36-file 静态位置类，视频数 0，不能作为扩容入口。

#### `cat × playing × outdoors`

`Videos of playing cats` 的 44 个直属成员已与 candidates 44/44 闭合；其中官方 metadata 的三个 outdoor 强线索——lizard 与两条 Lotti outdoor——也已审核。`Outdoor cats` 的 248 个直属文件及 forests/sand/snow/garden/grass 等后代主要是静态图片，不存在精确 `Videos of cats playing outdoors` 类。

`Videos of cats in Germany` 的 61 个直属视频也已全部存在于 candidates，title/SHA-1 新差集为 0。其两个非 `v` Lotti outdoor 文件已人工确认只是躺/坐而非 playing，并拒绝；两个 `v` 文件属于同一 Lotti filming series，只贡献 1 个 outdoor group。因此 cat 侧本轮没有任何合法的新 page-ID/SHA-1 候选。

可复现查询与差集步骤如下；所有递归必须保存 category path、跟随 `continue` 并防环：

```text
# 狗视频全集 / 玩耍父类；cat 侧替换 gcmtitle
action=query&generator=categorymembers&gcmtitle=Category:Videos_of_dogs&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2

action=query&generator=categorymembers&gcmtitle=Category:Playing_dogs&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2

# 先取九个直属子类，再逐子类执行上面的文件枚举
action=query&list=categorymembers&cmtitle=Category:Playing_dogs&
cmtype=subcat&cmlimit=max&format=json&formatversion=2
```

差集按 page ID/title 发现未见页面，按 SHA-1 排除同字节内容，再以 uploader、具名动物、日期地点、标题前缀、关键帧/音轨指纹建立传递 group；不同 SHA-1 绝不自动等于独立来源。推荐审核顺序：① 不再重复审已闭合的 44 个 cat-playing；② 对新雨中流浪狗只在需要 outdoor pair 时审核；③ 对狗全集 18 个差集先做 metadata 排除，不下载明显历史片/工作犬；④ 真正的 `playing×indoors` 扩容需要寻找新的、逐文件可许可的官方 Commons 视频分类或新上传，而不是继续递归静态 Playing/Indoor 类。所有新命中仍只进入 pending。

#### 本轮 pair-deficit 审核与可执行搜索结果

分类差集之外，又对现有 pending 中最可能补 pair 的 10 条生成全长 16 帧联系表；只有
`-01- Miniature Pinscher puppy.webm` 同时满足单狗、室内和持续 playing。密集前 2 秒的
16 帧显示幼犬持续用嘴和爪摆弄毛绒玩具，批准为
`dog|playing|indoors`，group 固定为
`commons-audited-unique:miniature-pinscher-puppy-runologe`。其余 9 条中，三条巴哥视频和
拔河片段含多人/多狗；Capuccino 与 `How to punish` 是人物主导互动；contrafeeloading 是
觅食任务；flyball 与 throw-the-stick 是户外。因此该批是 **1 approved / 9 rejected**，
没有把弱相关动作改写成 playing。

cat 侧保留 Ljubljana 2022-06-13 同日六段中的一条代表：全长和密集窗口均显示单只灰猫在
林地持续扑抓鼠，prompt 明确保留为 `a cat playing with a mouse outdoors`，不把 prey 偷换成
toy；六条统一为 `commons-series:ljubljana-cat-mouse-2022-06-13`，其余五条以同事件冗余拒绝。
该批是 **1 approved / 5 rejected**，只贡献一个 group。

最后执行仓库内 `configs/commons_pair_deficit_search.json` 的 8 个 coverage-driven 查询（dog indoor
toy/room 与 cat outdoor garden/grass/mouse），新差集只有 3 条，全部是犬神经科病例或人物
新闻片的 `cat` 关键词假阳性，已在 metadata gate 拒绝，未下载媒体。这个负结果与官方分类
审计一致：Commons 当前高精度池已接近穷尽，继续堆全文同义词不会快速补齐门槛。

当前权威候选为 **1,096 = 119 approved + 845 pending + 132 rejected**；v3 manifest 为
**119 records / 91 independent groups**，held `cat|playing|indoors=16/50`。训练侧 pairs 变为
`cat×indoors=28/10`、`cat×playing=5/10`、`playing×indoors=5/10`；后两者各仍差 5，正式
target=50 下仍为 **0 eligible C1**，所以不启动训练。manifest SHA-256 为
`d0586b4cc3e454f58f3de7055dc4b8e90e098b109e80e841427f7c59ed7f6a0a`，coverage SHA-256 为
`3dd24a5e6f19230f524697e095a64e4a873279207846322013fa6a4feaeb3618`。

#### 非 held pair 优先队列与视觉发现负结果

为避免把 held `cat|playing|indoors` 本身误算成训练侧 pair 支持，新增只读
`pair-priority` 报告并把语义约束写死：补 `cat|playing` 的候选必须是 **cat + playing +
outdoor 且非 indoor**；补 `playing|indoors` 的候选必须是 **dog + playing + indoor 且非
outdoor**。它还要求动物实体词和动作证据同时存在，将演员“扮演”、影片、软件、会议等语境
降权，且只输出正分、provenance 完整、时长至少 2 秒并可播放的 pending。该报告只排序，绝不
自动标签或批准。

首次宽松版本暴露出演员、软件和电影会被 `cat`/`playing` 词面误排到前列；在任何新媒体
决策前修正为上述非-held约束后，当前 **834 条 pending 的严格 pair-priority 输出为 0 条**。
这不是说余下数据绝无可用片段，而是说 metadata 已不能提供足以自动优先的正证据，必须用
更昂贵的视觉发现，并仍由完整时间审核裁决。

随后对低置信 outdoor-cat 发现池的 7 条生成 16 帧全时长联系表：Bengal tricks 是室内人物
引导剪辑；三条 Ljubljana park 是坐、卧、理毛或走路；pigeon 是明确捕猎；Larry 是新闻
蒙太奇；moths 主要是静止观察/捕猎。全部拒绝，没有把 chase/hunt 偷换成 playing。另对原
优先结果中的 Fur Lickers、Mimi part 2、Gatos 原版/稳定版复核：前者和 Gatos 均为多猫，
Mimi part 2 是已批准同猫同房间同玩具事件的冗余，四条全部拒绝。两份批次报告分别为
**0 approved / 7 rejected** 和 **0 approved / 4 rejected**。

候选状态因而更新为 **1,096 = 119 approved + 834 pending + 143 rejected**；approved、v3
manifest、coverage 与哈希均不变，仍为 119 records / 91 groups、held 16/50、训练 pairs
28/5/5、0 eligible C1。对全部 triples 排序也没有更可行的替代目标：第二名
`dog|playing|outdoors` 只有 1 个 held group，其他候选的 held/pair 支持更低。因此不降低
50/10 门槛，也不为了更易成功改换一个证据更弱的三元组。

### 动作同义词分类审计：toy/fetch/chase 只能用于召回

2026-08-14 又用 Commons Action API 审计了不直接以 `play` 命名的动作树，并把每个真实视频的
page ID、标题和原媒体 SHA-1 与当前 1,096-row candidates 比对。这里采用严格语义边界：
`chasing/pouncing/jumping/chewing/carrying/fetching` 可能分别是捕猎、跳跃、进食、工作或训练，
toy/ball 也可能只是物体共现；只有审核窗口内确实持续表现娱乐性互动时，才可标为 `playing`。
同理，缺少 indoor category 不能反推 outdoor。

#### 猫：结构化 toy 类没有新的独立视频

- [`Cat toys`](https://commons.wikimedia.org/wiki/Category:Cat_toys) 当前为 87 个直属 files / 4 个
  subcats，API 实测 16 个视频容器：round-track toy 1 条、Finchen 4 条、Finchen+Lotti 11 条。
  16/16 均已在 candidates；后 15 条又属于两个已知 Ies/Radevormwald 系列，不能按 clip 计独立
  来源。round-track（page ID 22864110，SHA-1
  `b75cecad2f79a52703e9898e6f884f1a516d1d42`）也已 approved，不是新差集。
- [`Cats with toys`](https://commons.wikimedia.org/wiki/Category:Cats_with_toys) 为 4 files / 4
  subcats，直属实际视频为 0；其 `Cats and catnip toys`（11/0）、`Cats carrying toys in mouth`
  （1/0）和 `Cats and balls`（20/1）直属实际视频也均为 0。Commons 没有可直接采用的
  cats-chasing/pouncing/batting 精确视频类。
- [`Cats playing with their prey`](https://commons.wikimedia.org/wiki/Category:Cats_playing_with_their_prey)
  为 12 files / 1 subcat，只有 2 个真实视频。lizard（page ID 68960767，SHA-1
  `705009486b5da6bb2caac9da2016fcd5ee615e10`）和 `Cat play 02`（99201738，
  `7fd4bcaa0e3d62ba78ca3331c91e776663447e09`）都已在 candidates；前者有明确 outdoor category，
  后者已因 Commons 标题大小写近重复拒绝。捕食/活体猎物也不能自动改写成 toy play。
- [`Videos of jumping cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_jumping_cats) 为
  5 files / 0 subcats，五条标题与 candidates 的差集为 0：三个已按原义 approved 为 `jumping`
  （`Cat jumping backwards`、`Katzensprung in Zeitlupe`、`Salto - Cat Jump`），`Sammy springt`
  与 `Petit saut de chat` 仍 pending。即使后两条通过，也应先保留 `jumping`；只有画面另有明确
  玩具/追逐互动，才可能贡献 cat-playing pair。

#### 狗：强 play 同义类多为静态图，真实视频偏户外

官方树的当前计数与直属视频实测如下：`Dogs chewing` 57 files / 1 subcat、视频 0；其
`Puppies orally exploring` 7/0、视频 0；`Dogs carrying toys` 32/3、视频 0；`Dogs carrying
sticks` 58/0、视频 0；[`Dogs playing tug-of-war`](https://commons.wikimedia.org/wiki/Category:Dogs_playing_tug-of-war)
16/1、视频 0；[`Dogs with toys`](https://commons.wikimedia.org/wiki/Category:Dogs_with_toys)
49/12、视频 0；`Dogs catching toys` 10/0、视频 0。它们不能按页面的 file 数扩充视频集。

[`Dogs fetching`](https://commons.wikimedia.org/wiki/Category:Dogs_fetching) 为 106 files / 1
subcat，只有两条真实视频，且都已在 candidates：

| 文件 | page ID / SHA-1 | 当前差集与场景先验 |
|---|---|---|
| `Dog Fetches Hero2 Stick From River.webm` | 47052929 / `d25d1699ce6adfea6f7045fe0a694f5645b8e589` | pending；标题明确 river/outdoor，不补 `playing×indoors`。 |
| `Playingfetchwithadog - beach ... .ogv` | 149615031 / `515ac4b72282dd2d9a5828aa0c5e26132120fddb` | 已 rejected；beach/outdoor 且主体过小。 |

`Dogs with balls` 为 114 files / 3 subcats，实际视频只有同日、同两犬、同 grassy yard 的两条
Goldendoodle/Black Lab 视频（page IDs 30032853/30032854；SHA-1 `6e19ec10962101fd59ba0c00f9a7c34a935f8db5` /
`b2cc96a6c046b56452eb8e797239c8a748427f3f`）；两条都已在 candidates，且最多是一个 outdoor
series group。`Dogs retrieving`（11/2）及 `Dogs retrieving training dummies`（15/0）直属均无
视频，并且其官方组织语义偏猎物/训练，不应作为 play 标签来源。

因此，本轮同义分类的 **page-ID/title 新差集为 0**；没有新下载，也没有改变 119-record
manifest 或 coverage。现有 `Puppy playing.webm` 的官方 description “playing in house” 与已
approved 的泡泡狗仍是最强 indoor 正例；`Throw the Stick!` 已因户外且两秒窗口内 play/fetch
不稳定而 rejected。下一轮若继续人工审核，只建议依次看：① 两个 jumping pending，仅判断是否
另有明确 play；② river fetch 只用于未来 outdoor support；③ 不再扫描上述零视频的静态 toy/
tug/chew 类。它们都不是补齐当前两个 pair 各差 5 groups 的可靠路线。

可复现查询如下；`titles` 可用 `|` 一次查询多个类，文件枚举须跟随 `continue`：

```text
# 类的直属 file/subcat 计数
action=query&prop=categoryinfo&
titles=Category:Cat_toys|Category:Cats_playing_with_their_prey|
Category:Videos_of_jumping_cats|Category:Dogs_chewing|Category:Dogs_fetching|
Category:Dogs_playing_tug-of-war|Category:Dogs_with_toys&
format=json&formatversion=2

# 对每个类枚举并获得 page ID / MIME / SHA-1
action=query&generator=categorymembers&gcmtitle=Category:Dogs_fetching&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories&
iiprop=url|mime|size|sha1|extmetadata&cllimit=max&format=json&formatversion=2
```

MIME 门禁接受 `video/*`，并仅对扩展名为 `.ogv` 的 `application/ogg` 作视频例外；不能接受任意
OGG。差集先按 page ID/title，再按 SHA-1 去字节重复，最后仍要按 uploader、具名动物、拍摄
日期/地点、标题前缀与关键帧/音轨指纹建立传递 group。不同 SHA-1 不代表独立动物或拍摄事件。

### Internet Archive 可否作为下一来源：仅限逐 item 小规模补召回

2026-08-14 直接调用 Internet Archive 官方
[`advancedsearch.php`](https://archive.org/help/aboutsearch.htm) 与
[`metadata/{identifier}`](https://archive.org/developers/metadata-schema/index.html) 做了小样本审计。
官方搜索文档说明 advanced search 查的是 **item metadata**，排序分页最多返回 10,000 条，深分页
应改用 cursor-based scraping API。因此关键词命中不是视频片段标注：长片说明、节目单、游戏
“Let's Play”和正文里的 cat/dog 都会形成假阳性。

为展示噪声边界，以下是本次 API 的实时响应，不是训练样本计数：

| 查询 | `numFound` | 审计结论 |
|---|---:|---|
| `mediatype:movies AND (cat OR kitten) AND (play OR playing) AND (outdoors OR outside OR garden OR yard OR grass)` | 440 | 未限字段时混入长合集、动画、电影与游戏；不能称为 440 个候选。 |
| 同上再加 `licenseurl:*` | 127 | 只证明 item metadata 有 license URL，不证明语义或上传者权利。 |
| `mediatype:movies AND (dog OR puppy) AND (play OR playing) AND (indoors OR inside OR house OR room)` | 1,342 | 同样有历史片、会议、动画、人物节目等大量假阳性。 |
| 同上再加 `licenseurl:*` | 260 | 仍不是 260 个 dog-playing-indoor 视频。 |

更严格的许可限定标题查询只找到一个表面相关 item：
[`2011-cats-kitten-playing-outside-barn`](https://archive.org/details/2011-cats-kitten-playing-outside-barn)，
标题为 `Cats Kitten Playing Outside Barn (c. 2011)`，`mediatype=movies`，creator 为 Katie Madonna
Lee，uploader metadata 为同名账号邮箱，`licenseurl=CC BY-NC-ND 4.0`。该 item 内并非一个 clip，
而是多个同一家庭/猫群视频；其中同名 original M4V 为 13,431,731 bytes，MD5
`02c557ef3c6a9c0cf521841b07feb784`、SHA-1
`59f428c18f002d80273f2c2982ca779961a9a47a`，另有 IA 生成的 MP4 derivative（4,535,186
bytes；SHA-1 `c5ea98704f9830b591375a98660d21d0c43e7b5f`）。**推断**：它可能补 outdoor cat，但标题是复数
cats/kittens、整个 item 共享拍摄来源，且 ND 条款对裁成 2 秒训练片段与模型派生物是否获准并不
清晰；在取得权利人许可前不采用。

另一个高词面命中
[`funny_or_die_video_08fb1ecea6`](https://archive.org/details/funny_or_die_video_08fb1ecea6)
确有 original MP4（20,429,797 bytes；SHA-1
`adb8e75c0b908b286ca961b56daa7cc7d950c8a1`），但 item 的 `licenseurl` 缺失且内容是商业喜剧短片；
可下载不等于可训练，直接排除。dog 侧没有发现同时具备明确标题、indoor scene 与可安全解释
许可的强候选；例如 Osceola 宠物节目虽然有 CC BY-NC-ND 3.0 和 original MOV，但说明只是称幼犬
“playful/house-broken”，并不证明片中 indoor playing，也包含人物/节目制作权，应排除。

#### 文件级 provenance 契约

对任何将来候选先查询：

```text
GET https://archive.org/metadata/{identifier}
```

必须核验 `metadata.identifier/mediatype/title/description/subject/creator/uploader/licenseurl`，再对
`files[]` 保存 `name/source/format/md5/sha1/mtime/size`。官方 schema 明确 `mediatype=movies` 只
表示 item 的主要内容是视频并触发相应展示/处理，不保证 item 中每个文件都是目标视频；
`licenseurl` 是 uploader 可编辑、非必填的“所选许可 URL”；`uploader` 是上传账号邮箱，不能
自动等同作者或权利人。

本仓库的候选 JSON 不持久化 uploader 邮箱，也不保存包含邮箱的完整 metadata 摘要，避免把公开
邮箱或其裸哈希变成可链接的个人标识。自动 source group 先按必须公开用于署名的 creator 保守
合并；uploader 只在实时 metadata 审计中验证存在。任何候选在批准前仍必须由人工把同上传者、
同动物、同拍摄系列或事件写入稳定 duplicate group；未完成该步骤不能进入训练 manifest。

实测 metadata 中 `files[].source=original` 对应上传原文件，`source=derivative` 对应 IA 生成的
MP4/OGV 等派生对象；derivative 不是新的版权授权，也不能冒充 original。下载地址是：

```text
https://archive.org/download/{identifier}/{percent-encoded-file-name}
```

下载前重新锁定 item metadata，下载后同时验证所选文件的 size、MD5 和 SHA-1；若用 derivative
做审核或训练，要同时保存 original 与 derivative 两套 name/hash/source。一个 item 可有多个
original 视频，必须按文件逐个审核，并把同 item、同 creator/uploader、同拍摄系列全部并入一个
传递 group，不能把十个文件算十个独立来源。

#### 权利结论与采用边界

`licenseurl` 缺失绝不表示公有领域；存在也只是 item metadata 声明，Internet Archive 托管和
可下载本身不授予训练、裁切、再发布或商业使用权。必须逐 item 验证上传者的权利链，并分别
处理 BY 署名、SA、NC 和 ND；ND 或权利链不明的 item 当前 fail-closed。媒体还可能含第三方音乐、
电视/艺术品、商标以及可识别人物；自由版权许可不自动清除隐私、肖像、表演或场地权利。

因此 IA **可以作为人工小规模补召回 API，但不作为新的批量数据源**。只有明确许可允许裁切和
模型训练、creator/source 可追踪、存在带 MD5/SHA-1 的 original 视频、clip-level 三因子审核
通过且 provenance/group 完整的 item 才能进入 pending→approved 流程。本轮没有安全可采用的
新样本，119-record manifest、coverage 与“0 eligible C1 / 不启动正式训练”边界均不改变。

本仓库随后把上述边界固化为 `code/video_dit/prepare_archive.py`：输入必须是人工列出的精确
identifier、`cat|dog` subject 和查询意图；程序只接受 `mediatype=movies`、非邮箱 creator、
兼容的 CC BY/BY-SA/CC0/Public Domain URL，以及**唯一一个**同时带正 size/mtime、MD5 和 SHA-1
的 original 视频。网络错误会中止，元数据不合格则写入结构化 rejection；通过者也只写成
`pending + review_required`，不会直接进入 Commons manifest 或 coverage。

2026-08-14 用固定 spec 实跑两条最相关 item，spec SHA-256 为
`c796a795406dfde67189dd713449348cc7afb1abc1cf62c9fc6ad0f11b200a32`：结果为 0 pending、
2 rejected；barn cats 因 BY-NC-ND 被许可门拒绝，`Playingwithmydogs` 的旧式 public-domain
URL 可被锚定规则识别，但 creator 缺失，因此被归属门拒绝。空候选 JSONL 的 SHA-256 为
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。因此这不是“搜索没找到
所以当作可用”，而是可复算地证明这两条不能进入后续媒体下载；训练 manifest 和 coverage
仍保持不变。

可复现查询示例：

```text
GET https://archive.org/advancedsearch.php?
  q=mediatype%3Amovies%20AND%20licenseurl%3A*%20AND%20
    %28cat%20OR%20kitten%29%20AND%20%28play%20OR%20playing%29%20AND%20
    %28outdoors%20OR%20outside%20OR%20garden%20OR%20yard%20OR%20grass%29&
  fl[]=identifier&fl[]=title&fl[]=description&fl[]=creator&fl[]=uploader&
  fl[]=licenseurl&fl[]=mediatype&rows=50&page=1&output=json

# dog 侧将三组语义词替换为：
# (dog OR puppy) AND (play OR playing) AND (indoors OR inside OR house OR room)
```

### 下一条可审计路线：Commons 全文索引差集，而不是新的大数据集

2026-08-14 用 Commons 官方 Action API 的 File namespace 全文搜索按页面创建时间倒序扫描，并
与当前 1,096 条 candidates 做标题连接。该路线比继续递归静态分类更可能发现近期上传，但仍是
metadata recall，不是标签器；`timestamp` 是 Commons 页面/索引时间，也不是拍摄时间。

当前精确响应为：`cat playing filetype:video` 221 hits，其中 118 个标题不在 candidates；
`dog playing filetype:video` 41 hits，其中 12 个标题不在 candidates；`puppy playing
filetype:video` 9 hits，其中 3 个标题不在 candidates；`playing indoors filetype:video` 仅 2
hits。差集数仍不能称为动物视频数：cat 查询的新标题大多是电影、政治讲话、人物演奏、游戏或
动画；dog 查询的新标题大多是军犬示范、历史片和动物无关正文。`playing in house` 的索引命中
更达 9,406，因 House/playing 多义而不再采用。

可复现请求如下；必须分页，并把 API page ID 作为主键、原媒体 SHA-1 作为内容键：

```text
action=query&list=search&srnamespace=6&
srsearch=puppy%20playing%20filetype%3Avideo&srlimit=500&
srinfo=totalhits&srprop=timestamp|snippet&srsort=create_timestamp_desc&
format=json&formatversion=2

action=query&pageids=<page-id-list>&prop=imageinfo|categories&
iiprop=mime|sha1|url|size|extmetadata&cllimit=max&format=json&formatversion=2
```

#### `playing × indoors`：唯一值得优先复核的新差集

[`Een mooie collectie jonge witte poedels-512309.ogv`](https://commons.wikimedia.org/wiki/File:Een_mooie_collectie_jonge_witte_poedels-512309.ogv)
（page ID 35108948）是本轮最强新方向。官方 description 明确为幼年白色贵宾犬在围栏后、并在
**kitchen floor** 上和多个物体玩耍；原媒体 SHA-1 为
`09e05106c5c1e92bd9fb0bdc7a3e912bf5d6b4f2`，`application/ogg`、5,755,150 bytes，许可为 CC
BY-SA 3.0 NL，署名/管理来源为 Polygoon-Profilti 与 Nederlands Instituut voor Beeld en Geluid。
检索当时它不在 1,096 条 candidates 的 title 集合中，是可合法逐文件审核的 metadata hit；
下段记录随后完成的导入和最终人工决定。

该 page-ID 命中随后通过新增的 `prepare_commons.py exact` 精确导入：spec 固定完整 `File:`
title、subject 和查询意图；API 回传 title、既有 subject/query contract、重复 spec 或输入输出
路径任一漂移都会 fail-closed。新行先保持 `review_required/pending`，再用同一批次 sidecar
锁定 page revision、原媒体 SHA-1/timestamp、CC BY-SA 3.0 NL 许可、署名和官方 240p
derivative。全长 2 fps 联系表确认它确实在厨房地板拍摄，但始终出现多只幼犬且有新闻剪辑，
没有稳定的单一主要犬主体两秒窗口，因此拒绝，不能贡献 `playing×indoors`。

同一轮还把现有 pending 中三个最强 dog-play 标题做了视觉发现，而不是继续依赖 metadata：

- `Labrador retriever with her 4 puppies`：母犬、四只幼犬和人物同时出现，拒绝；
- `Puppiesplaying-tokyoarea`：全长和 1 fps 密集检查均是两只幼犬互动，拒绝；
- `Ashley. Alaskan Malamute playing in the snow`：单犬雪地，但五个密集两秒窗口表现为牵引绳
  下奔跑/嗅闻且人物持续参与，无法把 playing 与 running/walking 清晰分开，拒绝。

四条审核已绑定 candidate snapshot `98693bb651cea6102d5df9f82abe5495c3a81f0a85709a6a06a435c887781da6`
和逐条原媒体 SHA-1。当前池为 **1,097 = 119 approved + 831 pending + 147 rejected**；v3
manifest、91 个独立组、held 16/50、训练 pairs 28/5/5 和 0 eligible C1 均不变。这是视觉门禁
排除高似然假阳性的负结果，不把标题里的 “playing” 当标签，也不据此启动训练。

审核前的风险已经明确：description 使用复数 puppies，新闻片可能有多段/剪辑。已经完成的
全长审核确认了这两项风险，因此不再分段或批准。即使当时存在合格窗口，也只能贡献同一
Polygoon/Open Beelden 拍摄事件的一个 group。

两个容易误判的新差集不建议导入：`Hoela-hond-517783.ogv`（35089168；SHA-1
`95d719222165578b1bfc4247597cea2852c0d0ee`）的官方描述是女孩玩呼啦圈、狗观看后拾起自行车圈，
地点为农场，主体/动作/场景均不满足 indoor pair；`Landbouwpaard kan koorddansen-520075.ogv`
（35089141；`796a92e39ca6d2f88f8d86a6b8054638d9d492a8`）是马走钢丝并与狗踢球，既非单狗主体也没有
indoor 证据。军犬示范 `31 SFS K-9 Demo` 虽标 public domain，但官方描述是工作犬训练，且带
`Milimedia files ... review needed`，不是 play 数据。

此前 MediaSearch 看似较新的两条实际已在 candidates，不能算新增：两幼犬 Tokyo 文件
`Puppiesplaying-tokyoarea-jan7-2020`（85691731；SHA-1
`8fa5a1a067a488b821f74db02c96c88258a0739e`；CC BY 4.0）仍是 pending 且明确多狗；泡泡狗
（15446135；`b6c1fd97f257c36d04941cdaeed6e3d34d23fe8f`）已经 approved。后者也说明不能仅按 page
ID 的文本形式查本地 JSON；须用规范化 title/description URL 与 provenance page ID 联合连接。

#### `cat × playing` 非 indoor：近期索引没有新的强正例

全文搜索的结构化复核仍只回到已审核的 lizard/Lotti outdoor。近期上传
`2023-12-16 at Kitten Academy-0nNV51d0vZk`（197214178；SHA-1
`4b0c3db023bad0848cd6b692900ada7a356d5ccc`；CC BY 3.0）也已在 candidates；官方描述是多只寄养
猫与照护者玩耍/拥抱，scene 未说明，并带 `License review needed (video)`，不能作为 outdoor 或
许可已通过的样本。`cat playing grass` 的两条命中分别是已拒绝的多猫 Somaliland 与历史家庭
影片；`cat playing garden` 除已知 lizard 外是历史电影和兽医病例。因此本轮没有新的高精度
cat-playing-outdoor page ID，继续全文扩词的边际收益低。

审核前唯一推荐动作是把 page ID 35108948 作为 **pending metadata hit** 导入 provenance
sidecar、核验许可页与原媒体版本，再做全时长/密集窗口人工审核；该动作现已完成并以拒绝结束。不要导入
上述两个历史新闻片假阳性，不要重复导入已在 candidates 的 Tokyo/泡泡狗/Kitten Academy，亦
不要把 118/12/3 个词面差集批量写入候选。本阶段没有批准 clip、没有改动 119-record manifest 或
coverage；当前仍为 0 eligible C1，正式训练继续不启动。

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
| G1 | Commons-1K | 逐文件 provenance 完整，source/pHash split 无泄漏；prompt 与输出主体/动作相关 | coverage 或 caption alignment 不足就继续审核/补搜，不训练 6K |
| G2 | Commons 扩展集（上限 6K，按 G1 coverage 决定） | ID-test 明显优于无条件基线；日志与 manifest 完整 | 普通泛化失败则不做组合宣称 |
| G3 | Commons C1 子集 | 每个留出 triple ≥50 独立 source groups；三因子正确率显著高于打乱 prompt，最近邻非复制 | 覆盖不足就缩小矩阵；仅随机 split 好看只报告内插 |
| G4 | C2 pair holdout | 独立报告更强的二元留出结果和失败案例 | 不用扩大数据掩盖 split 缺陷 |

最终目标不是生成一个“看起来像猫”的 cherry-picked MP4，而是能回答：模型是否在没有见过某个 `subject × action × scene` 三元组、也没有见过同源和近重复视频的情况下，仍能按照文本同时满足三个条件。

## 2026-08-13 执行记录

- 已取得并解析 VATEX 官方 `vatex_training_v1.0.json`：25,991 records，SHA-256
  `9a3b5f08e354d9543ef4f1ab004f9db9bc4e5da49d9692f0c8c6aa3bef9751c4`。
  caption 粗筛得到 cat 1,415、dog 600、cat+snow 25、dog+snow 71；这些数字只是
  metadata 候选量，不能代替视频级人工确认。
- 当前机器没有 Hugging Face 登录令牌；FineVideo 需显式接受 gated 条款，暂不使用。
  VideoUFO 是匿名可访问的；固定 revision 的完整 CSV 与最小尾部视频 tar 已在 2026-08-14
  下载并通过 SHA-256 校验，但 resolved 视频尚未经过逐条人工审核，因此未写进训练 manifest。
- 第一轮 Wikimedia Commons 搜索与分类树检索曾取得 799 个候选。已通过本地审核 UI 播放
  官方低码率 derivative，并用多帧时间联系表辅助检查；101 条自然视频已人工标为
  `approved`，统一转为 96×96、16 帧、2 秒 H.264 clip。另有 645 条 pending、53 条
  rejected，不以标题或搜索命中直接代替视频审核。approved 为 cat 59、dog 41、person 1；
  这条 person 是 `walking|beach`，其 3--5 秒单主体区间经完整播放和逐秒帧核验后进入
  manifest。person 仍有 105 条 pending，不能计入主体覆盖。
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
  priority 功能落地前的候选快照为 799。该快照为 101 approved、645 pending、53 rejected；approved
  为 cat 59、dog 41、person 1，person pending 105。分类遍历现逐 batch 原子保存 category
  queue、continuation、未处理 titles 与
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
- 为减少无效人工播放，新增只读的 `commons_review.py priority` 报告与 UI
  `--priority-order`。它只依据标题/描述把 `solo/runner/hiker` 等线索前移，把 crowd、
  first-person、timelapse、vlog 等明显风险后移；不修改标签或审核状态。按此流程前的顺序
  又检查 6 条人物候选，全部因人群、第一视角无可见人物、车站延时摄影或旅行 vlog 拒绝，
  因而 approved 仍为 101、C1 coverage 不变。105 条 person pending 中有 104 条通过时长与
  官方预览可播放门；其余 1 条不会进入 priority。优先队列第一条已在本地 UI 打开，但 derivative
  网络载入未完成，保持 pending 而非依据标题批准。
- priority 队列随后继续审核 6 条 person 候选：`Forest walking and outdoor cooking
  slovenia` 在 2598--2600 秒经 4 fps 联系表确认只有一名清晰人物持续走入森林，批准为
  `person|walking|forest`；另外 5 条因第一视角无独立人物、ASMR 主播/风景剪辑、多人市场、
  越野跑第一视角或动作并非 walking 而拒绝。当前状态因此为 102 approved、639 pending、
  58 rejected，person 为 2/99/16；98 条 person pending 可播放。新片已在下载前重验原媒体
  URL/SHA-1/timestamp，规范化为 H.264 96×96、8 fps、16 帧、2 秒，clip SHA-256 为
  `9fdb8f40357b21f071a6d596cfc4077c912df04adb262eab7ebb0a165a753030`，pHash 为
  `7d4d98fcc4989585`。v2 manifest 现为 102 records / 102 independent groups，SHA-256
  `90b0e4951a5b57644cc73ce9aa6cb03396551edfbd287d1192431120b3304b0e`；新增三元组仍只有
  1 个独立来源，0 eligible C1 triples，不能据此启动训练。
- 随后运行 8 个更窄的 `woman/man × walking/running × forest/snow/field` 全文查询，只有
  `woman walking forest` 新增 1 条；其余 7 个为 0。唯一命中是 43 分钟历史多州旅行合集，
  20 帧全时长联系表显示驾驶 POV、建筑、多人和互不连续场景，无法证明目标片段，故拒绝。
  候选池现为 800 = 102 approved + 639 pending + 59 rejected；person 为 2/99/17。
  这轮证据否定“继续堆窄全文查询即可扩到 Commons-1K”的假设，后续需使用更有结构的
  Commons 分类/人工策展来源，或在公开缺口的前提下缩小 C1 组合矩阵。
- 官方分类审计后实际导入 `Videos of hiking` 的 13 条直属/一级子类视频，以及
  `Videos of people jogging` 的直属视频。过程中发现 Commons 将 `Running form.ogv` 与
  `Spring shoes jogger Riverside Park jeh.ogv` 报为 `application/ogg`，旧 importer 因只认
  `video/*` 静默丢弃；修复后仅允许 `.ogv + application/ogg`，普通 `.ogg` 仍拒绝。jogging
  分类页 4 个成员中，3 个通过 MIME/逐文件许可门进入候选，Clinton 公有领域视频因缺
  `LicenseUrl` 保持 fail-closed。新审核的 hiking 3 条分别为无人物第一视角、瀑布末段多人、
  大型朝圣人群；jogging 3 条分别为主体过小、只拍腿部、先后出现多名跑者，全部拒绝。
  当前池为 816 = 102 approved + 649 pending + 65 rejected；person 为 2/109/23，仍未增加
  C1 coverage。priority/UI 现在通过 `--provenance` 只读合并 sidecar，能审核新导入项而不
  改写候选快照；108 条 person pending 通过时长和官方 preview 门，另 1 条不绕过门禁。
- 为避免 enrichment 与运行中的审核 UI 同时原子替换候选 JSONL，新增只读候选、逐 batch
  原子写独立 sidecar 的 `commons_review.py provenance`。最新真实运行已补齐 816/816 条：每条
  固化 Commons 描述页 revision、原媒体 URL/SHA-1/timestamp、时长和审核 derivative URL。
  下载前会再次查询并比对原媒体版本，发生覆盖就 fail-closed；sidecar 与候选 title 必须
  一一对应。最初 101 个 approved 已重新下载到**新的** provenance-complete 数据树，全部
  使用审核 derivative，并分别记录实际下载 URL/kind/SHA-256 与 UTC 时间；随后新增的
  forest-person clip 也走同一门禁，当前 102/102 个 clip 文件存在且规范化 SHA-256 实算零
  mismatch。第一版迁移曾发现许可快照应与 page revision 同次获取，而不能沿用更早候选字段；
  v2 sidecar 因此在同一个 API 响应中锁定 page revision、许可 URL/署名/描述页和原媒体版本。
  当前 v2 manifest SHA-256 为
  `90b0e4951a5b57644cc73ce9aa6cb03396551edfbd287d1192431120b3304b0e`，字段零缺失。原 101 条
  的许可分布为 CC BY-SA 4.0 43、CC BY
  3.0 18、CC BY 4.0 11、CC0 10、CC BY 2.0 8、CC BY-SA 3.0 6、CC BY-SA 2.0 4、
  CC BY-SA 3.0 es 1。旧 manifest 不与
  新契约混写。新的 `prepare_real_data.py coverage` 复用 split validator 的传递
  source/duplicate/pHash 分组；报告为 102 records / 102 independent groups / 0 eligible
  C1 triples。最接近门槛的是 `cat|sitting|indoors=13`（距 50 还差 37），其次
  `cat|playing|indoors=10`（差 40）、`cat|walking|outdoors=8`（差 42）。这证明当前不可
  构造 C1；下一步应按 deficit 排序审核/补搜，而不是启动训练。
- VideoUFO metadata-first 路径已经落地：全量 1,091,712 行 CSV 扫描得到 4,234 条
  bounded pending 候选与完整 3×3×4 coverage；最小 `VideoUFO_200.tar` 的 197 个 MP4
  已证明 member basename 与 CSV ID 可精确关联，并命中 1 个候选。当前猫组合稀疏，只有
  `cat|walking|grass=54` 达到 50 个 metadata 命中；经视频审核、pHash 合并后只会更少。
  三分片 pilot 随后证明随机扩片无法形成平衡动物覆盖；这一历史下一步已被否决。当前不再
  扩大 VideoUFO 分片，只推进 Commons-1K 审核/补搜；同样不能提前构造 C1 或启动正式训练。
- coverage deficit 驱动的下一轮把主候选定为 `cat|playing|indoors`。先对原有标题/描述高置信
  池逐片生成全时长 10--12 帧联系表，再对候选 2 秒窗口生成 8 帧密集联系表；首批批准 8 条、
  拒绝 2 条。随后导入官方 `Videos of playing cats` 44-file 精确视频类，与旧池差集只有 5 条
  OGV；人工审核后仅 `Cat Playing with String Toy 1 2014-02-04` 满足单猫、室内、明确 playing，
  其余 4 条因静止/多猫/剪辑拒绝。再检查精确类中 5 条非系列 pending，全部因多猫或不稳定片段
  拒绝。原始 approved 因此从 102 增到 111；当前候选池为 1,034 = 111 approved + 832 pending
  + 91 rejected，pending 仍不计训练或覆盖。
- 为补差额，8 个 `cat/kitten × toy/ball/string/laser/box/indoors` 同义全文查询又带来 213 条
  provenance-complete pending，但其中大量是 Cat 软件、人物名、乐队、医学、静态猫或动画。
  对 15 条视觉上最可能的自然猫视频做全时长联系表后，仍无新增合格 playing 样本；这些检索
  证明宽全文查询只能作低精度召回，不能自动标签。`Sitting cats` 的一手对照审计也显示其首
  500 个直属文件为 493 JPEG + 6 PNG + 1 个已知 WebM，故不把主目标切回 sitting。
- 人工 series 审计补齐 `Moomin1/2`、同日 Siamese Maine Coon、Self-Cleaning、Cat Play、
  German shepherd food、同日狗游泳等 7 组，并为新审核的 Kahli/Finchen/Lotti/Mimi 系列写入
  `duplicate_group`。进一步要求每个 approved 样本都必须有显式组：已知同一动物/系列共享
  series group，其余不能证明独立的样本按 uploader 保守合并。最终把 111 条记录从表面 111
  修正为 **79 个独立 source/series/uploader/pHash groups**；`cat|playing|indoors` 从 18 条记录
  修正为 **14 个独立 groups**，距 50 仍差 36；`cat|sitting|indoors` 为 11。正式 target=50 下
  仍为 0 eligible C1，故不启动训练。
- 下载/恢复契约同步加强：分类搜索的 `category_traversal_id` 不再误传进媒体 dataclass；审核
  后补写的 duplicate group 会同步进既有 manifest；被改为 rejected 的样本会从 manifest
  移除；prompt/三因子/片段起点/frames/fps/size 与既有 clip 不一致时 fail-closed，要求重建
  新数据树。该阶段的 **v3 历史快照**从锁定 derivative 重新规范化全部 111 条，逐条记录
  `clip_start_seconds/clip_frames/clip_fps/clip_size`；111/111 均为 H.264 96×96、8 fps、16 帧，
  clip SHA/provenance 零错误。manifest SHA-256 为
  `74cb501a4b1c18d877614f10e917a6f7a828b399ab663a2afe85555245550575`，coverage SHA-256 为
  `4122894dbfc7fc2303635196c8ca8b2634325f5ff2e7cf306746081719e0df85`。
- 随后按 holdout 后 pair deficit 而不是 held 数量继续执行。官方 `Playing dogs` 直属 107 个
  files 中只有 2 个真实视频：`Two poodles playing` 因多狗且户外拒绝；单狗室内追肥皂泡
  经全时长联系表和多个 2 秒窗口复核后批准。再导入 `Videos of dogs` 的 141 个直属视频，
  与既有候选差集新增 57 条；高置信标题中，office 多狗、户外球类/飞盘与主体过小者拒绝，
  两条不同上传者的单狗室内 mirror-play 视频通过完整时间复核。当前权威候选为
  **1,093 = 114 approved + 884 pending + 95 rejected**；v3 manifest 为 114 records / 82
  independent groups。拟留出 `cat|playing|indoors` 仍为 14/50；训练侧 pair 为
  `cat×indoors=24/10`、`cat×playing=3/10`、`playing×indoors=4/10`；这里排除了 held
  group 的全部成员，而不只是同名 triple。当前还有 held groups 混入 kneading/scratching/sitting
  三个普通 triples，正式 split 会拒绝，须在冻结 split 前消解。coverage CLI 现在直接
  输出每个潜在 held triple 的 atom/pair support、deficit 与 mixed-group 列表，并预注册 pair 下限 10，所以
  当前仍是 0 eligible C1。最新 manifest SHA-256 为
  `7e44ef027aeb4e6f91c2e919221b3ef54f937f3087aa0eb8d05199dbef66f95f`，coverage SHA-256 为
  `ee314cef84ff7f44ca39407722f38b22168b1731ce2698d3d29d419e46b5f797`。
- mixed-group 身份复核没有按文件盲拆。对 FASTILY 六条和 Nesnad 两条 approved clip 的规范化
  8 帧联系表逐一比较毛色、毛长、脸部花纹、体型与拍摄年份：橘色长毛、灰白幼猫、长毛虎斑、
  短毛虎斑等明显不同个体拆为具名 identity groups；疑似同一灰白幼猫的 straw/stick 两条仍
  保持同组。由此 mixed-held blocker 消失，但这只修正此前 uploader 级保守下界，不代表新增
  媒体。随后完整审核 Lotti outdoor 四条：两条 `v` 视频持续追逐人手中的树枝，批准为同一
  `radevormwald-lotti-outdoor` series，只贡献 1 个独立 group；另两条只是躺/坐在草地，拒绝。
  当前权威状态为 **1,093 = 116 approved + 880 pending + 97 rejected**；manifest 为 116
  records / 88 independent groups，held `cat|playing|indoors=15/50`，训练 pairs 为
  `cat×indoors=28/10`、`cat×playing=4/10`、`playing×indoors=4/10`，0 eligible C1。
  manifest SHA-256 为 `46bbe9eeee40bb5780365eb05929ef4ae78e5cd5eae94efa369f95a281fe8450`，
  coverage SHA-256 为 `707446e9a116d9be789735cc0c4c6c51384b75c2901cd3c5b0da2a82b2b9a5ea`。

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
