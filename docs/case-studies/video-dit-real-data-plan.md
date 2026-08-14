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

24 GB Mac 的下一步保持小而可审计：先把现有 127 approved 扩到 1K，优先补 `cat|playing|indoors` 及其训练侧 pair support；每条分别保存 Commons 描述页 revision、原媒体 timestamp/SHA-1、原文件 URL、作者、license URL、下载时间、实际下载对象（原文件或审核 derivative）的 URL/type/SHA-256、规范化 clip SHA-256、片段起点、预处理参数、人工标签和 source/duplicate group。只允许 `approved` 且这些字段完整的 clip 进入 manifest。1K 达标后再按 coverage table 决定是否扩到 6K；达不到每个 C1 triple 50 个独立 source groups，或训练侧任一 pair 达不到预注册的 10 个独立 groups，就缩小组合矩阵并公开缺口，不用来源不明的视频补数。

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

从现有 127 个 approved clips 出发，优先审核/补搜 `cat|playing|indoors` 及稀疏训练侧组合；只有逐文件许可、署名、来源、原始/裁切 SHA-256、人工三因子标签和 source/duplicate group 完整的 `approved` 样本才能进入 manifest。达到约 1K 后重新计算 source-group/pHash 闭包下的三元组覆盖：每个拟留出 C1 triple 至少 50 个独立 groups，且其每个训练侧 pair 至少 10 个独立 groups，才进入正式训练；否则缩小组合矩阵并公开缺口。当前 1,135 个候选中的 800 个仍为 pending，不能计入训练规模或覆盖。

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

### 视觉召回：冻结 CLIP poster 只决定看片顺序

标题/描述检索连续漏掉或误排真实动作后，本阶段增加了一个严格只读的视觉漏斗。权重直接来自
仓库已固定 commit 的 OpenAI CLIP 包所声明的官方 ViT-B/32 Azure URL；第一次不完整下载虽可被
后续错误拼接成可加载 JIT，但 SHA 不匹配，已丢弃。最终 checkpoint 为 353,976,522 bytes，
SHA-256 精确为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`，且
`torch.jit.load` 成功。不存在用随机权重或相邻模型替代的回退。

`visual_priority.py` 对当时 135 条 dog pending 的当前 Commons 320px poster 编码。每张 poster
下载前后都重验原媒体 SHA-1/timestamp，缓存键包含 title、媒体 SHA-1 和 poster URL；报告还绑定
候选快照 SHA `98693bb651cea6102d5df9f82abe5495c3a81f0a85709a6a06a435c887781da6`、
provenance SHA `8f922388f614b13f4fb13d926e709d637da858277352ba575de9fddd2030e38d`、
checkpoint SHA、全部正负 prompts 和每张 poster SHA。正 prompt 聚焦单犬室内与玩具互动；负
prompt 显式覆盖多犬、户外、人物主导、静止/睡觉和非真实动物。排序分数是正 prompt 平均 cosine
减最相似负 prompt cosine；feature 非有限或零向量会 fail-closed。Commons 429 后使用可识别
User-Agent 和 1.2 秒请求间隔，从原子缓存继续，没有绕过服务端限制。

这不是动作分类器：poster 无法证明时间、playing 或单主体连续性，报告明确
`training_ready=false`。实际漏斗为 **135 pending posters → top 24 人工看图 → 5 个完整官方
derivative 的 16 帧全长审核 → 0 approved**：

- rank 1 是狗观看两个实验屏幕，不是 playing；
- rank 2 是人在场的指令取物实验，不是娱乐性互动；
- rank 6/11/13 是同一 FASTILY Yorkshire Terrier 系列，分别为抚摸、近景静止和室内嗅闻，
  都没有稳定 playing 窗口；即使通过也不能算三个独立来源。

其余 top 24 的官方说明/画面已明确为医学步态、药物中毒、饮食饮水、吠叫训练、多犬活动或
户外任务，因此没有扩大完整视频下载。五条最终审核绑定当前 candidates SHA
`5c6b6223c6dcc9735db03ffbef98b7227d6d844819b76bb2769f81ad3dc52c16` 和逐条原媒体
SHA-1；状态更新为 **1,097 = 119 approved + 826 pending + 152 rejected**。由于 0 approved，
v3 manifest 与 coverage 仍是 119 records / 91 groups、held 16/50、训练 pairs 28/5/5、
0 eligible C1；不能启动训练。该结果也限定下一步：单 poster CLIP 已把“室内单犬”实验视频排到
前列，却不能检出 play 的时序语义；后续若继续视觉召回，应使用多帧/短视频 embedding 或人工
策展的新来源，而不是调 prompt 后反复查看同一 130 条 pending。

### 24 GB Apple Silicon 的多帧/短视频排序模型选择

2026-08-14 按官方论文、仓库与模型卡审计后，选择“两级漏斗”：全库继续复用已锁定的
OpenAI CLIP ViT-B/32，但改为 8--16 帧分数聚合；只对前 10--20% pending 使用 Microsoft
X-CLIP base patch32 的 8 帧模型复排。两级分数仍只决定人工看片顺序，不自动改写
subject/action/scene，不进入 manifest 或 coverage。

| 模型 | 权重、许可与输入 | 是否有时序建模 | 24 GB 本机判断 | 决策 |
|---|---|---|---|---|
| [OpenAI CLIP ViT-B/32](https://github.com/openai/CLIP)（[论文](https://arxiv.org/abs/2103.00020)，[模型卡](https://github.com/openai/CLIP/blob/main/model-card.md)） | 官方 repo MIT；`clip.load("ViT-B/32")` 输入单张 224 图。现有权重文件 353,976,522 bytes，SHA-256 已锁为 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`；还须保存官方 repo commit、采样时刻、prompts 与聚合规则。 | **没有**。逐帧编码后 mean/max/top-quartile 都是本项目工程聚合，不是 OpenAI 的视频模型。 | 约 151M 参数级；FP32 权重约 0.6 GB，逐小 batch 编码 8--16 帧在 24 GB unified memory 有充分余量。官方 README 只给 CPU/CUDA 用法，MPS 算子、吞吐和数值一致性必须本机 benchmark，失败时用 CPU。 | 全量第一阶段；复用现有已核验权重和 poster 管线，增加均匀多帧。 |
| [Microsoft X-CLIP base patch32](https://huggingface.co/microsoft/xclip-base-patch32)（固定 revision `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`；[论文](https://arxiv.org/abs/2208.02816)） | 官方 Microsoft HF 组织仓库非 gated、标 MIT；固定 config 为 224 输入、8 frames。`model.safetensors` 为 786,414,772 bytes，LFS 与本地 SHA-256 均为 `abf286e8cdd0612761c3e42d3a55eca998382dfa67a04a0f3fdcdfa4f150cdbb`。只保存 safetensors 与配置/tokenizer，不同时保存等价 `.bin`。模型 revision、Transformers、PyTorch 与 NumPy 版本全部锁定。 | **有**。论文使用跨帧通信和视频级时序聚合做视频文本对比/zero-shot；比独立帧 CLIP 更贴合 playing，但不能据架构推定在本数据上准确。 | 196,585,729 参数；本机 CPU 单条 8×224 synthetic pilot 完整 forward 为约 0.12 秒（不含解码），证明可运行但不代表真实批量吞吐。MPS 仍须 10 片 benchmark。 | 只复排 CLIP top 10--20%，避免对 800+ pending 全量承担更重解码和时序计算。 |
| [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)（[论文](https://arxiv.org/abs/2104.08860)） | 官方 repo 可锁 commit，论文比较 mean pooling、LSTM/Transformer 与 tight interaction；但官方 release/Google Drive 预训练链接当前失效，官方 issue [#120](https://github.com/ArrowLuo/CLIP4Clip/issues/120) 也记录 404。 | 有视频帧聚合版本。 | 旧 CUDA 导向依赖，官方未承诺 Apple MPS；更关键是无法从一手来源取得并锁定匹配 checkpoint。 | 不采用第三方镜像；权重不可审计即 fail-closed。 |
| [VideoMAE](https://github.com/MCG-NJU/VideoMAE)（[论文](https://arxiv.org/abs/2203.12602)，[HF 文档](https://huggingface.co/docs/transformers/model_doc/videomae)） | 官方代码主体 CC BY-NC 4.0；HF Kinetics base revision 可锁为 `488eb9a0565f257b32866000305c8178965eb9f6`。典型输入 16×224，base 约 86M 参数级。 | 有真正 tube/time 视频建模。 | 单视频推理内存可行是合理工程推断，但多 clip/crop 会增加成本；同样要验证 MPS。 | 不用于本任务文本排序：它没有文本编码器，Kinetics logits 只是固定动作类，不能计算任意 prompt 的 video-text similarity。 |

#### 选定漏斗的冻结协议

第一阶段每个候选在同一已锁定 derivative 上均匀取 8 帧；若短动作召回不足，再预注册为 16 帧，
不能看完结果后逐样本改变采样。每帧 CLIP feature 先 L2 normalize，分别保存 frame-level cosine、
mean、top-25% mean 和 max；排序前把 prompts 拆成 subject（单只 cat/dog）、action（持续玩玩具/
互动，而非 fetching/实验指令/静止）与 scene（room/home/indoor 对 outdoor/grass/snow）三组。
mean 会稀释短动作、max 会放大偶发误判，因此不得只报最有利的一个聚合值。

第二阶段对固定 top 10--20% 取同一 2 秒候选窗口内 8 个均匀帧，用 X-CLIP zero-shot 同一组正负
文本复排。报告必须绑定：输入 candidates/provenance hash、每个媒体 revision 和原 SHA-1、实际
derivative hash、帧时间戳、模型完整 revision、权重 SHA-256、依赖版本、device、prompts、分数及
排序规则。先在已有 approved/rejected 人工集上做不参与训练的排序评估，公开 recall@K 和不同
series/uploader groups 的 recall；只有对现有单-poster 基线有预注册增益，才继续下载新 top-K。

不能声称多帧 CLIP 具有时序理解，不能在没有本地标注评测时声称 X-CLIP 能识别 Commons 的
猫狗 playing，也不能声称 MPS 与 CUDA 数值/速度等价。模型代码/权重的 MIT 或 NC 许可不向输入
视频授予新权利，也不清除 Commons 文件的署名、SA、人物或第三方内容风险。最终仍须完整播放、
稳定两秒窗口、单主体、逐文件许可和 source/near-duplicate group 人工门禁；当前 0 eligible C1
与不启动正式训练的边界不变。

模型选择中还实际验证了一个不能靠模型名推断的陷阱：`microsoft/xclip-base-patch16-zero-shot`
固定 revision `6746d6a5202bf24300cd9bf36f457f174483d41b` 的真实 config 要求 **32 帧**；按 8 帧输入
会在 message-token reshape 处直接失败。因此不篡改 config，也不把 32 帧模型冒充 8 帧模型，
而改用上述 config 原生为 8 帧的 patch32 checkpoint。

#### X-CLIP 十片真实 pilot：时序结构不等于时序判别有效

实现的 `temporal_priority.py` 不直接读取会在长任务中变化的输入：candidates、provenance、上游
poster report 与模型文件均先冻结字节快照；上游报告必须精确绑定前两者，模型目录复制到私有临时
目录后再加载。每条媒体下载前及整批结束后重验 Commons 原媒体 SHA-1/timestamp 和 derivative URL，
缓存键绑定 title、媒体 SHA-1 与 URL。当前固定策略不是全长只抽 8 帧，而是对每条视频最多取 8 个
均匀分布的 2 秒窗口，每窗取 8 个 midpoint frames；保存 derivative、逐帧 SHA、时间戳、全部 logits、
正向/倒序 margin 和模型所有文件 SHA。窗口尾部留固定 0.5 秒容差，以避免容器 duration 比最后可解码
PTS 稍长；该规则对所有视频一致，不按结果调节。

先在单 poster 前 10 条 dog pending 上运行 CPU pilot。权重 SHA 为 `abf286…0cdbb`，窗口报告
SHA-256 为 `49f1648e06c8566205d033ea44c2eaee144b15e4db7737cace2f3d497682622f`。
每条最多 8 windows、每窗正反各一次 forward；最高分依次仍为动物模仿实验、门口站立的柴犬、
临床步态、神经疾病楼梯测试和生肉喂食。10 条最佳窗口的正向/倒序 margin 平均绝对差只有
`0.0120`、最大 `0.0241`，不能据此声称模型在该样本上有效利用了帧序。

人工门禁没有只看模型给出的 2 秒窗口：另对每条完整 derivative 做 16 帧全时长联系表，再看最佳
窗口的 8 帧密集联系表。结果 **10 rejected / 0 approved**；其余五条分别是药物中毒临床运动、
实验室叙事/字幕、受控选择实验、慢动作饮水和肌营养不良步态。审核后状态为
**1,097 = 119 approved + 816 pending + 162 rejected**；这是十片 X-CLIP pilot 结束时的历史快照。v3 manifest/coverage 未变，仍是
119 records / 91 groups、held 16/50、训练 pairs 28/5/5、0 eligible C1。

此外用缓存中 26 条已有人审终态的 dog 视频做独立诊断，其中严格 `dog|playing|indoors` approved
为 4 条、其余 22 条为 rejected。全长 8 帧版本的 average precision 为 `0.5238`；recall@1/3/5/10/15
分别为 `0.25/0.25/0.25/0.75/1.0`。这是小而有选择偏差的诊断集，不是泛化 benchmark；它只说明
X-CLIP 可把部分已知正例提前，却没有把新 pilot 的 precision 提高到产生 approved。当前不把它扩到
当时剩余 816 条全库下载，也不把“有时序模块”误报成“能识别 playing”。下一步应转向有视频级动作标注、
逐文件权利可审计的人工策展来源，或先建立更大的盲标 validation pool 再校准排序器，而不是继续在
同一 top-10 上调 prompt。

#### Cat 盲审 40：poster 高分与随机对照均为 0/20

为了避免只审核模型最高分后再解释结果，先对全部 587 条 cat pending 运行同一锁定的 OpenAI CLIP
ViT-B/32 poster pipeline，再在任何人工判断前固定两组：rank 1--20 为 high arm，从其余 567 条用
seed `20260814` 无放回抽取 20 条 random-control arm。随后打乱 40 条，只向审核队列暴露 blind ID、
title、完整视频 URL、duration 与媒体 revision；arm、rank 和 margin 只保存在与 queue SHA-256 绑定的
sealed key。完整 poster report SHA-256 为
`3dbd67c222a7778a9282e489bf2050b9083b655005b45b72c70badbf619e7bae`。看片前冻结的原始
queue/key SHA-256 分别为 `13f2fce8f37081bb9575415415469b23ccdd254b4f6174f13ce48268a2b70e81` /
`ac52c2bffc754bf513e8214054b2c3d74026e7f87ea481476e9eabde3e1437e7`。代码复核后又从精确的
pre-review candidate snapshot、同一 report/seed 和完全相同的 40 条 assignment 确定性迁移，给 queue
加入不泄露 arm 的 assignment commitment；当前 queue/key 为
`45f951a5fc015ea85c42622926b6606824f6f0412aee8c12486e7204a9d0c4b0` /
`238e047578618a096cffe0f99f322db2e7bbed9030c7abec4ffc26060077959f`。这项迁移让未来篡改 arm
必然失败，但属于审核后的结构加固，不能倒推成原始运行已有的密码学时间戳；原始 key hash 与实际
assignment 保留为本轮 pre-review 记录。

40 条官方 derivative 全部下载并按全时长 16 帧联系表盲审；7 条最接近目标者又做 40 帧密集检查，
拟通过者必须另有连续 2 秒窗口证据。最终 **40 rejected / 0 approved**。典型误报包括室外猫、吃猎物、
多猫/人手、静止特写、动画，以及“背景有玩具但猫只是走过”；不能把 poster 上出现球等同于 playing。
全部决定完成后才揭盲：high arm **0/20**，random control **0/20**，yield difference `0.0`；两组
Wilson 95% interval 都是 `[0, 0.1611]`。audit SHA-256 为
`c8465f4290cb47bc8b10508230bc0e37fe177c9d70396ec9eb77b5f45a88d537`，reveal report 为
`caa4d2f3e92e707d4c7a5a14b61c1167c9b8cb806a7b10900822f98845c077fb`。reveal 还强制绑定
40 条实际 derivative 的 media manifest（SHA-256
`7de3d0dc315ba0b7a992652b0330a48d0cc00ba0cf23697d56add93bdaa62007`），因此分组、人工决定与
审核时所见媒体不能只靠文件名人工并列。

这只是冻结 partial pool 的严格 target yield/precision，不是 full-pool recall；0/20 也不能证明剩余
候选没有正例。但它否决了“poster top-20 比随机抽样更值得继续下载”的当前假设。候选状态更新为
**1,097 = 119 approved + 776 pending + 202 rejected**；cat 为 72 approved / 547 pending /
123 rejected。v3 manifest 与 coverage 没有变化，仍是 119 records / 91 groups、held 16/50、
训练 pairs 28/5/5、0 eligible C1，因此不启动正式训练。下一步不能继续对同一 poster ranking 调 prompt；
应先取得视频级人工动作标签或新的逐文件权利可审计策展来源，再以新的、预注册盲池检验召回策略。

#### Commons Structured Data 差集：31 条新增、3 条非 playing 通过

分类树、全文关键词和 poster 排序之外，Commons 官方 Structured Data 还提供稳定 API 搜索语法
`haswbstatement:P180=<QID>`。固定查询
`filetype:video haswbstatement:P180=Q146`（cat）和 Q144（dog）在执行时分别返回 397 / 68 个视频；
与已有候选按精确 File title 去重后新增 21 cat + 10 dog。`P180` 只证明 media 被标注为描绘该主体，
不证明 playing 或 indoors，因此 31 条全部以 `review_required` 进入 pending，且每条重新锁 description
revision、原媒体 SHA-1/timestamp、许可和 derivative。配置固定在
`configs/commons_structured_depicts_search.json`，可恢复 state 不把 API 总命中数冒充新增数。

先按 description 选出 7 条可能包含稳定单动物片段，下载完整官方 derivative 并检查全时长 20 帧联系表。
结果 3 approved / 4 rejected：新通过的是单猫室内 walking、sitting、lying 各一条；拒绝项是场景不可判的
发情翻滚、人与狗共同主导和多犬室外。其余 24 条仍为 pending，没有按标题批量拒绝。七条 audit SHA-256
为 `82b785e13a1bdf0cab1b16bbaf130ae82094d106311955bd0b70a84a20cfb24c`。三条规范化结果均为
H.264、96×96、8 fps、16 帧、2 秒，122/122 clip SHA-256 复核无误。

在随后两个 exact person 候选导入和审核前，候选曾为 **1,128 = 122 approved + 800 pending + 206 rejected**；v3 manifest 为
**122 records / 93 independent groups**，manifest/coverage SHA-256 分别为
`443b9cebbf3eb29591d3ebcb636a5001cd7e6ca101c2eeb8987898d525ea434e` /
`f6692fb9182e1455c3ee5a6d2d53350e4fb9f5a4d0cdf5f7526f49c631f0f409`。held
`cat|playing|indoors` 仍为 16/50；整组移除 held 后，训练 pair 变成 30/5/5，仅 cat×indoors 增加，
两个 playing pair 完全没有改善，仍是 0 eligible C1。因此 structured depicts 是有效的 subject 差集，
不是动作标签来源，也不构成启动训练的理由。

### 下一来源门禁：有条件采用 Ego4D 的 annotation-first 小样本路线

以下审计日期为 **2026-08-14**。这里把三个条件同时当作硬门槛：有视频级动作描述或时间标注；
每个下载文件能回溯到稳定 ID 和适用的数据许可；能先筛 metadata 再定向取少量媒体。annotation
可下载、仓库代码开源或网页上能播放，都不等于底层视频可用于训练或允许发布派生 checkpoint。

| 来源 | 官方事实 | 对当前缺口的判断 | 决策 |
|---|---|---|---|
| [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | 220,847 个 12 fps、240 px VP9 WebM，174 个预定义的“人操作日常物品”动作类；官方直接托管 19.4 GB 分卷，但须登录并接受 research-use Data License Agreement。视频以 `1..220847` 编号，train/val/test 为 168,913/24,777/27,157，另有 label/object annotations。 | 媒体与标签可按 ID 对齐，但主体定义就是人和物，不提供 cat/dog playing；规模可装下不代表语义适合。 | **排除**，不为凑数量下载。 |
| [Charades](https://prior.allenai.org/projects/charades)（[README](https://prior.allenai.org/projects/data/charades/README.txt)，[license](https://prior.allenai.org/projects/data/charades/license.txt)） | 9,848 个室内日常活动 MP4、267 subjects、157 action classes、66,500 个时间段动作标注和 27,847 条描述；官方直接托管 13 GB 的 480p 版。CSV 逐视频保存 `id/subject/scene/script/descriptions/actions/start/end/length`，15 个 scene 都是室内。许可只允许 academic/non-profit/government research 或 evaluation，禁止营利用途、修改后公开和向第三方分发数据。 | 官方 class 文件中含 play 的仅 `Playing with a phone/camera` 与 `Working/Playing on a laptop`，没有 pet/cat/dog/generic playing；不能把词面 play 当成这里的动物玩耍动作。 | **目标主线排除**；也不把它包装成 `playing×indoors` 正例。 |
| [Ego4D](https://ego4d-data.org/docs/start-here/)（[CLI](https://ego4d-data.org/docs/CLI/)，[schema](https://ego4d-data.org/docs/data/annotations-schemas/)，[video/metadata](https://ego4d-data.org/docs/data/videos/)） | 完整 primary dataset 约 7.1 TB，annotations 约 2 GB、benchmark clips 约 1 TB。先接受 [data license](https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf)，官方称凭证审批约 48 小时且 14 天失效。`narrations.json` 以 `video_uid` 为键，逐条保存 `timestamp_sec/narration_text/annotation_uid`；master metadata 保存 `video_uid/duration_sec/scenarios/video_source/fps/codec/resolution`。CLI 可用 `--video_uids` 或 `--video_uid_file` 定向取 `clips`、`video_540ss` 或 canonical video。 | narration 可以先检索 cat/kitten/dog/puppy 与 play/chase/toy/fetch/tug 的共现，再按 UID 取视频；这是本轮唯一同时具备时间文本、稳定媒体 ID、许可入口和定向下载能力的候选。但第一视角 narration 命中仍不证明目标动物、playing 或 indoor，且项目级协议不是逐文件 CC 许可。 | **有条件采用**：仅在本项目主体接受许可后做内部、受限 pilot；未复核协议对训练产物的约束前，不发布 clips 或 checkpoint。 |
| [YouTube-8M](https://research.google.com/youtube8m/download.html) | 官方发布约 6.1M 视频的 TFRecord 特征；segment 版约 237K 人工验证片段、1,000 类。frame features 为 1.53 TB，video-level mean features 为 31 GB；记录是匿名 `id/labels`，segment 另有起止与 score。 | CC BY 4.0 可归于 Google 发布的 dataset/特征/标签，不能外推到底层 YouTube 视频；官方不直接托管原视频，匿名映射也会因下架变化。 | **排除为训练视频源**；少量 shard 只能做标签/特征实验。 |
| [ActivityNet 1.3](https://activity-net.org/download.html) | 200 个人类活动类；train 10,024 视频/15,410 instances，val 4,926/7,654，test 5,044。JSON 保存 YouTube provenance、duration、subset 与 `segment/label`，下载器可按 class/max duration/max samples 筛；失效视频补发链接需申请且只有七天。 | 时间动作标注强，但仍依赖 YouTube 或临时补发；官方页未给底层视频统一媒体许可，类别也不是猫狗玩耍。 | **排除**。 |
| [AVA / AVA-Kinetics](https://research.google.com/ava/download.html) | AVA v2.2 为 430 个 15 分钟 movie clips，每秒标人框与 80 个 atomic actions；CSV 是 `video_id/timestamp/person_box/action_id/person_id`。官方下载给 annotations，不直接给 430 部影片。 | 标签明确绑定 person，且 YouTube 电影媒体权利不能从 annotations 的 CC BY 4.0 推出。 | **排除**。 |
| [Animal Kingdom](https://sutdcv.github.io/Animal-Kingdom/Animal_Kingdom/)（[action README](https://sutdcv.github.io/Animal-Kingdom/Animal_Kingdom/action_recognition/README_action_recognition.html)，[论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.pdf)） | 30,100 个 action-recognition clips、140 个细粒度多标签动作、850+ species；AR annotation 用 `clip_id/clip_number/frame_number/clip_path/action_labels`，15.6 GB 视频 tar 可经表单下载。官方描述素材来自 YouTube 的野生动物纪录片、trap 和个人录像。 | 语义上可能补 `cat×playing`，但不是 indoor 主导，也不能下载前按 clip 定向取视频。更重要的是本轮官方项目页/repo 未展示明确 dataset/media license；公开表单不是训练或再分发授权。 | **在作者书面确认数据许可、训练与产物范围前排除**，不先下载 15.6 GB。 |

Ego4D 的最小 pilot 采用如下 fail-closed 顺序；它是**数据源可行性审计**，不是训练启动：

1. 先由实际使用者接受并保存当时的 Ego4D data license、访问批准时间和许可文件 SHA-256；若条款
   不能明确覆盖本地非商业训练与内部 checkpoint，就在下载媒体前停止。代码仓库的 MIT 许可不替代
   data license。
2. 只取 annotations 与 master metadata：

   ```bash
   ego4d --output_directory data/raw/ego4d_pair_pilot \
     --version v2_1 --datasets annotations --metadata
   ```

   在约 2 GB annotation 中检索 narration 的 `cat|kitten|dog|puppy` 与
   `play|playing|chase|toy|fetch|tug`，保留前后时间上下文，同时记录负场景词。先冻结全部命中、查询式、
   annotation release 与文件 SHA；关键词只排序 pending，不自动生成 subject/action/scene 标签。
3. 按 `video_uid` 去重并保守合并同一 `video_source`、同一拍摄段和相邻 narration。预注册最多 10 个
   UID，优先选择来自不同 source、时间文本明确且可能补 `cat×playing` 或非 cat 的
   `playing×indoors`；若 metadata 阶段为 0 命中，就以 0 结果结束，不改词后追分。
4. 只对冻结 UID 取 540-short-side 视频，并把下载上限设为 12 GB，为 2 GB annotations、裁片和临时文件
   留空间；超预算则停止而不是转下全量。媒体阶段**不调用官方 CLI 的非版本化下载命令**，而是执行下文
   `prepare_ego4d.py preflight` 和 `download`：先冻结官方 manifest 与每个 S3 VersionId，再做 exact GET
   和流式 byte cap。

5. 每个候选保存 license/version、`video_uid`、`annotation_uid`、narration 原文与 timestamp、
   `video_source`、实际下载文件 SHA-256、裁片起止、人工决定和拒绝原因。看片确认主体、持续 playing、
   indoor/outdoor、第三方屏幕/音乐与可识别人；同一 source/主体/房间/连续事件及近重复必须同组。
6. 只有出现至少 5 个**新的独立 approved groups**，且许可复核允许本地模型训练，才把它们加入下一版
   provenance manifest。否则记录为 0-yield/低-yield pilot 并结束。无论结果如何，当前
   **122 records / 93 groups、held 16/50、训练 pairs 30/5/5、0 eligible C1** 均不变，仍不启动正式训练。

仓库已经实现 `ego4d_import.py` 的离线 metadata seam，但没有代替实际使用者接受许可或下载数据。它要求
同时提供 `ego4d.json`、`narrations.json`、当时许可文件和本地 receipt；receipt 必须把许可 SHA-256 与
`permits_local_noncommercial_training=true` 显式绑定。scanner 只接受同一句 narration 中动物词
`cat|kitten|dog|puppy` 与动作词 `play|chase|toy|fetch|tug` 共现，验证 UID、annotation UID、duration、
timestamp 与 metadata 归属，按 participant（缺失时 origin video）保守分组，并最多冻结 10 个 UID。
输入必须含非空 annotation version/date，许可 receipt 必须给出完整 UTC 日期时间；每个 pass 的 narration
先按 timestamp 排序再保存前后上下文。UID 选择先按 indoor/outdoor 线索排序，再优先覆盖不同
participant/origin group，独立组不足时才按稳定 UID 顺序补满预算。
候选和报告绑定四份输入 SHA、完整查询策略与 candidate SHA，始终写
`training_ready=false`；没有真实许可 receipt 时这条 CLI 无法合法越过媒体下载门禁。

这个选择并不声称 Ego4D 已含目标三元组，也不声称其 consent/data agreement 等价于 Commons 的逐文件
自由许可。它只说明：在本轮候选里，Ego4D 是唯一能先用人工时间文本筛选、再以稳定 UID 定向取得受同一
协议约束的视频，并把每一步写入 provenance 的可执行路线。Charades 可追溯但语义错误；Animal Kingdom
语义较近但许可未闭合；YouTube-8M、ActivityNet 与 AVA 的 annotation 权利不能补齐底层媒体权利。

#### Ego4D `video_540ss` 的真实落盘与下载后验证契约

以下事实核对自 Ego4D 官方 CLI 仓库 commit
[`4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb`](https://github.com/facebookresearch/Ego4d/tree/4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb/ego4d/cli)，
而不是根据未执行的下载猜测；本轮没有取得凭证、接受协议或下载受限媒体。

- [官方 README](https://github.com/facebookresearch/Ego4d/blob/4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb/ego4d/cli/README.md)
  说明每个 dataset 放在 `<output>/<version>/<dataset>/`。源码进一步明确版本目录只取 major：
  `v2_1` 也写到 `<output>/v2/`。显式 `--metadata` 下载的主 metadata 则写在
  `<output>/ego4d.json`，不在 `v2/video_540ss/` 内。
- [`manifest.py`](https://github.com/facebookresearch/Ego4d/blob/4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb/ego4d/cli/manifest.py)
  先把该 release 的完整 manifest 写为 `<output>/v2/video_540ss/manifest.csv`。视频 row 以
  `video_uid` 作为 UID；`type=mp4|video` 时本地文件名强制为 `<video_uid>.mp4`，不沿用 S3
  object basename。`--video_uid_file` 是 whitespace-delimited UID 列表；视频 dataset 缺少任一请求
  UID 时 CLI 报错，不静默缩小集合。
- [`download.py`](https://github.com/facebookresearch/Ego4d/blob/4bd10ed40b4f8d8ad26344afc2c8526f7d1dedeb/ego4d/cli/download.py)
  在 `<output>/v2/video_540ss/manifest.ver` 保存 `uid,version,filename`；`version` 是下载时 S3 object
  version ID。它不是内容 hash。CLI 下载前 HEAD 每个 S3 object，累加 `content_length` 并显示预计 GiB；
  下载后完整性检查仅比较本地文件字节数与该 `content_length`。源码仍留有损坏后重试 TODO，且没有
  SHA-256 校验。
- [主 metadata schema](https://ego4d-data.org/docs/data/metadata/) 可验证 `video_uid`、
  `duration_sec`、`scenarios`、`video_source`、participant/origin 关系，以及 canonical video 的
  `fps/num_frames/video_codec/resolution/mp4_duration_sec`。后面这些是 canonical metadata，**不能**
  冒充 `video_540ss` 派生文件的精确 codec、分辨率、字节数或内容 hash；540ss 实际媒体属性必须下载后
  `ffprobe`。

因此原计划中的 `download_contract.max_bytes=12_000_000_000` 目前只是冻结的本地门禁，不是官方 CLI
已经实现的 `--max-bytes` 参数。实际获批后应按下面顺序执行，超出任一步就保持
`training_ready=false`：

1. 固定 Ego4D release、用于审计 manifest/layout 的官方源码 commit、本地 exact-version downloader
   与 boto3 lock、执行 UTC、license/receipt SHA、UID 文件内容与 SHA。保持 `≤10` 个 UID；不走
   官方 CLI 的 `-y`、`--skip-s3-checks`、`--bypass-existing` 或 latest-object fallback。
2. 在确认下载前，以 `manifest.csv` 中每个目标 UID 的 S3 path 做 HEAD，保存 object version ID 与精确
   `ContentLength`，按十进制 bytes 求和；总和大于 12,000,000,000 就停止。官方源码以 `bytes/1024^3`
   计算却显示为 GB，所以十进制 12 GB 硬帽对应提示值最多约 **11.17587**；交互值还只显示四位，适合
   人工 sanity check，不足以替代精确 byte gate。若不写预检 wrapper 或独立配额卷，就不能声称硬性
   12 GB 已执行。
3. 不直接调用官方 CLI 的非版本化 GET：它在 GET 前读取 version，但下载调用本身没有传 `VersionId`，
   object 若在 HEAD 与 GET 之间更新，同尺寸新对象可能被旧 `manifest.ver` 错绑。改由本地 wrapper 对每条
   显式传入预检 `VersionId`，边流式读取边执行单文件和累计 byte cap；响应 version/ETag/ContentLength
   任一漂移都停止，不回退到 latest object。完成后写兼容格式的 `manifest.ver`。
4. 下载后要求 UID 集合与 `<uid>.mp4` 文件集合完全相等，无额外/缺失/零字节文件；逐条核对
   `manifest.ver` 的 UID、文件名和预检 S3 version，再核对本地 byte size 与预检 `ContentLength`。
5. 对每条 MP4 另算 SHA-256 并运行 `ffprobe`，保存 container、video/audio codec、duration、fps、frame
   count、display/sample resolution。duration 只能与 `ego4d.json` 作带容差的一致性检查；不要强行要求
   540ss codec/resolution 等于 canonical metadata。
6. 生成只读 media manifest，绑定 `ego4d.json`、官方 dataset `manifest.csv` 的 S3 version/hash、
   `manifest.ver`、UID file、下载 receipt、license/receipt 和每个 MP4 的 SHA/bytes/ffprobe。随后才裁 2 秒片段并人工审 subject/action/
   scene；裁片再记录自己的 SHA 和 source/duplicate group。S3 ETag 可能是 multipart ETag，不能未经证明
   当作 MD5，更不能代替本地 SHA-256。

上述外部硬门现已由 `prepare_ego4d.py` 落地，而不是只停留在建议。`preflight` 先验证 scanner 的
metadata/narration/license/receipt/candidate/UID 绑定，从固定官方 S3 path 以 version-specific GET 冻结
`manifest.csv`，仅允许官方源码列出的 Ego4D media buckets，再逐 UID HEAD 锁定 version ID、精确
`ContentLength` 和 ETag；总和超过十进制 12 GB 就在视频下载前失败。`download` 对已锁 VersionId 做
精确 GET 并流式执行 byte cap，避免官方 CLI 的 HEAD→latest-GET 竞态；完成后写下载 UTC、boto3 版本、
plan hash、逐文件流式 SHA-256 与 `manifest.ver` hash。`verify` 再要求
receipt/manifest.ver/UID/filename/version/下载时 SHA 全部一致，
本地文件必须是普通非符号链接且 byte size 等于预检值，然后流式计算 SHA-256、以容差核对
`ego4d.json` duration（预注册为 `max(1.0 秒, metadata duration × 1%)`），并用 ffprobe 固化
container、video/audio codec、duration、fps、frame count 与
分辨率。三个报告继续是 `training_ready=false`；它们只把获得许可后的下载变成可审计输入，不能替代
用户接受协议，也不能替代 2 秒窗口人工审核。

#### 无需代用户接受新协议的替代源：没有新的标准数据集通过三项门禁

这次又核对了三个第一方、无需由 agent 代签协议即可查看的近邻来源；它们分别只满足“动作”“猫狗”或
“明确许可”的一部分，没有一个可替代 Commons：

- [DogCentric Activity Dataset 官方页](https://robotics.ait.kyushu-u.ac.jp/yumi/db/first_dog.html)
  给出匿名 FTP 的 320×240、48 fps 分段视频，并有 `playing with a ball` 等 10 个动作；四只狗佩戴
  GoPro，路线含住宅、河边公园、沙滩、田野和街道。这是语义最接近的标准 action dataset，但官方页未给
  dataset/media license 或训练与派生产物授权。匿名可下载不等于获得版权许可，因此**不下载、不纳入**。
- [COP3D 官方仓库](https://github.com/facebookresearch/cop3d) 有 4,200 个不同 cat/dog pet sequences，
  明确为 CC BY-NC 4.0，不要求新的访问协议；但压缩数据 322 GB、以约 20 GB chunks 发布，格式是
  sequence images/masks/point cloud，不是带 action/scene caption 的短视频。24 GB 本机无法在留出解压与
  审核空间的同时安全处理一个 20 GB chunk，也无法 metadata-first 定向找 playing，故**不采用**。
- [SA-V 官方仓库](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md) 直接发布
  51K MP4 与 643K masklets，标 CC BY 4.0；[官方数据页](https://ai.meta.com/datasets/segment-anything-video/)
  的 train JSON 有 `video_id/duration/frame_count/height/width/video_environment`，可给 Indoor/Outdoor，
  但官方明确没有 class labels，因此没有 cat/dog 或 playing 标签。没有可审计的主体/动作定向筛选键时，
  不能为当前缺口下载大池后再碰运气，故**不采用**。

结论分为事实与决策两层：确有“无需新协议”的公开动物视频源，但本轮未找到同时具备目标语义键、逐媒体
可审计权利和 24 GB 定向下载能力的**新标准数据集**。当前仍只有 Wikimedia Commons 满足可逐文件查询
许可、标题/description/categories/depicts 召回和按单文件下载；它不是标准 T2V benchmark，且每个命中仍
须人工审批。Ego4D 保持“用户自行接受许可后再做 metadata-first pilot”的备用路线，不能由本次研究替用户
跨过协议门槛，也不改变当前 0 eligible C1 / 不训练状态。

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

生成结果的人审也必须真正盲化，而不是先看 prompt 再给一条好看的 MP4 贴标签。当前
`generation_blind_review.py` 已实现未来正式实验的证据链：评估 spec 只能从重新执行
`validate_split()` 后逐字节一致的 manifest/report 派生，并在任何推理前冻结训练 manifest、
checkpoint SHA、EMA/raw、采样参数和至少两个明确 seed；每个 seed 必须生成完全相同的 prompt
集合，且推理 metadata 必须绑定同一 spec、checkpoint、训练 manifest 与输出 MP4 的 SHA-256。
创建 queue 时把 MP4 复制为
随机 blind ID，公开文件不含 prompt、split 或 seed，sealed key 用 commitment 绑定完整 assignment。
至少两名独立评审者从冻结 factor vocabulary 判断 subject/action/scene；三类词表都包含不可能成为
expected label 的 `absent_or_unclear`，避免无主体/无动作的失败样本被迫猜成真标签；揭盲时任一因子分歧都作为
ambiguous/incorrect，分别报告 IID/C1/C2 的三因子与完整 triple accuracy、Wilson 95% 区间，以及
同 seed 内按 factor triple 严格 derangement 的错配对照（无法错配即拒绝）。该工具尚未产生 C1 结果，因为当前 0 eligible C1；它是
防止未来用 cherry-picked artifact 冒充组合泛化的预注册评估门，而不是当前成功证据。

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

### 2026-08-14：训练侧 pair 的新 Commons 精确入口（全部 pending）

本节只补训练侧 `cat×playing`（必须明确 outdoor、不得是 held indoor）和
`playing×indoors`（必须是非 cat 主体），不继续重复已经闭合的 playing-cat 44 文件、dog-play
分类树、零视频 toy/fetch/tug 类或既有宽全文查询。审计只用了 Commons 官方 Action API、分类页和
文件页；与导入前的 **1,128-row candidates** 做精确标题差集后，下列文件均不在池中。分类/API 命中只是
召回，**没有任何一条因此 approved，也没有改变 5/10、5/10 或 0 eligible C1**。

#### 最优先：`Videos of children playing` 的两个明确 indoor 文件

[`Category:Videos of children playing`](https://commons.wikimedia.org/wiki/Category:Videos_of_children_playing)
当前为 **10 个直属 files / 0 subcats**；逐文件 `imageinfo` 为 6 个 `video/webm` 与 4 个
`.ogv + application/ogg`。这是未在既有计划中穷尽的新精确 action 类；其中只有下面两条同时有官方
indoor 正线索，最适合作为 `person×playing×indoors` 的 metadata-first pending：

| pending 文件 | 官方固定身份与许可 | 官方语义证据 / 审核边界 |
|---|---|---|
| [`Muzeo MALI Limo (Peruo).webm`](https://commons.wikimedia.org/wiki/File:Muzeo_MALI_Limo_(Peruo).webm) | page ID `98156360`；SHA-1 `e643a5f2ecc94b76adedc8ece9817d88a1e494b1`；11,316,831 bytes；32.468 s；CC BY-SA 4.0；artist `RG72` | categories 同时含 `Videos of children playing`、`Children playing inside` 与 `Videos of interiors of buildings`。这是强 scene/action metadata，但仍须全长确认主要人物、持续 playing、稳定 2 秒窗口和第三方可识别人。 |
| [`Knabo ludas en infanludejo en la aĉetcentro Kristall (Tjumeno).webm`](https://commons.wikimedia.org/wiki/File:Knabo_ludas_en_infanludejo_en_la_a%C4%89etcentro_Kristall_(Tjumeno).webm) | page ID `107074981`；SHA-1 `d516704074ed84d70e51f6c1e919640e89622985`；7,534,479 bytes；25.503 s；CC BY-SA 4.0；artist `RG72` | 官方 description 是男孩在购物中心室内游乐场玩耍，categories 含 `Boys playing`、`Interiors of shopping malls in Tyumen`。同样只可 pending；口罩儿童、其他访客和人物权利必须逐窗审核。 |

第三条较弱候选
[`Speech-in-action-degree-of-hand-preference-for-grasping-predicts-speech-articulation-competence-in-Video1.ogv`](https://commons.wikimedia.org/wiki/File:Speech-in-action-degree-of-hand-preference-for-grasping-predicts-speech-articulation-competence-in-Video1.ogv)
为 page ID `36916325`、SHA-1 `4a599638f9ac5b07af9ba18684353c30c41f636b`、4,184,040 bytes、
22.689 s、CC BY 4.0；它属于 `Videos of children playing` 与 `Lego`，但官方 description 为空且没有
indoor category，只应在前两条之后做 poster/全长 scene 复核，不能预标 indoors。

两条强候选来自同一 artist `RG72`，但 metadata 给出的事件分别是 2014-08-03 Lima 与 2021-06-24
Tyumen。不同日期/国家是“可能独立事件”的证据，不自动等于两个 source groups；在保存 revision、原
SHA-1、完整时间联系表和人物/场景记录前仍按保守 grouping。即使两条都通过，这个 10-file 小类也不能
单独把 5/10 扩到长期 50 级规模；可把同一分类 API 查询作为增量上传监视入口。

#### `Videos of playing animals` 与兔子线索：可召回，但不能直接补当前门禁

[`Category:Videos of playing animals`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_animals)
当前为 **6 个直属 files / 1 subcat**；唯一子类是已经闭合的 `Videos of playing cats`。直属六条是
meerkat、两条 dwarf-hamster disc、black-bear cubs、African wild dogs 和 fossas。两条 hamster
分别为 page IDs `21925981/21929186`、SHA-1
`96bd7c1a933759b92f892d6b46f806c201b83ff0` / `7ac311eb074c4657d8cd5306c06fa55d53ade466`，
均为 CC0、同一 uploader、同日且同为多只 hamster 在 exercise disc 上跑；最多一个 series/source
group，而且“running on disc”与 indoors 都未由官方 metadata 证明。其余四条的官方描述指向 zoo、
wildlife park 或 private game reserve，也没有 indoor 正证据。因此该分类适合保留为低成本增量监视，
当前没有一条可直接计入 `playing×indoors`。

独立文件
[`Conill béliere jugant i esperant menjar.ogv`](https://commons.wikimedia.org/wiki/File:Conill_b%C3%A9liere_jugant_i_esperant_menjar.ogv)
为 page ID `77688284`、SHA-1 `e459edec9a764ede18747ae93e1bc46010ef2a24`、6,801,897 bytes、
45.650 s、CC BY-SA 4.0；官方 description 明确为 lop rabbit playing/waiting for food，但 categories
只有 lop rabbits/self-published，没有 indoor/outdoor 证据。更重要的是当前精确导入 taxonomy 只接受
`cat|dog|person`，所以它只能作为未来扩 taxonomy 的 pending，不得冒充当前 pair support。
Action API 的精确 File-namespace 查询 `"rabbit playing" filetype:video` 当前正好 **1 hit**，即该
page ID；因此这条路线的可审计上限就是一个 scene 未知文件，不具备扩展性。

可复现 API（读取 `continue`，并以 page ID 为主键、原 SHA-1 为内容键）：

```text
# 分类精确计数
action=query&prop=categoryinfo&titles=Category:Videos_of_children_playing|
Category:Videos_of_playing_animals&format=json&formatversion=2

# 直属文件、MIME、SHA-1、许可、描述与 categories
action=query&generator=categorymembers&gcmtitle=Category:Videos_of_children_playing&
gcmtype=file&gcmlimit=max&prop=imageinfo|categories|revisions&
iiprop=mime|sha1|url|size|extmetadata&cllimit=max&rvprop=ids|timestamp&
format=json&formatversion=2
```

执行顺序是：① 精确导入上述两个 indoor 强候选为 `pending/review_required`；② 锁 page revision、原媒体
timestamp/SHA-1、artist、LicenseUrl 和 attribution，再做全长及密集 2 秒审核；③ 人物可识别、儿童、
多主体、剪辑或 action 不稳定任一失败即 reject；④ Lego 文件只在 scene 视觉证据通过后考虑；⑤
hamster/rabbit 不扩 taxonomy、不计 coverage。CC BY/CC BY-SA 是文件页的版权许可声明，要求相应署名、
许可链接及适用的 share-alike；它不替代可识别人物的隐私/人格权审核，见 Commons 官方
[`Personality rights warning`](https://commons.wikimedia.org/wiki/Template:Personality_rights)。

不把图库 API 当作替代扩容路线：[`Pixabay API documentation`](https://pixabay.com/api/docs/) 与
[`Pixabay Terms`](https://pixabay.com/service/terms/)、以及 Pexels 官方
[`API FAQ`](https://help.pexels.com/hc/en-us/categories/900001326143-Pexels-API) 对自动化使用另设条件，
并不授权未经许可为机器学习批量收集媒体。本轮不申请 key、不调用这些 API、不下载其媒体；站点级
“free to use”也不能替代逐 item provenance、人物权利和可复现 source-group 证据。

#### 执行结果：两个 indoor person 候选均未通过人物权利门禁

上述两个强 metadata hit 已通过 `prepare_commons.py exact` 精确导入并补齐 sidecar provenance。
导入时还纠正了 page ID `98156360` 的规范 File title：API 返回的是
`File:Muzeo MALI Limo (Peruo).webm`，而不是带额外句点的旧转写；原 SHA-1、description URL 和
page ID 均未改变。两条官方 240p derivative 分别完成全时段 12 帧联系表审核：

- `Muzeo MALI Limo` 是博物馆大厅内大量儿童与成人的群体活动，没有稳定的单一主要人物 2 秒窗口；
- `Knabo ... Kristall` 是一个戴口罩的儿童持续操作玩具，动作与室内场景相关，但仍是可识别未成年人。

按照本节在看片前写定的“儿童、可识别人物或多主体任一失败即拒绝”，两条均标为 rejected，不能因为
它们恰好补 `playing×indoors` 就放宽规则。审核 audit 绑定候选快照 SHA-256
`9641ee51a6c29634edb60f374698a301e428746ddf99e3521e8cebd248368bb9`，audit 自身 SHA-256 为
`d1402052da3f4e93ee92061a4995149212bf8d7eb30140312e51920014164d1e`。当前候选为
**1,130 = 122 approved + 800 pending + 208 rejected**；v3 manifest 未改变，重新计算的 coverage
仍逐字节保持 SHA-256 `f6692fb9182e1455c3ee5a6d2d53350e4fb9f5a4d0cdf5f7526f49c631f0f409`：
122 records / 93 groups、held 16/50、训练 pair 30/5/5、0 eligible C1。下一来源必须避开儿童和
群体人物，优先寻找逐文件权利可审计的单犬 indoor playing；本轮没有启动训练。

为避免拒绝证据只留在 gitignored 本机数据里，可携带证据保存在
[`artifacts/video-dit/person-playing-indoor-review/evidence.json`](artifacts/video-dit/person-playing-indoor-review/evidence.json)：
它逐条固化 page ID、page revision、原媒体 SHA-1、实际审核 derivative URL/字节数/SHA-256、下载 UTC、
12 个抽帧时刻、联系表 SHA-256、决定和理由；同目录两张低分辨率联系表按源文件 CC BY-SA 4.0 及
记录的 RG72 署名发布。原 derivative 不提交 Git，但可由固定 URL 下载并以记录的 SHA-256 复核。
导入器现在也会把 Commons `pageid` 写入新 candidate；新的 live API 导入或 provenance enrichment
若缺少正整数 page ID 会 fail-closed，review sidecar 同样持久化该值。上述两条本地 sidecar 已补为
`98156360/107074981`，当前 sidecar SHA-256 是
`9490d2c91e72f00d2f6e8b0436cbc5d88ad79d1c61c8e1a43c2a906632da4537`；旧记录允许无 page ID，
避免把历史 manifest 冒充已迁移，但新审核证据必须按本段 tracked artifact 锁定。

#### 下一批 pair 来源复核：官方高置信新增差集为 0

在上述两个儿童文件已经导入并拒绝后，再以当前 **1,130-row candidates** 的规范 `File:` title
集合做精确差集。按 Unicode title 升序、每行一个 title 且末尾保留换行的 UTF-8 字节列表 SHA-256 为
`d166e7e6f8759e4ec31710cf82588f25148ca4d87d3fce66127f6fe8880ba272`，且没有重复 title。本轮只查
Wikimedia Commons 官方 REST 搜索、Action API、分类页与文件页，优先级依次为
`dog|playing|indoors`、无儿童且单一成年人的 `person|playing|indoors`、
`cat|playing|outdoors`；没有把搜索摘要或分类成员资格直接当成标签。

官方 REST title 搜索 `File: dog playing indoors`、`File: dog playing inside` 均返回 0 个视频 title；
`File: puppy playing house` 唯一视频命中是已在池中的
[`Puppy playing.webm`](https://commons.wikimedia.org/wiki/File:Puppy_playing.webm)。成年 person 的
`File: adult playing indoors`、`File: person playing indoors`、`File: people playing indoors`，以及
outdoor cat 的 `File: cat playing outdoors`、`File: cat playing outside`、
`File: kitten playing outdoors`，在每个查询最多 100 个官方结果内均为 0 个 `.webm/.ogv` title。
更宽的 `dog play/puppy play/dog toy` 只召回已收录犬视频、犬展、训练演示或标题歧义内容，没有同时
由官方 metadata 证明“单犬 + 持续 playing + indoors”的新文件。已穷尽的精确类仍是
[`Playing dogs`](https://commons.wikimedia.org/wiki/Category:Playing_dogs)、
[`Videos of cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_cats) 与
[`Videos of playing cats`](https://commons.wikimedia.org/wiki/Category:Videos_of_playing_cats)；这次没有
通过改写查询重复导入它们的成员。

为确认结果不是同名漏判，Action API 对当前五个最强 indoor-dog 文件重新读取固定身份：

| 已在池中的文件（不是新增 pending） | page ID | 原媒体 SHA-1 | 文件页许可 | 排除出“新增差集”的原因 |
|---|---:|---|---|---|
| [`Puppy playing.webm`](https://commons.wikimedia.org/wiki/File:Puppy_playing.webm) | `27014028` | `733210c6cc9a1c2d54b93011b164cbbc1f2e10d8` | CC BY-SA 3.0 | 已 approved；官方 description 明确 puppy playing in house。 |
| [`-01- Miniature Pinscher puppy.webm`](https://commons.wikimedia.org/wiki/File:-01-_Miniature_Pinscher_puppy.webm) | `195852685` | `7a64bf681973f0283eacad95f8d3a51a327133fb` | CC BY-SA 4.0 | 已 approved；不是新 source。 |
| [`Пузыри собака.ogv`](https://commons.wikimedia.org/wiki/File:%D0%9F%D1%83%D0%B7%D1%8B%D1%80%D0%B8_%D1%81%D0%BE%D0%B1%D0%B0%D0%BA%D0%B0.ogv) | `15446135` | `b6c1fd97f257c36d04941cdaeed6e3d34d23fe8f` | CC BY-SA 3.0 | 已 approved；不是新 source。 |
| [`Dog mirror test.ogv`](https://commons.wikimedia.org/wiki/File:Dog_mirror_test.ogv) | `10794688` | `09f4aa6919dafe8befa7e8ac036bde137e66d774` | CC BY 2.0 | 已 approved；同类 mirror-play 已做 grouping 审核。 |
| [`Dog mirror test 3.ogv`](https://commons.wikimedia.org/wiki/File:Dog_mirror_test_3.ogv) | `10794633` | `f98d207ce9f96882be010fc8924c56e76cd50dd9` | CC BY 2.0 | 已 approved；同类 mirror-play 已做 grouping 审核。 |

因此本轮**没有追加媒体候选，也没有任何文件从 pending 变成 approved**；这比把弱命中包装成新数据更
符合预注册门禁。coverage 仍是训练 pair `cat×indoors=30/10`、`cat×playing=5/10`、
`playing×indoors=5/10`，held `cat|playing|indoors=16/50`，**0 eligible C1**。下一次只接受
官方文件页同时给出 action/scene 正证据、单一非儿童主体、逐 item 许可与原 SHA-1 的新 title；同一
uploader、同一动物、同日连续上传或同一拍摄事件先保守并为一个 source group。未经全长和密集 2 秒
窗口审核的命中一律保持 `pending/review_required`，不得计 coverage，也不启动正式训练。

#### 从“play”全文改为游戏分类交集：三个成年单人 indoor pending

上一轮的 0 差集只覆盖英文 title/description 查询；它不能证明 Commons 没有被按具体游戏分类、但
标题不含 `play` 的视频。因此本轮改走官方分类结构。Commons 官方
[`Videos of people playing`](https://commons.wikimedia.org/wiki/Category:Videos_of_people_playing)
当前只有 `Videos of children playing` 一个子类、没有成年直属视频，故这条显式 people/activity 树的
成年差集仍是 0。随后审计
[`Videos of board games`](https://commons.wikimedia.org/wiki/Category:Videos_of_board_games) 的 7 个
直属文件：7 个 title 均不在当前 1,130-row 集合，但原视频只显示棋盘、游戏画面、手或前臂，或同时
出现两名玩家；官方 metadata 也没有同时证明“单一成年 person + indoors”。例如 `Carrom playing`
虽明确是 playing，但全长只见一只前臂；`9x9 Go` 是 tournament 且两名玩家的手交替出现。这 7 条
都不能因为场景看似室内就直接预标 person/indoors。

真正可执行的新入口是官方
[`Videos of billiards`](https://commons.wikimedia.org/wiki/Category:Videos_of_billiards)：24 个直属视频的
规范 title 与 1,130-row 集合差集为 **24**。Action API 固定 page ID、原媒体 SHA-1 与许可后，又对
官方原视频做完整低频联系表，而不是只看搜索摘要。`Quick Pool Break` 只在末尾出现手指，`Piqué`
只显示手和球杆，`Carom billiards` 有两名选手及裁判，赛事新闻片通常也有多名人物；这些都不进入
下面的高优先 pending。只有三条历史演示持续以一名成年男子为主要主体，且室内台球桌、击球动作与
单人构图在完整时序中稳定：

| pending 文件 | 官方固定身份与许可 | action / scene / 人物与 grouping 边界 |
|---|---|---|
| [`Arie Bos Nederlands kampioen biljarten driebanden-524299.ogv`](https://commons.wikimedia.org/wiki/File:Arie_Bos_Nederlands_kampioen_biljarten_driebanden-524299.ogv) | page ID `35084511`；SHA-1 `5335eb4274cb9b7d455912bd76ba4cfdf4df9c2c`；9,866,592 bytes；CC BY-SA 3.0 NL；Polygoon-Profilti / Nederlands Instituut voor Beeld en Geluid；官方日期 1938-04-04 | 官方 description 明确荷兰三库仑台球冠军 Arie Bos 逐项 demonstration；原视频联系表显示单一成年男子和台球桌贯穿多数镜头，scene 为明确室内布景。只能 `pending/review_required`；同题的 page `84556061`、SHA-1 `1392ac…` WebM 是同一 Arie Bos 新闻片路线，先按同一 source/near-duplicate group，不能算第二组。 |
| [`De vermaarde handbiljarter J. Heilbron-508928.ogv`](https://commons.wikimedia.org/wiki/File:De_vermaarde_handbiljarter_J._Heilbron-508928.ogv) | page ID `35084522`；SHA-1 `42a1aecbcd18c66b3ee385e6cc6698e41f3495cf`；19,275,705 bytes；CC BY-SA 3.0 NL；同一官方馆藏；官方日期 1934-08-24 | 官方 description 明确一名男子在 Haarlem 用手指而非球杆演示多种 billiards shots；完整 240p derivative 联系表为单一成年男子、室内台球桌和持续击球。与 Arie Bos 是不同人物/日期/事件，但仍须按原影片标识、pHash 和片段重叠确认后才能声明独立 source group。 |
| [`Biljartdemonstratie door H.J. Robijns.webm`](https://commons.wikimedia.org/wiki/File:Biljartdemonstratie_door_H.J._Robijns.webm) | page ID `88210054`；SHA-1 `9e77a6e777c534214343bf029e34dd8f6942b933`；28,366,770 bytes；文件页标为 Public Domain；Polygoon Hollands Nieuws / Nederlands Instituut voor Beeld en Geluid；官方日期 1932-01-11 | 官方 description 明确世界冠军 H.J. Robijns 演示手势与击球，但也说明片头由 I.L. Pinto 介绍。完整联系表存在大量稳定单人 Robijns 窗口；导入后必须只选无助手、无剪辑的密集两秒窗口。1932/1934/1938 是三个不同事件的正证据，不替代近重复闭包。 |

上述三条只是在看片前规则下值得进入 exact import 的 **pending**，不是 approved，更没有立即贡献三个
groups。执行时应只导入表中规范 title，锁 source revision、原 SHA-1、许可/署名和官方日期，再为每条
生成完整联系表及候选两秒窗口；人物切换、字幕/剪辑污染、主体不足或 action 不连续即 reject。Arie Bos
的 OGV/WebM 变体必须合并；同一 Polygoon 馆藏不自动等于同一拍摄源，但也不能仅凭不同文件名拆组。
人物均为 1930 年代公开新闻片中的已具名成年人，风险显著低于儿童或当代无 release 人物；这仍不把
文件版权许可等同于人格权保证。

dog 结构化交集也做了独立负查：官方 CirrusSearch
`haswbstatement:P180=Q144 incategory:"Playing dogs" filetype:video` 只返回
[`Two poodles playing.ogv`](https://commons.wikimedia.org/wiki/File:Two_poodles_playing.ogv)（page ID
`10794007`、SHA-1 `72dd708bd5433482aadbf194b48af015ab04a8df`、CC BY-SA 2.0），它已在当前池并因多犬、
outdoors 被拒绝。放宽为 dog depicts + `playing` 的 5 个视频 title 也全部已在 1,130-row 集合；
`Puppies playing` 精确类是 0 个视频。因此 dog 结构化高置信新增差集仍为 **0**。

这次发现 pending 不改变权威 coverage：训练 pair 仍为 `cat×indoors=30/10`、
`cat×playing=5/10`、`playing×indoors=5/10`，held `cat|playing|indoors=16/50`，
**0 eligible C1**；在 exact import、逐窗审核、source/pHash 闭包和重新生成 manifest 前不得把三条台球
视频计入 5/10 缺口，也不得启动正式训练。

### 2026-08-14：SA-V 官方审计——可做 305-ID 可行性 pilot，不能直接计独立来源

结论先行：Meta 的 Segment Anything Video Dataset（SA-V）比不透明网页抓取更适合做**有许可、固定 ID、
有界数量**的候选发现实验，但当前公开版本不能直接解决本计划最关键的 source-group 门禁。官方
[`SA-V 数据页`](https://ai.meta.com/datasets/segment-anything-video/)说明视频和 annotations 采用
CC BY 4.0，约 51K 视频来自签约第三方组织的 crowdworkers；官方
[`SAM 2 仓库的数据说明`](https://github.com/facebookresearch/sam2/blob/main/sav_dataset/README.md)
给出 50,583 个 train MP4、155 个 val 和 150 个 test 视频。可是公开 JSON 只有不透明的
`video_id`、时长/帧数/尺寸、`video_environment`（Indoor/Outdoor）、split 和 masklet 统计；官方同时明确
mask 是 class-agnostic，**没有类别或文本标签**。因此 SA-V 只能按粗粒度 indoor/outdoor 预筛，不能用
官方 metadata 查询 cat、dog、person 或 playing，更不能把自动 caption/classifier 的结果直接当真值。

#### 许可、下载与 100–500 条有界机制

- 官方 landing page 和仓库均把**视频及 annotations**声明为 CC BY 4.0；但 SAM 2 论文 dataset card 还明确
  要求用户在下载或使用前同意下载页的 Terms of Use。CC BY 声明不能替代对当时 ToU 的保存与复核，
  本轮没有接受条款、没有请求下载链接、也没有下载媒体。正式 pilot 必须保存接受日期、ToU 快照、
  dataset/version 标识和归因文本，不能仅记录“CC BY”。见官方论文
  [`Appendix H.2 / Distribution`](https://arxiv.org/html/2408.00714)及
  [`下载入口`](https://ai.meta.com/datasets/segment-anything-video-downloads/)。
- 最干净的 bounded pilot 不是先吞下 50,583 个 train MP4，而是下载官方 **val 155 + test 150 = 305 个
  video IDs**。仓库说明这两部分提供 24 fps JPEG frames、6 fps PNG masks 与 split ID list；305 正好落在
  预注册的 100–500 范围内。若为了现有审核工具把 frames 封装为 MP4，必须同时保留原 archive SHA-256、
  原 `video_id`、frame-list hash、转换命令和派生 MP4 SHA-256，并注明发生格式转换。
- train 端是每个 `{video_id}.mp4` 配 `{video_id}_manual.json`，部分另有 `_auto.json`。官方 repo 没有
  文档化一个受支持、带版本/checksum contract 的“按 subject/action 查询并只取 100–500 个 train MP4”
  API；官方 SAM 3 数据准备说明也把 SA-V 下载描述为从 Meta 页面取得 **dynamic links**，并以
  `videos_fps_6.tar` 链接为例。参见 Meta 官方
  [`SAM 3 SA-V preparation`](https://github.com/facebookresearch/sam3/blob/main/scripts/eval/silver/README.md#sa-v)。
  Dataset Explorer 的当前前端确实使用 cursor/`first` 查询并返回签名媒体 URL；2026-08-14 的只读探测
  以 `video_environment=Indoor, first=500` 请求得到 501 个唯一 `video_id`、仍有下一页（ID 列表
  SHA-256 `b568e8554c4a4eee85fd31fbad4da074eff6d4a1c2f4cde5e7294b7ff9b7c18a`，未请求 URL、未下载
  视频）。这证明技术上可以分页浏览，但返回数的 off-by-one、动态签名 URL 和未文档化 API 都意味着
  它不能作为稳定 acquisition contract；若以后使用，必须本地截断、锁 cursor/ID/响应 hash，并按 exact
  ID 下载后重算媒体 hash。

#### source group、近重复和人物权利的硬边界

官方论文说明 SA-V 的 train/val/test 是按**视频作者及其地理位置**拆分，以尽量减少相似物体重叠；这使
官方 val+test pilot 比随机抽取 train 更适合测试 pipeline。然而作者/位置字段没有随上述 JSON schema
公开。更关键的是，dataset card 明确说虽然视频文件彼此 distinct，仍有若干视频拍摄了同一 subject
matter，并明确回答实例间关系没有显式提供；原始视频也没有保留，发布物只是统一重采样为 24 fps 的
MP4。参见官方论文 [`§5.2 与 Appendix H.2`](https://arxiv.org/html/2408.00714)。所以：

- `sav_XXXXXX` 不等于独立 uploader、作者、动物或拍摄事件；连续 ID 也不能被解释为或排除同一 series。
- 发布说明没有 per-video 原始来源 ID、作者 ID、capture-session ID、官方内容 hash 或 duplicate cluster。
  pilot 必须计算发布字节 SHA-256、时间 pHash/embedding，人工复核同一动物、背景、镜头和连续事件；
  但这些仍只能解决 exact/near duplicate，不能恢复缺失的作者来源关系。
- 在 Meta 不提供 author/capture-group mapping 前，SA-V 命中只能保持 `pending/review_required`。严格计数时
  最保守做法是把全部 SA-V 候选视为 **1 个未验证 source group**；不能因为得到 305 个 ID 就贡献 305 个
  pair-support groups。可以向 `segment-anything@meta.com` 请求稳定 version/checksum manifest 和经过匿名化
  的 author/session group IDs，再决定是否放宽。

人物与隐私方面，dataset card 记录了 face-blurring model、crowdworker consent forms、第三方对所需
notice/consent 的承诺、撤回机会及内部 review；这比普通网络视频的 provenance 强。但官方没有声称零可识别
人物，也没有给每个 `video_id` 发布人物授权/同意记录，且 metadata 不标人、年龄或未成年人。因此人脸模糊
不能替代本项目的全长审核：任何未充分模糊的可识别人、儿童、旁观者、屏幕/住址等敏感信息继续 fail-closed；
只有 `cat|dog|person × action × Indoor|Outdoor` 经外部模型召回、两阶段人工逐窗确认、权利审核和
source/near-duplicate 闭包后，才可从 pending 进入候选审核。

SA-V 的下一步仅是一个**不计 coverage 的 305-ID feasibility pilot**：衡量 cat/dog/person 与 playing 的
命中率、解码/存储成本、人物权利拒绝率、近重复簇比例，并验证 frame-to-clip 与 provenance 流程；不能把
val/test 当成已经满足 C1 的数据，也不能因此启动训练。该审计完成时的 Commons 快照是
**1,130 titles**，训练 pair 为 `cat×indoors=30/10`、`cat×playing=5/10`、
`playing×indoors=5/10`，held `cat|playing|indoors=16/50`，**0 eligible C1**；紧随其后的台球
审核把当前快照推进到 1,132 / 124 approved / `playing×indoors=7/10`，但仍为 0 eligible C1。
本节没有新增 SA-V pending/approved media，所有 SA-V 结论均为 pending-only。

### 2026-08-14 执行结果：两条台球视频通过，pair support 从 5 增到 7

上述 billiards 路线已经按预注册边界执行，而不是停在搜索建议。exact import 最终只保留两条
CC BY-SA 3.0 NL 文件：Arie Bos 与 J. Heilbron。H.J. Robijns 文件页只有 Public Domain Mark、没有
本项目 Commons provenance gate 要求的非空 `license_url`，因此导入过程 fail-closed，未通过修改
通用许可门禁把它塞入数据。这个处理也与 Creative Commons 的官方说明一致：
[`Public Domain Mark`](https://creativecommons.org/public-domain/pdm/) 是标识已知公版作品的标签，
不是像 [`CC0`](https://creativecommons.org/public-domain/cc0/) 那样的权利放弃/贡献工具。

两条通过许可门的原 OGV 均重新下载并逐字节核对 Commons SHA-1；完整 20-frame 联系表和每条
`[8,10)` 秒、8 fps 的 16-frame 密集窗口显示：一名已具名成年人是唯一人物，持续演示台球击球，
场景稳定为室内，且所选窗口无剪辑。二者分别绑定
`commons-event:polygoon-arie-bos-billiards-1938-04-04` 与
`commons-event:polygoon-heilbron-billiards-1934-08-24`，没有因为同一馆藏/制片方而冒充同一事件，
也没有把 Arie 的 OGV/WebM 变体拆成两组。原媒体、review derivative、规范化 96×96×16 clip、
页面 revision、page ID、原 SHA-1、许可和审核图 hash 均记录在
[`commons-billiards-review/evidence.json`](artifacts/video-dit/commons-billiards-review/evidence.json)。

重建后的权威数据为：candidates **1,132 = 124 approved + 800 pending + 208 rejected**；v3 manifest
**124 records / 95 independent groups**，manifest SHA-256
`06b75c5684c5430f13ecd1639a1e5274a18c333610ab4e6afda2ac648dcc3479`，coverage SHA-256
`95fea2349d0e5a0c19c2ee8523d27bf978eea6bc153a943cec5adbc7a9cdf27a`。held
`cat|playing|indoors` 仍是 **16/50**；整组排除 held 后，训练 pair 为
`cat×indoors=30/10`、`cat×playing=5/10`、`playing×indoors=7/10`。因此仍是
**0 eligible C1**，这两条确实缩小了 pair 缺口，但不允许启动正式训练。

### 2026-08-14 执行结果：Internet Archive 定向 pilot 全部不进入训练

为避免只看 metadata，本轮增加 `prepare_archive_media.py`：冻结候选 bytes/hash，在网络请求前检查
总字节硬帽；只接受绑定 `archive.org/details/<id>` 与 `archive.org/download/<id>/<original>` 的
原文件；流式核对 IA `size + SHA-1 + MD5`，再记录本地 SHA-256 与 ffprobe 的 codec/尺寸/fps/帧数/
时长。任一下载、digest 或 probe 失败都会清理 staging，最终 manifest 明确
`training_ready=false`。

4 个 exact specs 中，两个现代家庭上传只声明 PDM，按上面的许可规则在 metadata 阶段拒绝；剩余
2 条共 75,256,255 bytes 在 100,000,000-byte hard cap 内完整验证。全长审核结论仍为 0 approved：

- `Playing with Ruby Dog Sep 2012 Philly` 的狗确实玩球，但全片为公园/喷泉户外并出现成年人，不能
  支撑 `dog|playing|indoors`。
- `Cat Neve hunting a Mechanic Toy -Dog` 明确室内，猫会跟随/潜行接近机械玩具狗；但完整联系表及
  0/4/8/16/22 秒附近密集窗口均没有清晰、持续的扑抓或玩耍，源标题本身也是 “hunting”，所以按
  action fail-closed，而不是为了补数改成 playing。

候选、媒体 manifest、两条本地媒体 SHA-256 与三张审核联系表的哈希由
[`archive-pair-pilot-v3/evidence.json`](artifacts/video-dit/archive-pair-pilot-v3/evidence.json)
绑定。该 pilot 没有写入 Commons manifest、没有改变 coverage，也没有触发训练。

### 2026-08-14：下一条可执行路线——三个独立日期的单人室内 bowling pending

本轮先锁当前 **1,132 candidates / 124 approved / 95 groups** 的规范 title 集合；按 Unicode title
升序、每行一个 title 且保留末尾换行的 UTF-8 列表 SHA-256 为
`9914f8391b183dfde64442a0d850ba09c305f5a57d6d5a3f389ced4df566ea62`。所有差集均相对这一快照，
搜索或分类成员只作召回，不作标签或批准决定。

cat 路线先做结构化负查，避免重复此前 44-file playing-cat 类和宽全文查询。Commons Action API 的
`Playing cats ∩ Outdoor cats`、`Playing cats ∩ Cats on grass`、`Cats with toys`、
`Cats and balls`、`Cats and catnip toys`、`Cats and stuffed animals` 的视频交集均为 0；
`Playing kittens` 唯一视频、`Cats playing with their prey` 两个视频都已经在 1,132-row 池中。
`Cat toys` 的 16 个视频也全部是池内已审核的 Finchen/Lotti 系列或 `round track toy`；
`Cats on grass` 的两个视频正是已经全长拒绝的 Lotti lying/sitting 片段。因此这些官方分类的高置信
**cat-playing 新 title 差集为 0**，不能通过换一个类别名重复膨胀 source groups。

接着审计未在旧计划中穷尽的 Commons 官方
[`Videos of bowling`](https://commons.wikimedia.org/wiki/Category:Videos_of_bowling)。其 16 个直属视频
相对 1,132-title 集合的精确差集为 **16/16**。公共领域老片继续受当前非空 `license_url` 门禁约束；
球道机器、空球道、第一视角和只见手/球的文件不能标为 person。对剩余具名成年运动员视频读取
Action API 的 page ID、原媒体 SHA-1、许可、作者、日期和 description，并直接审核官方原视频完整
2 fps 联系表。下面三条在不依赖标题猜测的情况下，同时具备单一成年主体、完整投球动作和明确室内
球道；它们最适合作为下一批 exact import 的 `pending/review_required`：

| pending 文件 | 官方固定身份与许可 | 原视频证据与 source/session 边界 |
|---|---|---|
| [`20220715 Kyle Troup two-handed bowling delivery.webm`](https://commons.wikimedia.org/wiki/File:20220715_Kyle_Troup_two-handed_bowling_delivery.webm) | page ID `120676309`；SHA-1 `651d94c548394b53458746f871cb879d2cc5ff16`；3,881,121 bytes；CC BY-SA 4.0；artist `RCraig09`；官方日期 2022-07-15 | 官方 description 明确 Kyle Troup 在 Delaware Mid-County Lanes 的 pro-am 做 two-handed delivery；完整 2 fps 联系表只有一名运动员且大部分背对镜头，投球从助跑到随挥连续，室内球道无歧义。只能 pending；选择两秒窗口时仍须排除脸部转向、其他参赛者进入和剪辑。 |
| [`20230630 Backup ball two-handed delivery - Eric Jones.webm`](https://commons.wikimedia.org/wiki/File:20230630_Backup_ball_two-handed_delivery_-_Eric_Jones.webm) | page ID `133882229`；SHA-1 `11d415ee95d53b33e6096aa385326a634ebf7445`；1,665,536 bytes；CC BY-SA 4.0；artist `RCraig09`；官方日期 2023-06-30 | 官方 description 明确 Eric Jones 在同一球馆另一场 pro-am 投出 backup ball，并说明原片已剪到 relevant content；完整联系表是单人背向镜头的连续投球。它与 2022 视频同 uploader/venue，但日期、运动员和赛事 session 不同；仍须媒体 pHash/镜头重叠审核后才能声明独立 group。 |
| [`20240226 François Lavoie.webm`](https://commons.wikimedia.org/wiki/File:20240226_Fran%C3%A7ois_Lavoie.webm) | page ID `146021730`；SHA-1 `48469e533a318757ac60f771875c909bc2545415`；3,918,130 bytes；CC BY-SA 4.0；artist `RCraig09`；官方日期 2024-02-26 | 官方 description 明确 François Lavoie 在同一 Delaware 球馆的 pro-am；完整联系表显示投球阶段是一名成年人、室内球道和持续动作，但结尾本人转脸且邻道出现局部人物。只能从前段选无脸、无旁观者的密集两秒窗口，否则 reject。 |

这三条不是三条已证明的独立 groups。预注册 grouping 应先按
`RCraig09 + Mid-County Lanes + capture date/event` 建 session key：三个日期暂为三个 provisional sessions；
同日的其他运动员文件（例如 2019-08-18 两条、2024-06-21 两条）必须各自合并为一个 session，不能按
运动员或文件数拆组。同一场馆背景会导致高视觉相似，必须在规范化前同时跑时间 pHash、embedding、
相邻帧和拍摄机位复核；若无法排除同一长录像的切片关系，就继续合并。CC BY-SA 4.0 是版权许可，
不等于人物 release；三人是官方 description 具名的成年职业运动员、处于公开 pro-am 场景且候选窗口
尽量不显示脸，这只降低而不消除人格权风险。任何儿童、清晰旁观者或正脸窗口继续 fail-closed。

可执行顺序为：exact import 三个规范 title并锁 page revision/SHA-1/LicenseUrl/attribution；生成完整联系表
与至少三个两秒候选窗口；先做人物权利和单主体审核，再做 action/scene 审核；最后做跨三条及与现有
124 clips 的 source/pHash 闭包。只有全部通过才可使 `playing×indoors` 从 7 最多增到 10；现在仍是
训练 pairs `cat×indoors=30`、`cat×playing=5`、`playing×indoors=7`，held
`cat|playing|indoors=16/50`，**0 eligible C1**，不得启动正式训练。

### 2026-08-14：标准视频数据集的媒体权利与 source-group 审计

这次只接受数据集维护方的下载页、许可、schema 或官方仓库作为证据，并把三个条件分开检查：媒体是否
明确获准用于本项目的训练且能否再分发；是否有稳定 media ID；是否公开了足以合并同一作者、参与者或
拍摄 session 的 group ID。代码仓库的开源许可不自动覆盖视频，annotation 的许可也不自动覆盖底层媒体。
结论是：这些标准集里没有一个可以在不接受额外协议、不补 grouping 或不改变目标语义的情况下，直接
并入当前可再分发的 Commons manifest。

| 数据集 | 官方权利与身份事实 | source/session 闭包与本任务结论 |
|---|---|---|
| [Ego4D](https://ego4d-data.org/docs/start-here/)（[metadata schema](https://ego4d-data.org/docs/data/metadata/)、[videos](https://ego4d-data.org/docs/data/videos/)、[data-license PDF](https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf)） | 官方要求先审阅并签署 data license，凭证经批准后发放且约 14 天失效；协议是不可转让、不可再许可的受限使用权，不是逐媒体 CC 许可。master metadata 有稳定 `video_uid`，并公开 `origin_video_id`、`video_source`、`fb_participant_id`、`video_component_uid`、`concurrent_video_set_id` 和 `physical_setting_name`；canonical clip 还能回溯 `clip_uid → video_uid`。 | **唯一仍值得执行的受限 annotation-first pilot**。group key 至少以 participant 为闭包，再用 origin video、concurrent set、physical setting 与时间连续性继续合并，不能按 clip 数计独立来源。只有实际使用者保存当时签署的协议/receipt，并确认本地非商业生成训练及 checkpoint 条款后，才能按前文 exact-version gate 下载；媒体、派生 clip 和 checkpoint 均不得默认公开。它仍只是候选发现路线，不证明含 cat/dog playing 或 indoor。 |
| [Charades](https://prior.allenai.org/projects/charades)（[README/schema](https://prior.allenai.org/projects/data/charades/README.txt)、[license](https://prior.allenai.org/projects/data/charades/license.txt)） | 官方许可允许 academic/non-profit/government research 和 evaluation，但明确禁止修改后公开数据及向第三方分发；只允许论文所需短片段或静帧。CSV 同时给稳定 video `id`、唯一 `subject` ID、15 类 indoor `scene`、script/descriptions 和时间 action，因此按 `subject` 合并可以闭合“同一演员多视频”泄漏。 | **provenance 最完整，但目标语义仍不成立**。官方 action vocabulary 中的 play 是 phone/camera 或 laptop，不是本计划的 generic playing，更没有 cat/dog；不能为补 `playing×indoors` 把词面命中重标。可作独立的 person-indoor 方法学对照，但不能并入当前三元组门禁，也不能发布其派生媒体包。 |
| [Something-Something V2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)（[research-use agreement](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf)） | 官方给 1–220847 的稳定视频编号和 train/val/test annotation；许可只授予内部、非营利 Research Use 的处理权，禁止发布、分发、转让或向第三方披露数据，并明确不保证众包内容的隐私、公开形象或第三方知识产权。 | 官网只报告由大量 crowd actors 创建，公开 annotation/download schema 没有逐视频 actor/uploader/session ID；数字 video ID 不能当 source group。主体又是手与日常物品，scene 也未形成可审计的 indoor/outdoor 标签。**权利受限、grouping 缺失、语义不匹配，排除。** |
| [AVA / AVA-Kinetics](https://sites.research.google/gr/ava/download/) | 官方 CSV 有稳定 YouTube `video_id`、timestamp、person box、action ID，以及仅用于相邻帧关联同一人的 `person_id`；下载页把其列出的 datasets 标为 CC BY 4.0，并说明 AVA 取 430 个连续 movie segments、AVA-Kinetics 又合入 Kinetics。 | `person_id` 不是跨视频演员/uploader/session ID；按 `video_id` 可阻止同一电影 clip 跨 split，却不能证明不同 YouTube ID、影片版本或 Kinetics 切片独立。官方页也没有逐影片的底层权利链或 subject×scene 标签。**不能把 annotation 页的统一声明当成逐电影来源审计，排除。** |
| [Kinetics-400/600/700](https://github.com/cvdfoundation/kinetics-dataset) | 官方 CVDF 仓库把 Kinetics 定义为 YouTube URL/clip 集，annotation 以 YouTube ID 和起止时间定位；当前镜像托管 K400/K700-2020 bytes，同时明确警告 K400 validation 大量进入 K700 training，跨版本直接训练/评测会泄漏。 | YouTube ID 能把同一原视频的多个时间窗合并，却没有 uploader/capture-session group、逐 item 许可或统一媒体再分发权利；不同频道转载/剪辑也不会靠 ID 自动闭合。类别只有单一 action class，无 subject×scene 真值。**可做受控动作预训练对照，不能作本次 C1 证据。** |
| [TAO](https://taodataset.org/)（[官方下载说明](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md)、[仓库许可](https://github.com/TAO-Dataset/tao/blob/master/LICENSE)） | TAO 联邦组合 Charades、LaSOT、ArgoVerse、AVA、YFCC100M、BDD100K、HACS 等不同来源。官方仓库的 MIT 文件明确只适用于 code；下载说明甚至保留媒体权利人/出镜者的删除请求入口，没有给全部视频一份统一媒体许可。 | dataset/video ID 可以回到成员集，却不能替代每个成员集及每条原媒体的许可、uploader/session 和近重复闭包；object tracks 也不是 action/scene 标签。**异构权利链无法以 TAO 名义一次闭合，排除。** |
| [BDD100K](https://bdd-data.berkeley.edu/download.html) | 官方下载页许可研究、教育、非营利用途下使用、修改和分发数据，并提供稳定的 40 秒 driving-video 文件及 GPS/IMU；商业权利仅限指定成员或另行许可。 | 公开下载/schema 没有可把连续行车片段合并的 driver、vehicle、route 或 capture-session ID；文件名不是独立 session 证明。内容又是 ego-driving outdoor scene，不提供目标 animal/person playing 标签。**许可比多数 web 集清楚，但 grouping 与语义均不合格，排除。** |

因此下一步优先级不变：若实际使用者愿意接受并保存 Ego4D 当时的限制性协议，就执行前文最多 10 UID、
12 GB 上限的 metadata-first pilot；否则继续 Commons 的逐媒体许可路线。Charades 只保留为 person-indoor
对照设计参考，不当作缺口数据；其余标准集不下载。该审计完成时的快照仍是
**1,132 candidates / 124 approved / 95 groups**，held `cat|playing|indoors=16/50`，训练 pairs
`cat×indoors=30/10`、`cat×playing=5/10`、`playing×indoors=7/10`，仍为 **0 eligible C1**；
紧随其后的 bowling 执行结果如下。

### 2026-08-14 执行结果：三条 bowling 视频通过，`playing×indoors` 达到预注册下限

上述 bowling 路线已经完整执行。三个规范 title 均以 exact import 进入 pending，锁定 Commons page ID、
页面 revision、原媒体 SHA-1、CC BY-SA 4.0 license URL、署名和实际审核 derivative SHA-256；随后分别生成
完整时长 12-frame 联系表和 8 fps 的密集两秒窗口。人工复核确认三个窗口都只有一名具名成年人，动作从
助跑、出手到随挥连续，场景为室内球道且没有剪辑。Kyle Troup 与 Eric Jones 使用 `[1,3)` 秒窗口，
François Lavoie 使用 `[0.5,2.5)` 秒窗口以避开结尾转脸和邻道人物。

三条来自同一 uploader 和球馆，但分别属于 2022-07-15、2023-06-30、2024-02-26 的不同运动员/赛事日期；
因此先按 `uploader + venue + capture date/event` 建三个保守 session group，而不是按文件或运动员名字直接
计数。规范化后它们的 pHash 分别为 `dc1c3c76ce5e7e3c`、`1b3b1b5b1b39292f`、
`7c7d5c1cbc9c939a`，并未触发现有 Hamming ≤ 4 的近重复闭包。完整 provenance、clip SHA-256、审核理由
和 6 张已跟踪审核图由
[`commons-bowling-review/evidence.json`](artifacts/video-dit/commons-bowling-review/evidence.json)
绑定。

重建后的当前权威状态是：candidates **1,135 = 127 approved + 800 pending + 208 rejected**，candidate
SHA-256 `c89bc31afa06f15fd9a3bf157b92f0f35c90ad9beeee4881beef2fddb8103b93`；v3 manifest
**127 records / 98 independent groups**，manifest SHA-256
`fa500e664f2c8518c87085c469f8199c52de6f9e84a79db08b9c7a3fa115242c`，coverage SHA-256
`4bf2778af68423465e45c6419c049f158457f07e8187fe41b9ca298ffef85886`。held
`cat|playing|indoors` 仍为 **16/50**；整组排除 held 后，训练 pair 为
`cat×indoors=30/10`、`cat×playing=5/10`、`playing×indoors=10/10`。这轮确实关闭了一个 pair
缺口，但 held triple 还差 34 个独立 groups，`cat×playing` 还差 5 个，故仍为 **0 eligible C1**。
没有生成 split、没有启动正式训练，也不能据此回答 unseen compositional generalization。

### 2026-08-14：Structured Data on Commons 的 cat-playing 严格负结果

本轮把结构化审计的基线更新到当前 **1,135** 个规范 title；title 按 Unicode 升序、逐行写入并保留
末尾换行后的 SHA-256 是 `13e97e60a8ec003bc4f5eb8c913fdfb6d711206025841f8ec238e288a40a1346`。
只使用 Commons/Wikidata 的官方 Action API 数据，不把搜索排序、文件名或 category 命中当成动作真值。

Commons 搜索谓词
[`haswbstatement:P180=Q146 filetype:video`](https://commons.wikimedia.org/w/index.php?search=haswbstatement%3AP180%3DQ146+filetype%3Avideo&title=Special:MediaSearch&type=video)
返回 **397** 个带 Structured Data on Commons `depicts: cat (Q146)` 的视频。逐个读取这 397 个
MediaInfo entity 的全部 `P180` statements 后，共有 49 个不同的 depicts item；其中包括 kitten、
drinking、grooming、sleeping、eating、yawn、interior space 和 outside，但没有
[`playing (Q1150958)`](https://www.wikidata.org/wiki/Q1150958)。官方搜索中
`P180=Q146 ∩ P180=Q1150958 ∩ filetype:video`、cat 与 `game (Q11410)`、cat 与
`play and playthings (Q64893424)` 的交集也都为 **0**。这不是说 397 条视频中绝对没有玩耍，而是说明
SDC 目前没有提供可据以补充 playing 标签的结构化正例；缺失 statement 不能反推负标签。

相对 1,135-title 基线，397 条中只有以下 **8 个精确 title 差集**。page ID、原媒体 SHA-1 和许可均来自
各文件页的 `imageinfo/extmetadata`；其中 Commons 对 public-domain 文件没有返回 `LicenseUrl`，所以它们
还会触发当前非空 license URL 的 fail-closed 门禁。

| SDC cat 视频差集 | page ID / 原媒体 SHA-1 / 许可 | 为什么不是新的 cat-playing 候选 |
|---|---|---|
| [`Cat birth 2.ogv`](https://commons.wikimedia.org/wiki/File:Cat_birth_2.ogv) | `1003409` / `d4d5174d8e9a53b20168115ed2618c97bbc2c885` / Public domain，API `LicenseUrl` 为空 | 官方 description 与类别都是猫分娩，并含猫和狗；不是 playing、不是单主体。 |
| [`Black and white cat drinks from a puddle.ogv`](https://commons.wikimedia.org/wiki/File:Black_and_white_cat_drinks_from_a_puddle.ogv) | `11136217` / `843688c33f6ea3c7b94abf12818788b5ead25520` / Public domain，API `LicenseUrl` 为空 | 官方 description、`Videos of drinking cats` 类和 SDC `drinking` statement 一致，只证明 drinking。 |
| [`Yawning kitten.ogv`](https://commons.wikimedia.org/wiki/File:Yawning_kitten.ogv) | `22623990` / `8418efa0bf0600ae8ba5043ad015c7537f5f3b32` / Public domain，API `LicenseUrl` 为空 | 官方 description、类别和 SDC statement 都是 yawn；不是 playing。 |
| [`The boxing cats (Prof. Welton's).ogv`](https://commons.wikimedia.org/wiki/File:The_boxing_cats_(Prof._Welton%27s).ogv) | `23863226` / `6e68ef52e7e03d33fb9741dc6b1a30de1255a10a` / Public domain，API `LicenseUrl` 为空 | 官方页把它标为 1894 年 Edison/Library of Congress 的默片，标题是复数 cats；既没有 SDC playing statement，也不满足自然单猫、稳定两秒 playing 契约，不能把 staged boxing 重标为玩耍。 |
| [`INTERNATIONALE KATTENTENTOONSTELLING IN DE DIERENTUIN-PGM4011556.webm`](https://commons.wikimedia.org/wiki/File:INTERNATIONALE_KATTENTENTOONSTELLING_IN_DE_DIERENTUIN-PGM4011556.webm) | `84558066` / `3ad6802c06b3f9865a969f2411e6594c79c046e7` / Public domain，API `LicenseUrl` 为空 | 官方 description 是 1939 年国际猫展，明确大量猫与参观者；多主体、历史剪辑，且没有 playing 证据。 |
| [`Internationale kattententoonstelling.webm`](https://commons.wikimedia.org/wiki/File:Internationale_kattententoonstelling.webm) | `84580846` / `797054845be1c6071fb5d25fdd5d2a9104dffe61` / Public domain，API `LicenseUrl` 为空 | 官方 description 是 1948 年海牙猫展，类别含 cat show/grooming；多主体且不是 playing。 |
| [`A male cat grooms a female kitten 2021-08-25 August 25.webm`](https://commons.wikimedia.org/wiki/File:A_male_cat_grooms_a_female_kitten_2021-08-25_August_25.webm) | `109487074` / `fe9330a761b094af2e6b9cee91b77a8eb2a45057` / Public domain，API `LicenseUrl` 为空 | 官方 description 与 `Videos of grooming cats` 类明确是两只猫 grooming；动作和单主体都不合格。 |
| [`Reismand met kat die onraad ruikt en er zo snel mogelijk weer uit wil.ogg`](https://commons.wikimedia.org/wiki/File:Reismand_met_kat_die_onraad_ruikt_en_er_zo_snel_mogelijk_weer_uit_wil.ogg) | `153466011` / `72aaccdf470f9e6560a91a3f749bc5c35607af15` / CC BY-SA 4.0 / [`LicenseUrl`](https://creativecommons.org/licenses/by-sa/4.0) | 荷兰语官方 description 是猫在藤制旅行篮里并想尽快出去；SDC depicts 为 cat + pet carrier，没有 playing statement 或稳定玩耍动作证据。 |

另用 Commons category namespace 对德、法、西、葡、荷、意、波、俄、乌、中、日、韩、土耳其语的
“猫 + 玩耍”词形做官方 API 查找，没有发现一个独立于既有英文 category tree 的猫玩耍视频类。
这与此前已经闭合的 `Videos of playing cats` 44 files、`Cat toys`、prey、Lotti/Finchen 系列结果一致；
多语言词面不是新的 source tree，更不能绕过 page ID/SHA-1 差集。

因此本轮是严格的 **0 个高置信新增 / 0 个可执行 pending**。八个差集命中都不写入 candidates，
不改变 approved、groups、coverage 或 pilot split。当前权威状态仍为 **1,135 candidates / 127 approved /
98 groups**，held `cat|playing|indoors=16/50`，训练 pairs 为 `cat×indoors=30/10`、
`cat×playing=5/10`、`playing×indoors=10/10`，仍是 **0 eligible C1**。

### 2026-08-14：预注册 low-coverage C1 pilot 已执行，结果失败

按用户明确授权，在不把它冒充正式确认实验的前提下，用当前 provenance-locked 127-record v3
manifest 启动了一次低 coverage 诊断。训练开始前，协议已经以 Git commit `536d0f9` 固化；其
SHA-256 为 `5ace4b109ec353223c03ab631641dead334fdba5d9fb6bba621c26a0c53a4313`。确定性 split
使用 seed `20260814`，得到 train/validation/IID/C1=`94/9/5/19`；held triple 的 16 个独立组，
以及与它共享 source、显式 group 或 pHash Hamming≤4 的整个传递闭包都未进入训练。pilot 训练侧
pair support 是 `cat×indoors=23`、`cat×playing=5`、`playing×indoors=8`，因此 spec 明确写入
`evaluation_tier=pilot` 与 `confirmatory_gate.passes=false`。

模型从零训练：KL-f8 latent `16×12×12`、patch 3、hidden 256、8 heads、4 组 spatial/temporal
blocks、CLIP tokens-v2 cross-attention、velocity prediction，共 11,988,772 参数。固定训练
10,000 updates、batch 8、lr `1e-4`、EMA `0.99`，耗时 2,267.9 秒。只使用预注册的 final
step-10000 EMA checkpoint，SHA-256 为
`25147a4cc6a273205c2acbaf303558872e0345cbd11c68bb1afa0417971b5db5`；first/last 500-step
loss 均值从 `0.58087` 降到 `0.24438`。这证明优化确实推进，不证明自由生成成功。

评估严格生成 12 prompts×4 seeds=`48/48`，DDIM-50、start 999、CFG 2.5；没有额外或缺失 MP4，
也没有按结果挑 checkpoint、prompt、seed 或视频。两名彼此隔离的盲评员均完整检查 48×16 帧，
但 48 项中只有 1 项三因子意见完全一致。把分歧按错误计算后：

| split | samples | subject | action | scene | unanimous triple | Wilson 95% |
|---|---:|---:|---:|---:|---:|---:|
| IID | 20 | 0.0% | 5.0% | 55.0% | 0.0% | [0.0%, 16.11%] |
| C1 | 28 | 7.14% | 0.0% | 28.57% | 0.0% | [0.0%, 12.06%] |

预注册 full-grid strict derangement 还有一个设计缺陷：每个 seed 有 7 个同一 C1 factor triple、只有
5 个 IID triples，所以 28-vs-20 的全网格一对一全异匹配在数学上不可能。揭盲器先 fail-closed；
随后只对不看结果、按 factor 计数确定的最大平衡子集做严格 derangement，40 项纳入、8 个 C1 项明确
排除。该子集 matched 与 deranged triple accuracy 都是 0%，gain=0。这是透明的 pilot 修补，不能
反称原预注册控制已完整满足。独立 frozen CLIP 诊断同样失败：top-1 `14.58%`、shuffled `12.5%`、
gain 仅 `2.08` 个百分点，mean margin `-0.01259`；最近训练视频相似度均值/最大为 `0.892/0.933`。
对最高三组的逐帧联系表核查未见逐帧像素复制，但生成结果是强烈形变、碎片化拼贴，并吸附训练视频的
宽泛动物/场景纹理。

因此本 pilot 的结论是 **失败**：训练 loss 下降，但 IID 和 C1 的自由文本绑定都没有建立，不能把失败
单独归因于 unseen composition。最合理的并列解释是：94 个训练 clips 对 11.99M from-scratch 模型过少；
生成质量/容量与优化仍不足；`cat×playing=5`、`playing×indoors=8` 的 pair support 太弱。完整报告、
损失抽样、确定性第一条生成 MP4 与最近邻图已跟踪在
[`artifacts/video-dit/c1-pilot-v1/results/`](artifacts/video-dit/c1-pilot-v1/results/)。正式结论仍必须
等待 held≥50、每个训练 pair≥10、零 source/event/near-duplicate 泄漏后重新预注册并从头执行。

### 2026-08-14：现有 pending cat 池的高精度复审仍不足以触发正式门禁

在 low-coverage pilot 失败后，本轮没有继续扩大低精度网页搜索，而是回到当前锁定的
1,135-row Commons candidates 快照，只复审其中 `review_status=pending` 且 `subject=cat`
的 **563** 条。为避免把检索 query 当成动作真值，筛选只使用规范 title 和 Commons 官方
description；`play/toy/ball/laser/mouse/string/feather/chase/wrestle/romp/game/kitten` 词面共命中
**89** 条。

这 89 条中的高分项主要是与目标无关的词面假阳性，例如球类运动、玩具乐器、游戏预告、会议录屏、
历史片和多猫合集。真正带猫动作证据的少数项也不能提供新独立 group：

- `Play fight between cats` 的官方 description 明确是同窝两只幼猫，不满足单主体契约；
- `2023-12-16 at Kitten Academy` 明确是多只 foster cats 与人的长视频，不满足稳定单猫窗口的先验；
- `Interframe motion interpolation` 引用第三方 Pexels 猫玩玩具片段，但 Commons 文件本身是算法对比派生版，
  不能代替原媒体许可与 source/event 锁定；
- 大量 `LJUBLJANA doma` 项来自同一长期上传/动物系列，即使其中个别窗口经全长复审后可重标为 playing，
  在没有新的动物身份和拍摄 event 证据前也只能作为一个保守 group，不能用文件数膨胀 coverage。

因此，对当前 pending 池继续调低排序阈值不是可行的正式数据路线：它会增加人工审核量，但不会可靠地
把 held `cat|playing|indoors` 从 **16** 提升到 **50**，也不会把训练侧 `cat×playing` 从
**5** 提升到 **10**。正式实验仍精确缺 **34 个 held 独立 groups** 和 **5 个 non-held
`cat×playing` groups**。下一批数据必须来自新的、逐文件权利清晰且能稳定归并动物身份/拍摄事件的来源；
若没有这类外部授权或用户自有视频，则不得通过放宽单主体、动作、scene 或 group 定义来伪造门禁通过。

### 2026-08-14：Internet Archive 许可过滤结果与正式数据阻断

本轮不再重复已经闭合的 Commons 分类树，只查询 Internet Archive 官方
[`Advanced Search API`](https://archive.org/advancedsearch.php) 和逐 item
[`metadata API`](https://archive.org/metadata/catvideo_0003)。对 `cat playing`、`kitten playing`、
`cat plays`、`kitten plays`、`cat with toy`、`cat toy`、`playing with a cat`、
`playing with cat`、`playing with kitten` 九组 title/description 精确词组，加
`mediatype:movies AND licenseurl:*` 后，官方 API 分别返回 `8/4/5/0/0/3/2/0/0` 条。逐 item 排除
NC、ND、NC-SA、NC-ND、仅 Public Domain Mark、动画/音乐/转载和非真实猫后，只剩 **1 个可执行
pending、0 approved**；它不在当前 1,135-row title 池中：

- [`catvideo_0003`](https://archive.org/details/catvideo_0003)：官方 metadata 的 creator 为
  `AprilDolphin`、date 为 `2020-02-06`、许可为
  [`CC0 1.0`](https://creativecommons.org/publicdomain/zero/1.0/)。原文件
  `20200206_174812.mp4` 为 163,639,292 bytes、75.78 s、SHA-1
  `295ee52bed9e1f3b1bcdb479357cd29eac08518c`、MD5
  `ef5868a16164bafa01d84bbfa65f5361`；IA 审核 derivative
  `20200206_174812.ia.mp4` 为 7,442,654 bytes、75.83 s、SHA-1
  `243acffbc7c472da75f4bb89b9a3c8c7b8d6af03`。本轮读取的 7,074-byte metadata 响应 SHA-256 为
  `347034513f6b3011dd551cf69b774e27836c2cf0d09eb1c6be527a5037bfa301`。title 是
  `20200206 174812`，description 只写 `Footage of cat playing`，所以 subject/action 只是召回
  证据，scene 尚未证明。为避免复制公开邮箱，source key 取 uploader 精确 UTF-8 值的 SHA-256：
  `archive-uploader-sha256:898c6f29f8ed4b4484fd62bccb7f74a83c35f686599055c2f0006deecc588ddf`，
  再与 creator/date/item ID 联合闭包；本 item 最多贡献 **1** 个 provisional group，不能按原文件、
  derivative 或多个窗口重复计数。

exact import 必须固定 `archive.org/details/catvideo_0003` 与
`archive.org/download/catvideo_0003/20200206_174812.mp4`，把单 item 下载硬帽显式提高到
`170,000,000` bytes，先核对 IA size/SHA-1/MD5，再流式重算本地 SHA-256 与 ffprobe。随后以完整时间
联系表和密集窗口确认：单只真实猫、连续至少两秒主动玩耍、明确 indoor/outdoor、无剪辑、可识别人或
敏感屏幕；最后与该 uploader 的其他 items 及现有 127 clips 做 source/event、时间 pHash 和 embedding
闭包。任一项不明即 reject，不得因 description 含 playing 直接批准，也不得看片后选择性分配到 held
或 train。

其他高相关结果不能使用：`lady_20191207` 与 `bandit_baby` 只有
[`Public Domain Mark`](https://creativecommons.org/public-domain/pdm/)；`GodingFamilyHomeMovie09`、
`MsKitka`、`catplayssticksgamewithowner` 分别是 CC BY-NC、CC BY-NC-ND、CC BY-NC；
`cat-playing-a-piano-public-domain` 虽标 CC BY 4.0，但官方 metadata 明确是 1931 动画的 colour
restoration/manipulation。不能把 PDM 当授权、删除 NC/ND 条件或把动画当真实 held triple。

因此当前正式门禁仍是 held `cat|playing|indoors=16/50`（缺 **34**）、训练侧
`cat×playing=5/10`（缺 **5**）、`playing×indoors=10/10`，仍为 **0 eligible C1**。即使唯一 IA
pending 审核通过，也远不能补齐门禁。Commons 官方分类/SDC 与本轮 IA 许可过滤都没有发现可一次扩到
确认性规模的公开来源；下一条现实路径必须是用户自有或逐视频明确授权、允许本项目训练及派生物分发的
拍摄征集：至少另取 34 个 held 独立猫/家庭/session groups，并另取 5 个不属于 held triple 的
`cat×playing` groups。每条仍须稳定 media ID、书面许可/署名、原媒体 digest、
uploader/animal/session group 和全长审核，不能把同一家庭、同一只猫或同次拍摄拆成多个 groups。

### 2026-08-14 执行结果：IA `catvideo_0003` 只补入一个 non-held group

上述 IA 条目随后已按 exact ID 执行，不再只是 pending 计划。`prepare_archive.py` 从官方
metadata API 生成 1 条锁定 candidate，candidate/spec SHA-256 分别为
`1f446272c30b9fbcd8de2e08d2a9d7fd6ce5cca520009ac00f8c4306a9cf4a51` 和
`2a33af0a424c381ab5b65e869a2131f5803906c3a6a1339be68088698fdc7485`。下载前的 170,000,000-byte
硬帽大于锁定的 163,639,292 bytes；流式下载后 size、SHA-1 和 MD5 均与 IA 文件记录一致，
本地 SHA-256 为 `73b0ef30c798ba3c62467609484718c36c412115dbcab6af0684f62b82c5e216`。

完整 75.776122 秒视频显示一只虎斑猫在户外树干与灌木下活动。全长 24-frame 联系表与
`[10,12)` 秒的 16-frame/8-fps 密集窗口显示它连续起身、伸爪和拨弄低处枝条；窗口内只有一只猫、
无剪辑，地面、树干与灌木使 outdoor scene 无歧义。因此它通过审核的标签是
`cat|playing|outdoors`，**不是** held `cat|playing|indoors`。选定 clip 规范化为 16 帧、8 fps、
96×96，SHA-256 `7f18a83afa469c6ba678856130b91bd9c3e3db7079fbfd9e55cb3444c5802dbb`，
pHash `860cbc6860a5a521`；它与原 127 clips 的最小 pHash Hamming 距离为 21。

分组沿用 importer 的 privacy-safe 契约：上一节中的 uploader hash 是搜索时的实时审计下界，
不持久化；真正写入 manifest 的 `source_id`/`duplicate_group` 是公开 creator `AprilDolphin`
的规范化 SHA-256 group，以后发现同 creator/item series 必须传递合并。审核证据在
[`archive-cat-playing-review-v1/evidence.json`](artifacts/video-dit/archive-cat-playing-review-v1/evidence.json)，
并跟踪全长联系表、密集窗口和实际训练 clip。

重建 coverage 后，合并 provenance manifest 为 **128 records / 99 independent groups**，manifest
SHA-256 `75775aa9776a8cb3cd1448be25b17ac3c229578bba8dfce6db637101f18836d8`，coverage SHA-256
`36ae275e03d972648ec08c943c89f7424b962665e210c04689296ba3c4ce4071`。held 仍为 **16/50**；
训练侧现为 `cat×indoors=30/10`、`cat×playing=6/10`、`playing×indoors=10/10`。所以这轮只把
non-held `cat×playing` 缺口从 5 缩到 **4**；held 仍缺 **34** groups，`eligible_c1_triples=[]`，
仍不得启动正式训练。

### 2026-08-14：授权自有视频的 fail-closed 导入入口已实现

公开来源审计已经证明，剩余 34 个 held groups 不能靠调低 Commons/IA 搜索阈值获得。为了让
用户自有或逐条书面授权的素材能直接进入同一 provenance 契约，现已新增
`prepare_local_contribution.py`，但它不把“用户给了一个文件”等同于“可训练”。

契约分为三个公开 CLI seam：

1. `create SPEC RECEIPT OUTPUT`：校验原媒体 SHA-256、UTC 时间、权利人/署名、许可 URL，以及
   `model_training/create_derivatives/redistribute_normalized_clips` 三个显式 grant；绑定稳定的
   contributor/animal/capture-session IDs，产生 16 帧、8 fps、96×96 的两秒 clip，并把锁定的
   完整原视频、全长导航联系表和密集窗口一起保存在 bundle 中。这一阶段固定为
   `training_ready=false`；24 帧联系表只用于导航，不能代替完整播放原视频。
2. `approve STAGED REVIEW OUTPUT`：必须用同一 manifest/receipt/clip/两张联系表的 SHA-256 绑定
   人工决定，且 full-duration、single-subject、continuous-action、scene 和 no-edit 五个审核门坎
   都必须显式为 `true`。
3. `merge BASE APPROVED... --output NEW_DATASET`：不原地修改 base，而是在临时目录重新解析 rights
   receipt 的授权、身份和媒体契约，重验 clip、完整原视频、联系表、human-review 和 manifest 绑定后，
   把每个 approved bundle 的完整审计副本写入 `contribution-audit/`，再原子提交可移动数据树。
   `BASE` 只能是固定的 pristine Commons 数据树；CLI 拒绝把旧 merge 结果再次作为增量 BASE，扩容时
   必须从同一 BASE 一次传入全部 approved bundles，防止上一代完整审计副本在下一代静默丢失。

模板位于 `code/video_dit/configs/local_contribution_{spec,receipt,review}.example.json`，完整命令见
`code/video_dit/README.md`。`RealVideoRecord` 现在可选持久化 `contribution_id`/
`rights_receipt_path`/`rights_receipt_sha256`，老 manifest 仍可读，新 contribution 则必须三字段 all-or-none。
相同的稳定 ID 会导出相同 source/session group；但代码无法自动发现贡献者故意或误把同一动物、
家庭或连续事件换成新 ID，因此正式 split 前仍保留人工身份/事件审计和 pHash/embedding 近重复门禁。
这一实现消除了授权素材到 manifest 之间的手工 JSON 环节，但没有伪造新数据：当前仍是
128 records / 99 groups，held 16/50，`cat×playing=6/10`，0 eligible C1。
