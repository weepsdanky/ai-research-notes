# Video DiT Learning Lab

这是一个与 `code/image_dit` 完全分开的、可以在 Apple Silicon Mac 上从零训练的教学型 Video DiT。它生成的主结果是一个真正沿时间变化的 MP4：猫、狗或穿彩色外套的人会在雪地、草地、夜景或影棚中行走、弹跳或静止。扩散去噪过程另存为诊断 GIF，不会再冒充“动画”。

这不是把图像 DiT 连续跑八次。视频始终采用 `[B,C,T,H,W]` 张量；每一帧独立 patchify，完整空间 Transformer 和完整时间 Transformer 交替运行，时间注意力会同时看到同一空间位置的全部帧。

## 这套代码保留了什么

| 论文设计 / trick | 可读实现 |
|---|---|
| Latte Variant 1（空间块→时间块交替） | `video_dit.py:VideoDiT.blocks` |
| uniform per-frame 2D patch embedding | `VideoDiT._patchify` |
| 固定 2D 空间 + 1D 时间 sin/cos 位置 | `get_2d_sincos_pos_embed`、`get_1d_sincos_pos_embed` |
| diffusion timestep + prompt condition | `TimestepEmbedder`、`StructuredConditionEmbedder` |
| adaLN-Zero residual gates | `AxisBlock` |
| 零初始化输出层 | `FinalLayer`、`initialize_weights` |
| epsilon prediction（论文默认）/ v-prediction（质量消融） | `diffusion_loss`、`--prediction-type` |
| learned-range variance + VLB | `--learn-sigma`、`DiffusionSchedule.model_variance` |
| classifier-free guidance | faithful 档默认 10% condition dropout、`forward_with_cfg` |
| EMA | faithful 档默认 0.9999；concept 档复制 raw 权重 |
| ancestral DDPM 与 DDIM | `sample_loop`、`ddim_sample_loop` |
| 可恢复训练 | 完整 model / EMA / optimizer / RNG checkpoint |
| 真实视频输出 | `save_video_mp4`（H.264、8 FPS） |

主要依据是 [Latte 论文](https://arxiv.org/abs/2401.03048) 与其[官方实现](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/models/latte.py)。完整的设计取舍、一手来源、计算预算与能力边界见 [`../../docs/case-studies/video-dit-mac-design.md`](../../docs/case-studies/video-dit-mac-design.md)。

## 为什么先用可控合成视频

训练目标不是 MNIST，也不是彼此无关的静态图片。`dataset.py` 会实时渲染有连续轨迹的 32×32 RGB clip，并提供四个可检查的条件：

```text
subject = cat | dog | person
color   = orange | brown | red | blue | black
motion  = walk_right | walk_left | bounce | stand
scene   = snow | grass | night | studio
```

英文或中文自然短句会被映射到这套有限语法，例如：

- `an orange cat walking right in the snow`
- `a brown dog walking left on grass`
- `a person in a red coat walking in the snow to the right`
- `一只黑猫在雪地里向左走`

UI 会显示解析后的四个条件。这是为了在一台 Mac 上学清楚 Video DiT，而不是假装这个 248 万参数的小模型拥有开放词表。未出现在上述视觉词表里的对象不会被训练出来。

parser 是关键词映射，不是文本编码器：必须写出 cat / dog / person（或已支持的中文别名），否则报错；未知形容词会被忽略。缺省动作是向右走，缺省场景是雪地，缺省颜色是 person 红色、动物橙色。因此 UI 中的 parsed factors 是实际条件，不要从原始句子自行推测。

## 安装与测试

需要 Python、PyTorch 与 `ffmpeg`。仓库使用 `uv` 锁定依赖：

```bash
cd code/video_dit
uv sync --extra dev
uv run pytest -q
```

媒体测试会实际调用 `ffmpeg` / `ffprobe`，并验证 MP4 可以解码且包含正确数量的时间帧，而不是 mock 掉编码器。

## 先做 dry-run

CPU 和 MPS dry-run 都只进行一次真实 optimizer update，不会偷偷开始长训练：

```bash
uv run python train.py --dry-run --device cpu \
  --batch-size 2 --output-dir runs/dry-cpu

uv run python train.py --dry-run --device mps \
  --batch-size 2 --output-dir runs/dry-mps
```

每个目录都包含训练恢复用 `video_dit.pt`、安全推理用 `video_dit_inference.pt`、`summary.json` 与 `step_log.csv`。dry-run 的 checkpoint 只证明数据、模型、loss、反传和 checkpoint 链路正确；它没有学会画东西，不能当演示模型。

## 已完成的本机训练

当前可直接演示的是明确标注的 **canonical overfit + concept tricks + v-prediction** 实验：

```bash
uv run python train.py \
  --mode concept --curriculum canonical --prediction-type velocity \
  --device mps --max-steps 3000 --batch-size 8 \
  --hidden-size 128 --num-pairs 4 --num-heads 4 --patch-size 4 \
  --learning-rate 2e-4 --timesteps 1000 \
  --ema-decay 0.99 --horizontal-flip-prob 0 \
  --sample-every 1000 --checkpoint-every 1000 \
  --output-dir runs/canonical-velocity
```

这次完整运行的恢复 checkpoint 是 `runs/canonical-velocity/video_dit.pt`，同权重的可携带安全推理版是 `demo_artifacts/video_dit_inference.pt`：2,472,112 参数，3,000 updates，最后一个 mini-batch loss 为 0.01833。它只反复学习三段固定 clip，不是留出集、多样性或组合泛化结果。使用 seed 42、100-step DDIM 与有效 CFG=1 生成三个训练内 prompt 后，简单运动加权质心的水平位移为：

| prompt | 期望方向 | 测得位移（pixel） |
|---|---:|---:|
| orange cat / snow | 右 | +6.58 |
| brown dog / grass | 左 | -4.36 |
| red-coated person / snow | 右 | +7.49 |

位移符号和三个条件一致，说明 MP4 确实有时间运动；该指标是这三个 overfit 样本的工程诊断，不是视频质量基准。仓库内可直接查看的生成文件在 `demo_artifacts/samples/`：

- `model_cat_orange_walk_right_snow_seed-42.mp4`；
- `model_dog_brown_walk_left_grass_seed-42.mp4`；
- `model_person_red_walk_right_snow_seed-42.mp4`；
- 每个 MP4 的 `*_denoising-diagnostic.gif`，以及 `contacts/` 联系表与汇总 `temporal_metrics.json`；包含完整请求配置的逐次 JSON 保留在本机 `runs/`。

完整 optimizer/RNG checkpoint 与日志仍在被 Git 忽略的 `runs/`；安全推理权重、代表性 MP4、诊断 GIF、联系表和校验值已随仓库交付，见根目录 `ARTIFACTS.md`。

## 其他训练配置

`mode` 和 `prediction_type` 是正交开关：`concept` / `faithful` 控制 learned variance、condition dropout/CFG 训练能力与 EMA；`epsilon` / `velocity` 单独控制扩散预测目标。上面的已训模型因此是 concept + velocity，不能称为 Latte 论文 recipe。

Latte-faithful 档默认 epsilon，保留 learned variance、VLB、10% condition dropout、CFG 与 EMA，适合审计论文技巧：

```bash
uv run python train.py \
  --mode faithful --curriculum canonical --device mps \
  --max-steps 60000 --batch-size 16 \
  --hidden-size 128 --num-pairs 4 --num-heads 4 --patch-size 4 \
  --learning-rate 1e-4 --timesteps 1000 \
  --sample-every 5000 --checkpoint-every 5000 \
  --output-dir runs/faithful-60k
```

上面的 60k 是尚未完成的长程实验建议，不是已实测质量结果。当前 CLI 默认是 `faithful + canonical + epsilon + 3000 steps + batch 8`；它保留论文参数化，但不保证在同样 3,000 steps 下达到上述 concept + velocity checkpoint 的可辨识度。checkpoint 会记录 mode、curriculum 和 `prediction_type`，推理自动采用对应参数化。

canonical 验收通过后，再单独运行更难的 `factorial` 课程；它实时组合全部主体、颜色、动作和场景，用于学习组合泛化：

```bash
uv run python train.py \
  --mode faithful --curriculum factorial --device mps \
  --max-steps 100000 --batch-size 16 \
  --sample-every 5000 --checkpoint-every 5000 \
  --output-dir runs/factorial-100k
```

checkpoint 和 `summary.json` 都记录 curriculum；恢复训练时必须显式使用相同课程，避免把两种实验混成一条不可解释的曲线。

想先只看核心架构，可用 concept 档：固定 posterior variance、无 condition dropout/CFG，EMA 接口退化为每步复制 raw 权重，便于保持相同 checkpoint 格式。它仍可以和 `--prediction-type epsilon` 或 `velocity` 任一目标组合。

```bash
uv run python train.py \
  --mode concept --device mps --max-steps 10000 --batch-size 16 \
  --sample-every 2000 --checkpoint-every 2000 \
  --output-dir runs/concept-10k
```

恢复训练时模型结构参数必须与 checkpoint 一致：

```bash
uv run python train.py \
  --mode faithful --device mps --max-steps 100000 --batch-size 16 \
  --resume runs/faithful-60k/video_dit.pt \
  --output-dir runs/faithful-100k
```

`--max-steps` 表示恢复后的总 step，而不是额外 step。为了兑现“接着完全相同的下一批训练”，resume 会校验 device type、schedule、模型/mode/curriculum/prediction type，以及 seed、batch size、worker 数、水平翻转概率、learning rate、weight decay、有效 EMA decay 和梯度裁剪；需要改变这些实验条件时应开始一个新 run。默认 FP32，避免在第一次可复现实验里把混合精度误差和模型问题混在一起。早期短核函数基准在 batch 8 时约为 0.058 秒/step；它不包含渲染、验证采样、Metal 首次编译与温控，也不是当前版本的端到端吞吐。正式墙钟时间请以 `step_log.csv` 和 `summary.json` 为准。

## CLI 推理

训练完成后，默认读取 checkpoint 的 EMA 槽生成最终动画；concept checkpoint 中该槽每步复制 raw 权重，并未执行指数平均：

```bash
uv run python inference.py demo_artifacts/video_dit_inference.pt \
  --prompt "a person in a red coat walking in the snow to the right" \
  --seed 42 --sampling-steps 100 --cfg-scale 1 \
  --device mps --output-dir runs/canonical-velocity/inference
```

输出文件名包含解析条件、seed、请求 hash 与唯一 run id，不会因不同采样配置或并发请求互相覆盖；其中最重要的是：

- `*.mp4`：模型生成的时间轴，播放它才是在看动画；
- `*_denoising-diagnostic.gif`：采样轴诊断，每一格都横向展示完整视频时间轴；
- `*.json`：checkpoint、prompt、解析条件、seed、采样 steps、CFG 和耗时；
- `*_temporal-metrics.json`：质心路径、首尾位移、相邻帧/首尾 MSE 与平滑比率。

同一 seed、DDIM `eta=0` 与相同参数会得到可复现结果。CFG 必须配合训练时的 condition dropout；concept checkpoint 会自动按无 CFG 方式运行。

## Web UI

```bash
uv run python web_ui.py \
  --device mps --server-name 127.0.0.1 --server-port 7861 \
  --output-dir runs/canonical-velocity/web_ui
```

浏览器打开 `http://127.0.0.1:7861/`。界面将最终 MP4 标为 **Final temporal animation (MP4)**，将诊断标为 **Denoising diagnostic (GIF — diffusion time, not animation time)**。另有明确标注的 dataset preview，可在没有训练 checkpoint 时检查 prompt 到真实时间运动的映射；它只是数据渲染器预览，不是模型生成结果。

CLI/UI 只用 `weights_only=True` 加载 `video_dit_inference.pt`。含 optimizer 和 RNG 的 `video_dit.pt` 采用 pickle，仅应由训练器恢复自己生成且可信的文件；不要把未知 checkpoint 传给 `--resume`。

如果 7860 正被 image DiT UI 使用，视频 UI 使用 7861，二者不会互相覆盖。

## 应该观察哪些学习信号

建议固定三组 prompt 与 seed，每 5,000 steps 对照：

1. 1k steps：loss 有限，输出不再是 adaLN-Zero 的全零初值；
2. 10k steps：相邻帧相似但前景有非零位移；
3. 30k steps：左右方向和雪/草/夜景多数正确；
4. 60k steps：猫、狗、红外套人物可区分；
5. 只有确认不是 normalization、unpatchify 或 prompt mapping bug 后，才扩展到 100k。

可以亲手做的消融包括：`--no-learn-sigma`、condition dropout `0 ↔ 0.1`、CFG `1 ↔ 2.5 ↔ 5`、patch `4 ↔ 8`、`num-pairs 1 ↔ 4`，以及 EMA/raw 权重对比。

## 能力边界

这个本地模型能诚实展示时序注意力、条件视频扩散、真实 MP4 与主体运动，但当前训好的 checkpoint 只是三个 canonical clip 的 closed-vocabulary overfit；它不能生成照片级人物或理解任意文本。若需要开放词表、真实猫狗和人物场景，应在本项目之外使用已在大规模视频上预训练的模型。设计报告给出了 Apple 官方 [MLX Wan2.1](https://github.com/ml-explore/mlx-examples/tree/main/video/wan2.1) 作为可选外部路线；当前代码并没有 Wan backend 、backend selector 或其权重下载逻辑，也不把它作为本教学项目的依赖。

不要把本目录称为 Latte-1 checkpoint 复现：Latte 论文的大实验用了 8×A100 80 GB；发布的约 1B T2V checkpoint 还依赖 T5-XXL/PixArt 风格条件。这里保留的是论文中最值得学习、又能在当前 Mac 上实际跑通的 Variant-1 框架与训练技巧。

## 真实视频与未见组合实验（进行中）

真实数据路线的规范和数据集取舍见
[`video-dit-real-data-plan.md`](../../docs/case-studies/video-dit-real-data-plan.md)。当前已经实现并验证：

- JSONL manifest 强制保存 source ID、原页面、许可、署名、SHA-256；
- source group、人工 duplicate group 与 64-bit 感知指纹构成传递 group；近重复既不跨
  split，也不会被重复计作独立来源；
- IID、C1 三元组合留出、C2 二元组合留出的确定性切分和 fail-closed 验证；
- ffmpeg 真实 MP4 解码，以及缓存冻结文本 embedding / VAE latent 的数据接口；
- 官方 OpenAI CLIP ViT-B/32 冻结文本塔，encoder ID 中记录完整权重 SHA-256；
- 真实 manifest 的训练 checkpoint 会记录 manifest hash、text encoder ID；
- IID/C1/C2 分开报告逐因子、三元组、打乱 prompt 对照和训练最近邻相似度。

准备和验证已有 manifest：

```bash
uv run python prepare_real_data.py split source.jsonl split.jsonl \
  --c1 cat,running,snow --c2 dog,snow --report split-report.json

uv run python prepare_real_data.py validate split.jsonl \
  --c1 cat,running,snow --c2 dog,snow \
  --min-c1-source-groups 50 --min-train-pair-source-groups 1
```

验证器按 source/duplicate 的传递闭包计数，不把同源裁出的多个 clip 当成多个独立
样本。正式 CLI 默认要求每个 C1 三元组至少 50 个独立来源；IID 三元组必须也存在于
训练集。`split-report.json` 会输出 train/IID/C1/C2 的三元组 coverage table 和训练
pair coverage。小规模管线测试可以显式降低门槛，但不能据此宣称组合泛化。

Wikimedia Commons 的许可可审计候选采用“搜索→人工审核→下载”两阶段流程。搜索结果
永远是 `pending`，不会直接进入训练：

```bash
uv run python prepare_commons.py search \
  configs/commons_g0_search.json data/commons_g0/candidates.jsonl

# 可选：按 Commons 分类树扩展候选，再批量补齐时长和官方 240p derivative：
uv run python prepare_commons.py category "Videos of cats" cat \
  data/commons_g0/candidates.jsonl --max-depth 1
uv run python commons_review.py enrich data/commons_g0/candidates.jsonl

# 候选正被审核 UI 使用时，改写独立 sidecar，不与 UI 竞争同一 JSONL：
uv run python commons_review.py provenance \
  data/commons_g1/candidates.jsonl data/commons_g1/provenance-v2.jsonl

# 在浏览器逐条播放官方低码率 derivative，填写真实标签和最佳 2 秒起点：
uv run python commons_review_ui.py data/commons_g0/candidates.jsonl \
  --server-port 7862

# 把一批已完成的人工决策固化为候选快照 SHA、媒体 SHA-1 和审核字段；
# titles.json 是严格的 Commons title 数组，pending/缺失标题会失败：
uv run python commons_review.py audit data/commons_g1/candidates.jsonl \
  data/commons_g1/cat-playing-audit.json --titles titles.json \
  --provenance data/commons_g1/provenance-v2.jsonl

# 只排序能补 held 之外训练 pair 的候选：cat-playing 必须有 outdoor 证据，
# playing-indoors 必须是 dog；报告不会改写任何审核字段：
uv run python commons_review.py pair-priority \
  data/commons_g1/candidates.jsonl data/commons_g1/pair-deficit-priority.jsonl \
  --provenance data/commons_g1/provenance-v2.jsonl \
  --target-pair 'cat|playing' --target-pair 'playing|indoors'

# 人工检查原页面与视频，把正确的 subject/action/scene/prompt 改好，
# 仅将确认样本的 review_status 改为 approved。
uv run python prepare_commons.py download \
  data/commons_g0/candidates.jsonl data/commons_g0/clips \
  data/commons_g0/source.jsonl

# 新 provenance 契约写入全新的数据树；下载前会重新验证媒体 SHA-1/timestamp：
uv run python prepare_commons.py download \
  data/commons_g1/candidates.jsonl data/commons_g1_provenance_v3/clips \
  data/commons_g1_provenance_v3/source.jsonl \
  --provenance data/commons_g1/provenance-v2.jsonl

uv run python prepare_real_data.py coverage \
  data/commons_g1_provenance_v3/source.jsonl --target-per-triple 50 \
  --min-train-pair-source-groups 10 \
  --report data/commons_g1_provenance_v3/coverage.json

# 只补当前 C1 训练侧 pair 缺口；所有搜索结果仍是 pending：
uv run python prepare_commons.py search \
  configs/commons_pair_deficit_search.json \
  data/commons_g1/candidates.jsonl \
  --state data/commons_g1/pair-deficit-search.state.json

# 把官方 API 审计得到的精确 File: title 可恢复地追加为 pending；
# API 回传 title 或既有 subject/query contract 不一致时会拒绝：
uv run python prepare_commons.py exact \
  configs/commons_pair_exact_candidates.json data/commons_g1/candidates.jsonl

# 冻结 CLIP 只对当前媒体版本的 320px poster 排序；报告不能批准动作/场景，
# top-K 仍须下载完整 derivative 并人工检查连续两秒。checkpoint SHA 必须为
# 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af：
uv run python visual_priority.py \
  data/commons_g1/candidates.jsonl data/commons_g1/provenance-v2.jsonl \
  data/commons_g1/visual_posters/dog data/commons_g1/dog-visual-priority.json \
  --checkpoint data/models/ViT-B-32.pt --subject dog \
  --positive 'a single dog playing with a toy inside a home' \
  --negative 'multiple dogs or puppies together' --request-delay 1.2

# 对锁定 poster 排序的固定 top-K 再用原生 8-frame X-CLIP 扫描最多 8 个
# 均匀 2 秒窗口；报告保留逐窗口、倒序诊断及所有模型/媒体/帧哈希：
uv run python temporal_priority.py \
  data/commons_g1/candidates.jsonl data/commons_g1/provenance-v2.jsonl \
  data/commons_g1/dog-visual-priority.json data/commons_g1/temporal_cache/dog \
  data/commons_g1/dog-temporal-priority.json \
  --model-dir /path/to/microsoft-xclip-base-patch32-fixed-revision \
  --subject dog --limit 20 --positive 'a single dog playing indoors' \
  --negative 'a dog sleeping standing or sitting still'

# Internet Archive 只接受人工列出的精确 item ID；输出仍是 metadata-only pending，
# 许可、creator、唯一 original 视频以及 size/MD5/SHA-1 任一不完整都会拒绝：
uv run python prepare_archive.py configs/archive_pair_candidate_specs.json \
  data/archive_pair_pilot/candidates.jsonl \
  data/archive_pair_pilot/scan-report.json
```

下载只接受 `approved`、标签完整且来源为 Wikimedia HTTPS 的记录；若审核时已缓存
官方低清 derivative，训练预处理会使用同一 derivative。输出 clip 路径相对于 manifest，
因此移动整个数据目录后仍可加载。

新的下载会 fail-closed 要求 `enrich`/`provenance` 写入 Commons 描述页 revision 以及原媒体的
timestamp/SHA-1，并分别保留原页面、原始媒体 URL、逐条 license URL、署名、下载 UTC
时间，以及**实际下载对象**（原文件或审核 derivative）的 URL/type/SHA-256、规范化 clip
SHA-256 和感知 hash。页面 revision、原媒体版本和 derivative hash 是三个不同对象，不会
混称为“原始媒体 hash”。旧版 101-record manifest 曾缺少其中若干独立字段，不能冒充为
满足新契约；当前 v3 的 119 条已全部重新 enrich、重新规范化，并记录片段起点、预处理
参数和显式来源组。按上传者保守初分后，又用关键帧审核明显不同的猫身份并保留疑似同猫
合并，当前为 91 个独立组；`cat|playing|indoors` 只有 16/50。coverage 还预注册每个 held
triple 的训练侧 pair 下限为 10；当前 `cat×indoors=28/10`、`cat×playing=5/10`、
`playing×indoors=5/10`，mixed-held group 已经消解，但只补 held triple
仍不能放行。错误搜索命中（例如机器人
狗、纪录片或主体过小）必须拒绝。搜索按 API page 原子保存候选和 continuation token；
网络 429 或进程中断后可继续当前 query，已完成 query 不会重复抓取。当前 G1 Commons
候选池为 1,097 条，目前 119 条 approved、816 条 pending、162 条 rejected；pending 绝不
计作训练数据或组合覆盖。人物审核目前批准 2 条：单人海滩行走，以及固定机位下单人走入
森林的 2598--2600 秒片段；另有多轮候选因人群、第一视角、延时摄影、旅行 vlog 或动作不符
而拒绝，person 仍有 109 条
pending。标题启发式只能排序人工队列，绝不自动批准或生成标签：

```bash
uv run python commons_review.py priority data/commons_g1/candidates.jsonl \
  data/commons_g1/person-review-priority.jsonl --subject person \
  --provenance data/commons_g1/provenance-v2.jsonl
uv run python commons_review_ui.py data/commons_g1/candidates.jsonl \
  --status pending --subject person --priority-order \
  --provenance data/commons_g1/provenance-v2.jsonl --server-port 7863
```

当前 109 条 person pending 中有 108 条满足“时长至少 2 秒且存在官方预览”的可播放审核门，
因此 priority 报告只列这 108 条；余下 1 条不会绕过媒体门禁。

8 个更窄的 `woman/man × walking/running × forest/snow/field` Commons 全文查询只新增
1 条 43 分钟多州旅行合集；20 帧全时长联系表确认它混有驾驶 POV、建筑、多人和不连续
场景，已拒绝。该结果说明窄全文查询不能作为 Commons-1K 的高吞吐扩容方法。

结构化分类导入还发现 Commons 会把视频 `.ogv` 报为 `application/ogg`；导入器现按扩展名
只接受 `.ogv + application/ogg`，普通 `.ogg` 音频仍拒绝。priority/UI 可只读合并 provenance
sidecar，因此新候选不必改写 candidates 就能获得时长与官方 preview。审核 UI 在批准时
强制填写稳定的 duplicate/source group：已知同一动物、拍摄系列或事件必须共享组；尚不能
证明独立的文件按 uploader 保守合并。该字段进入 manifest，
并与 source ID、pHash 一起用于防泄漏分组。`Videos of hiking`
新增 13 条，`Videos of people jogging` 在去重与许可门后新增 3 条；已审核的 6 条均因无
可见人物、多人、主体过小或身体严重裁切而拒绝，剩余仍为 pending。

缓存官方 CLIP embedding。checkpoint 可以从 OpenAI CLIP 官方仓库中
`ViT-B/32` 对应的 Azure URL 获取；不要用随机向量替代失败的下载：

```bash
uv run python cache_text_embeddings.py source.jsonl embedded.jsonl \
  --checkpoint /path/to/ViT-B-32.pt --embedding-kind tokens
```

冻结 VAE latent 路线使用 MIT 许可的 `stabilityai/sd-vae-ft-mse` KL-f8；必须固定
revision。编码采用 posterior mode（确定性），存储前乘 `0.18215`，推理解码前除回去：

```bash
uv run python vae_latents.py embedded.jsonl embedded-latent.jsonl \
  --model stabilityai/sd-vae-ft-mse \
  --revision c01e0282cf0708e7fdecf07fb5f9089f5bc61c79 \
  --device mps --frames 16 --size 96

uv run python train.py --dataset real --manifest embedded-latent.jsonl \
  --text-conditioning tokens --text-embedding-size 512 \
  --space latent --no-learn-sigma \
  --frames 16 --image-size 12 --patch-size 2 --hidden-size 256 \
  --num-heads 8 --num-pairs 4 --horizontal-flip-prob 0 --sample-every 0

uv run python inference.py runs/latent/video_dit_inference.pt \
  --prompt "a dog swimming in a pond" --device mps \
  --text-checkpoint /path/to/ViT-B-32.pt \
  --text-conditioning tokens \
  --vae-model stabilityai/sd-vae-ft-mse \
  --vae-revision c01e0282cf0708e7fdecf07fb5f9089f5bc61c79

# 只用于诊断最高噪声区间；正式结果仍应从默认 t=999 开始：
uv run python inference.py runs/latent/video_dit_inference.pt \
  --prompt "a dog swimming in a pond" --start-timestep 900 # 其余参数同上

# 固定同一 seed 生成至少两个 prompt 后，测 CLIP 检索与训练集最近邻：
uv run python condition_alignment.py runs/latent/inference \
  --checkpoint /path/to/ViT-B-32.pt \
  --train-manifest embedded-latent.jsonl
```

`tokens` 模式缓存 77 个冻结 CLIP hidden states，并在每个空间/时间 block 使用零初始化
gate 的 cross-attention；encoder ID 的 `#tokens-v2` 表示 padding hidden states 已明确清零。
`condition_alignment.py` 要求每个 seed 都有完全相同且唯一的 prompt 集合；它先在
每个 seed 内做 matched 检索和循环打乱 prompt 对照，再跨 seed 汇总 Wilson 95% 区间，
防止把随机噪声差异误算成 prompt 效果。目录内 checkpoint hash、权重种类、训练 step、
采样 steps、起始 timestep、latent/pixel 裁剪策略、CFG、prediction type 和文本 encoder
也必须一致。CLIP 检索与最近邻只是
诊断代理，不替代对主体、动作和时间连续性的目视审核。
旧推理 metadata 若没有生成时写入的 `checkpoint_sha256`，会被明确视为不可验证，不能
混入 locked comparison；请用当前 `inference.py` 重新生成。

latent 采样不会把预测的 clean latent 裁剪到 `[-1,1]`；该裁剪只适用于归一化 pixel。
训练视频上的条件绑定可用 `teacher_forced_alignment.py` 在固定噪声/时间步下诊断，但它
不代表从纯噪声自由生成成功。若该诊断只在最高噪声尾部失败，可做显式、可审计的实验：

```bash
uv run python train.py ... --initialize-from runs/latent/video_dit_inference.pt \
  --initial-weights ema --timestep-sampling high-noise \
  --high-noise-start 900 --high-noise-fraction 0.75
```

`--initialize-from` 仅加载 safe inference checkpoint 的指定权重，并重置 optimizer 和
step；它不同于严格恢复训练的 `--resume`。timestep 分布会写入 optimization contract。

若要定位语义在反向轨迹的哪个阶段出现，可加 `--save-trajectory-videos`。每个捕获状态
会另存一个带 `trajectory_timestep` 和完整生成契约的 MP4/JSON，然后逐时间步评估：

```bash
uv run python condition_alignment.py runs/latent/inference \
  --checkpoint /path/to/ViT-B-32.pt --artifact-mode trajectory \
  --trajectory-timestep 494 --output runs/latent/inference/alignment-t494.json
```

轨迹 MP4 是扩散时间诊断，不是额外生成的动画；不同 timestep 不能混在同一检索 block。

真实像素视频训练入口：

```bash
uv run python train.py \
  --dataset real --manifest embedded.jsonl --text-embedding-size 512 \
  --mode faithful --device mps --max-steps 1 --batch-size 1 \
  --frames 16 --image-size 96 --patch-size 8 \
  --hidden-size 128 --num-heads 4 --num-pairs 2 \
  --horizontal-flip-prob 0 --sample-every 0 --checkpoint-every 0 \
  --output-dir runs/real-smoke
```

本机已用一条人工检查过的 Wikimedia Commons CC BY 3.0 猫跳跃 clip 跑通上述
96×96、16 帧、learned-variance、CLIP 条件训练更新：MPS 单步约 1.96 秒，loss
1.02395。这只是 **G0 管线和性能基线**，没有证明模型已学会生成，更没有形成 C1
实验。随后已取得固定 revision 的 SD VAE safetensors，完成 100/100 latent cache、
10,000-step latent 训练和 VAE 推理解码。进一步的 token cross-attention 消融与 10-clip
11.99M/10k capacity 对照也已完成；之后的全时间 velocity 10k 运行将多 seed prompt
检索提升到 17/30，并生成了有真实跨帧变化的动物/场景视频。因此 G0 的基础运动/小集合
拟合 gate 有条件通过，但文本三因子门禁仍未通过。结果、混淆项与失败边界记录在学习计划。
目前只开始 G1 数据准备，没有启动 G1 正式训练。固定 revision 的 VideoUFO 完整 CSV
已校验并扫描 1,091,712 行，得到 4,234 条待审核三因子候选；最小官方视频分片也已取得，
197 个 MP4 的 member basename 可与 CSV ID 精确关联，但只命中 1 条候选。猫组合 coverage
尤其稀疏，人工审核与 pHash 分组后尚不可能满足 C1 门禁，因此没有启动 1K 训练。

替代源审计也已完成：OpenVid 的官方文件名到 ZIP 映射让定向取媒体在工程上可行，但其
视频混合继承 Panda、ChronoMagic、Open-Sora-Plan、CelebV-HQ 等上游来源条件，不能把
仓库顶层的 license 标签当成逐 clip 权利证明；Panda-70M 和 InternVid 主要发布网页视频
metadata，并带非商业/上游权利限制。因此这些来源只作 metadata coverage 参考，不进入
当前 checkpoint。训练主线改为 Commons-first：把现有 119 approved、逐文件记录许可和
署名的样本扩到 1K，优先补 person 与稀疏猫狗组合；达不到每个 C1 triple 50 个独立
source groups 时就缩小矩阵并公开缺口，不用来源不明的视频补数。完整证据与固定 revision
见学习计划。

复现 metadata 扫描与 tar 解析（原始文件位于仓库外/ignored 的 `data/raw/videoufo`）：

```bash
cd code/video_dit
uv run --offline python prepare_videoufo.py scan \
  ../../data/raw/videoufo/VideoUFO.csv \
  ../../data/raw/videoufo/candidates.jsonl \
  ../../data/raw/videoufo/coverage.json \
  --expected-sha256 a456192ca88a320080f840c5ec007b0ce6c1852a98e27d0dfad61e2b1f357b52
uv run --offline python prepare_videoufo.py resolve \
  ../../data/raw/videoufo/candidates.jsonl \
  ../../data/raw/videoufo/coverage.json \
  ../../data/raw/videoufo/resolved-200.jsonl \
  ../../data/raw/videoufo/resolved-200-report.json \
  --shard-checksums ../../data/raw/videoufo/shard-checksums.json \
  ../../data/raw/videoufo/VideoUFO_tar/VideoUFO_200.tar
```

`shard-checksums.json` 是 `{文件名: 官方 SHA-256}` 的 JSON object；resolve 缺少它或
hash 不符会直接失败。`source_id` 使用去掉末尾 `.clip-index` 的原始视频 ID，避免同一
YouTube 源的 sibling clips 被拆到不同 split。

本地三分片 pilot（`1/2/200`）已完成：8.46 GB（7.88 GiB）的 tar 映射出 49 条候选、46 个
原视频 source groups，其中 person 43、dog 5、cat 1。49 条媒体已按 checksum 一次验证
每个 archive 后安全提取，总计约 30 MB，全部保持 `pending`。随机分片明显偏向 person，
不能据此启动平衡训练；`VideoUFO_3.tar` 下载到约 3.92 GB 时因 TLS handshake timeout
失败且上游客户端删除 partial，未计入 pilot。

```bash
uv run --offline python prepare_videoufo.py extract \
  ../../data/raw/videoufo/resolved-pilot.jsonl \
  ../../data/raw/videoufo/review-pilot.jsonl \
  ../../data/raw/videoufo/review-media \
  --shard-checksums ../../data/raw/videoufo/shard-checksums.json \
  ../../data/raw/videoufo/VideoUFO_tar/VideoUFO_1.tar \
  ../../data/raw/videoufo/VideoUFO_tar/VideoUFO_2.tar \
  ../../data/raw/videoufo/VideoUFO_tar/VideoUFO_200.tar
```

官方 VATEX v1.0 训练 annotations 也已在本机完整下载并解析，SHA-256 为
`9a3b5f08e354d9543ef4f1ab004f9db9bc4e5da49d9692f0c8c6aa3bef9751c4`；其中
25,991 个视频 ID 中，caption 匹配 cat/dog 的视频约为 1,415/600，匹配
cat+snow/dog+snow 约为 25/71。这是候选规模统计，不是人工确认标签；VATEX 官方不
分发视频本体，因此视频本体的来源权利仍须逐条审计。
