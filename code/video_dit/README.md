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

# 自由文本 latent checkpoint 还必须显式绑定训练时的 frozen dependencies：
uv run python web_ui.py runs/c1/video_dit_inference.pt \
  --device mps --server-port 7861 --text-checkpoint data/models/ViT-B-32.pt \
  --vae-model stabilityai/sd-vae-ft-mse \
  --vae-revision c01e0282cf0708e7fdecf07fb5f9089f5bc61c79
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

# 用 Commons Structured Data 的 depicts QID 补足分类/全文检索漏项；
# Q146=cat、Q144=dog，只作为 subject 召回，动作和场景仍须人工审核：
uv run python prepare_commons.py search \
  configs/commons_structured_depicts_search.json \
  data/commons_g1/candidates.jsonl \
  --state data/commons_g1/structured-depicts-search.state.json

# 把官方 API 审计得到的精确 File: title 可恢复地追加为 pending；
# API 回传 title、正整数 page ID 或既有 subject/query contract 不一致时会拒绝：
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
uv sync --extra temporal
uv run python temporal_priority.py \
  data/commons_g1/candidates.jsonl data/commons_g1/provenance-v2.jsonl \
  data/commons_g1/dog-visual-priority.json data/commons_g1/temporal_cache/dog \
  data/commons_g1/dog-temporal-priority.json \
  --model-dir /path/to/microsoft-xclip-base-patch32-fixed-revision \
  --subject dog --limit 20 --positive 'a single dog playing indoors' \
  --negative 'a dog sleeping standing or sitting still'

# 在看片前冻结 poster 高分组与确定性随机对照；queue 不含 arm/rank/margin，
# 只有全部视频完成 approved/rejected 后才用 sealed key 揭盲：
uv run python blind_review_pool.py create \
  data/commons_g1/candidates.jsonl data/commons_g1/provenance-v2.jsonl \
  data/commons_g1/cat-visual-priority.json \
  data/commons_g1/cat-blind40-queue.json data/commons_g1/cat-blind40-key.json \
  --subject cat --high-count 20 --control-count 20 --seed 20260814
uv run python blind_review_pool.py reveal \
  data/commons_g1/cat-blind40-queue.json data/commons_g1/cat-blind40-key.json \
  data/commons_g1/candidates.jsonl data/commons_g1/cat-blind40-audit.json \
  data/commons_g1/cat-blind40-media.json data/commons_g1/cat-blind40-result.json \
  --target-triple 'cat|playing|indoors'

# Internet Archive 只接受人工列出的精确 item ID；输出仍是 metadata-only pending，
# 许可、creator、唯一 original 视频以及 size/MD5/SHA-1 任一不完整都会拒绝：
uv run python prepare_archive.py configs/archive_pair_candidate_specs.json \
  data/archive_pair_pilot/candidates.jsonl \
  data/archive_pair_pilot/scan-report.json

# Ego4D 只有在实际使用者接受并保存数据许可后才能运行。scanner 先冻结
# metadata/narrations/许可/receipt 四份字节快照，按时间排序 narration 上下文，
# 优先覆盖不同 participant/origin group，最多输出 10 个 UID；不下载视频、
# 不自动生成标签，输出仍为 training_ready=false：
uv run python ego4d_import.py \
  data/raw/ego4d_pair_pilot/ego4d.json \
  data/raw/ego4d_pair_pilot/narrations.json \
  data/private/ego4d-license.pdf data/private/ego4d-license-receipt.json \
  data/ego4d_pair_pilot/candidates.jsonl data/ego4d_pair_pilot/video-uids.txt \
  data/ego4d_pair_pilot/scan-report.json \
  --max-video-uids 10
# 只有 scanner 报告和许可复核都通过后，UID 文件才可进入下面的本地
# preflight/download wrapper；不要把它直接交给官方媒体 CLI。报告固定
# video_540ss、12 GB hard stop，超预算不得回退到全量下载。

# 获得用户自己的 Ego4D/AWS 访问后，preflight 会以 version-specific GET 冻结
# 官方 v2_1 manifest，再逐 UID HEAD 并锁定精确 S3 version/ContentLength/ETag；
# 这一步先执行十进制 12,000,000,000-byte hard gate：
uv run python prepare_ego4d.py preflight \
  data/ego4d_pair_pilot/candidates.jsonl data/ego4d_pair_pilot/video-uids.txt \
  data/ego4d_pair_pilot/scan-report.json data/ego4d_pair_pilot/download-plan.json \
  --aws-profile ego4d

# 人工确认 plan 后，由本工具按已冻结的 VersionId 做 exact GET；不会退回 latest
# object，也不会在流式字节数超过 plan/hard cap 后继续写入：
uv run python prepare_ego4d.py download \
  data/ego4d_pair_pilot/download-plan.json \
  data/raw/ego4d_pair_pilot/v2/video_540ss \
  data/ego4d_pair_pilot/download-receipt.json \
  --aws-profile ego4d

# 下载后再绑定 metadata duration、下载 receipt、manifest.ver、逐文件 SHA-256
# 和 ffprobe 媒体属性；输出仍然 training_ready=false：
uv run python prepare_ego4d.py verify \
  data/ego4d_pair_pilot/candidates.jsonl data/ego4d_pair_pilot/video-uids.txt \
  data/ego4d_pair_pilot/scan-report.json data/raw/ego4d_pair_pilot/ego4d.json \
  data/ego4d_pair_pilot/download-plan.json data/ego4d_pair_pilot/download-receipt.json \
  data/raw/ego4d_pair_pilot/v2/video_540ss/manifest.ver \
  data/raw/ego4d_pair_pilot/v2/video_540ss \
  data/ego4d_pair_pilot/verified-media.json
```

下载只接受 `approved`、标签完整且来源为 Wikimedia HTTPS 的记录；若审核时已缓存
官方低清 derivative，训练预处理会使用同一 derivative。输出 clip 路径相对于 manifest，
因此移动整个数据目录后仍可加载。

新的下载会 fail-closed 要求 `enrich`/`provenance` 写入 Commons 描述页 revision 以及原媒体的
timestamp/SHA-1，并分别保留原页面、原始媒体 URL、逐条 license URL、署名、下载 UTC
时间，以及**实际下载对象**（原文件或审核 derivative）的 URL/type/SHA-256、规范化 clip
SHA-256 和感知 hash。页面 revision、原媒体版本和 derivative hash 是三个不同对象，不会
混称为“原始媒体 hash”。旧版 101-record manifest 曾缺少其中若干独立字段，不能冒充为
满足新契约；Commons v3 的 127 条已全部重新 enrich、重新规范化，并记录片段起点、预处理
参数和显式来源组。按上传者保守初分后，又用关键帧审核明显不同的猫身份并保留疑似同猫
合并；随后又加入 1 条逐媒体锁定的 Internet Archive CC0 猫视频、1 条逐文件锁定的 Figshare
CC BY 4.0 室内玩耍猫视频，以及 3 条 origin/API/rendition 全部锁定的 PeerTube 室内玩耍猫视频。
当前合并 manifest 为 133 records / 104 个独立组；`cat|playing|indoors` 为 20/50。
coverage 还预注册每个 held
triple 的训练侧 pair 下限为 10；当前 `cat×indoors=30/10`、`cat×playing=6/10`、
`playing×indoors=10/10`，mixed-held group 已经消解，但只补 held triple
仍不能放行。错误搜索命中（例如机器人
狗、纪录片或主体过小）必须拒绝。搜索按 API page 原子保存候选和 continuation token；
网络 429 或进程中断后可继续当前 query，已完成 query 不会重复抓取。当前 G1 Commons
候选池为 1,135 条，目前 128 条 approved、797 条 pending、210 条 rejected；pending 绝不
计作训练数据或组合覆盖。人物审核目前批准 7 条：单人海滩行走、固定机位下单人走入
森林的 2598--2600 秒片段、两条单人室内台球片段，以及三条不同日期/赛事 session 的单人
室内 bowling 片段。另有多轮候选因人群、
第一视角、延时摄影、旅行 vlog 或动作不符而拒绝，person 仍有 109 条
pending。标题启发式只能排序人工队列，绝不自动批准或生成标签：

此前一次 exact 导入加入两个有明确 indoor-playing metadata 的 person 视频；全时段联系表显示
一个是多人博物馆场景，另一个是可识别儿童。两条均按预注册的人物权利/单主体门禁拒绝，
所以当时的 manifest 与 30/5/5 训练 pair coverage 没有变化，也没有启动训练；后续台球审核
才把 `playing×indoors` 推进到 7/10。

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
混入 locked comparison；当前版本还会写入 MP4 文件名和 `video_sha256`，防止 prompt
metadata 与后来被替换的视频错绑。请用当前 `inference.py` 重新生成旧 artifact。

正式 IID/C1/C2 结果不能只看 CLIP 或挑选好看的 MP4。达到数据门禁后，先从已经重新验证的
split manifest 和 validation report 派生不可手填的评估 spec，再原子生成完整
prompt × seed 网格，并把视频复制为不暴露 prompt、split 或 seed 的随机 blind ID：

```bash
uv run python generation_blind_review.py spec \
  split.jsonl split-report.json generation-evaluation-spec.json \
  --checkpoint runs/c1/video_dit_inference.pt --weights ema \
  --sampling-steps 50 --start-timestep 999 --cfg-scale 2.5 \
  --seeds 17 29 43 61
# 生成器只在所有 prompt × seed 都成功后提交输出目录；中途失败不留下可供挑选的半成品。
uv run python generate_evaluation_grid.py \
  runs/c1/video_dit_inference.pt generation-evaluation-spec.json runs/c1/inference \
  --device mps --text-checkpoint data/models/ViT-B-32.pt \
  --text-conditioning tokens --vae-model stabilityai/sd-vae-ft-mse \
  --vae-revision c01e0282cf0708e7fdecf07fb5f9089f5bc61c79
uv run python generation_blind_review.py create \
  runs/c1/inference generation-evaluation-spec.json split.jsonl split-report.json \
  runs/c1/blind-media runs/c1/blind-queue.json runs/c1/sealed-key.json \
  --seed 20260814
```

低于 `50/10` 正式门禁时，只能在训练前明确加 `spec --pilot` 创建诊断性 pilot；报告会固化
实际 held/pair coverage 并把 `confirmatory_gate.passes` 写为 `false`。pilot 无论成功或失败都
不能改写成正式 unseen-composition 结论，也不能在看过生成后更换 checkpoint、prompt 或 seed。

至少两名互相独立的评审者完整播放 blind MP4，只能从 queue 固定的 subject/action/scene
词表选择；每个因子都额外提供永远不可能是正确答案的 `absent_or_unclear`，避免强迫评审者
把垃圾帧猜成某个真实标签。决定分别保存为 `blinded generated-video factor review v1` JSON。揭盲要求
两份决定覆盖完整队列、媒体 SHA 未变化、sealed assignment commitment 未变化；任一因子
分歧都按错误而不是择优仲裁：

```bash
uv run python generation_review_ui.py \
  runs/c1/blind-queue.json runs/c1/blind-media runs/c1/rater-a.json \
  --reviewer-id rater-a --server-port 7863
# 第二名评审者在独立会话使用不同 reviewer-id 和 decision 文件。

uv run python generation_blind_review.py reveal \
  runs/c1/blind-queue.json runs/c1/sealed-key.json runs/c1/blind-media \
  runs/c1/blind-human-evaluation.json \
  --decisions runs/c1/rater-a.json runs/c1/rater-b.json
```

报告分别保留 IID/C1/C2 的 subject、action、scene、完整 triple accuracy、Wilson 95% 区间，
并把同一 seed 的 expected factor triple 做严格 derangement 作为显式对照；若无法构造不同
三元组的错配就直接拒绝评估。这个人工门禁仍须与训练集最近邻
报告共同解释；它不能单独排除记忆复制。

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
当前 checkpoint。训练主线改为 provenance-first：从现有 127 条 Commons approved 与 1 条经同等门禁审核的 IA CC0 视频继续扩展，逐文件记录许可和
署名的样本扩到 1K，优先补 person 与稀疏猫狗组合；达不到每个 C1 triple 50 个独立
source groups 时就缩小矩阵并公开缺口，不用来源不明的视频补数。完整证据与固定 revision
见学习计划。

最新一轮从 Commons 台球分类补入两条不同人物、日期与事件的 CC BY-SA 历史演示，经完整
视频联系表和 `[8,10)` 秒密集窗口审核后标为 `person|playing|indoors`。该历史快照的 v3 manifest 当时为
124 records / 95 independent groups；held `cat|playing|indoors` 仍为 16/50，训练侧
`cat×indoors=30/10`、`cat×playing=5/10`、`playing×indoors=7/10`，仍是 0 eligible C1，
所以没有启动正式训练。审核截图和逐媒体 hash 在
`docs/case-studies/artifacts/video-dit/commons-billiards-review/`。

随后又对 Commons `Videos of bowling` 的三个不同日期/赛事候选执行 exact import、完整视频
联系表和密集两秒窗口审核；三条均为单一具名成年人持续投球、室内球道，且按
`uploader + venue + capture date/event` 保守划为三个 session。当前 v3 manifest 为
127 records / 98 independent groups；`playing×indoors` 已达到 10/10，但
`cat×playing` 仍为 5/10，held `cat|playing|indoors` 仍为 16/50，因此仍是 0 eligible C1，
没有启动正式训练。逐媒体 provenance、规范化 clip hash 和审核图见
`docs/case-studies/artifacts/video-dit/commons-bowling-review/`。

再随后，Internet Archive exact-license 审计找到 `catvideo_0003`：原始 MP4 为 CC0 1.0，
官方 size/SHA-1/MD5 与本地流式下载结果一致。完整 75.78 秒联系表与 `[10,12)` 秒的
16-frame 密集窗口显示一只真实猫在户外灌木下连续伸爪/拨弄，因此只批准为
`cat|playing|outdoors`，不计入 held triple。它与现有 clips 的最小 pHash Hamming 距离为 21；
按 creator 保守归为一个新 group。合并 manifest 因此达到 128 records / 99 groups，
`cat×playing` 从 5 增到 6，held 仍是 16/50，仍为 0 eligible C1。审核 MP4、联系表与
逐对象 hash 见 `docs/case-studies/artifacts/video-dit/archive-cat-playing-review-v1/`。

再随后，Figshare article `28287896` version 2 的官方 API 直接托管三条 CC BY 4.0 实验 MP4。
逐文件核对官方 size/MD5、article snapshot、下载 URL 与本地 SHA-256 后，只有 file `57852742`
的 `[40,42)` 秒通过动作门：一只虎斑猫在室内连续翻滚、伸爪并拨弄玩具。另两条只是观察/找盒子，
没有为了补数重标成 playing。三条虽肉眼上是不同猫和房间，但官方没有公开 cat/session 映射，因此
全部保守视为一个 article-level group。`prepare_figshare.py create/merge` 会在每个边界重验
article/version/DOI/CC BY、file ID/size/MD5/SHA、人工审核与 group 契约，并把完整 audit 携带到新
数据树。执行后的 manifest 为 **129 records / 100 groups**，held `cat|playing|indoors=17/50`；
训练侧仍为 `cat×indoors=30/10`、`cat×playing=6/10`、`playing×indoors=10/10`，所以仍是
0 eligible C1、不得启动正式训练。证据见
`docs/case-studies/artifacts/video-dit/figshare-object-permanence-review-v1/`。

```bash
uv run --offline python prepare_figshare.py create \
  data/figshare_cat_object_permanence_v1/article.json \
  'data/figshare_cat_object_permanence_v1/media/Supplementary Video 3 violation event.mp4' \
  ../../docs/case-studies/artifacts/video-dit/figshare-object-permanence-review-v1/evidence.json \
  data/figshare_cat_object_permanence_v1/bundle-v1
uv run --offline python prepare_figshare.py merge \
  data/commons_g1_provenance_v3/source.jsonl \
  data/figshare_cat_object_permanence_v1/bundle-v1 \
  --output data/formal_c1_expanded_figshare_v1
```

再随后，Sepia Search 只用于发现候选，实际证据全部回到各 PeerTube origin 的官方 API。
五条强标题候选中，军方水印转发视频因上传者权利不清直接拒绝，Mercury 片段因动作只是翻滚/
自我接触而拒绝；只有 Buster 与 Žofinka 两条通过。两条均为单猫、室内、连续玩玩具，账号快照分别
明确把动物称为上传者的猫/具名 kitten，许可分别为 CC0 1.0 与 CC BY-SA 4.0。它们来自不同 origin、
账号、动物与事件，pHash 距离当前 base 分别为 23 和 19。

`prepare_peertube.py create` 会锁 video/account API bytes、许可、canonical origin/download URL、
rendition size/SHA-256、完整联系表、密集窗口与 2 秒规范化 clip；`merge` 再重放同一语义验证、重算
candidate-to-base pHash，并携带完整 bundle。随后同样审核了 PeerTube 上的 Neptune：上传者用第一人称
命名这只猫并说明自己在抛玩具，`[3.5,5.5)` 中一只黑猫连续追踪、扑抓和抱住毛绒玩具；
CC BY 4.0、原站 API/账号/媒体与完整视觉证据均已锁定，距固定 129-record base 的最近 pHash 距离为 13。

从固定 Figshare base 一次性重建三个 PeerTube bundle 后，manifest 为 **132 records / 103 groups**，held
`cat|playing|indoors=20/50`；训练侧仍为 `30/6/10`，所以 non-held `cat×playing` 仍缺 4 groups，
`eligible_c1_triples=[]`，正式训练仍未启动。审核资产见
`docs/case-studies/artifacts/video-dit/peertube-cat-playing-review/`。

```bash
uv run --offline python prepare_peertube.py create VIDEO.json ACCOUNT.json SOURCE.mp4 REVIEW.json BUNDLE
uv run --offline python prepare_peertube.py merge \
  data/formal_c1_expanded_figshare_v1/source.jsonl \
  data/peertube_cat_playing_v1/buster-bundle \
  data/peertube_cat_playing_v1/zofinka-bundle \
  data/peertube_held_v3/neptune-bundle \
  --output data/formal_c1_expanded_peertube_v2
```

在用户明确授权后，另行执行了一次**不满足正式数据门禁的预注册诊断 pilot**；它不是上述“正式训练”。
协议在训练结束前由 commit `536d0f9` 固化，held triple 的 source/duplicate/pHash 传递闭包全部排除。
11.99M 参数 latent text Video DiT 从零训练 10,000 updates 后，固定 final EMA 生成完整
12 prompts×4 seeds 网格。两名独立 prompt-hidden 评审的 IID/C1 unanimous triple accuracy 都是
0%；frozen CLIP top-1 14.6% 对 shuffled 12.5%，同样没有可信 prompt binding。完整结论是 pilot
失败：loss 下降但自由生成仍形变、碎片化，且 94 个训练 clips 与 `cat×playing=5`、
`playing×indoors=8` 都太弱。证据见
`docs/case-studies/artifacts/video-dit/c1-pilot-v1/results/`；正式确认仍要求 held≥50、每个 pair≥10
并重新预注册。

随后又在**当前 132-record 数据快照**上执行了新的预注册诊断 pilot。训练前 commit `a46e418`
锁定 split、10,000-step 训练配方和 13 prompts×4 seeds；生成前 commit `b1bfd4f` 再锁定 final EMA
checkpoint、DDIM-50、CFG 2.5 与完整 52-video inventory。101 条 train clips 的 loss 从前 500
步均值 0.584 降至后 500 步 0.255，但两名独立 prompt-hidden 评审只在 IID 得到 3/24
unanimous triple 命中；C1 为 0/28，subject/action accuracy 都是 0，scene 为 42.9%。frozen CLIP
prompt top-1 为 11.5%，反而低于 shuffled 19.2%。最近训练视频的 top-3 视觉审计没有发现逐帧
复制，但显示生成集中吸附到少数训练视频的颜色/纹理模式。结论仍是失败：它说明模型有很弱的
训练分布内条件信号，却不能证明未见组合泛化。可携带证据见
`docs/case-studies/artifacts/video-dit/c1-current-pilot-v2/results/`。正式门禁仍是 held≥50 且所有
held-excluded pair≥10；当前 held=20、`cat×playing=6`，因此不会把这次诊断当成正式 C1 成功。

为继续补 `cat×playing`，又对 pending 池中两个标题最强的 Commons 记录做了完整时段 24-frame
审核。`Play fight between cats` 始终是同窝两猫在室内床上打闹；Kitten Academy 记录则包含多只
foster cats 和可见人物。两条都违反冻结的单主体/稳定身份契约，已原子标为 rejected，coverage
增量为 0。联系表与锁定的 page/revision/原媒体 SHA-1、review derivative SHA-256 见
`docs/case-studies/artifacts/video-dit/commons-multi-subject-negative-review/`。

同一轮还批准了 `Ocicat on Cat Wheel`，但严格标为 `cat|walking|indoors`，没有为了补缺口改成
playing。完整 63.847 秒只有一只猫和一个固定室内场景，`[10,12)` 的 16-frame 窗口显示连续
四足步行且无剪辑。重新从 Commons base 构建 Figshare 与三个 PeerTube bundles 后，portable
manifest 为 133 records / 104 groups；held 仍为 20，训练 pair 为 `31/6/10`，所以正式 C1 仍不
放行。审核与重建 hash 见 `docs/case-studies/artifacts/video-dit/commons-ocicat-wheel-review/`。

截至本轮审计，正式门禁已到达外部素材阻断：所有已批准 non-held `cat|playing` 共 8 records，
按 Nairobi/Lotti 等同 series 闭包后只有 6 groups；现有公开来源与优先审核队列没有更多合格记录。
继续正式 C1 至少需要用户提供或逐视频取得书面许可的 4 个独立非室内 playing groups 和 30 个
独立室内 playing groups。精确缺口、已耗尽路线与每条素材必须满足的 receipt/identity/session/
完整时段/pHash 契约记录在
`docs/case-studies/artifacts/video-dit/formal-c1-data-blocker-v1/evidence.json`。

Internet Archive 的有界媒体验证入口如下；它在联网前检查总字节硬帽，流式核对 IA 的
SHA-1/MD5/size，保存本地 SHA-256 与 ffprobe 媒体契约，并始终输出
`training_ready=false`：

```bash
uv run --offline python prepare_archive.py \
  configs/archive_pair_candidate_specs_v2.json \
  data/archive_pair_pilot_v3/candidates.jsonl \
  data/archive_pair_pilot_v3/scan-report.json
uv run --offline python prepare_archive_media.py \
  data/archive_pair_pilot_v3/candidates.jsonl \
  data/archive_pair_pilot_v3/media \
  data/archive_pair_pilot_v3/media-manifest.json \
  --max-total-bytes 100000000
```

该 pilot 的 4 个定向 item 中，两个现代家庭视频只使用 Public Domain Mark，许可门禁直接
拒绝；Creative Commons 明确说明 PDM 是给已经属于公版的作品加标签，不是把作品投入公版的
法律工具。其余两个许可相容视频下载后也因 full-duration 语义不满足目标而拒绝：Ruby 狗视频
是户外，Neve 猫视频没有清晰持续的 playing 动作。审核证据在
`docs/case-studies/artifacts/video-dit/archive-pair-pilot-v3/`，因此 IA 本轮没有改变 coverage。

#### 自有/逐视频授权素材的两阶段导入

公开数据源不足以补齐正式 C1 时，使用 `prepare_local_contribution.py`。它不会把一份许可
JSON 直接当成已审训练数据：`create` 只生成 `training_ready=false` 的待审 bundle，包含锁定
原媒体 SHA-256、rights receipt、规范化 16-frame clip、锁定的完整原视频、全长 24-frame 联系表和密集两秒窗口。
`approve` 必须再绑定完整审核决定；任一主体/动作/scene/无剪辑确认不是显式 `true`，或哈希不匹配，
就不能进入 `merge`。

复制三个 `configs/local_contribution_*.example.json` 到与原 MP4 同一输入目录，先把 receipt 中的
`media_sha256` 替换为原文件实算值：

```bash
cd code/video_dit
shasum -a 256 /path/to/contribution/source.mp4

uv run --offline python prepare_local_contribution.py create \
  /path/to/contribution/spec.json \
  /path/to/contribution/receipt.json \
  data/local-contributions/luna-play-001-staged
```

先播放并完整查看 `source-media/` 中的原视频；`review/full-contact.jpg` 只用于快速导航，不能替代
逐段播放。再查看 `review/selected-window.jpg`，并用 staged `evidence.json`、
`source.jsonl` 和 receipt 中的实算哈希替换 review 模板里的占位值：

```bash
uv run --offline python prepare_local_contribution.py approve \
  data/local-contributions/luna-play-001-staged \
  /path/to/contribution/review.json \
  data/local-contributions/luna-play-001-approved

uv run --offline python prepare_local_contribution.py merge \
  data/commons_g1_provenance_v3/source.jsonl \
  data/local-contributions/luna-play-001-approved \
  --output data/formal-c1-expanded-v1

uv run --offline python prepare_real_data.py coverage \
  data/formal-c1-expanded-v1/source.jsonl --target-per-triple 50 \
  --min-train-pair-source-groups 10 \
  --report data/formal-c1-expanded-v1/coverage.json
```

`merge` 不原地修改基础数据；它重建一个新的可移动目录，并重验每个 clip、完整原视频、receipt、
联系表、human review 与 manifest 哈希。每个批准 bundle 的完整副本会保存在
`contribution-audit/`，因此移动或删除输入目录后仍可独立复核。`BASE` 必须始终是固定的原始
Commons 数据树；为避免丢掉旧审计链，CLI 会拒绝把上一次 merge 输出继续作增量 BASE。每次扩容都应
从同一 BASE 传入截至当时的**全部** approved bundles。`contributor_id`/`animal_id`/
`capture_session_id` 只用于生成稳定哈希组；同一动物与同次拍摄不得为了 coverage 拆成多个
session。软件能强制相同 ID 的分组一致，但不能自动识别人为改名的同一动物/事件；正式 split 前仍需
人工身份/事件审计与 pHash/embedding 近重复检查。这条链路只解决可审计导入，不会降低 held≥50
和每个训练 pair≥10 的正式门禁。

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
