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
