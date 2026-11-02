# Video DiT on a 24 GB Apple Silicon Mac: an implementable design

> 时期：Phase 0 — 架构与可运行基线（2026-08-13）  
> 状态：历史设计基线；实现仍可运行，但不代表自然视频训练已经成功  
> 顺序：[总索引](README.md) → 本文 → [Phase 0 数据版图](02-phase-0-training-data-landscape.md) → [Phase 1 真实数据计划](03-phase-1-real-data-and-compositional-plan.md)  
> Status: implemented educational model and local evidence record, 2026-08-13
> Target machine: 24 GB M5 Mac
> Scope: a small trainable Video DiT for learning; pretrained text-to-video is documented only as an external option

## Executive decision and implemented result

The repository implements one deliberately bounded lane:

1. **Educational Video DiT** — train from scratch on the Mac. It uses 8-frame, 32×32 RGB symbolic videos and a small, paper-grounded Latte Variant 1 backbone. A parsed condition controls subject, color, left/right/bounce/stand motion, and scene. It exports an actual temporal MP4 and a separately labelled denoising GIF.
2. **Pretrained natural-language video is outside this project.** Apple MLX Wan2.1 is documented later as one optional external route. There is no Wan backend, backend selector, downloader, or Wan Web UI integration in `code/video_dit`.

**Local measurement.** A completed `concept + canonical + velocity` run reached 3,000 optimizer updates with 2,472,112 parameters. Its full resume checkpoint remains local at `code/video_dit/runs/canonical-velocity/video_dit.pt`. A tensor/state-only inference checkpoint and representative MP4/GIF/contact-sheet evidence are distributed under `code/video_dit/demo_artifacts/`; hashes are recorded in `ARTIFACTS.md`, so a fresh clone can run inference without retraining.

Do **not** describe the educational lane as open-vocabulary text-to-video. Its “language” is a finite, inspectable symbolic grammar. Do **not** display diffusion sampling states as if they were animation frames. The primary artifact is an MP4 whose frames are the generated temporal axis; a denoising preview is an optional diagnostic with a different label.

This boundary is necessary because the original Latte-quality recipe used 8 × A100 80 GB GPUs, while the official Latte-1 text-to-video checkpoint is about 1B parameters and includes a T5-XXL text encoder. Those systems are useful references, not plausible from-scratch training targets for a 24 GB Mac. The relevant paper facts are in the [Latte experiment setup](https://arxiv.org/html/2401.03048#S4.SS1), and the released text-to-video configuration is in the [official Latte-1 transformer config](https://huggingface.co/maxin-cn/Latte-1/blob/main/transformer/config.json).

## Evidence labels used below

- **Paper fact** means a claim stated by the Latte paper.
- **Official-code fact** means a value observed in the official Vchitect or Apple MLX source/configuration.
- **Local measurement** means a result measured on the target development Mac, not a paper result.
- **Design decision** means this project's deliberately smaller choice.
- **Inference** means a conclusion derived from cited facts; it is not presented as a source claim.

## What Latte actually contributes

### Latent diffusion and video tokens

**Paper fact.** Latte encodes each video frame with a pretrained VAE, performs diffusion over the video latents, and uses a Transformer as the denoiser. It trains both the simple epsilon-prediction loss and the variational term needed to learn reverse-process covariance. The output is reshaped back to the same video-latent geometry and contains predicted noise and covariance. See [Latte §3.1–3.2](https://arxiv.org/html/2401.03048#S3).

For latent video

\[
V_L \in \mathbb{R}^{B\times F\times C\times H\times W},
\]

uniform frame patch embedding produces

\[
z \in \mathbb{R}^{B\times F\times S\times D}, \qquad
S=(H/p)(W/p).
\]

The important computational trick is to avoid attention over a flat sequence of length \(F S\):

- spatial attention reshapes to \((B F)\times S\times D\);
- temporal attention reshapes to \((B S)\times F\times D\).

With 8 frames and 64 spatial tokens per frame, full attention would operate on a length-512 sequence. Factorized attention instead operates on 8 batches of length 64 and 64 batches of length 8. This preserves direct spatial and temporal interactions while making a small Mac implementation practical.

### The four Latte variants

**Paper fact.** [Latte §3.2](https://arxiv.org/html/2401.03048#S3.SS2) evaluates four decompositions:

1. **Variant 1, interleaved fusion:** alternate a complete spatial Transformer block and a complete temporal Transformer block.
2. **Variant 2, late fusion:** place the spatial half of the network before the temporal half.
3. **Variant 3:** perform spatial attention followed by temporal attention inside every Transformer block.
4. **Variant 4:** split attention heads between parallel spatial and temporal paths, then add their outputs.

**Paper fact.** Variant 1 performs best; Variant 4 uses roughly one quarter of the FLOPs of the first three variants but performs worst. The paper attributes the differences to how strongly the spatial and temporal representations are decoupled. See the [variant ablation and analysis](https://arxiv.org/html/2401.03048#S4.SS2).

**Design decision.** The educational model therefore uses Variant 1: four spatial/temporal pairs, eight complete Transformer blocks in total.

### Paper-grounded best practices

**Paper fact.** The final best-practice set is [summarized in Latte §4.3](https://arxiv.org/html/2401.03048#S4.SS3):

- Variant 1;
- uniform per-frame 2D patch embedding instead of temporally compressed 3D tubes;
- scalable adaptive layer normalization, S-AdaLN;
- absolute temporal positional embedding rather than RoPE for fixed-length videos;
- joint video-image training.

The official base implementation clarifies several details:

- its S-AdaLN block is implemented as DiT-style **adaLN-Zero**, including shift, scale, and zero-initialized residual gates;
- it uses fixed 2D spatial and fixed 1D temporal sine/cosine embeddings;
- the temporal embedding is added before the first temporal block;
- class dropout defaults to 0.1 for classifier-free guidance;
- the final projection and its modulation are zero-initialized;
- learned sigma doubles the output channels;
- CFG is applied to the four epsilon channels while leaving the variance output untouched.

These are directly visible in [`models/latte.py`](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/models/latte.py).

### Model scale in the paper

**Paper fact.** [Latte Table 2](https://arxiv.org/html/2401.03048#S4.SS1) reports:

| Model | Total Transformer blocks | Hidden width | Heads | Parameters |
|---|---:|---:|---:|---:|
| Latte-S | 12 | 384 | 6 | 32.48M |
| Latte-B | 12 | 768 | 12 | 129.54M |
| Latte-L | 24 | 1024 | 16 | 456.81M |
| Latte-XL | 28 | 1152 | 16 | 673.68M |

In the base official implementation, those blocks are consumed in spatial/temporal pairs. Thus Latte-XL's 28 total blocks mean 14 spatial plus 14 temporal blocks, not 28 of each.

### Paper model versus the released Latte-1 T2V checkpoint

This distinction matters when deciding what “faithful to Latte” means.

**Paper fact.** For the text-to-video extension, the authors use Variant 1, initialize spatial blocks from PixArt-α 512×512, and train on Vimeo25M. See [Latte §4.5](https://arxiv.org/html/2401.03048#S4.SS5).

**Official-code fact.** The later public Latte-1 T2V checkpoint is a different, larger PixArt-style implementation:

- 28 configured layers;
- the source constructs **28 spatial and 28 temporal blocks**, rather than 28 total;
- 16 heads × 72 dimensions = width 1152;
- patch size 2 over a 64×64 latent grid, corresponding to 512×512 pixels with an 8× VAE;
- four input latent channels and eight output channels for learned variance;
- `ada_norm_single`, the PixArt conditioning form;
- 4096-dimensional T5 caption features and a 1152-dimensional cross-attention projection.

Sources: [Latte-1 config](https://huggingface.co/maxin-cn/Latte-1/blob/main/transformer/config.json) and the official [`models/latte_t2v.py`](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/models/latte_t2v.py).

**Official-code fact.** The public base training script is CUDA/DDP-only and explicitly rejects its text-to-video mode. Consequently, the published repository does not provide a Mac-ready reproduction of Latte-1's text training. See [`train.py`](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/train.py).

The educational implementation therefore reproduces the paper's conceptual Variant 1 and DiT conditioning; it is not presented as the released 1B Latte-1.

## Educational model specification

### Concrete target

**Design decision.** Use the following fixed first milestone:

| Item | Value |
|---|---|
| Video tensor | `[B, 3, 8, 32, 32]`（代码采用 channel-first） |
| Video duration | 8 frames at 8 FPS, one generated second |
| Domain | synthetic symbolic cat, dog, and red-coated person |
| Patch embedding | uniform 2D patchify independently per frame |
| Backbone | Latte Variant 1 |
| Depth | 4 spatial + 4 temporal blocks, interleaved |
| Position | fixed 2D spatial + fixed 1D temporal sin/cos |
| Condition | diffusion timestep + structured prompt embedding |
| Block conditioning | adaLN-Zero |
| Diffusion target | epsilon or velocity, independently selected from the mode |
| Guidance | faithful defaults to 10% condition dropout and supports CFG; concept disables both |
| Sampling | DDPM for reference; DDIM for interactive inference |
| Output | real MP4 plus optional denoising diagnostic |
| Measured model size | 2,472,112 concept parameters; 2,478,304 faithful parameters |

The implemented first milestone remains pixel-space. A VAE would add compression and reconstruction error before the learner has verified time attention, condition injection, and diffusion; a framewise autoencoder remains a possible later experiment, not a current component.

### Forward path

```text
RGB video [B,3,F,H,W]
  -> per-frame patch projection
  -> [B,F,S,D]
  -> add fixed spatial position
  -> spatial block on [(B*F),S,D]
  -> reshape + add fixed temporal position
  -> temporal block on [(B*S),F,D]
  -> repeat four times
  -> adaLN-conditioned final projection
  -> unpatchify [B,Cout,F,H,W]
```

The same timestep and prompt condition modulate both members of each spatial/temporal pair. Each block keeps its own LayerNorm, attention, MLP, and modulation parameters; this is what makes it Variant 1 rather than Variant 3.

### Two independent configuration axes

`mode` and `prediction_type` are orthogonal in the implementation. The mode selects the surrounding training tricks; `--prediction-type epsilon|velocity` independently selects the diffusion target. The default CLI configuration is `faithful + canonical + epsilon`, 3,000 updates, batch 8. That default is useful for inspecting the paper-style parameterization, but it is not claimed to match the recognizability of the completed concept + velocity run at the same budget.

#### `concept`

- fixed posterior variance;
- no label dropout or CFG;
- the EMA checkpoint slot copies raw weights every update rather than averaging;
- supports either epsilon or velocity prediction;
- adaLN modulation remains because it is the central way timestep conditioning reaches every block.

This mode answers: how do patch tokens, spatial attention, temporal attention, and reverse diffusion produce a video?

#### `faithful`

- adaLN-Zero residual gates and zero output initialization;
- learned-range posterior variance;
- epsilon is the paper-oriented default; velocity remains an explicit non-paper ablation;
- prediction MSE plus the VLB variance term when learned variance is enabled;
- 10% condition dropout and CFG;
- EMA 0.9999;
- DDPM and DDIM;
- gradient clipping and complete resume checkpoints.

This mode answers: which stability and quality choices surround the core architecture? The learned-variance formulation follows the paper's training description and the official base implementation's doubled output channels. `faithful` does not by itself force `prediction_type`; paper-oriented commands must retain the default epsilon explicitly or implicitly.

## Synthetic symbolic prompt dataset

### Why synthetic first

The purpose is to learn video diffusion, not dataset plumbing or copyrighted web-video curation. Programmatic clips provide exact labels for subject identity, motion, direction, foreground centroid, background, and frame-to-frame ground-truth trajectory. They also let tests distinguish a genuine generated animation from a sequence of unrelated images.

### Implemented prompt parser

The parser maps English/Chinese keywords into exactly four finite factors:

```text
subject = cat | dog | person
color   = orange | brown | red | blue | black
motion  = walk_right | walk_left | bounce | stand
scene   = snow | grass | night | studio
```

Example supported prompts include `an orange cat walking right in the snow`, `a brown dog walking left on grass`, and `a person in a red coat walking in the snow to the right`. The subject is mandatory and an unsupported subject raises a validation error. Unknown adjectives are ignored; missing motion defaults to `walk_right`, scene to `snow`, and color to red for a person or orange for an animal. The UI displays the resolved four-factor condition. This is keyword parsing, not natural-language understanding.

### Visual vocabulary

- **Cat:** triangular ears, compact body, curved thin tail.
- **Dog:** floppy/rectangular ears, longer muzzle, thicker tail.
- **Red-coated person:** circular head, red torso, two legs whose phase alternates during walking.
- **Snow:** pale blue/white ground and slowly descending sparse flakes.
- **Grass:** green ground with stable texture.
- **Night:** dark blue background and a fixed small moon/star field.

The renderer uses deterministic PIL primitives at the target resolution, with stable identity geometry and coat color across all eight frames. It does not currently add anti-aliasing or randomize size, starting position, hue, or gait phase beyond factor/seed-driven rendering.

### Motion generator

Generate motion from explicit continuous trajectories before rasterization:

- left/right walk: constant horizontal velocity plus small vertical gait oscillation;
- bounce: parabolic or sinusoidal vertical motion;
- leg/tail/ear phases: smooth periodic functions, never independent per-frame randomness;
- snowflakes: deterministic velocity per flake and wraparound at the frame boundary.

This produces a learnable distribution with real temporal structure. Randomly positioning a sprite independently in every frame would train flicker, not motion.

### Implemented curricula and absent splits

Implementation starts with an explicitly labelled **canonical overfit
curriculum** containing one fixed clip each for an orange cat, a brown dog, and
a red-coated person. This is an end-to-end recognizability gate, not a diversity
claim. After that gate passes, the **factorial curriculum** below is the separate
compositional experiment; the selected curriculum is stored in every checkpoint
and summary.

The current code generates examples from `(base_seed, index)` instead of storing frames. `CanonicalVideoDataset` repeats exactly three deterministic clips. `SyntheticVideoDataset` draws the four factors independently from the complete finite vocabulary for the harder `factorial` curriculum. There is currently no stored JSONL manifest, validation split, compositional holdout, or diversity benchmark. Horizontal flip does update the left/right condition so augmentation does not corrupt labels.

## 24 GB M5 Mac compute plan

### Local measured baseline

**Local measurement.** An earlier model revision with 8-frame 32×32 RGB input and eight alternating spatial/temporal blocks produced the following isolated training-step microbenchmark on the target Mac:

| Batch | Seconds/step | Clips/second | Frames/second |
|---:|---:|---:|---:|
| 8 | 0.0584 | 137.0 | 1,095.9 |
| 16 | 0.1207 | 132.6 | 1,060.5 |
| 32 | 0.2436 | 131.4 | 1,051.0 |

These are core-step microbenchmarks, not promised end-to-end training times. Dataset rendering, validation samples, MP4 encoding, checkpoints, first-use Metal compilation, and thermal behavior add wall time.

Throughput was nearly flat across these batch sizes. The implemented CLI now defaults to batch 8, which was also used for the completed quality run; the table remains an earlier kernel-level measurement rather than a current wall-clock promise.

### Original planning estimates

| Stage | Batch | Updates | Seen clips | Benchmark-only time | Purpose |
|---|---:|---:|---:|---:|---|
| smoke | 8 | 1,000 | 8,000 | ~58 s | shapes, loss decrease, MP4 export |
| first motion | 16 | 10,000 | 160,000 | ~20 min | direction and temporal continuity |
| useful model | 16 | 60,000 | 960,000 | ~2.0 h | recognizable symbolic subjects |
| quality run | 16 | 100,000 | 1,600,000 | ~3.35 h | stronger prompt fidelity and diversity |

These rows were planning estimates, not completed quality results. The actual completed concept + velocity run reached 3,000 steps in 1,120.7 seconds for the resumed 2,900-update segment, according to `summary.json`; renderer, sampling, checkpointing, MPS compilation and machine load make it inappropriate to extrapolate that number linearly to 100k.

### Memory and checkpoint policy

At 2,472,112–2,478,304 parameters, FP32 weights occupy about 9.4 MiB. Weights, gradients, EMA, and Adam first/second moments are still only on the order of tens of MiB; activations dominate but remain modest at 32×32 and eight frames. A 24 GB machine therefore has ample capacity for this educational configuration.

- prefer MPS FP32 for the first reproducible run;
- make mixed precision opt-in only after loss/gradient parity checks;
- use the implemented batch-8 default for the first run and benchmark larger batches locally;
- the implemented 3,000-step default samples and checkpoints every 1,000 steps; long runs should use a wider interval;
- each completed run writes a trusted resume checkpoint with raw model, EMA, optimizer, step, configuration, RNG states and explicit data/optimization contracts, plus a separate compact `video_dit_inference.pt` that CLI/UI load with `weights_only=True`;
- the renderer runs through the PyTorch data loader on CPU; the default uses zero workers and does not add a custom prefetch queue.

### Completed 3,000-step canonical run

The successful local training command was:

```bash
cd code/video_dit
uv run python train.py \
  --mode concept --curriculum canonical --prediction-type velocity \
  --device mps --max-steps 3000 --batch-size 8 \
  --hidden-size 128 --num-pairs 4 --num-heads 4 --patch-size 4 \
  --learning-rate 2e-4 --timesteps 1000 \
  --ema-decay 0.99 --horizontal-flip-prob 0 \
  --sample-every 1000 --checkpoint-every 1000 \
  --output-dir runs/canonical-velocity
```

**Local measurement.** The checkpoint contains 2,472,112 parameters and 3,000 updates; the final mini-batch loss was 0.0183286. With seed 42, 100-step DDIM and effective CFG=1, the generated motion-weighted horizontal centroid displacement was +6.58 pixels for the orange cat walking right in snow, −4.36 for the brown dog walking left on grass, and +7.49 for the red-coated person walking right in snow. These signs match the three conditions and establish nonzero directed temporal motion. They are diagnostics on memorized canonical examples, not held-out video metrics.

The full resume checkpoint is `code/video_dit/runs/canonical-velocity/video_dit.pt`. The repository distributes the same EMA model for safe inference as `code/video_dit/demo_artifacts/video_dit_inference.pt`, together with three MP4 files, three denoising-diagnostic GIFs, contact-sheet PNGs, and `temporal_metrics.json` under `demo_artifacts/samples/`. Per-generation JSON, optimizer state, RNG state and logs remain in the gitignored `runs/` tree. See `ARTIFACTS.md` for hashes and the exact portability boundary.

The CLI default is deliberately different: `faithful + canonical + epsilon`, 3,000 updates, batch 8. That default preserves the paper-oriented epsilon/learned-variance recipe but is not promised to produce the same recognizability within 3,000 steps. A 60k or 100k faithful/factorial run remains a proposed experiment, not a completed result.

## Real MP4 inference and evaluation

### Output contract

For one request, produce:

```text
run/
  model_<condition>_seed-<seed>_<request-hash>_<run-id>.mp4
  model_<condition>_seed-<seed>_<request-hash>_<run-id>_denoising-diagnostic.gif
  model_<condition>_seed-<seed>_<request-hash>_<run-id>.json
  model_<condition>_seed-<seed>_<request-hash>_<run-id>_temporal-metrics.json
```

The MP4 contains the model's eight output frames in temporal order and is the primary result. The GIF advances over diffusion sampling states; each GIF frame is a contact sheet of the complete animation time axis. The JSON records the checkpoint, parsed condition, seed, sampler configuration, effective CFG behavior, prediction type and elapsed time. `inference.py` also writes the temporal metric JSON; individual PNG frames and a standalone contact sheet are not part of its regular output contract.

### Structural checks

Use `ffprobe` or a decoder library to assert:

- MP4 is readable from start to finish;
- H.264 with `yuv420p` and fast-start metadata is used;
- resolution is 32×32 before optional nearest-neighbor display scaling;
- frame rate is 8 FPS;
- the decoded frame count is eight for the default model.

### Temporal and prompt checks

Because the synthetic renderer provides ground truth, evaluate more than “the file opens”:

- **direction accuracy:** sign of generated foreground-centroid displacement agrees with left/right condition;
- **motion energy:** adjacent-frame pixel difference is nonzero but lower than a shuffled-frame baseline;
- **background stability:** nonmoving background regions have low temporal variance;
- **identity stability:** subject color and silhouette statistics remain consistent across frames;
- **scene adherence:** snow/grass/night palette classifier matches the condition;
- **red-coat adherence:** the person's torso stays red across the clip;
- **diversity:** multiple seeds for the same prompt are not pixel-identical.

The implemented evaluator currently computes a motion-weighted horizontal centroid path, endpoint displacement, adjacent-frame MSE, endpoint MSE and their ratio. Palette, identity, diversity and multi-seed accuracy checks above remain useful future evaluations; they are not implemented acceptance results. No held-out correctness percentage is claimed for the canonical checkpoint.

### UI behavior

The implemented Web UI exposes:

- prompt with supported grammar examples;
- parsed subject/color/motion/scene JSON;
- seed, DDIM sampling steps, CFG scale, diagnostic snapshot count, FPS, and checkpoint;
- primary MP4 player titled **Final temporal animation (MP4)**;
- GIF titled **Denoising diagnostic (GIF — diffusion time, not animation time)**;
- downloadable generation metadata;
- a separately labelled dataset renderer preview that is never represented as model output.

This naming prevents the earlier confusion between content animation and diffusion progress.

Run the distributed trained checkpoint on its separate port with:

```bash
cd code/video_dit
uv run python web_ui.py \
  --device mps --server-name 127.0.0.1 --server-port 7861 \
  --output-dir runs/canonical-velocity/web_ui
```

Open `http://127.0.0.1:7861/`. With no positional argument, the UI loads the distributed `demo_artifacts/video_dit_inference.pt`; pass a different safe inference checkpoint explicitly when comparing runs.

## Optional external route: Apple MLX Wan2.1

Apple's [`ml-explore/mlx-examples` Wan2.1 implementation](https://github.com/ml-explore/mlx-examples/tree/main/video/wan2.1) is one defensible external path for broader prompts on Apple Silicon. It is not a backend implemented or promised by this repository.

**Official-code facts:**

- it supports text-to-video and image-to-video;
- the Wan2.1 T2V 1.3B model uses about 10 GB RAM for an unquantized 81-frame run with cache disabled, or about 14 GB with the cache;
- on an M4 Max, its published measurement is about 90 seconds for one DiT iteration at 81 frames;
- the documented example uses 832×480, 81 frames, 50 steps, CFG 5, and exports MP4;
- setting guidance to 1 skips the unconditional CFG pass and roughly halves computation per step;
- `ffmpeg` is required for saving videos.

The exact primary source is the [Apple MLX Wan2.1 README](https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/README.md); model weights come from the [official Wan-AI 1.3B repository](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).

**Inference.** At the published M4 Max rate, 50 × 90 seconds is approximately 75 minutes for DiT iterations alone; text encoding, VAE decode, compilation, and MP4 output add overhead. The target is an M5 rather than the measured M4 Max, so no exact runtime should be promised until a local smoke test is recorded. The 10–14 GB reported runtime footprint suggests the 1.3B lane can fit in 24 GB, but macOS and other applications still require headroom.

Illustrative external commands follow the cited MLX interface and must be run from that external project after its own installation:

```bash
# Standard quality-oriented example; expect a long run.
python txt2video.py \
  'A person wearing a red coat walks through the snow' \
  --size 832x480 --frames 81 --steps 50 --guidance 5.0 --seed 42 \
  --no-cache --output red_coat_snow.mp4

# A cheaper functional smoke test; quality is not the reference setting.
python txt2video.py \
  'A cat walks through the snow' \
  --frames 9 --steps 4 --guidance 1.0 \
  --no-cache --output cat_smoke.mp4
```

The second command is a project smoke-test proposal, not a source quality recommendation. Its purpose is to verify installation and MP4 export before committing to an hour-scale run.

Wan remains outside the educational project. A future integration would need a separately scoped backend selector, lazy installation/download, runtime warnings, and its own acceptance tests; none of those components exists in the current code.

## Honest capability limits

### What the educational model can demonstrate

- a Transformer denoiser over video tensors;
- frame patchification and unpatchification;
- factorized spatial and temporal self-attention;
- fixed spatial/temporal positions;
- diffusion timestep conditioning through adaLN-Zero;
- structured conditional generation and CFG;
- learned variance and hybrid loss as visible, removable tricks;
- DDPM/DDIM sampling;
- coherent, genuinely temporal symbolic cats, dogs, and a red-coated person for the three completed canonical-overfit examples;
- end-to-end training and MP4 inference on the target Mac.

### What it cannot honestly promise

- photorealistic people or animals;
- arbitrary natural-language understanding;
- identity consistency across long videos;
- unseen object concepts outside the symbolic renderer;
- Latte-1, HunyuanVideo, or Wan-level quality from local from-scratch training;
- reproduction of the paper's benchmark scores or its 8 × A100 training regime.

### What an external pretrained route could add

Wan2.1 supplies broad natural-language and visual priors learned elsewhere, so it can attempt realistic cat, dog, and red-coated-person prompts. It still does not guarantee anatomy, motion, text adherence, or temporal identity on every seed. On a 24 GB Mac it is a separate slow demonstration route, not this project's training target or UI backend.

## Implementation acceptance record

- [x] Video project is a separate folder from the image DiT project.
- [x] The model consumes and emits explicit `[B,C,T,H,W]` tensors and alternates complete spatial/temporal blocks.
- [x] Shape tests cover the five-dimensional forward path and reject a wrong temporal extent.
- [x] adaLN-Zero and the final projection begin at the intended zero initialization.
- [x] CFG arithmetic has a deterministic unit test; concept checkpoints disable CFG during inference because they were trained without condition dropout.
- [x] Learned-range variance produces a finite VLB term and the implementation detaches the prediction branch for that term.
- [x] Horizontal flip updates left/right labels.
- [x] Checkpoint resume reproduces the next loss and weights with restored RNG state when device type and the validated data/optimization contracts match; mismatched device, seed, batch size, workers, flip probability, learning rate, weight decay, EMA or grad clipping fail closed.
- [x] A CPU dry-run performs a real optimizer update and saves a complete checkpoint.
- [x] Media tests invoke ffmpeg/ffprobe and validate a decodable, correctly sized multi-frame MP4.
- [x] MP4 animation time and GIF diffusion time are visually and semantically separated in CLI metadata and the Web UI.
- [x] The local 3,000-step canonical-overfit checkpoint generated three directed-motion MP4s; measured displacement signs match all three training conditions.
- [ ] The completed checkpoint is not held out and does not establish factorial generalization, diversity, or open-vocabulary text-to-video.
- [ ] No dedicated test currently proves cross-frame causal influence by perturbing one input frame with temporal blocks enabled/disabled.
- [ ] The MPS training path completed locally, but there is no portable automated MPS test because CI hardware may not provide MPS.
- [ ] Wan2.1 is only an external route; no backend integration is implemented.

## Primary sources

- Xin Ma et al., [“Latte: Latent Diffusion Transformer for Video Generation”](https://arxiv.org/abs/2401.03048), and its [HTML full text](https://arxiv.org/html/2401.03048).
- TMLR/OpenReview, [Latte publication record](https://openreview.net/forum?id=ntGPYNUF3t).
- Vchitect, [official Latte repository](https://github.com/Vchitect/Latte) and audited commit [`ad4b85a`](https://github.com/Vchitect/Latte/tree/ad4b85a69bd26c66eebd65f4323718c80e7cf449).
- Vchitect, [base Latte implementation](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/models/latte.py).
- Vchitect, [Latte T2V implementation](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/models/latte_t2v.py).
- Vchitect, [UCF101 official training config](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/configs/ucf101/ucf101_train.yaml).
- Vchitect, [official Latte T2V sample config](https://github.com/Vchitect/Latte/blob/ad4b85a69bd26c66eebd65f4323718c80e7cf449/configs/t2x/t2v_sample.yaml).
- maxin-cn, [base Latte checkpoints](https://huggingface.co/maxin-cn/Latte/tree/main) and [Latte-1 text-to-video checkpoint](https://huggingface.co/maxin-cn/Latte-1/tree/main).
- Hugging Face Diffusers, [LattePipeline API](https://huggingface.co/docs/diffusers/api/pipelines/latte).
- Apple MLX, [Wan2.1 text-to-video and image-to-video implementation](https://github.com/ml-explore/mlx-examples/tree/main/video/wan2.1).
- Wan-AI, [official Wan2.1 T2V 1.3B weights](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and [Wan2.1 source repository](https://github.com/Wan-Video/Wan2.1).
