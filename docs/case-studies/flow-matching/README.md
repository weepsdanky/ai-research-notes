# Flow Matching 学习路径：从 Video DiT/DDPM 到 LPM Base Model

> 目标：不扩大当前 1.6M Video DiT；保留 backbone、数据、条件与 VAE，单独替换生成过程，并在一次公平 A/B 中掌握 Flow Matching。

## 1. 完成标准

完成本模块后，应能独立做到：

1. 从 $z_t=(1-t)z_0+t\epsilon$ 推导 $\dot z_t=\epsilon-z_0$；
2. 解释 conditional path 的逐样本 velocity target，和 marginal velocity field 的关系；
3. 区分 DDPM 的离散 posterior 与 Flow 的连续 ODE，代码中不交叉调用；
4. 说清本仓库的方向：训练 path 是 data $t=0\to1$ noise，生成用负 $\Delta t$ 从 $1\to0$；
5. 用 Euler/Heun 在 10–30 步从噪声生成视频，并按 NFE 公平比较；
6. 解释 epsilon prediction、diffusion v-prediction 与 linear-flow velocity 的差异；
7. 在 baseline 稳定后，单变量测试 timestep sampling 与 time shift。

本模块交付物：

- 数学与代码地图：[math-and-engineering.md](./math-and-engineering.md)
- 一手资料卡：[primary-source-notes.md](./primary-source-notes.md)
- 公平实验协议：[video-dit-flow-case-study.md](./video-dit-flow-case-study.md)
- 实现：[`code/video_dit/flow_matching.py`](../../../code/video_dit/flow_matching.py)
- 训练入口：[`code/video_dit/train_flow.py`](../../../code/video_dit/train_flow.py)
- 采样入口：[`code/video_dit/sample_flow.py`](../../../code/video_dit/sample_flow.py)
- 单元/组合测试：[`code/video_dit/tests/test_flow_matching.py`](../../../code/video_dit/tests/test_flow_matching.py)

## 2. 最短论文顺序

只精读四项，其余按问题查：

1. **[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)**：主论文。读 §2–4，回答 CNF、probability path、conditional objective 与 OT path。
2. **[Flow Straight and Fast](https://arxiv.org/abs/2209.03003)**：用直线路径理解 rectified flow、粗步 ODE 与 reflow；不要把“直线 conditional path”误写成“学习后的所有 marginal trajectories 必然直线”。
3. **[Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)** + [官方代码](https://github.com/facebookresearch/flow_matching)：把 path、scheduler、loss、solver 分成四个接口，并核对符号。
4. **[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)**：只读 rectified-flow formulation、timestep sampling 与 resolution-dependent shift，作为工程消融来源，不直接照搬到小视频。

按需补：

- CNF/ODE 背景：[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)。只需理解 $\dot z_t=v_\theta(z_t,t,c)$ 与数值积分。
- minibatch OT/CFM 变体：[Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482)。baseline 不先引入 OT coupling。
- 视频系统：[LTX-Video](https://arxiv.org/abs/2501.00103)、[HunyuanVideo](https://arxiv.org/abs/2412.03603)。只提取 latent flow、time shift、训练/采样约定；先完成本地线性 path。

停止规则：第一轮不读 stochastic interpolants、Schrödinger bridge、DMD 或 consistency distillation。它们不是验证基础 CFM 闭环的依赖。

## 3. 四个学习阶段

### Phase 0：公式与接口（半天）

- 手推 linear path、velocity、反向 Euler 与 Heun；
- 运行公式/solver tests；
- 在代码里确认没有 beta、$q(z_{t-1}\mid z_t,z_0)$ 或 DDIM update。

```bash
cd code/video_dit
uv run pytest -q tests/test_flow_matching.py
```

### Phase 1：训练闭环（半天）

```bash
uv run python train_flow.py --dry-run --device cpu \
  --batch-size 2 --output-dir runs/flow-dry
```

验收：一次真实 optimizer update；checkpoint 明确写入 `generative_process`、flow path 和 time-sampling contract；模型输出固定为 $C$ channels。

### Phase 2：canonical overfit（1–2 天）

用与 DDPM canonical run 相同的模型、数据、batch、优化器预算和 seed：

```bash
uv run python train_flow.py \
  --device mps --curriculum canonical --max-steps 3000 --batch-size 8 \
  --hidden-size 128 --num-pairs 4 --num-heads 4 --patch-size 4 \
  --learning-rate 2e-4 --time-sampling uniform --time-shift 1 \
  --output-dir runs/canonical-flow

uv run python sample_flow.py runs/canonical-flow/video_dit_flow_inference.pt \
  --prompt "an orange cat walking right in the snow" \
  --device mps --solver euler --steps 20 --seed 42
```

验收：10/20/30 Euler NFE 都能生成；20-step 输出至少达到当前 canonical DDPM 的可辨识主体/方向门槛。若失败，先检查时间方向、latent scaling、time embedding scale 和 condition，而不是加模型。

### Phase 3：solver 与 timestep 消融（1–2 天）

固定 checkpoint，先比较 Euler；Heun 必须按 NFE 比较。随后重新训练独立 runs：

- uniform, shift 1（baseline）；
- logit-normal, shift 1；
- uniform, shift $>1$；
- 最后才组合 logit-normal + shift。

完整预注册表见 [video-dit-flow-case-study.md](./video-dit-flow-case-study.md)。

## 4. 进入音频条件前的 gate

- [ ] path endpoints、velocity 符号、反演性质测试通过；
- [ ] CPU dry-run 和 MPS short run 都通过；
- [ ] 10–30 NFE 从 pure noise 生成，不调用 DDPM posterior；
- [ ] 至少一个 canonical Flow checkpoint 达到方向/主体可辨识；
- [ ] DDPM vs Flow 使用同一评测 seeds/prompts，并报告 wall time 与 NFE；
- [ ] 能解释为什么两种 training loss 的数值不能横向比较；
- [ ] 将结论写成“支持什么/不能推出什么”，再进入 speech-driven conditioning。
