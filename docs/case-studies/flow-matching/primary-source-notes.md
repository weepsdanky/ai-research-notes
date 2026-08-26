# Flow Matching 一手资料笔记

> 范围：只收录经典/权威论文与作者官方实现；检索日期 2026-08-27。本文是数学讲义、代码实验与 case study 的事实底稿。每节明确区分“论文/代码事实”和“本地建议”。

## 0. 最短阅读路线

必读顺序：

1. [Neural ODE](https://arxiv.org/abs/1806.07366) 的 ODE 与 Continuous Normalizing Flow（CNF）部分；只需理解状态由向量场连续演化、ODE 解把 base distribution 推到 data distribution。
2. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) 的 §2–4；重点是 probability path、continuity equation、FM/CFM 等价和 Gaussian conditional path。
3. [Flow Straight and Fast / Rectified Flow](https://arxiv.org/abs/2209.03003)；重点是直线路径、velocity regression、reflow 与少步 Euler 的关系。
4. [Stable Diffusion 3 技术报告](https://arxiv.org/abs/2403.03206) 的 §2、§3、§5.1、§5.3.2；重点是 linear RF、logit-normal timestep sampling 和 resolution-dependent shift。
5. [LTX-Video](https://arxiv.org/abs/2501.00103) 的 §2.5；把上述目标迁移到 video latent，并观察 token-count-dependent shift。
6. 对照官方实现：[Meta Flow Matching](https://github.com/facebookresearch/flow_matching)、[TorchCFM](https://github.com/atong01/conditional-flow-matching)、[RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)、[EDM sampler](https://github.com/NVlabs/edm/blob/main/generate.py)、[SD3 micro-reference](https://github.com/Stability-AI/sd3-ref)、[LTX-Video](https://github.com/Lightricks/LTX-Video)、[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)。

按需再读：[Conditional Flow Matching / OT-CFM](https://arxiv.org/abs/2302.00482)、[Stochastic Interpolants](https://arxiv.org/abs/2303.08797)、[FFJORD](https://arxiv.org/abs/1810.01367)。它们不是完成第一版 linear-flow Video DiT 的前置阻塞。

## 1. CNF、probability path 与 Flow Matching

### 论文事实

Neural ODE 以神经网络参数化连续动力系统

\[
\frac{d z_t}{dt}=u_\theta(z_t,t),
\]

并由 ODE solver 从初值求轨迹；论文也用 instantaneous change-of-variables 构造 CNF（[Neural ODE](https://arxiv.org/abs/1806.07366)）。[FFJORD](https://arxiv.org/abs/1810.01367) 进一步以 Hutchinson trace estimator 扩展 likelihood 训练，但生成用的第一版 Flow Matching 不需要先实现 density/trace。

Flow Matching 指定一条边缘分布路径 \(p_t\)，再学习生成它的边缘向量场 \(u_t\)。密度与速度满足 continuity equation

\[
\partial_t p_t(x)+\nabla\cdot\left(p_t(x)u_t(x)\right)=0.
\]

直接计算边缘 \(u_t\) 通常困难。Flow Matching 论文改为采样 tractable conditional path \(p_t(x\mid x_1)\)，回归其已知 conditional vector field：

\[
\mathcal L_{\mathrm{CFM}}
=\mathbb E\left[\left\|v_\theta(x_t,t)-u_t(x_t\mid x_1)\right\|^2\right].
\]

其关键定理是 FM 与 CFM 的目标相差与参数无关的常数，因此梯度相同（[Flow Matching](https://arxiv.org/html/2210.02747)、[推广后的 CFM 定理](https://arxiv.org/html/2302.00482)）。训练因此是 **simulation-free regression**：训练时不必先解 ODE；采样时才解 ODE。

“Conditional Flow Matching”中的 conditional 首先指对单个 endpoint/coupling 条件化的概率路径，不等同于“文本条件生成”。文本、类别或音频条件可作为网络的额外输入，这是另一层 conditioning。

### 本地建议

- 第一版只实现 fixed linear path，不实现 likelihood、divergence、adjoint 或 adaptive solver。
- 单测应守住三个接口：`path.sample(x0, noise, t) -> xt`、`path.target(x0, noise, t) -> velocity`、`solver.step(model, x, t, dt) -> x_next`；不要把路径、网络和 solver 写成一个函数。
- Meta 官方库可作为 API 参照：其源码按 [`path`](https://github.com/facebookresearch/flow_matching/tree/main/flow_matching/path)、[`loss`](https://github.com/facebookresearch/flow_matching/tree/main/flow_matching/loss)、[`solver`](https://github.com/facebookresearch/flow_matching/tree/main/flow_matching/solver) 分层，并提供图像训练例子；不要为了第一版把整库复制进本仓库。

## 2. Linear path、velocity target 与时间方向

### 论文事实

采用“data 在 \(t=0\)，noise 在 \(t=1\)”约定：

\[
z_t=(1-t)z_0+t\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
\]

对 \(t\) 求导立即得到

\[
u_t=\frac{d z_t}{dt}=\epsilon-z_0.
\]

训练目标为 \(\|v_\theta(z_t,t,c)-(\epsilon-z_0)\|_2^2\)。生成从 \(z_1\sim\mathcal N(0,I)\) 向 \(z_0\) 反向积分；当 \(\Delta t>0\) 表示正的步长幅值时，Euler 更新是

\[
z_{t-\Delta t}=z_t-\Delta t\,v_\theta(z_t,t,c).
\]

LTX-Video 原文明确采用上述路径、target 与反向更新（[LTX-Video §2.5](https://arxiv.org/html/2501.00103#S2.SS5)）。Flow Matching 原论文常用相反端点：noise 在 \(t=0\)、data 在 \(t=1\)（[Flow Matching §4](https://arxiv.org/html/2210.02747#S4)）；于是速度和积分方向都反号，但数学等价。HunyuanVideo 官方 CLI 因此显式暴露 [`--flow-reverse`](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/config.py)。

### 本地建议

- 全仓固定 `t=0 clean, t=1 noise`，checkpoint 保存 `time_convention`；只在适配外部 checkpoint 时转换。
- 用解析轨迹做 solver 测试：若 oracle 恒定返回 \(\epsilon-z_0\)，任意步数 Euler 都应在浮点误差内从 \(\epsilon\) 回到 \(z_0\)。
- 代码变量区分 `dt_signed=t_next-t` 与 `step_size=abs(dt_signed)`；避免在“反向时间 + 手工负号”中重复翻转。

## 3. epsilon prediction、diffusion v 与 flow velocity

### 论文事实

一般 Gaussian path 写成

\[
z_t=\alpha_t z_0+\sigma_t\epsilon,
\qquad
u_t=\dot\alpha_t z_0+\dot\sigma_t\epsilon.
\]

因此“预测 \(\epsilon\)”与“预测 flow velocity”只是同一路径下可换算的参数化，但换算依赖 \(\alpha_t,\sigma_t\) 及其导数；SD3 报告给出了 CFM 到 noise-prediction weighted loss 的一般换算（[SD3 §2](https://arxiv.org/html/2403.03206#S2)）。

对 linear path：

\[
u=\epsilon-z_0,\quad
\hat z_0=\frac{z_t-t\hat\epsilon}{1-t},\quad
\hat u=\frac{\hat\epsilon-z_t}{1-t},
\]

或

\[
\hat\epsilon=\frac{z_t-(1-t)\hat z_0}{t},\quad
\hat u=\frac{z_t-\hat z_0}{t}.
\]

端点附近的除法会病态，这也是直接预测 flow velocity 更干净的工程接口。

还要区分 diffusion 文献中的 **v-parameterization**：对 variance-preserving path \(\alpha_t^2+\sigma_t^2=1\)，通常定义

\[
v_{\mathrm{diff}}=\alpha_t\epsilon-\sigma_t z_0,
\]

它不是一般意义的 \(dz_t/dt\)。该参数化用于少步稳定性可追溯到 [Progressive Distillation](https://arxiv.org/abs/2202.00512)；SD3 同时比较 epsilon、diffusion-v 与 RF velocity，不能因都写作 `v` 就混为同一 target（[SD3 §3/§5.1](https://arxiv.org/html/2403.03206#S3)）。

### 本地建议

- 枚举名用 `epsilon`、`diffusion_v`、`flow_velocity`，禁止只写 `v`。
- 第一版只训练 `flow_velocity`；实现换算函数和 property tests 是为了理解，不是为了同时训练三种 target。
- `DDPM posterior`、`alphas_cumprod`、posterior variance 不属于 linear Flow ODE sampler；保留旧 DDPM 代码用于公平 baseline，但不要从 Flow sampler 调用。

## 4. Euler、Heun 与 NFE

### 论文/官方代码事实

对带符号步长 \(h=t_{i+1}-t_i\)，Euler 为

\[
x_{i+1}=x_i+h f_\theta(x_i,t_i).
\]

Heun（explicit trapezoid）先预测

\[
\tilde x_{i+1}=x_i+h f_\theta(x_i,t_i),
\]

再校正

\[
x_{i+1}=x_i+\frac h2\left[f_\theta(x_i,t_i)+f_\theta(\tilde x_{i+1},t_{i+1})\right].
\]

Euler 是一阶；Heun 是二阶但通常每步两次网络调用。EDM 的论文 Algorithm 2 与[官方 `generate.py`](https://github.com/NVlabs/edm/blob/main/generate.py)给出生成模型中成熟的 Euler + 二阶校正写法；官方 [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)同时提供 fixed-step Euler 和 adaptive Heun。

### 本地建议

- 首轮只实现 deterministic fixed-grid Euler/Heun；报告 **NFE**，不能只报 steps。通常 \(N\) 个状态区间的 Euler 约 \(N\) NFE，标准 Heun 约 \(2N\) NFE（末端是否省一次要写清楚）。
- 先比较相同 NFE，再补相同步数比较；否则 Heun 的优势可能只是多算了一倍网络。
- 10/20/30 steps 固定 seed、同一初始噪声和同一 checkpoint，记录 loss、wall time、峰值内存、视频指标及样本。

## 5. Timestep sampling、flow shift 与 noise shift

### 论文事实

原始 RF/CFM 常令 \(t\sim U(0,1)\)。SD3 系统比较后发现，中间噪声水平对高分辨率生成更重要；其 `logit-normal(0,1)` 在论文多项设置中优于 uniform RF，并且少步退化较慢（[SD3 §3.1/§5.1](https://arxiv.org/html/2403.03206#S3.SS1)）。采样方式是

\[
s\sim\mathcal N(m,\sigma^2),\qquad t=\operatorname{sigmoid}(s).
\]

SD3 还给出 resolution-dependent timestep shift：

\[
t' = \frac{\alpha t}{1+(\alpha-1)t},
\]

其中 \(\alpha>1\) 把分布推向更高噪声；论文在 1024² 训练/采样中使用 \(\alpha=3\)，但其推导基于简化的 constant-image 假设，作者也明确说这不是普适配方（[SD3 §5.3.2](https://arxiv.org/html/2403.03206#S5.SS3.SSS2)）。

LTX-Video 采用 log-normal/logit-space timestep 思路，并按 token 数把训练分布向高噪声移动，同时对分布尾部做 percentile clamp；多分辨率训练保持近似 token budget，并随机丢弃 0–20% tokens（[LTX-Video §2.5](https://arxiv.org/html/2501.00103#S2.SS5)）。HunyuanVideo 公开推理默认 `flow-shift=7.0`、Euler、50 steps，并把 reverse/solver/shift 都做成显式参数（[官方 README](https://github.com/Tencent-Hunyuan/HunyuanVideo)、[scheduler wiring](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/inference.py)）。这些数值属于各自模型配方，不能跨模型直接照搬。

### 本地建议

- 首个 teacher baseline 用 uniform \(t\)，端到端正确后才消融 `logit_normal(m=0,s=1)`；shift 从 1.0 开始，只做小网格并同时用于训练/验证协议。
- 将三件事分开命名：`train_timestep_distribution`、`train_timestep_shift`、`inference_time_grid/shift`。论文中的 “noise schedule/shift” 经常同时指不同对象。
- 记录每个 batch 的 \(t\) histogram 与按 timestep bin 的 validation MSE；只看全局平均 loss 会掩盖端点饥饿。
- 不复制 LTX/Hunyuan 的 shift 数值；本地 toy clip 的 token 数、VAE latent 统计与训练分辨率都不同。

## 6. Rectified Flow、OT-CFM 与 Stochastic Interpolants 的边界

### 论文事实

[Rectified Flow](https://arxiv.org/abs/2209.03003) 从任意 coupling \((X_0,X_1)\) 的直线插值学习 ODE；reflow 用前一轮 ODE 产生的新 coupling 再训练，使轨迹进一步变直、粗离散更准确。它不意味着“只要训练 linear path，学到的边缘 ODE 轨迹必然逐样本完全直”。

[OT-CFM](https://arxiv.org/abs/2302.00482) 通过 minibatch optimal-transport coupling 构造更简单的 flow，并把 CFM 推广到非 Gaussian source、Schrödinger bridge 等设置。它解决的是 endpoint coupling/path geometry，不是第一版视频 backbone 正确性的必要条件。

[Stochastic Interpolants](https://arxiv.org/abs/2303.08797) 用更一般的随机插值统一 transport ODE 与一族 forward/backward SDE，并给出 quadratic regression objectives。它适合在需要系统统一 flow/diffusion 或加入可调随机性时阅读；linear deterministic teacher 阶段不必实现。

### 本地建议

- 第一版只做 independent pairing：每个 data latent 配独立 Gaussian noise。
- 在 10–30 步 teacher 仍明显受 solver error 限制时，再评估 OT coupling 或 reflow；不要把它们加入最初的 DDPM→Flow 因果比较。

## 7. 视频模型中的工程采用

### 一手资料事实

| 系统 | 可由论文/官方代码确认的 Flow 相关做法 | 可读入口 |
|---|---|---|
| Stable Diffusion 3 | linear rectified flow；直接预测 velocity；logit-normal 等非均匀 timestep sampling；高分辨率 timestep shift；同时进行了 epsilon/v/RF 大规模比较 | [论文](https://arxiv.org/abs/2403.03206)，[官方 SD3 micro-reference](https://github.com/Stability-AI/sd3-ref/blob/master/sd3_impls.py) |
| LTX-Video | \(z_t=(1-t)z_0+t\epsilon\)、target \(\epsilon-z_0\)；token-count-dependent high-noise shift；约等 token 数的多分辨率/时长训练；论文 20-step 结果 | [论文 §2.5](https://arxiv.org/html/2501.00103#S2.SS5)，[官方 scheduler wiring](https://github.com/Lightricks/LTX-Video/blob/main/ltx_video/inference.py) |
| HunyuanVideo | 公开仓库以 FlowMatch scheduler 装配推理，暴露 `shift`、`reverse`、`solver`；默认示例 50-step Euler、shift 7.0 | [论文](https://arxiv.org/abs/2412.03603)，[config](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/hyvideo/config.py)，[scheduler](https://github.com/Tencent-Hunyuan/HunyuanVideo/tree/main/hyvideo/diffusion/schedulers) |

这些系统还同时改变了 VAE、latent 维度、数据、条件、训练规模和 backbone，不能把它们的质量提升归因于 Flow Matching 单一因素。

### 本地建议：公平 DDPM → Flow 替换

固定：Video DiT backbone、参数量、VAE、latent normalization、数据 split、条件、batch/global samples、optimizer、EMA、训练 wall-clock/step budget、评估 seeds。SD3 在训练集子集上全局统计 latent mean/std 后归一化（[论文 Appendix B.2](https://arxiv.org/html/2403.03206#A2.SS2)）；本地实验应复用当前 Video DiT 的同一 latent 统计，不能在 Flow 组悄悄重估或改 scaling。

只替换：

```text
DDPM: q_sample + epsilon target + DDIM/DDPM sampler
Flow: linear path + flow_velocity target + Euler/Heun ODE sampler
```

最低验收：

1. path 两端、\(v=\epsilon-z_0\)、反向单步符号和 oracle solver 测试全部通过；
2. sampler 中不存在 DDPM posterior 调用；
3. 10/20/30-step Euler 与相同 NFE 的 Heun/基线公平比较；
4. checkpoint 保存 path、time convention、prediction type、train timestep distribution/shift、inference grid/solver；
5. 能用一张表解释任何外部实现的 `t=0/1` 端点、网络 target 与 solver 更新方向。

## 8. 经典论文与官方实现索引

| 优先级 | 资料 | 只读什么 |
|---|---|---|
| A | [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) | ODE、CNF、instantaneous change of variables；跳过非生成任务细节 |
| A | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | FM/CFM、Gaussian path、OT displacement path |
| A | [Flow Straight and Fast](https://arxiv.org/abs/2209.03003) | linear interpolation、velocity regression、reflow、Euler |
| A | [Scaling Rectified Flow Transformers / SD3](https://arxiv.org/abs/2403.03206) | parameterization 关系、timestep sampling、shift、工程消融 |
| A | [LTX-Video](https://arxiv.org/abs/2501.00103) | 视频 latent 中的 RF 训练与 schedule |
| B | [Improving and Generalizing Flow-Based Models with Minibatch OT](https://arxiv.org/abs/2302.00482) | generalized CFM、I-CFM、OT-CFM |
| B | [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) | Euler/Heun、NFE、solver/schedule 解耦 |
| C | [Stochastic Interpolants](https://arxiv.org/abs/2303.08797) | ODE/SDE 统一视角；按需阅读 |
| C | [FFJORD](https://arxiv.org/abs/1810.01367) | likelihood CNF；只有需要 density estimation 时再读 |

官方代码优先看：[Meta `flow_matching`](https://github.com/facebookresearch/flow_matching)、[TorchCFM](https://github.com/atong01/conditional-flow-matching)、[RectifiedFlow](https://github.com/gnobitab/RectifiedFlow)、[NVLabs EDM](https://github.com/NVlabs/edm)、[`torchdiffeq`](https://github.com/rtqichen/torchdiffeq)、[SD3 micro-reference](https://github.com/Stability-AI/sd3-ref)、[LTX-Video](https://github.com/Lightricks/LTX-Video)、[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)。Meta Flow Matching 与原始 RectifiedFlow 仓库为非商业/CC 类许可，SD3 micro-reference 为 MIT，LTX-Video 为 Apache-2.0；复用代码前应分别检查仓库当前 LICENSE。
