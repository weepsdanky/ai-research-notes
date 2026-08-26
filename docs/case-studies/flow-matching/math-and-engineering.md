# Flow Matching：最小数学与工程笔记

## 1. 一条 path、一个 target

本仓库固定约定：$z_0$ 是 data/latent，$\epsilon\sim\mathcal N(0,I)$ 是 $t=1$ 的 noise。

$$
z_t=(1-t)z_0+t\epsilon,
\qquad t\in[0,1].
$$

对 $t$ 求导：

$$
v_t=\frac{dz_t}{dt}=-z_0+\epsilon=\epsilon-z_0.
$$

所以一次 Conditional Flow Matching 训练是：

1. 采 $z_0,c$；
2. 独立采 $\epsilon$ 和 $t$；
3. 构造 $z_t$；
4. 最小化 $\|v_\theta(z_t,t,c)-(\epsilon-z_0)\|_2^2$。

网络看到同一个 $z_t$ 时，回归的最优解是 conditional targets 的条件均值；这给出 transporting marginal probability path 的 velocity field。训练 target 可逐样本算，不需要在训练中先解 ODE。

## 2. 为什么生成时符号没有变

训练 target 始终是 forward velocity $\epsilon-z_0$。生成仍调用同一个 $v_\theta$，只是积分网格从 $t=1$ 降到 $0$：

$$
z_{t+\Delta t}=z_t+\Delta t\,v_\theta(z_t,t,c),
\qquad \Delta t<0.
$$

因此不是把网络输出手工取负，而是 solver 的 $\Delta t$ 为负。若另一个实现把 noise 设为 $t=0$、data 设为 $t=1$，它会得到相反 target；先看 endpoints，再看符号。

## 3. Euler 与 Heun

Euler（每步 1 NFE）：

$$
z'=z_t+\Delta t\,v_\theta(z_t,t,c).
$$

Heun / explicit trapezoid：

$$
\tilde z=z_t+\Delta t\,v_\theta(z_t,t,c),
$$

$$
z'=z_t+\frac{\Delta t}{2}\left[v_\theta(z_t,t,c)+v_\theta(\tilde z,t+\Delta t,c)\right].
$$

Heun 每步 2 NFE，因此 $N$ steps 是 $2N$ NFE。报告质量时比较 20 Euler NFE 与 20 Heun NFE（10 steps），而不是都写 20 steps。

## 4. 三种 velocity 不要混名

| 名称 | path | target / 含义 |
|---|---|---|
| DDPM epsilon prediction | $z_t=\alpha_tz_0+\sigma_t\epsilon$ | $\epsilon$ |
| diffusion v-prediction | 同上 | $\alpha_t\epsilon-\sigma_tz_0$（常见 VP 约定） |
| 本模块 linear-flow velocity | $z_t=(1-t)z_0+t\epsilon$ | $\epsilon-z_0$ |

一般 path 的真实导数是 $\dot\alpha_tz_0+\dot\sigma_t\epsilon$。只有在线性 $\alpha_t=1-t,\sigma_t=t$ 时才简化成 $\epsilon-z_0$。

线性 path 可从 $(z_t,v)$ 在所有 endpoints 稳定恢复两端：

$$
z_0=z_t-tv,
\qquad
\epsilon=z_t+(1-t)v.
$$

## 5. timestep sampling 与 shift

baseline：$t\sim U(0,1)$。logit-normal 是先采 $u\sim\mathcal N(\mu,\sigma^2)$，再令 $t=\operatorname{sigmoid}(u)$，改变中间时间的训练密度。

本地 shift：

$$
\hat t=\frac{s t}{1+(s-1)t}.
$$

$s>1$ 将质量推向 noisy endpoint，并保持 0/1 不变。它是训练时间分布变换，不是 solver step。小视频不应直接继承大分辨率模型的 shift；先把 $s=1$ 跑成 baseline。

## 6. 工程 checklist

- **时间嵌入**：外部 math 用 $t\in[0,1]$；送入现有 DiT 前显式乘 `model_time_scale=1000`。训练和推理必须相同。
- **输出通道**：Flow 只预测 $C$ channel velocity；禁用 DDPM `learn_sigma` 的 $2C$ 输出。
- **VAE contract**：复用同一缓存 latent、VAE ID 与 scaling factor；不要为了 Flow 重新 encode 一套样本。
- **time histogram**：记录 mean/std，复杂策略还应保存 histogram；否则 shift 配错不易发现。
- **precision**：第一轮 FP32；AMP、solver state dtype 和模型 dtype 分开消融。
- **EMA/CFG**：可以复用，但 CFG 会把每次 field evaluation 的实际模型 batch 翻倍；同时报告 NFE 和 wall time。
- **loss**：DDPM epsilon MSE 与 flow velocity MSE 的 target scale 不同，不能用绝对 loss 判优。
- **clipping**：ODE baseline 不逐步 clamp latent；pixel preview 只在保存前处理显示范围。
- **复现**：固定 model init seed、训练 batch 顺序、eval initial noise、prompt 与 seed。

## 7. 代码地图

| 数学对象 | 本地入口 |
|---|---|
| path / target / endpoint inversion | `flow_matching.py:FlowMatchingSchedule` |
| CFM objective | `flow_matching.py:flow_matching_loss` |
| Euler / Heun reverse solve | `flow_matching.py:flow_sample_loop` |
| 同一 Video DiT/data/VAE 训练 | `train_flow.py` |
| 20-step structured sample | `sample_flow.py` |
| 不变量 | `tests/test_flow_matching.py` |
