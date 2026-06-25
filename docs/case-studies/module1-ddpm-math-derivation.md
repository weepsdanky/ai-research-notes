# Module 1: DDPM 数学推导

本文整理 Denoising Diffusion Probabilistic Model (DDPM) 的训练侧推导，并对照本仓库的 MNIST 实验代码：

- 代码：`code/diffusion/train_mnist_ddpm.py`
- 学习计划：`docs/case-studies/5_lpm-case-study.md`
- 原论文：Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models", 2020
- 极简参考：`cloneofsimo/minDiffusion`

目标不是记公式，而是能从概率图模型和高斯代数独立推到：

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon - \epsilon_\theta(x_t,t)
\right\|^2
\right].
$$

## 1. 记号

数据样本记为 $x_0 \sim q(x_0)$。这里 $x_0$ 不是一个标量，而是一整张图像张量。MNIST 中它的形状是：

$$
x_0 \in \mathbb{R}^{1\times 28\times 28}.
$$

DDPM 构造一条长度为 $T$ 的加噪链：

$$
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T.
$$

下标 $t$ 表示第几个噪声时间步，不是幂。$x_t$ 的意思是“原图 $x_0$ 加噪 $t$ 步之后的图像”。$T$ 是总步数，代码里是 `NUM_TIMESTEPS = 1000`。

我们会同时看到两个概率分布字母：

- $q$：固定的 forward noising process，也就是人为设计的加噪过程，不学习参数。
- $p_\theta$：学习出来的 reverse denoising process，也就是神经网络定义的反向去噪过程。

$\theta$ 表示神经网络的所有可学习参数。在代码里，$\theta$ 就是 `UNet().parameters()` 里那些卷积核、线性层权重和 bias。

### 1.1 $\beta_t$ 是什么

$\beta_t$ 是第 $t$ 个加噪步骤加入的噪声方差。注意是 variance，不是 standard deviation。

代码里：

```python
BETA_START = 1e-4
BETA_END = 0.02

def linear_beta_schedule(timesteps: int):
    return torch.linspace(BETA_START, BETA_END, timesteps)
```

意思是：

$$
\beta_1,\beta_2,\ldots,\beta_T
$$

从很小的 $10^{-4}$ 逐渐增加到 $0.02$。早期少加噪，晚期多加噪。

### 1.2 $\alpha_t$ 是什么

DDPM 定义：

$$
\alpha_t = 1-\beta_t.
$$

如果 $\beta_t$ 是“这一步新加入的噪声方差比例”，那么 $\alpha_t$ 是“这一步保留下来的旧信号方差比例”。

这里的 $1$ 是归一化后的总方差尺度。直觉是：如果当前变量已经大致有单位方差，我们希望下一步仍然大致保持单位方差，而不是方差越来越爆炸。

假设：

$$
\operatorname{Var}(x_{t-1}) \approx I,
\qquad
z_t \sim \mathcal{N}(0,I),
$$

并且用线性形式：

$$
x_t = a_tx_{t-1} + b_tz_t.
$$

因为 $x_{t-1}$ 和 $z_t$ 独立，所以：

$$
\operatorname{Var}(x_t)
=
a_t^2\operatorname{Var}(x_{t-1})
+
b_t^2\operatorname{Var}(z_t)
\approx
(a_t^2+b_t^2)I.
$$

为了让总方差仍然约等于 $I$，需要：

$$
a_t^2+b_t^2=1.
$$

DDPM 选择：

$$
b_t^2=\beta_t,
\qquad
a_t^2=\alpha_t=1-\beta_t.
$$

所以：

$$
a_t=\sqrt{\alpha_t},
\qquad
b_t=\sqrt{\beta_t}.
$$

这就是为什么采样公式里出现 square root。高斯分布的第二个参数是方差 $\beta_t I$，真正乘在标准高斯噪声上的是标准差 $\sqrt{\beta_t}$。

### 1.3 $\bar{\alpha}_t$ 是什么

$\bar{\alpha}_t$ 读作 alpha bar t，定义为从第 1 步到第 $t$ 步所有 $\alpha$ 的连乘：

$$
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s
=
\alpha_1\alpha_2\cdots\alpha_t.
$$

这里 $s$ 只是乘积里的哑变量，和积分里的 $dx$ 类似。它的作用是避免和终点 $t$ 混在一起。

代码中：

```python
alpha_bar = torch.cumprod(alpha, dim=0)
```

`torch.cumprod` 是 cumulative product，累计乘积。如果：

```python
alpha = [a1, a2, a3, a4]
```

那么：

```python
torch.cumprod(alpha, dim=0) = [a1, a1*a2, a1*a2*a3, a1*a2*a3*a4]
```

也就是：

$$
[\bar{\alpha}_1,\bar{\alpha}_2,\bar{\alpha}_3,\bar{\alpha}_4].
$$

先记住结论：$\bar{\alpha}_t$ 会变成 $x_0$ 在 $x_t$ 中剩下的信号方差比例，$1-\bar{\alpha}_t$ 会变成累计噪声方差比例。这个不是定义出来的，而是从下一节的递推推出来的。

## 2. Forward diffusion: 固定的物理过程

Forward diffusion 是人为规定的加噪过程。它不是模型，不需要训练。每一步定义为：

$$
q(x_t \mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right).
$$

这个公式的意思是：在已知 $x_{t-1}$ 的情况下，$x_t$ 是一个高斯随机变量，它的均值是 $\sqrt{\alpha_t}x_{t-1}$，协方差是 $\beta_t I$。

写成采样形式就是：

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}z_t,
\qquad
z_t \sim \mathcal{N}(0,I).
$$

两种写法完全等价，因为一般地：

$$
y \sim \mathcal{N}(m,\sigma^2 I)
\quad
\Longleftrightarrow
\quad
y=m+\sigma z,\quad z\sim\mathcal{N}(0,I).
$$

在这里：

$$
m=\sqrt{\alpha_t}x_{t-1},
\qquad
\sigma^2=\beta_t,
\qquad
\sigma=\sqrt{\beta_t}.
$$

物理直觉：每一步把旧状态的振幅乘以 $\sqrt{\alpha_t}<1$，再加入一份温度为 $\beta_t$ 的热噪声。因为 $\alpha_t+\beta_t=1$，如果输入已经近似单位方差，输出也会维持近似单位方差。经过足够多步，原始结构消失，$x_T$ 接近标准高斯白噪声。

### 2.1 从 $q(x_t \mid x_{t-1})$ 推到 $q(x_t \mid x_0)$

先看前两步，不跳步。

第一步：

$$
x_1
=
\sqrt{\alpha_1}x_0
+
\sqrt{\beta_1}z_1.
$$

第二步：

$$
x_2
=
\sqrt{\alpha_2}x_1
+
\sqrt{\beta_2}z_2.
$$

把 $x_1$ 代进去：

$$
x_2
=
\sqrt{\alpha_2}
\left(
\sqrt{\alpha_1}x_0+\sqrt{\beta_1}z_1
\right)
+
\sqrt{\beta_2}z_2.
$$

展开：

$$
x_2
=
\sqrt{\alpha_1\alpha_2}x_0
+
\sqrt{\alpha_2\beta_1}z_1
+
\sqrt{\beta_2}z_2.
$$

后两项都是独立高斯噪声的线性组合。高斯的线性组合仍然是高斯，这是这里说“线性变换”的意思。它们的均值都是 0，方差相加：

$$
\alpha_2\beta_1 + \beta_2
=
\alpha_2(1-\alpha_1)+(1-\alpha_2)
=
1-\alpha_1\alpha_2.
$$

所以可以把后两项合成一个标准高斯 $\epsilon$：

$$
\sqrt{\alpha_2\beta_1}z_1
+
\sqrt{\beta_2}z_2
=
\sqrt{1-\alpha_1\alpha_2}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

于是：

$$
x_2
=
\sqrt{\alpha_1\alpha_2}x_0
+
\sqrt{1-\alpha_1\alpha_2}\epsilon.
$$

这已经显示出模式：两步之后，信号比例是 $\alpha_1\alpha_2$，噪声比例是 $1-\alpha_1\alpha_2$。

推广到第 $t$ 步：

$$
q(x_t \mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right).
$$

也就是：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I).
$$

这就是重参数化形式。所谓重参数化，就是把“从某个均值和方差的高斯采样”改写成“确定性均值项 + 标准高斯噪声乘标准差”。它把随机性全部放到 $\epsilon$ 里。

代码中对应 `forward_diffusion`：

```python
alpha_bar_t = noise_schedule.alpha_bar[t].view(-1, 1, 1, 1)
eps = torch.randn_like(x0)
xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
```

这里 `view(-1, 1, 1, 1)` 是为了让每个 batch 样本自己的 $\bar{\alpha}_t$ 能广播到整张图像上。一个 batch 里每张图可以抽到不同的 $t$。

### 2.2 一般情形的归纳推导

假设对 $t-1$ 成立：

$$
x_{t-1}
=
\sqrt{\bar{\alpha}_{t-1}}x_0
+
\sqrt{1-\bar{\alpha}_{t-1}}\epsilon_{t-1},
\qquad
\epsilon_{t-1}\sim\mathcal{N}(0,I).
$$

第 $t$ 步：

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}z_t,
\qquad
z_t\sim\mathcal{N}(0,I).
$$

因为 $\beta_t=1-\alpha_t$，也可以写成：

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{1-\alpha_t}z_t.
$$

代入归纳假设：

$$
x_t
=
\sqrt{\alpha_t\bar{\alpha}_{t-1}}x_0
+
\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon_{t-1}
+
\sqrt{1-\alpha_t}z_t.
$$

由于：

$$
\alpha_t\bar{\alpha}_{t-1}=\bar{\alpha}_t,
$$

信号项变成：

$$
\sqrt{\bar{\alpha}_t}x_0.
$$

噪声项是两个独立标准高斯的线性组合：

$$
\sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\epsilon_{t-1}
+
\sqrt{1-\alpha_t}z_t.
$$

它的均值是 0，方差是：

$$
\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t)
=
1-\alpha_t\bar{\alpha}_{t-1}
=
1-\bar{\alpha}_t.
$$

所以噪声项可以写成：

$$
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

最终得到：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon.
$$

这一步是 DDPM 训练能高效实现的关键，原因有两个。

第一，训练 loss 是对随机时间步 $t$ 的期望：

$$
\mathbb{E}_{t,x_0,\epsilon}[\cdots].
$$

如果没有闭式公式，每次训练一个样本到第 $t$ 步，都要真的运行：

$$
x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_t.
$$

这会让一次训练样本的加噪成本从 $O(1)$ 变成 $O(t)$，最多接近 $O(T)$。

第二，有了闭式公式，我们不仅能直接得到 $x_t$，还知道这次混入的真实噪声 $\epsilon$。所以训练标签天然就是：

$$
\epsilon.
$$

这就是代码里能直接写：

```python
xt, eps = forward_diffusion(x0, t, noise_schedule)
pred_eps = model(xt, t)
loss = F.mse_loss(pred_eps, eps)
```

物理意义：所有噪声时间切片都可以直接从同一个干净状态 $x_0$ 抽样得到。$t$ 只是在设置信噪比，不需要真的模拟完整历史。

## 3. Reverse diffusion: 学习反向去噪链

Forward process 是：

$$
x_0 \rightarrow x_1 \rightarrow \cdots \rightarrow x_T.
$$

生成时我们想反过来：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_0.
$$

最终的 $x_0$ 就是生成出来的图像。训练数据里的 $x_0$ 是真实 MNIST；采样时的 $x_0$ 是模型生成的 MNIST-like 图像。

起点选择标准高斯：

$$
p(x_T)=\mathcal{N}(0,I).
$$

这里不用 $\theta$，因为起点分布不是学出来的，就是固定白噪声。

### 3.1 为什么 joint distribution 写成一串乘积

模型定义为：

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t).
$$

先把乘积写开：

$$
p_\theta(x_{0:T})
=
p(x_T)
p_\theta(x_{T-1}\mid x_T)
p_\theta(x_{T-2}\mid x_{T-1})
\cdots
p_\theta(x_0\mid x_1).
$$

这就是一个反向 Markov chain。Markov 的意思是：从 $x_t$ 走到 $x_{t-1}$ 时，只看当前状态 $x_t$，不再显式依赖更远的未来 $x_{t+1},x_{t+2},\ldots$。

这个乘积不是魔法，只是链式法则加上 Markov 假设。一般链式法则会写成：

$$
p(x_T,x_{T-1},\ldots,x_0)
=
p(x_T)p(x_{T-1}\mid x_T)p(x_{T-2}\mid x_{T-1},x_T)\cdots.
$$

Markov 假设把：

$$
p(x_{T-2}\mid x_{T-1},x_T)
$$

简化成：

$$
p(x_{T-2}\mid x_{T-1}).
$$

于是得到上面的乘积形式。

### 3.2 为什么写 $p_\theta$

$p_\theta$ 表示“由参数 $\theta$ 控制的概率分布”。在 DDPM 里，$\theta$ 是神经网络参数。网络输入 $x_t$ 和时间步 $t$，输出反向高斯分布的均值参数，或者等价地输出噪声预测。

代码里：

```python
pred_eps = model(x, t_tensor)
```

这里的 `model` 就是 $\epsilon_\theta(x_t,t)$。它不是直接输出 $x_{t-1}$，而是先输出噪声估计，再通过公式构造 $p_\theta(x_{t-1}\mid x_t)$ 的均值。

### 3.3 每一步反向分布为什么也设成高斯

DDPM 设：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

原因有两个。

第一，forward process 每一步是小幅高斯扰动。当 $\beta_t$ 很小时，反向一步也可以用高斯近似。物理上就是“小时间步的扩散反演”。

第二，高斯和高斯之间的 Kullback-Leibler divergence 可以解析计算。这会让训练目标变成均值匹配，最后简化成 MSE。

### 3.4 方差的“简单选择”是什么意思

反向高斯有两个东西：

$$
\mu_\theta(x_t,t)
\quad
\text{and}
\quad
\Sigma_\theta(x_t,t).
$$

最基础 DDPM 主要学习均值，方差用固定规则。常见选择有三种：

1. 简单固定为 forward variance：

   $$
   \Sigma_\theta(x_t,t)=\beta_t I.
   $$

   你的代码就是这个版本。采样时加：

   ```python
   noise = torch.randn_like(x)
   x = x + torch.sqrt(beta_t) * noise
   ```

   这里 `noise` 是：

   $$
   z\sim\mathcal{N}(0,I).
   $$

   `torch.sqrt(beta_t) * noise` 就是给 $x_{t-1}$ 加标准差为 $\sqrt{\beta_t}$ 的高斯扰动。

2. 固定为真实 posterior variance：

   $$
   \Sigma_\theta(x_t,t)=\tilde{\beta}_t I,
   $$

   其中：

   $$
   \tilde{\beta}_t
   =
   \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
   $$

   这比直接用 $\beta_t$ 更贴近真实反向后验。

3. 学习方差：

   $$
   \Sigma_\theta(x_t,t)
   \text{ is predicted by the neural network.}
   $$

   Improved DDPM 会讨论这种做法。它通常改善 likelihood 和采样效率，但会让训练目标更复杂。

你当前代码选的是最容易理解的第 1 种：只学习噪声预测，也就是主要学习反向均值；采样噪声强度直接用 $\beta_t$。

## 4. 真实后验 $q(x_{t-1}\mid x_t,x_0)$

这一节是 DDPM 推导里最容易卡住的地方。先区分两个分布。

### 4.1 $q(x_{t-1}\mid x_t)$ 和 $q(x_{t-1}\mid x_t,x_0)$ 的区别

$q(x_{t-1}\mid x_t)$ 的意思是：只知道当前 noisy image $x_t$，问上一时刻 $x_{t-1}$ 的分布是什么。

这个分布很难直接写，因为一个 noisy image 可能来自很多不同的干净图像。数学上：

$$
q(x_{t-1}\mid x_t)
=
\int
q(x_{t-1}\mid x_t,x_0)
q(x_0\mid x_t)
dx_0.
$$

这里的 $q(x_0\mid x_t)$ 依赖真实数据分布。对 MNIST 来说，它问的是：“看到这张带噪图片后，原来的干净数字可能是什么？”这不是一个简单高斯。

而 $q(x_{t-1}\mid x_t,x_0)$ 的意思是：不仅知道当前 noisy image $x_t$，还知道它来自哪张原图 $x_0$。训练时我们确实知道 $x_0$，因为数据集给了真实图片。所以这个条件后验可以解析写出。

代码层面：

- 训练时：我们有 `x0`，自己采样 `t` 和 `eps`，构造 `xt`。
- 推导时：用 $q(x_{t-1}\mid x_t,x_0)$ 得到最优反向一步应该长什么样。
- 采样时：我们没有真实 $x_0$，所以用模型预测的噪声 $\epsilon_\theta$ 间接估计反向均值。

### 4.2 用 Bayes 公式写出真实后验

目标是：

$$
q(x_{t-1}\mid x_t,x_0).
$$

用 Bayes 公式：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

由于 forward process 是 Markov chain，给定 $x_{t-1}$ 后，$x_t$ 不再需要知道 $x_0$：

$$
q(x_t\mid x_{t-1},x_0)
=
q(x_t\mid x_{t-1}).
$$

所以：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

如果只关心它作为 $x_{t-1}$ 的函数，分母 $q(x_t\mid x_0)$ 不含 $x_{t-1}$，只是归一化常数。因此：

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
$$

这句话的数学意义是：posterior = likelihood times prior，再归一化。

### 4.3 两个高斯相乘

把两个因子写出来：

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right),
$$

$$
q(x_{t-1}\mid x_0)
=
\mathcal{N}
\left(
x_{t-1};
\sqrt{\bar{\alpha}_{t-1}}x_0,
(1-\bar{\alpha}_{t-1})I
\right).
$$

为了看清楚代数，先把 $x_{t-1}$ 记成 $y$。忽略常数项，两个高斯的 log density 相加：

$$
-\frac{1}{2\beta_t}
\left\|
x_t-\sqrt{\alpha_t}y
\right\|^2
-
\frac{1}{2(1-\bar{\alpha}_{t-1})}
\left\|
y-\sqrt{\bar{\alpha}_{t-1}}x_0
\right\|^2.
$$

这是一个关于 $y$ 的二次型。把二次项和一次项收集起来：

$$
-\frac{1}{2}
\left[
\left(
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar{\alpha}_{t-1}}
\right)
\|y\|^2
-
2
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right)
\cdot y
\right]
+ C.
$$

一个高斯的 canonical form 是：

$$
-\frac{1}{2}
\left[
A\|y\|^2 - 2B\cdot y
\right]
+C,
$$

它的方差是 $A^{-1}$，均值是 $A^{-1}B$。

这里：

$$
A
=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar{\alpha}_{t-1}}
=
\frac{1-\bar{\alpha}_t}{
\beta_t(1-\bar{\alpha}_{t-1})
}.
$$

所以方差是：

$$
\tilde{\beta}_t
=
A^{-1}
=
\frac{
1-\bar{\alpha}_{t-1}
}{
1-\bar{\alpha}_t
}
\beta_t.
$$

均值是：

$$
\tilde{\mu}_t(x_t,x_0)
=
A^{-1}
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0
\right).
$$

代入 $A^{-1}=\tilde{\beta}_t$ 并化简：

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha}_t
}x_0
+
\frac{
\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})
}{
1-\bar{\alpha}_t
}x_t.
$$

因此：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right).
$$

物理直觉：如果你同时知道“最初干净状态 $x_0$”和“当前带噪状态 $x_t$”，那么上一时刻 $x_{t-1}$ 的最佳高斯估计是两边信息的精度加权平均。噪声越小的一边，权重越大。

## 5. Evidence Lower Bound (ELBO): 为什么训练目标是 KL 之和

Evidence lower bound 通常缩写为 ELBO，中文可以叫“证据下界”或“变分下界”。它的作用是：真实数据似然 $\log p_\theta(x_0)$ 很难直接算，于是我们找一个可以计算、可以优化的下界。

先定义 Kullback-Leibler divergence，缩写 KL divergence：

$$
D_{\mathrm{KL}}(q\|p)
=
\mathbb{E}_{x\sim q}
\left[
\log\frac{q(x)}{p(x)}
\right].
$$

它衡量分布 $q$ 和 $p$ 的差异。$D_{\mathrm{KL}}(q\|p)\geq 0$，当且仅当两个分布相同才等于 0。注意它不是对称距离，一般：

$$
D_{\mathrm{KL}}(q\|p)\neq D_{\mathrm{KL}}(p\|q).
$$

### 5.1 目标：最大化数据似然

我们希望模型给真实图像 $x_0$ 高概率：

$$
\log p_\theta(x_0).
$$

但模型实际定义在整条反向链上：

$$
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
$$

如果只要 $x_0$ 的概率，需要把所有隐藏变量 $x_1,\ldots,x_T$ 积掉：

$$
p_\theta(x_0)
=
\int
p_\theta(x_{0:T})
dx_{1:T}.
$$

所以：

$$
\log p_\theta(x_0)
=
\log
\int
p_\theta(x_{0:T})
dx_{1:T}.
$$

这个积分高维且难算。

### 5.2 Jensen 不等式在这里怎么用

Jensen 不等式说：如果 $f$ 是凹函数，那么：

$$
f(\mathbb{E}[Y])
\geq
\mathbb{E}[f(Y)].
$$

$\log$ 是凹函数，所以：

$$
\log \mathbb{E}[Y]
\geq
\mathbb{E}[\log Y].
$$

现在人为乘除同一个 forward distribution：

$$
q(x_{1:T}\mid x_0).
$$

从：

$$
\log p_\theta(x_0)
=
\log
\int
p_\theta(x_{0:T})
dx_{1:T}
$$

变成：

$$
\log p_\theta(x_0)
=
\log
\int
q(x_{1:T}\mid x_0)
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
dx_{1:T}.
$$

这就是：

$$
\log p_\theta(x_0)
=
\log
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

令：

$$
Y=
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}.
$$

用 Jensen：

$$
\log p_\theta(x_0)
\geq
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

右边就是 ELBO：

$$
\mathrm{ELBO}
=
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{
p_\theta(x_{0:T})
}{
q(x_{1:T}\mid x_0)
}
\right].
$$

最大化 ELBO 就是在最大化 $\log p_\theta(x_0)$ 的下界。训练时通常最小化负 ELBO：

$$
L_{\mathrm{vlb}}
=
-\mathrm{ELBO}.
$$

这里 vlb 是 variational lower bound。

### 5.3 把负 ELBO 展开

从：

$$
L_{\mathrm{vlb}}
=
\mathbb{E}_{q}
\left[
\log q(x_{1:T}\mid x_0)
-
\log p_\theta(x_{0:T})
\right].
$$

先展开模型分布：

$$
\log p_\theta(x_{0:T})
=
\log p(x_T)
+
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t).
$$

然后把 forward chain 用“从 $x_T$ 反向看、但仍然条件在 $x_0$ 上”的方式分解：

$$
q(x_{1:T}\mid x_0)
=
q(x_T\mid x_0)
\prod_{t=2}^{T}
q(x_{t-1}\mid x_t,x_0).
$$

这个等式来自链式法则。它不是模型采样方向，而是为了代数整理方便。取 log：

$$
\log q(x_{1:T}\mid x_0)
=
\log q(x_T\mid x_0)
+
\sum_{t=2}^{T}
\log q(x_{t-1}\mid x_t,x_0).
$$

代回负 ELBO：

$$
\begin{aligned}
L_{\mathrm{vlb}}
=
\mathbb{E}_{q}
\Big[
&
\log q(x_T\mid x_0)
+
\sum_{t=2}^{T}
\log q(x_{t-1}\mid x_t,x_0)
\\
&
-
\log p(x_T)
-
\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t)
\Big].
\end{aligned}
$$

把 $t=1$ 的模型项单独拿出来：

$$
\sum_{t=1}^{T}\log p_\theta(x_{t-1}\mid x_t)
=
\log p_\theta(x_0\mid x_1)
+
\sum_{t=2}^{T}\log p_\theta(x_{t-1}\mid x_t).
$$

于是：

$$
\begin{aligned}
L_{\mathrm{vlb}}
=
\mathbb{E}_{q}
\Big[
&
\log \frac{q(x_T\mid x_0)}{p(x_T)}
\\
&
+
\sum_{t=2}^{T}
\log
\frac{
q(x_{t-1}\mid x_t,x_0)
}{
p_\theta(x_{t-1}\mid x_t)
}
\\
&
-
\log p_\theta(x_0\mid x_1)
\Big].
\end{aligned}
$$

现在每一项都能看成 KL 或 reconstruction term。

第一项：

$$
L_T
=
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\;\|\;
p(x_T)
\right).
$$

它要求 forward noising 的最终分布 $q(x_T\mid x_0)$ 接近标准高斯先验 $p(x_T)$。如果 $T$ 足够大，这一项基本由 noise schedule 决定，不需要神经网络学习。

中间项：

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right).
$$

它要求模型的一步反向分布接近真实后验。这里才是 DDPM 训练的核心。

最后一项：

$$
L_0
=
-\log p_\theta(x_0\mid x_1).
$$

它是最后一步从轻微噪声 $x_1$ 还原到干净图像 $x_0$ 的 reconstruction loss。

所以 DDPM 论文写成：

$$
L_{\mathrm{vlb}}
=
L_T
+
\sum_{t=2}^{T}L_{t-1}
+
L_0.
$$

这不是“直接给出”的公式，而是把负 ELBO 代入两个链式分解后，把 log ratio 按照 KL divergence 的定义分组得到的。

## 6. 从 KL 到预测噪声

中间项是：

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\;\|\;
p_\theta(x_{t-1}\mid x_t)
\right).
$$

两边都是高斯。真实后验是：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
x_{t-1};
\tilde{\mu}_t(x_t,x_0),
\tilde{\beta}_t I
\right).
$$

模型反向分布是：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\sigma_t^2 I
\right).
$$

两个高斯的 KL 有解析公式。若：

$$
q=\mathcal{N}(m_q,\Sigma_q),
\qquad
p=\mathcal{N}(m_p,\Sigma_p),
$$

则：

$$
D_{\mathrm{KL}}(q\|p)
=
\frac{1}{2}
\left[
\log\frac{\det\Sigma_p}{\det\Sigma_q}
-d
+
\operatorname{tr}(\Sigma_p^{-1}\Sigma_q)
+
(m_p-m_q)^T\Sigma_p^{-1}(m_p-m_q)
\right].
$$

这里 $d$ 是维度。对一张 MNIST 图，$d=1\times 28\times 28$。

如果反向方差 $\sigma_t^2 I$ 是固定的，不由神经网络学习，那么上面公式里只有最后的均值差项依赖 $\theta$。所以对模型训练来说：

$$
L_{t-1}
\propto
\frac{1}{2\sigma_t^2}
\left\|
\tilde{\mu}_t(x_t,x_0)
-
\mu_\theta(x_t,t)
\right\|^2.
$$

这就是“KL 变成均值匹配”的精确意思。

但直接让网络预测 $\mu_\theta$ 不够直观。DDPM 改成让网络预测混进 $x_t$ 的噪声 $\epsilon$。

利用闭式加噪公式：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
$$

可以反解：

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
}{
\sqrt{\bar{\alpha}_t}
}.
$$

把这个 $x_0$ 代入真实后验均值：

$$
\tilde{\mu}_t(x_t,x_0)
=
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha}_t
}x_0
+
\frac{
\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})
}{
1-\bar{\alpha}_t
}x_t.
$$

先处理 $x_0$ 那项。因为：

$$
\bar{\alpha}_t
=
\alpha_t\bar{\alpha}_{t-1},
$$

所以：

$$
\sqrt{\bar{\alpha}_t}
=
\sqrt{\alpha_t}\sqrt{\bar{\alpha}_{t-1}}.
$$

代入 $x_0$：

$$
\frac{
\sqrt{\bar{\alpha}_{t-1}}\beta_t
}{
1-\bar{\alpha}_t
}
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon
}{
\sqrt{\bar{\alpha}_t}
}
=
\frac{
\beta_t
}{
\sqrt{\alpha_t}(1-\bar{\alpha}_t)
}
x_t
-
\frac{
\beta_t
}{
\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}
}
\epsilon.
$$

再加上原来的 $x_t$ 项：

$$
\frac{
\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})
}{
1-\bar{\alpha}_t
}x_t.
$$

$x_t$ 的总系数是：

$$
\frac{\beta_t}{\sqrt{\alpha_t}(1-\bar{\alpha}_t)}
+
\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}.
$$

通分：

$$
\frac{
\beta_t+\alpha_t(1-\bar{\alpha}_{t-1})
}{
\sqrt{\alpha_t}(1-\bar{\alpha}_t)
}.
$$

因为 $\beta_t=1-\alpha_t$：

$$
\beta_t+\alpha_t(1-\bar{\alpha}_{t-1})
=
1-\alpha_t+\alpha_t-\alpha_t\bar{\alpha}_{t-1}
=
1-\bar{\alpha}_t.
$$

所以 $x_t$ 的总系数是：

$$
\frac{1}{\sqrt{\alpha_t}}.
$$

因此真实后验均值可以写成：

$$
\tilde{\mu}_t(x_t,\epsilon)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon
\right).
$$

于是我们让模型预测噪声：

$$
\epsilon_\theta(x_t,t)
\approx
\epsilon.
$$

对应的模型均值写成：

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right).
$$

这正是采样代码中的核心更新：

```python
pred_eps = model(x, t_tensor)

coeff1 = 1.0 / torch.sqrt(alpha_t)
coeff2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
x = coeff1 * (x - coeff2 * pred_eps)
```

因为 $1-\alpha_t=\beta_t$，所以 `coeff2` 就是：

$$
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}.
$$

## 7. 为什么可以用 $L_{\mathrm{simple}}$

把 $\tilde{\mu}_t$ 和 $\mu_\theta$ 的表达式相减：

$$
\tilde{\mu}_t - \mu_\theta
=
\frac{\beta_t}{
\sqrt{\alpha_t}\sqrt{1-\bar{\alpha}_t}
}
\left(
\epsilon_\theta(x_t,t)-\epsilon
\right).
$$

因此 KL 项等价于带权重的噪声预测 MSE：

$$
L_{t-1}
=
\frac{
\beta_t^2
}{
2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)
}
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
+ C,
$$

其中 $C$ 与 $\theta$ 无关。

Ho et al. 发现去掉这个时间相关权重后，经验效果更好，于是使用简化目标：

$$
L_{\mathrm{simple}}
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right].
$$

代码中对应：

```python
t = torch.randint(0, NUM_TIMESTEPS, (x0.size(0),), device=DEVICE)
xt, eps = forward_diffusion(x0, t, noise_schedule)
pred_eps = model(xt, t)
loss = F.mse_loss(pred_eps, eps)
```

关键理解：模型不是直接生成数字，而是在任意噪声等级 $t$ 下学习“这张图里混入的高斯噪声是哪一份”。

## 8. Score matching 视角

Score 是概率密度对样本的 log gradient：

$$
\nabla_x \log p(x).
$$

它是一个向量场。直觉上，它指向“让 log probability 增加最快”的方向。对图像分布来说，score 告诉你：当前这张 noisy image 应该往哪个方向移动，才更像真实数据。

Score matching 指训练模型去匹配这个 score field，而不是直接建模归一化概率密度 $p(x)$。DDPM 和 denoising score matching 的关系是：在每个噪声等级 $t$，预测噪声 $\epsilon$ 等价于预测 score 的一个缩放版本。

对固定 $x_0$，有：

$$
q(x_t\mid x_0)
=
\mathcal{N}
\left(
\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right).
$$

它对 $x_t$ 的 score 是：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-
\frac{
x_t-\sqrt{\bar{\alpha}_t}x_0
}{
1-\bar{\alpha}_t
}.
$$

由

$$
x_t-\sqrt{\bar{\alpha}_t}x_0
=
\sqrt{1-\bar{\alpha}_t}\epsilon,
$$

得到：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-
\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}.
$$

因此：

$$
\epsilon
=
-
\sqrt{1-\bar{\alpha}_t}
\nabla_{x_t}\log q(x_t\mid x_0).
$$

所以预测 $\epsilon$ 等价于预测 score 的一个缩放版本。采样时，模型沿着高概率数据流形的方向把噪声拉回去。

Annealed Langevin dynamics 指“逐渐降低噪声温度的 Langevin 采样”。DDPM 的采样也在做类似的事：从高噪声状态开始，每一步使用当前噪声等级下学到的向量场，逐渐把样本推向数据分布。

## 9. 采样算法

训练完成后，采样从标准高斯开始：

$$
x_T \sim \mathcal{N}(0,I).
$$

对 $t=T,T-1,\ldots,1$：

$$
z \sim \mathcal{N}(0,I)
\quad
\text{if } t>1,
\qquad
z=0
\quad
\text{if } t=1.
$$

然后：

$$
x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z.
$$

你的代码选择 $\sigma_t=\sqrt{\beta_t}$。更精细的 DDPM 实现常用 posterior variance：

$$
\sigma_t^2
=
\tilde{\beta}_t
=
\frac{
1-\bar{\alpha}_{t-1}
}{
1-\bar{\alpha}_t
}
\beta_t.
$$

这也是改进 epoch 20 样本质量时可以尝试的第一批工程点之一。

## 10. 三种预测目标：$\epsilon$、$x_0$、$v$

同一个 UNet 可以输出三种不同的训练目标。网络结构可以不变，变化的是：

1. 训练时 target 是什么。
2. 采样时如何把网络输出换算成 $\epsilon_\theta$ 或 $\hat{x}_0$。

先统一记号：

$$
a_t=\sqrt{\bar{\alpha}_t},
\qquad
b_t=\sqrt{1-\bar{\alpha}_t}.
$$

前向加噪公式是：

$$
x_t=a_tx_0+b_t\epsilon.
$$

这一个式子可以解出不同预测目标之间的关系。

### 10.1 预测噪声 $\epsilon$

目标：

$$
\epsilon_\theta(x_t,t)\approx \epsilon.
$$

训练代码就是当前脚本：

```python
xt, eps = forward_diffusion(x0, t, noise_schedule)
pred_eps = model(xt, t)
loss = F.mse_loss(pred_eps, eps)
```

采样时也最直接，因为 DDPM 采样公式本身需要 $\epsilon_\theta(x_t,t)$：

```python
pred_eps = model(x, t_tensor)
x = coeff1 * (x - coeff2 * pred_eps)
```

优点：

- DDPM 原始实现最常用。
- 训练目标简单，target 总是标准高斯噪声。
- 和 score matching 的关系最直接。

缺点：

- 不同 $t$ 的视觉意义不直观。模型输出看起来像噪声，不像图像。
- 在低噪声区域，真实需要修正的噪声很小，但 target 仍然是整幅 $\epsilon$。

### 10.2 预测干净图像 $x_0$

目标：

$$
x_{\theta}(x_t,t)\approx x_0.
$$

如果模型直接预测 $\hat{x}_0$，训练代码会变成：

```python
xt, eps = forward_diffusion(x0, t, noise_schedule)
pred_x0 = model(xt, t)
loss = F.mse_loss(pred_x0, x0)
```

采样公式仍然需要 $\epsilon_\theta$，所以要把 $\hat{x}_0$ 换回噪声预测。

从：

$$
x_t=a_tx_0+b_t\epsilon
$$

移项：

$$
b_t\epsilon=x_t-a_tx_0.
$$

除以 $b_t$：

$$
\epsilon=\frac{x_t-a_tx_0}{b_t}.
$$

所以如果模型输出 $\hat{x}_0$：

$$
\hat{\epsilon}
=
\frac{x_t-a_t\hat{x}_0}{b_t}
=
\frac{
x_t-\sqrt{\bar{\alpha}_t}\hat{x}_0
}{
\sqrt{1-\bar{\alpha}_t}
}.
$$

代码上大概是：

```python
alpha_bar_t = noise_schedule.alpha_bar[t].view(-1, 1, 1, 1)
sqrt_ab = torch.sqrt(alpha_bar_t)
sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)

pred_x0 = model(xt, t)
pred_eps = (xt - sqrt_ab * pred_x0) / sqrt_omab
```

反过来，如果模型预测 $\epsilon_\theta$，可以换算出 $\hat{x}_0$：

$$
\hat{x}_0
=
\frac{
x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)
}{
\sqrt{\bar{\alpha}_t}
}.
$$

优点：

- 输出是干净图像，视觉上更容易 debug。
- 可以直接 clip $\hat{x}_0$ 到图像范围，例如 $[-1,1]$。

缺点：

- 当 $t$ 很大时，$x_t$ 几乎是纯噪声，直接从纯噪声猜 $x_0$ 难度很高。
- 换算 $\epsilon$ 时要除以 $b_t=\sqrt{1-\bar{\alpha}_t}$，在非常小噪声时需要小心数值稳定性。

### 10.3 预测 velocity $v$

velocity 参数化定义：

$$
v = a_t\epsilon - b_tx_0.
$$

这不是凭空定义的。把 $x_t$ 和 $v$ 放在一起：

$$
\begin{bmatrix}
x_t \\
v
\end{bmatrix}
=
\begin{bmatrix}
a_t & b_t \\
-b_t & a_t
\end{bmatrix}
\begin{bmatrix}
x_0 \\
\epsilon
\end{bmatrix}.
$$

这个矩阵是一个旋转矩阵，因为：

$$
a_t^2+b_t^2
=
\bar{\alpha}_t+(1-\bar{\alpha}_t)
=
1.
$$

旋转矩阵的逆等于转置，所以：

$$
\begin{bmatrix}
x_0 \\
\epsilon
\end{bmatrix}
=
\begin{bmatrix}
a_t & -b_t \\
b_t & a_t
\end{bmatrix}
\begin{bmatrix}
x_t \\
v
\end{bmatrix}.
$$

于是：

$$
x_0 = a_tx_t - b_tv,
\qquad
\epsilon = b_tx_t + a_tv.
$$

如果模型预测 $v_\theta$，训练 target 是：

$$
v = a_t\epsilon-b_tx_0.
$$

代码大概是：

```python
alpha_bar_t = noise_schedule.alpha_bar[t].view(-1, 1, 1, 1)
sqrt_ab = torch.sqrt(alpha_bar_t)
sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)

xt, eps = forward_diffusion(x0, t, noise_schedule)
target_v = sqrt_ab * eps - sqrt_omab * x0
pred_v = model(xt, t)
loss = F.mse_loss(pred_v, target_v)
```

采样时换回 $\epsilon$：

```python
pred_v = model(x, t_tensor)
pred_eps = sqrt_omab * x + sqrt_ab * pred_v
x = coeff1 * (x - coeff2 * pred_eps)
```

也可以换回 $\hat{x}_0$：

```python
pred_x0 = sqrt_ab * x - sqrt_omab * pred_v
```

优点：

- $v$ 是 $x_0$ 和 $\epsilon$ 的旋转组合，不是简单偏向某一端。
- 在不同信噪比下 target 尺度更均衡。
- 现代扩散模型常用它来改善训练稳定性。

缺点：

- 物理直觉比 $\epsilon$ 和 $x_0$ 更绕。
- 采样和可视化时总要换算回 $\epsilon$ 或 $x_0$。

### 10.4 三者效果区别

| 预测目标 | 训练 target | 采样时需要 | 直觉 | 常见效果 |
| --- | --- | --- | --- | --- |
| $\epsilon$ | 加入的标准高斯噪声 | 直接使用 | 学噪声/score | DDPM 经典选择，简单稳定 |
| $x_0$ | 干净图像 | 换算成 $\epsilon$ | 学还原图像 | 易 debug，高噪声处更难 |
| $v$ | $a_t\epsilon-b_tx_0$ | 换算成 $\epsilon$ 或 $x_0$ | 旋转坐标 | SNR 更均衡，现代模型常用 |

你的当前 MNIST 脚本用的是 $\epsilon$ prediction。对 Module 1 来说，这正合适，因为它最贴近 DDPM 原论文和 $L_{\mathrm{simple}}$ 推导。

## 11. 代码到数学的索引

| 数学对象 | 代码位置 | 含义 |
| --- | --- | --- |
| $\beta_t,\alpha_t,\bar{\alpha}_t$ | `NoiseSchedule` | 噪声调度 |
| $q(x_t\mid x_0)$ | `forward_diffusion` | 直接采样任意噪声等级 |
| $\epsilon_\theta(x_t,t)$ | `UNet.forward` | 预测混入的噪声 |
| $L_{\mathrm{simple}}$ | `F.mse_loss(pred_eps, eps)` | 简化训练目标 |
| $p_\theta(x_{t-1}\mid x_t)$ | `sample` | 学到的反向去噪链 |
| $x_T\sim \mathcal{N}(0,I)$ | `torch.randn(...)` | 采样起点 |

## 12. Module 1 应掌握到什么程度

按 LPM case study 的 Module 1，你需要达到：

1. 能从 $q(x_t\mid x_{t-1})$ 独立推到 $q(x_t\mid x_0)$。
2. 能解释为什么训练时随机采样 $t$，而不是真的跑完整 forward chain。
3. 能从 ELBO 的 KL 项推到“高斯均值匹配”。
4. 能把均值匹配重参数化成预测 $\epsilon$ 的 MSE。
5. 能手写 Algorithm 1 训练循环和 Algorithm 2 采样循环。
6. 能说清楚 $\epsilon$ prediction、$x_0$ prediction、$v$ prediction 的换算关系。

如果这些都能自己推一遍，你已经完成 Module 1 的数学主线。之后看 Improved DDPM 时，重点就会变成两个工程问题：更好的 noise schedule，以及是否学习/选择更好的反向方差。

## 13. epoch 20 样本不好看的数学解释

epoch 20 的样本已经出现了笔画结构，但不少还不是稳定数字。这通常不是推导错了，而是学习到的 score field 还粗糙。

在每个 $t$ 上，模型都要学习一个向量场：

$$
x_t \mapsto \epsilon_\theta(x_t,t).
$$

采样会连续调用这个向量场约 1000 次。某一步的小偏差会被后续步骤继承和放大，所以视觉质量比训练 loss 更敏感。

常见改进：

1. 把训练从 20 epoch 提到 100 epoch。
2. 把学习率从 `1e-3` 降到 `2e-4`。
3. 用 EMA 权重采样。
4. 用 posterior variance $\tilde{\beta}_t$ 替代简单的 $\beta_t$。
5. 在 UNet block 中加入 GroupNorm、residual connection 或 attention。

这些是工程质量问题，不改变上面的数学主线。
