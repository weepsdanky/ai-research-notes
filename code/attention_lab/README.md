# Attention Lab：从四个 token 到最小 MHA 训练

目标是理解 decoder-only 多头 causal self-attention 的输入、输出和参数更新。先手算一个
4-token 例子，再对照代码，最后在 16GB Mac mini 上进行可控的最小训练。
这个项目独立于 [`qwen_next_lab`](../qwen_next_lab/)，不会修改其代码和训练数据。

## 阅读顺序

1. 打开 [01_four_token_attention.ipynb](01_four_token_attention.ipynb)，先预测每个 cell 的结果，再运行。依次手算 QK、mask、softmax、加权 V，再观察多头 shape 和一次梯度更新。
2. 逐行阅读 [attention.py](attention.py) 中的 `CausalMHA`；把每个张量形状写在纸上。
3. 阅读同一文件中的 `TinyLM` 和 [train.py](train.py)：从字符 token 到 logits、next-token loss、backward、生成文本。
4. 运行测试，再跑固定小文本训练。比较第一步和最后一步 loss，并阅读真实生成结果。
5. 回到 [qwen_next_lab/model.py](../qwen_next_lab/model.py)，区分基础 MHA 与 RoPE、RMSNorm、output gate、QSA。

每步的学习问题和手算参考都在 notebook 中。完成标准：能够手推第二个 token 的 attention 输出、解释每个 shape、验证不泄露未来 token，且说清一次训练更新了哪些参数。

## 在 Mac mini 运行

本地工作站只准备文件；下列安装、测试、notebook 和训练交给 Mac mini。将
[MAC_MINI_GOAL.md](MAC_MINI_GOAL.md) 中的 prompt 发给远程 ChatGPT / Codex。
使用本目录独立环境，避免影响已有 Qwen 训练环境：

```bash
cd /Users/mark/Documents/Github/ai-research-notes/code/attention_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
LAB_DEVICE=mps python -m unittest -v
jupyter lab 01_four_token_attention.ipynb
```

在 notebook 中从上往下逐 cell 执行。命令行完整执行检查：

```bash
python -c 'import nbformat; from nbclient import NotebookClient; p="01_four_token_attention.ipynb"; n=nbformat.read(p, as_version=4); NotebookClient(n, timeout=120, kernel_name="python3").execute()'
```

先做 CPU 训练闭环，再在 MPS 上单独用新目录运行：

```bash
python train.py --device cpu --steps 300 --out runs/cpu
python train.py --device mps --steps 300 --out runs/mps
```

输出目录不能已存在，防止覆盖。脚本打印参数量、step、train loss、耗时与生成文本，并保存 checkpoint 和配置；也可使用 `python train.py --help` 查看参数。生成使用 greedy decoding。

## 实验边界

- 训练使用脚本内置、重复的小文本，字符级 tokenizer；这是**训练集过拟合**，不声称验证集泛化或通用语言能力。
- 模型是一层 decoder block、4 heads、宽度 32；学习参数包括 token/position embeddings、Q/K/V、输出投影、MLP、norm 和语言模型 head。实际参数数目由脚本打印。
- 这是纯 PyTorch 参考实现；它没有 Qwen 的 GDN/MoE/GR/QSA、没有训练大语料，也没有 KV cache。生成每步重算窗口，因此不能用它报告缓存加速。
- 不同设备及版本可能存在浮点差异。先确认测试和 finite loss，再看训练 loss 与真实生成；如果失败须记录原因。
- `runs/`、虚拟环境及 notebook checkpoint 均被忽略；结果可写 `RESULTS.md`，不要提交训练权重或环境目录。

## 下一课（先不实现）

实现 MHA incremental KV cache，并比较同一参数与位置编码下 cached / full-prefix logits；然后从 MHA 推到 MQA/GQA。代码结构和训练结果先在 Mac mini 上验证，避免把未经验证的参考实现当实验结论。
