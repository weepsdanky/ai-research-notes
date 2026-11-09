# Mac mini goal prompt

把以下内容发给 Mac mini 上 `ai-research-notes` 项目中的新任务：

> 请创建一个 goal，并实际执行，不要停在计划：在这台 16GB Mac mini 上验证 `code/attention_lab` 的四 token notebook 和最小 MHA 训练，带我从公式到代码理解每一步。
>
> 先确认运行主机和 Git 状态，保护已有修改；安全获取包含 `code/attention_lab` 的最新 repo commit。若当前 checkout 正用于别的任务，不要覆盖或中断其工作，使用独立目录或安全的 git worktree。读取仓库指令和本目录 README。
>
> 在本目录独立虚拟环境安装依赖；记录 Python、PyTorch、macOS、机器和 MPS 可用性。运行 CPU 单元测试；MPS 可用时也运行 MPS 测试。用 nbclient 在正确工作目录从头执行 notebook，每个 cell 必须通过，并报告手算对照结果。遇到问题先修复并重新运行相关验证，不要修改或跳过断言来掩盖问题。
>
> 执行一次短 CPU smoke train，然后在 MPS 可用且资源允许时运行 300-step MPS 训练；如果 MPS 不可用则解释原因并使用 CPU。记录实际模型参数量、设备、步骤、训练损失、耗时、峰值内存（如可获得）及真实生成文本。初步训练应展示 loss 变化，但不可承诺一定收敛。
>
> 把实际命令、测试结果、训练数据和生成样本写入 `code/attention_lab/RESULTS.md`。明确说明语料是重复的小文本，只验证过拟合/训练闭环，不代表泛化；当前生成没有 KV cache，不能声称缓存加速。
>
> 在本任务中以教学方式带我读 notebook：先给我每一步问题让我预测，结合实际运行结果解释 shape、QK 点积、mask、softmax、加权 V、多头拆分、next-token loss 与反向传播。不要一次塞进 GDN、QSA、MoE 的细节。若需我的回答，可以先完成独立验证和训练再互动。
>
> 不下载完整语料、不租 GPU、不更改系统 MPS 内存保护；不提交数据、checkpoint 或虚拟环境。若代码需要修复，提交源码及报告到独立分支并报告 commit；未经明确要求不推送权重。
>
> 只有 notebook 与测试通过、训练和结果报告真实完成，才将 goal 标记完成。若 `create_goal` 工具不可用，明确说明并继续任务，不要假称创建成功。请先创建 goal 然后立即开始检查。
