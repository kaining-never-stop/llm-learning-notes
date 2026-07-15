# LLM 学习笔记

这里收录语言模型、强化学习与后训练相关的学习笔记。内容以 Markdown 保存，公式使用 LaTeX 编写，可以在线阅读，也可以下载原文。

## 内容分类

<div class="grid cards" markdown>

-   **从数学讲清后训练**

    从目标函数、概率分布与优化过程出发，理解后训练算法为什么成立。

    [进入分类](categories/math-post-training.md)

-   **后训练基础**

    补齐模型结构、训练链路与实现基础，为后续学习 SFT、DPO、PPO、GRPO 建立统一底座。

    [进入分类](categories/post-training-basics.md)

</div>

## 最新收录

### PyTorch 训练底层与 CausalLM 手撕

从 Tensor、Parameter、计算图和 AdamW 开始，完整串起：

- `input_ids -> logits -> logprob/loss` 的前向链路；
- `loss.backward() -> parameter.grad -> optimizer.step()` 的参数更新链路；
- CausalLM 的 Shift、Mask、Logprob 与 Cross Entropy；
- 从 Attention、RoPE、RMSNorm 到最小 Llama 风格 Transformer 的手写实现。

[阅读文章](post-training-basics/pytorch-training-causallm.md){ .md-button .md-button--primary }

## 从数学讲清后训练

### 在分布视角下理解语言模型后训练

从“语言模型是一个序列分布”出发，将生成过程展开成自回归概率树，并在同一视角下比较：

- SFT 如何沿外部示范路径提高 Token 概率；
- RL 如何从当前策略采样，并根据 Reward 调整路径概率；
- OPD 如何在 Student 访问的状态上学习 Teacher 分布；
- On-Policy Data 为什么可能减少遗忘并改善泛化。

[阅读文章](post-training/distributional-view.md){ .md-button .md-button--primary }

### DPO 为什么只做偏好分类，却“自带” KL 约束？

这篇来自一次手推过程中产生的问题：DPO 看起来只是在拟合 Chosen 与 Rejected 的偏好关系，为什么 Loss 中却自然出现了当前策略与 Reference Policy 的概率比？

文章从 KL-Regularized Reward Maximization 出发，依次推导：

- KL 约束下最优策略的解析形式；
- 为什么同一 Prompt 下的回答共享同一个 $Z(x)$；
- $Z(x)$ 如何在 Bradley–Terry Preference Model 中抵消；
- DPO Loss 为什么继承了原始 RLHF 目标的 KL 结构。

文中同时附有三页原始手稿。

[阅读文章](post-training/dpo-implicit-kl.md){ .md-button .md-button--primary }

## 获取原文

- [下载全部笔记（ZIP）](https://github.com/kaining-never-stop/llm-learning-notes/archive/refs/heads/main.zip)
- [查看 GitHub 仓库](https://github.com/kaining-never-stop/llm-learning-notes)
- [查看全部 Markdown 原文](https://github.com/kaining-never-stop/llm-learning-notes/tree/main/docs)

更多下载方式见[获取笔记](download.md)。
