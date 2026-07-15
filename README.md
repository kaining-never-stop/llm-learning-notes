# LLM Learning Notes

欢迎来到雷姆哈滋邦德的频道!!! 这里收录了我平时学习LLM、RL与Post-Train的学习笔记。欢迎大家学习、指正!! (正文保留 Markdown 原稿，数学公式使用 LaTeX 编写)

## 在线阅读

- [LLM 学习笔记](https://kaining-never-stop.github.io/llm-learning-notes/)

## 从数学讲清后训练

- [在分布视角下理解语言模型后训练](https://kaining-never-stop.github.io/llm-learning-notes/post-training/distributional-view/)

  从自回归序列分布和概率树出发，梳理 SFT、RL、On-Policy Distillation 的训练对象、更新信号，以及它们在泛化与灾难性遗忘上的差异。

- [DPO 为什么只做偏好分类，却“自带” KL 约束？](https://kaining-never-stop.github.io/llm-learning-notes/post-training/dpo-implicit-kl/)

  从一次手推过程出发，完整推导 KL-Regularized RL、最优策略、同一 Prompt 下共享的 $Z(x)$，以及 DPO Loss 中隐含的 KL 结构。文章附有三页原始手稿。

## 后训练基础

- [PyTorch 训练底层与 CausalLM 手撕](https://kaining-never-stop.github.io/llm-learning-notes/post-training-basics/pytorch-training-causallm/)

  从 Tensor、计算图、梯度与 AdamW 出发，串起 CausalLM 的 Logprob、Loss 和参数更新，并进一步手写最小 Llama 风格 Transformer。

## 获取笔记

- [下载全部笔记（ZIP）](https://github.com/kaining-never-stop/llm-learning-notes/archive/refs/heads/main.zip)
- [查看 Markdown 原文](docs/)

也可以克隆仓库，后续通过 `git pull` 获取更新：

```bash
git clone https://github.com/kaining-never-stop/llm-learning-notes.git
```
