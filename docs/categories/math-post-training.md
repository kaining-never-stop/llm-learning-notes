# 从数学讲清后训练

这个分类从目标函数、概率分布与优化过程出发，重点回答后训练算法中的公式为什么成立，以及这些公式最终如何改变模型的生成分布。

## 收录内容

### [在分布视角下理解语言模型后训练](../post-training/distributional-view.md)

从“语言模型是一个序列分布”出发，将生成过程展开成自回归概率树，并在同一视角下理解 SFT、RL 与 OPD。

### [DPO 为什么只做偏好分类，却“自带” KL 约束？](../post-training/dpo-implicit-kl.md)

从 KL-Regularized Reward Maximization 出发，推导最优策略、Reward 表达与 DPO Loss 之间的关系。
