# LLM 学习笔记

这里收录语言模型、强化学习与后训练相关的学习笔记。内容以 Markdown 保存，公式使用 LaTeX 编写，可以在线阅读，也可以下载原文。

## 已收录

### 在分布视角下理解语言模型后训练

从“语言模型是一个序列分布”出发，将生成过程展开成自回归概率树，并在同一视角下比较：

- SFT 如何沿外部示范路径提高 Token 概率；
- RL 如何从当前策略采样，并根据 Reward 调整路径概率；
- OPD 如何在 Student 访问的状态上学习 Teacher 分布；
- On-Policy Data 为什么可能减少遗忘并改善泛化。

[阅读文章](post-training/distributional-view.md){ .md-button .md-button--primary }

## 获取原文

- [下载全部笔记（ZIP）](https://github.com/kaining-never-stop/llm-learning-notes/archive/refs/heads/main.zip)
- [查看 GitHub 仓库](https://github.com/kaining-never-stop/llm-learning-notes)
- [下载本文 Markdown 原文](https://raw.githubusercontent.com/kaining-never-stop/llm-learning-notes/main/docs/post-training/distributional-view.md)

更多下载方式见[获取笔记](download.md)。
