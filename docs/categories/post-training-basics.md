# 后训练基础

这个分类整理进入 SFT、DPO、PPO、GRPO 等后训练方法之前，需要先掌握的模型结构、训练链路与实现基础。

## 收录内容

### [PyTorch 训练底层与 CausalLM 手撕](../post-training-basics/pytorch-training-causallm.md)

从 Tensor、计算图、梯度与 AdamW 出发，串起 CausalLM 的 Logprob、Loss 和参数更新，并进一步手写最小 Llama 风格 Transformer。
