# 01 PyTorch 训练底层

## 00 前言

这份文档适合想要开始学习 LLM 算法，但目前只具备基础 Python 知识、还没有真正训练过模型的人。这里的内容都是我在学习过程中逐步积累下来的笔记：不仅记录“一个概念是什么”，也尽量把它放回真实的训练链路中，去理解它为什么存在、输入输出是什么、最终又会影响到哪里。

全文的带学路径为：

1. 定义清楚当前阶段的学习目标，先知道这一章到底要解决什么问题；
2. 熟悉必要的前置概念，避免读源码时被 Tensor、Parameter、计算图、梯度等名词打断；
3. 结合源码理解原理讲解中容易被省略、但工程中一定会遇到的细节；
4. 通过手写练习把“看懂了”转换成“自己能写出来”；
5. 最后通过经典问题和吃透标准，检查自己是否真的建立了完整认识。

这条路径的重点不是背下所有 API，也不是第一次就读懂源码里的所有优化分支，而是逐渐建立一条稳定的主线：模型如何从 `input_ids` 得到 `logits`，如何从 `logits` 得到 `loss`，又如何通过 `backward()` 和 `optimizer.step()` 真正改变参数。只要这条主线能够讲通，后面学习 SFT、DPO、PPO、GRPO 时，就不会只看到一堆彼此分散的公式。

### 怎么使用这份文档

- 第一遍先顺着主流程阅读，不必停下来记住每个源码细节；
- 第二遍重点跟 Shape，给每个关键张量标出 `[B, T, H]`、`[B, T, V]` 等维度；
- 第三遍自己手写练习，再对照文档检查 Shift、Mask、Dtype 和梯度更新顺序；
- 最后尝试脱离笔记回答文末问题，回答不清楚的地方再回到对应章节补齐。

### 写在开始之前

千里之行，始于足下。

这一部分第一次学起来有些慢是正常的。训练底层同时涉及线性代数、计算图、模型结构和优化器状态，很难只看一遍就全部连起来。遇到 Shape 对不上、分不清 `backward()` 和 `step()`、或者读源码时频繁断线，都不代表学不会，只说明这些概念还没有在脑中形成一条完整链路。

不要急着追求一次吃透。先确保今天比昨天多讲清楚一个环节：可能是 Labels Shift，可能是 `.grad` 的来源，也可能只是终于能解释为什么要先 `log_softmax` 再 `gather`。这些局部理解最终会连在一起。等你能独立把一次训练更新从前向传播讲到参数更新时，后面的后训练算法会明显更容易理解。

## 本阶段要解决的问题

这一阶段只解决一个底层问题：一次训练到底发生了什么。

必须能从 `input_ids -> logits -> logprob/loss -> loss.backward() -> parameter.grad -> optimizer.step()` 完整讲下来。后面所有 SFT、DPO、PPO、GRPO，本质上都是把 loss 换掉，把数据来源换掉，把训练系统变复杂，但底层更新参数的机制仍然是这条链。

## 源码锚点

- [PyTorch AdamW](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py)
- [PyTorch Adam](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py)
- [Transformers Llama CausalLM](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [TRL GRPOTrainer](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)

## 核心概念

### 一些概念

包含：

- 1、Tensor
    - 至少包含张量值、grad、shape、dtype、device、requires_grad、grad_fn
- 2、Parameters :
    - 一种特殊Tensor , 关键区别在于 , 挂到nn.Module里后 , 会自动出现在model.parameters()里 , 从而可以被传给optimizer管理
    - 通常表示模型中需要训练的参数
    - 一般满足 : requires_grad = True  ,  is_leaf = True , grad_fn = None , backward 后 p.grad 不为None
- 3、Module:
    - 可以包含Parameter、buffer、sub Module 、 forward逻辑
- 4、Optimizer
    - 负责保存参数引用、保存超参数、保存优化器状态、读取p.grad、更新参数p

### 串一遍流程

流程：

- 1、logits labels
    - 某个位置的hidden state.shape = [H]
    - lm_head.weight.shape = [V, H]
    - logits = W · h
    - (token k 对应的是 W_vocab[k, : ]这一行 , 这一行和h越接近 , 点积越大 , 那么token k 的logit 就越高
    - W_vocab 是一个共享的词表解码器 , h负责表示当前位置的上下文语义是什么 , W_vocab 负责判断这个语义状态更像哪个 next token)
- 2、shift
    - shift_logits.shape = [B, T-1, V]
    - shift_labels.shape = [B, T-1]
    - 对于某一个位置 (b , t) , shift_logits[b, t, :]是一个长度为V的向量
    - (表示模型在这个位置对整个词表 V 个 token 的预测分数)
- 3、log_softmax
    - log_probs = F.log_softmax(shift_logits, dim = -1)
    - 得到log_probs.shape = [B, T-1, V]
    - (表示每个位置、每个vocab token的log prob)
- 4、gather
    - `selected_logps = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)`
    - 得到selected_logps.shape = [B, T-1]
    - (表示每个位置 给正确的预测token的分数)
- 5、loss scalar
    - per_token_loss.shape = [B, T-1]
    - (每条样本 , 每个预测位置, 都有一个loss , 但训练需要一个标量)
    - 1️⃣没有mask时
    - loss = per_token_loss.mean()
    - 2️⃣有mask时
    - 公式见脑海:D
    - 然后得到了一个标量loss
- 6、backward
    - 它对每个参数元素 , 都有一个偏导
    - 然后 , 对每个参与当前 loss 计算的可训练参数张量 , 计算一个和它shape一致的.grad
    - 比如对logits的梯度(见笔记本) , 继续传播到 lm_head、Transformer blocks、embedding等参数上
    - 1️⃣lm_head.weight.grad 只负责更新输出词表头
    - 2️⃣dL / dh 会继续进入到Transformer , 然后一层一层往前传
    - (对于后面的传播都是todo , 要细细研究)

### 分界线(以下是琐碎的点)

### PyTorch calculation graph params

包含：

- 1、requires_grad
    - 表示是否需要计算梯度
- 2、grad_fn
    - 表示该 tensor 由哪个可导操作产生
    - 用户直接创建出来的叶子张量的 grad_fn 为 None
- 3、is_leaf
    - 计算图的起点 , 通常是模型参数
- 4、.grad
    - 反向传播后保存梯度的地方
    - 对于中间张量 , 默认不会保存梯度 , 如果要看的话 , 需要使用.retain_grad()
- 5、detach()
    - 保留数值 , 但是切断计算图 , 后续的 loss 不会通过 old_logps 反传给原来的模型参数
- 6、with torch.no_grad():
    - 这段 forward 不构建计算图 , 不保存中间激活 , 不能backward , 省显存

### logits  and  log probability

logits 是模型最后一层 `hidden_states @ W_vocab^T` 得到的词表分数，还没有经过 softmax，也不是 log 之后的概率。

对第 `t` 个位置：

- hidden state：`h_t`，shape `[hidden_size]`
- vocab head：`W_vocab`，shape `[vocab_size, hidden_size]`
- logits：`z_t = W_vocab h_t`，shape `[vocab_size]`
- probability：`p_t = softmax(z_t)`
- logprob：`log p_t = log_softmax(z_t)`

后训练里经常说“拿到 token 的 logprob”，实际过程是：

1. forward 得到所有位置的 logits。
2. 对 logits 做 `log_softmax`。
3. 用真实生成 token 的 id 去 `gather` 对应位置的 logprob。
4. mask 掉 prompt、padding、无效位置。

### Adam

维护 “历史梯度方向” 和 “历史梯度大小” 的滑动平均 。
假设当前第 t 次 optimizer update 时 :

- 1、W 的梯度g_t 存在 W.grad 里
- 2、一阶动量 :
    - 历史梯度的指数滑动平均
    - 公式: m_t = beta1 * m_{t-1} + (1-beta1) * g_t
    - 用途 : 不要让一个mini-batch的噪声梯度把更新方向带偏
- 3、二阶动量:
    - 历史梯度平方的指数滑动平均
    - 公式 : v_t = beta2 * v_{t-1} + (1 - beta2) * g_t ^ 2
    - 用途 : 如果某个参数方向上的梯度一直很大 , 会自动把这维的步子缩小一点 ( by  1 / sqrt(v_i) )
- 4、对于反向传播可以展开讲讲
    - loss是一个标量 : L(theta)
    - 反向传播得到的是 : dL/dtheta = [dL/dtheta_1 , dL/dtheta_2, ... , dL/dtheta_N]
    - 某一维的梯度很大指的是 : 某一个 theta_i 稍微变一点 , loss变化就很大
    - 每个 theta 都有自己的小方向信号 , optimizer 根据这些信号同时更新整个参数张量
- 5、Adam 核心更新
    - theta_i = theta_i - lr * m_i / ( sqrt(v_i) + eps )
- 6、gradient accumulation 时 , 是怎么step的
    - 先积累grad , 后面做一次.step()

### loss 是标量，梯度是每个参数上的方向信号

训练时不是直接“看 loss 大小就知道每个 token 怎么改”，而是：

1. loss 是一个标量。
2. autograd 沿计算图反向求导。
3. 每个参数 `W` 得到 `dL/dW`。
4. optimizer 用 `W = W - lr * dL/dW` 或 AdamW 的变体更新。

如果某个 token 的 loss 希望提高目标 token 概率，反向传播会让该 token 的目标词 logits 梯度方向与其他词不同。真正发生变化的不是 token 本身，而是产生该 token logits 的模型参数。

### 梯度下降是什么意思

梯度 `dL/dW` 表示：如果参数 `W` 增大一点，loss 会怎么变。

- `dL/dW > 0`：增大 `W` 会让 loss 变大，所以更新时减小 `W`。
- `dL/dW < 0`：增大 `W` 会让 loss 变小，所以更新时增大 `W`。

所以参数更新常写成：

```text
W_new = W_old - learning_rate * gradient
```

AdamW 不是简单 SGD，但核心仍然是利用梯度方向，只是加了动量、二阶矩估计和 decoupled weight decay。

### Cross entropy

交叉熵损失的标准写法 , 在NTP任务下的写法 (todo)。
真实情况下的计算 用 log_softmax 来实现:

- 1、为什么不先softmax再log
    - 因为直接softmax可能存在很大/小的logit导致overflow/underflow
    - 而log_softmax会做一个稳定化处理 , like: (todo)

## 源码走读任务

tips: 读源码只关注以下几点:

- 1️⃣训练底层 , AdamW怎么从 p.grad 更新参数
- 2️⃣causallm , input_ids 怎么变成 logits  , labels 怎么变成
- 3️⃣后训练 logprob ,  logits 怎么通过 log_softmax + gather 变成 token logprobs
- 4️⃣工程诊断

### 任务 1：从 AdamW 源码理解 optimizer.step()

#### 1、认知

loss.backward() 把梯度写到每个参数的 .grad
-> optimizer.step() 遍历 optimizer 管理的参数 , 找到 p.grad is not None 的参数
->读取 p.grad , 结合AdamW维护的 exp_avg、exp_avg_sq、step 来进行参数更新

#### 2、AdamW 继承于 Adam

初始化时把 decoupled_weight_decay = True  传给父类
-> 只是在 weight decay 语义上做了解耦

- step : 一个计数器 , 记录这个参数已经被optimizer更新过几次 .
- bias correction :  前几次滑动平均会偏小 , 所以要校正
- AdamW 为什么叫 decoupled weight decay。
- 一阶动量、二阶动量

#### 3、step() 主流程

```python
def step(self):
    # optimizer里的参数组 , 可以给不同参数设置不同的学习率
    for group in self.param_groups:
        # 只有grad is not None的参数才会被更新
        params_with_grad = []
        # 每个参数的梯度
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []

        self._init_group(
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            ...,
            state_steps,
        )

        adam(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            ...,
            state_steps,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            betas=group["betas"],
            # ...
        )

```

#### 4、_init_group收集梯度

```python
for p in group["params"]:
    if p.grad is not None:
        params_with_grad.append(p)
        grads.append(p.grad)

        state = self.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])
        state_steps.append(state["step"])

```

#### 5、p.grad 变化

刚创建时 , 为None
->执行完backward() 后 ,  p.grad 变为和 p 同维的 tensor
->optimizer.zero_grad(set_to_none = True) 后 , 设置为None
->set_to_none = False 后 , 设置为全0 tensor , 但下一次optimizer仍然会把它视为有grad的参数

标准顺序一般为:

```python
optimizer.zero_grad(set_to_none = True)
outputs = model(**batch)
loss = outputs.loss
loss.backward()  # 计算 loss 对参与当前计算的可训练 leaf parameter 的梯度 , 并累积到p.grad
optimizer.step()  # 只读 p.grad 然后依据 optimizer state 去更新参数
```

```text
总结 :
optimizer.step() 遍历 optimizer 的 parameter groups , 找到 p.grad is not None 的参数 , 把这些参数、对应的 p.grad 以及 AdamW 的一阶动量、二阶动量、step 等状态组织起来 , 然后执行 AdamW 更新

AdamW 和 Adam 的关键区别在于:
weight decay不进入一阶动量和二阶动量 , 而是独立对参数做衰减
具体来说, weight decay就是让参数不要变得太大,把参数往0的方向拉.
但是Adam会把它加到梯度里,它会被动量 & 自适应学习率影响
AdamW独立做weight decay,然后再做梯度更新
```

### 任务 2：从 CausalLM 源码理解 labels shift

#### 1、LlamaForCausalLM源码

```text
"""
1、config项

backbone , 不带语言模型输出头的主体结构 , 负责:
input_ids
-> token embedding
-> Transformer decoder layers
-> final hidden_states

lm_head , 把hidden_states投影到vocab_size:
hidden_states[B, T, H]
lm_head  H->V
logits[B, T, V]
"""

self.model = LlamaModel(config)
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size , bias = False)

"""
2、forward

outputs的输入:
input_ids : [B, T] , 索引
attention_mask : [B, T] , 参与注意力的token为1
position_ids : [B, T] , 位置编号
past_key_values : KV Cache
inputs_embeds : 可以直接传 embedding 向量
labels : [B, T] , 传了后 , 会算 CausalLM loss
use_cache : 是否返回 KV Cache
logits_to_keep : 训练时通常需要全序列的 logits , 推理时只需要最后一个位置的 logits
**kwargs
"""

#这个model是LlamaModel(config),即没有lm_head的主体
#同一个forward,可以用来推理 / 训练
#推理:入参不需要labels / outputs.loss = None
#训练:入参需要labels / outputs.loss 有值
#一个输出对象,有多个字段: last_hidden_state、past_key_values、hidden_states、attentions (hidden_states表示每一层的hidden_state的集合)

outputs = self.model(
    ....
)

#先拿到hidden_states
#logits_to_keep 控制 lm_head 对哪些序列位置计算 logits,选出一部分token位置,送进lm_head,得到每个位置的logits
#slice_indices : 由 logits_to_keep 得到, slice_indices = slice(-logits_to_keep, None)
 hidden_states = outputs.last_hidden_state
 logits = self.lm_head(hidden_states[:, slice_indices, :])

 loss = None
 if labels is not None:
     loss = self.loss_function(logits = logits , labels = labels, vocab_size = ...)

 return CausalLMOutputWithPast(
     loss = loss,
     logits = logits,
     past_key_values = outputs.past_key_values,
     hidden_states = outputs.hidden_states,
     attentions = outputs.attentions
     )

```

#### 2、`logits_to_keep` 、`attention_mask` 和 `labels=-100` 的区别

- 1️⃣attention_mask：
    - 控制 attention 里哪些 token 可见。
    - (它不一定让这些位置完全不计算 hidden state , 而是让其余 token 不去 attention 这些位置)
- 2️⃣labels=-100 / loss_mask：
    - 控制哪些位置参与 loss。
- 3️⃣logits_to_keep：
    - 控制哪些位置过 lm_head 计算 logits。

#### 3、整个流程串起来过一遍

1️⃣__init__ , 从 config 中载入 model、lm_head、vocab_size , 并做 post_init

```text
LlamaForCausalLM
├── self.model: LlamaModel backbone
└── self.lm_head: hidden_size -> vocab_size
```

2️⃣forward

第一步, 调用backbone

```text
input_ids
-> embedding
-> position / RoPE
-> causal mask + attention_mask
-> 多层 decoder layer
-> final norm
-> last_hidden_state
```

第二步, 取最后的 hidden_state

```python
hidden_states = outputs.last_hidden_state
```

第三步, 决定保留哪些logits位置

```python
#整数->保留最后logits_to_keep个位置
#index/tensor -> 按给定位置取
slice_indices = slice(-logits_to_keep,None) if isinstance(logits_to_keep,int) else logits_to_keep
```

第四步, 通过 lm_head 得到 logits
第五步, 有labels , 就计算loss
第六步, 返回结构化输出

### 任务 3：从后训练 logprob 逻辑理解 gather

#### 1、后训练为什么需要 logprob

- 后训练时 , 是把 prompt + completion 喂给模型, 让模型在 teacher forcing 模式下计算:
    - 每个completion token在它对应的 prefix 下的 logprob

```text
    SFT:
    需要 assistant token 的 logprob
    loss = - mean(logprob)

    DPO:
    需要 chosen / rejected response 的 sequence logprob
    比较 policy 相对 reference 是否更偏 chosen

    PPO / GRPO:
    需要 completion token 的 per-token logprob
    用 current / old / ref logprob 构造 ratio、KL、policy loss
```

#### 2、后训练 logprob 的完整计算链路

```text
forward
-> logits
-> F.log_softmax(logits, dim = -1)
-> log_probs
-> torch.gather(log_probs, dim = -1 , index = labels.unsqueeze(-1)).squeeze(-1)
-> selected_logps
```

#### 3、三类mask

- attention mask:
    - forward时backbone内计算注意力时 , 哪些token可以被看见
- labels = -100:
    - 控制计算CE loss 时哪些位置可以忽略
- completion mask:
    - 在后训练时 , 控制哪些 token 属于response / completion
    - (主要是控制哪些token的logprob被统计、哪些token的loss被计算、哪些token用于 advantage / reward / KL / ratio)

#### 4、TRL 的 memory-efficient selective_log_softmax 源码

- 一句话:
    -   selective_log_softmax 解决的是 [不需要所有 token 的 logprob , 只需要已生成 token 那一列的 logprob]
- 简单来说:
    - todo
- 朴素写法:

```python
# 存在的问题: 产生两个大张量
# 核心优化目标: 不要完整保存 logits.log_softmax(dim=-1)的结果
log_probs = logits.log_softmax(dim = -1)
selected_logps = torch.gather(
    log_probs,
    dim = -1,
    index = index.unsqueeze(-1),
).squeeze(-1)
```

- 我们先熟悉一个数学概念:
    -  log_softmax(z_y) = logit(z_y) - logsumexp(z)
    - 所以其实可以不用先算完整的 logits.log_softmax(-1)
    - 而是可以先算:
        - selected_logits = gather(logits, index)
        - normalizer = logsumexp(logits, dim = -1)
        - selected_logps = selected_logits - normalizer
    - 对比一下两种:
        - 朴素版: logits [B,T,V] -> log_probs [B,T,V] -> gather -> [B,T]
        - 改进后: logits [B,T,V] -> selected_logits [B,T,1] -> logsumexp [B,T] -> selected_logps [B,T]
- 有哪些缺陷:
    - 1️⃣ 目前只对 float32 / float64 , 因为低精度下 logsumexp
        - 原因: 低精度下 , 大量小数相加会产生明显误差,
    - 2️⃣ 为什么朴素版的没有?
        - 原因: 因为朴素版用 F.log_softmax , 是一个高度优化的基础算子 , 但改进版是手动算
        - 它会去做:
        - max-subtraction ( 先把最大的logit取出来 , 所有的logits去减掉它, 可行因为 softmax对整体平移不敏感)
        - 稳定的 reduction ( 首先,这是聚合操作,在取max logit,求sum_exp时都会用,但是V很大时误差会累积,所以可以先分块求局部max,再合并得到全局max,再分块求sum,总之有更稳定的方法)
        - 合理的 dtype / accumulation 处理 (精度变换)
        - 避免真的先 softmax 再 log (先softmax得到的概率可能会很小,再取log可能得到-inf,这个问题两种都解决了)

#### 5、在 GRPOTrainer 中看 selected logprob 如何进入 old/ref/current logprobs、ratio、KL、loss

(todo 等学了后面的后训练算法后再看这部分)

## 手写练习

### 练习 1：写 `masked_mean`

要求支持 `[batch, seq]` 的值和 mask：

```python
import torch

def masked_mean(values, mask, dim=None, eps=1e-8):
    """
    values:
    Tensor，常见 shape:
    [B, L] -> 每个 token 一个值
    [B, L, H] -> 每个 token 一个向量

    mask:
    Tensor，常见 shape:
    [B, L]
    1 表示有效位置，0 表示无效位置

    dim:
    None: 对所有有效位置求均值
    1: 对序列维求均值，返回 [B] 或 [B, H]

    eps:
    防止 mask 全 0 时除以 0
    """
    #确保在一个device, 并可以相乘
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    masked_values = values * mask
    #如果一个batch全是无效的token,那么分母为0,直接除会NaN,所以clamp_min
    if dim is None:
        numerator = masked_values.sum()
        denominator = mask.sum().clamp_min(eps)
    else:
        numerator = masked_values.sum(dim = dim)
        denominator = mask.sum(dim=dim).clamp_min(eps)
    return numerator / denominator
```

验收：

- mask 全 0 不出现 NaN。
- dim 为 `None` 和 `1` 都能工作。
- 能解释 SFT、DPO、GRPO 中哪里会用到它。

### 练习 2：写 `selective_log_softmax`

要求：

```python
def selective_log_softmax(logits,labels,ignore_index=-100):
    """
    logits : [B, L, V]
    labels : [B, L]
    return:
        selected_logps : [B, L]
        labels 为 ignore_index 的位置返回 0
    """
    #不等于 ignore_index 的 labels 位置置为 True
    valid_mask = labels.ne(ignore_index)
    #取反,那么ignore_index的位置就会被置换成0 , 因为-100不是一个合法id,为了做索引操作,进行一步转换
    safe_labels = labels.masked_fill(~valid_mask,0)

    log_probs = F.log_softmax(logits,dim=-1)

    selected_logps = torch.gather(
        log_probs,
        dim = -1,
        index = safe_labels.unsqueeze(-1)
    ).squeeze(-1)

    #通常情况下,会把ignore_index的位置的logps设置为不影响后续聚合的中性值
    selected_logps = selected_logps.masked_fill(~valid_mask,0.0)
    return selected_logps
```

- 输入 logits 和 token ids。
- 输出每个位置所选 token 的 logprob。
- 不对 prompt token 计算 loss。

面试追问：

- 为什么不能先 softmax 再 log。
    - 因为数值不稳定 , softmax中有 exp(logits) , 很大时会 overflow , 很小时会underflow ; log_softmax用更稳定的等价形式
- `log_softmax` 为什么数值更稳定。
    - (todo)
- vocab 很大时这一步显存瓶颈在哪里。
    - 完整的 materialize log_probs 会带来很大的峰值显存
    - (本来 logits 很大 , 又生成了一个同样很大的 log_probs , 显存爆炸 , 这也就是peak memory)
    - 因为后训练里 , 通常只需要被采样 or 被监督 token 的 logprob
    - 所以TRL用memory-efficient selective log softmax 来减少峰值显存

### 练习 3：写一个Sequence logprob

后训练语境里基本≈Response logprob , 这个 Sequence 特指 response / completion 部分：

```python
def sequence_logprobs(logits,labels,response_mask,ignore_index=-100):
    """
    logits : [B,L,V]
    labels : [B,L]
    response_mask : [B,L]
        0 表示 prompt / padding token
        1 表示 response / completion token
    return :
        token_logps : [B,L-1]
        seq_logps : [B]
    """
    # contiguous() 把 tensor 在内存里整理成连续存放的格式,只改变底层内存布局
    # 因为切片得到的是 view,可能是指向原 Tensor 的部分区域,而不是连续内存
    shift_logits = logits[:,:-1,:].contiguous()
    shift_labels = labels[:,1:].contiguous()
    shift_mask = response_mask[:,1:].contiguous()

    token_logps = selective_log_softmax(
        shift_logits,
        shift_labels,
        ignore_index = ignore_index,
    )
    # 两个条件 : label不能是ignore_index & 必须属于 response/completion
    # 同时满足 labels 不是 -100 、此位置属于 response , 才为 True
    valid_mask = shift_labels.ne(ignore_index) & shift_mask.bool()
    # 把 mask 转成浮点类型,然后 * 就可以把 false 位置的logps变为0.0
    token_logps = token_logps * valid_mask.to(token_logps.dtype)

    # seq prob = token prob 连乘 , log 后变成 sum
    seq_logps = token_logps.sum(dim=-1)
    return token_logps,seq_logps
```

### 练习 4：写一个最小 CausalLM train_step

不要求训练大模型，但要求接口像真实训练：

```python
def train_step(model, batch, optimizer,max_grad_norm=None):
    """
    batch:
        input_ids : [B,L]
        attention_mask : [B,L]
        labels : [B,L]
    return:
        loss.item()
        grad_norm
    整个过程:
        参数 θ
        → hidden states
        → logits
        → loss
        → backward 得到梯度
        → 更新参数 θ
        → 下一次 forward 时 logits 发生变化
    """
    # 切换到训练模式 : 1️⃣Dropout 2️⃣BatchNorm(会更新)
    model.train()
    # 把梯度清成 None , 而不是 0 , 更省显存、速度更快
    optimizer.zero_grad(set_to_none=True)
    # forward
    outputs = model(
        input_ids = batch["input_ids"],
        # 有的话取出来,没有的话返回None
        attention_mask = batch.get("attention_mask",None),
        labels = batch.get("labels",None),
    )

    # 如果算了 loss, 就返回 outputs 里的 loss
    if hasattr(outputs,"loss") and outputs.loss is not None:
        loss = outputs.loss
    else:
        #如果outputs有logits属性，就取outputs.logits； 否则就把outputs本身
        logits = outputs.logits if hasattr(outputs,"logits") else outputs
        #手动算loss
        loss = causal_lm_loss_from_logits(logits,batch["labels"])

    loss.backward()
    #初始化梯度范数
    #norm_type : L1(绝对值求和,更多用于诊断梯度总绝对量、稀疏分析: 可以分析梯度质量是否分散、梯度是不是稀疏集中在一些位置)、L2(平方求和再开方,和“欧式步长”更对应,并且对大梯度更敏感,参数变化量:-lr*g, L2距离衡量:lr*||g|| , 限制max L2, 就是限制这一步参数的移动距离)
    #global(把模型所有梯度加起来算)、per_layer(每一层分开算)、per_parameter(每个参数张量分开算)
    grad_norm = None
    #梯度裁剪,按比例缩小,不超过max_grad_norm
    #更新时: param = param - lr * grad 可能出现情况: 1️⃣loss出现inf/nan 2️⃣梯度异常大导致loss爆炸 3️⃣参数更新过猛,参数移动距离大,会去往另一个区间
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )

    optimizer.step()

    # 这里是打日志,切出autograd计算图 (避免显存无法释放、OOM)
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": None if grad_norm is None else float(grad_norm.detach().cpu())
    }
```

验收：

- `zero_grad -> forward -> loss -> backward -> grad_norm -> step` 顺序正确。
- 能说清楚每一步如果漏掉会发生什么。
- 能把这个 train_step 映射到 SFTTrainer 的训练流程。

### 练习 5：手撕 Transformer

#### 〇、前置

- 定位:
    - Llama-style Transformer
- 流程:
    - PreNorm RMSNorm
    - -> Self-Attention with RoPE and possibly GQA
    - -> Residual add
    - -> RMSNorm
    - -> SwiGLU MLP
    - -> Residual add
- 和经典Transformer差异(前经典 , 后Llama):
    - Norm: LayerNorm / RMSNorm
    - Norm位置: post-norm / pre-norm
    - 位置编码: 绝对位置编码 / RoPE
    - Attention: MHA / 支持GQA & MQA
    - FFN: Linear -> ReLU/GELU -> Linear     /    SwiGLU
    - outputs: encoder & decoder    /   decoder-only CausalLM
- shape:
    - B = batch size
    - T = seq_len
    - H = hidden_size
    - V = vocab_size
    - Nh = num_attention_heads
    - Nkv = num_key_value_heads
    - D = head_dim = H / Nh
    - I = intermediate_size
- 完整的 forward:
    - input_ids [B,T]
    - -> embed_tokens -> hidden_states [B,T,H]
    - -> LlamaDecoderLayer ✖️ N   ->  hidden_states [B,T,H]
    - -> final RMSNorm -> hidden_states [B,T,H]
    - -> lm_head -> logits [B,T,V]
- 单层Decoder内部 (每层内都做了残差连接):
    - q_proj  -> [B,T,Nh * D] -> view / transpose -> q : [B,Nh,T,D]
    - k_proj  -> [B,T,Nkv * D] -> view / transpose -> k : [B,Nkv,T,D]
    - v_proj  -> [B,T,Nkv * D] -> view / transpose -> v : [B,Nkv,T,D]
    - RoPE(q,k)
    - repeat_kv(k,v)  if  GQA
    - attention score : [B,Nh,T,T]
    - attention output : [B,Nh,T,D]
    - transpose/reshape -> [B,T,H]
    - o_proj -> [B,T,H]
- 一些概念:
    - view: 同一块内存用新shape来解释 , 内存布局不兼容时需要先contiguous
    - (作为对象时 , 表示共享storage , 但是 metadata 不同的两个tensor)
    - transpose: 交换两个维度 , 只会改 stride , 不改真实 storage
    - reshape: 能不复制 , 就返回 view ; 如果内存不连续 , 就复制一份新内存
    - contiguous: 把一个 tensor 在内存中整理成“连续存放”的形式
    - tips: (一个 tensor 由 真实数据内存 ➕ 可解释元信息 组成 , 其中元信息包括 : 1️⃣shape / size 2️⃣ stride : 表示某个维度上索引➕1 , 需要在内存上跳多少个元素3️⃣storage_offset4️⃣dtype5️⃣device6️⃣requires_grad . 常见的可能只改变 metadata 的操作: transpose、permute、view、reshape、unsqueeze、squeeze、expand , 如果后面需要做 storage上的操作 , 就需要 contiguous 操作)

#### 一、RMSNorm

- Norm:
    - 针对什么问题:
        - 在一个 block 中 ,  x -> attention/MLP -> residual add -> 下一层
        - 每一层的输出都可能让 hidden state 的尺度变大 or 变小
        - 造成尺度变换的原因 : 1️⃣Linear proj 2️⃣Residual add 3️⃣MLP激活 4️⃣层数累积
    - 方法:
        - 对每个 token 的 hidden 向量 , 在 hidden_size 维度上做归一化
    - 数学:
        - 先取 hidden 向量的均值 : μ = (1 / H) * Σ x_i
        - 再算方差 : σ² = (1 / H) * Σ (x_i - μ)²
        - 然后归一化 :   γ_i * (x_i - μ) / sqrt(σ² + ε) + β_i
        - (1️⃣ γ_i  可学习缩放系数 2️⃣  β_i 可学习偏置参数 3️⃣ ε 防止除0)
- 原理:
    - 只用均方根做归一化 , 不减均值 (分子、分母都不减均值)
    - 同时因为不减均值 , 所以不用 偏置参数
- 公式:
    - y_i = γ_i * x_i / sqrt(mean(x_i^2) + ε)
- 代码:

```python
class LlamaRMSNormMini(nn.Module):
    def __init__(self,hidden_size: int,eps: float = 1e-6):
        super().__init__()
        # 可学习缩放系数 γ_i
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # 用高精度,避免数值风险
        x = hidden_states.to(torch.float32)
        # 求平方平均,保持[B,T,1],方便后面广播相乘
        variance = x.pow(2).mean(dim=-1,keepdim=True) #[B,T,1]
        # rsqrt() = 1 / sqrt()
        x = x * torch.rsqrt(variance + self.eps) # [B,T,H]

        return (self.weight.float() * x).to(input_dtype) # [B,T,H] , 注意精度回归一下
```

#### 二、RoPE

- 传统做法:
    - 原理:
        - 把 position embedding 加到 hidden_states 上
    - 缺陷:
        - 1️⃣表达的是绝对位置 , 不直接显式体现相对关系 , 但是自然语言里很多关系由相对位置决定
        - 2️⃣泛化到超出训练长度的长上下文通常比较差
            - 假如训练时只学到最大长度 2048 , 后续推理中如果长度扩展到 4096 or 更大 , 尤其是现在长 cot、代码生成、数学推理中 , 往往上下文变长是肯定的
            - (可以用插值、随机初始化来解决 , 但是模型因为没有经过专门的训练学习 , 所以效果不好)
        - 3️⃣位置和语义相加在同一个hidden_state中 , 二者会直接耦合
- 原理:
    - 把位置信息通过旋转注入到 query_states、key_states中
- 具体原理:
    - Attention 分数计算 :
        - score(i,j) = Q_i @ K_j  , 表示 第 i 个 token 对第 j 个 token的关注程度
    - 让位置直接融入 Q、K:
        - 这样可以直接感知相对位置
        - 看一个最简单的二维旋转 :
            - q_embed = q * cos + rotate_half(q) * sin
            - k_embed = k * cos + rotate_half(k) * sin
        - 扩展到高维:
            - q.shape = [B,Nh,T,D]  k.shape = [B,Nkv,T,D]
            - 把最后一维两两分组 , 但是不同组用不同频率 , 这样就可以对短距离、长距离位置变化都进行感知
            - 频率:
                - 每一组都有一个频率 freq_i , 通过频率来决定不同的 theta
                - freq_i = 1 / base ^ (2i / head_dim)
                - 其中 base一般为10000 , i 表示第几组 , head_dim 为每个 attention_head的维度
                - 这样设置后:
                    - 1️⃣较高频率组 , pos 变化一点 , 角度就变化很多 (用来捕捉局部、短距离差异)
                    - 2️⃣较低频率组 (用来捕捉长距离、全局位置关系)
            - 某一组视角下的score计算:
                - (q1 cosθ_i - q2 sinθ_i) * (k1 cosθ_j - k2 sinθ_j)
                - + (q2 cosθ_i + q1 sinθ_i) * (k2 cosθ_j + k1 sinθ_j)
                - 变形后:
                - (q1 k1 + q2 k2) * cos(θ_i - θ_j) + (q1 k2 - q2 k1) * sin(θ_i - θ_j)
                - 自然而然 ! 相对位置就体现出来了
            - 改进后的切半版:
                - 为什么:
                    - todo
                - 前情提要:
                    - q = q_proj(hidden_states)
                    - k = k_proj(hidden_states)
                    - reshape -> [B,Nh,T,D]   [B,Nkv,T,D]
                    - RoPE 作用在 D 上
                    - 假设某个 head 的一个向量 x = [x0,x1,...,x7]
                - 做法:
                    - 前半 : [x0,x1,x2,x3]  后半: [x4,x5,x6,x7]
                    - rotate_half (x)  =  [-x4,-x5,-x6,-x7,x0,x1,x2,x3]
                    - 这样的切分 , 和 freq 的 cat 严格匹配 , 不用改很多
                    - 布局 : (x0,xD/2) , (x1,xD/2+1), ...
                    - 具体的见代码
            - 代码:

```python
def rotate_half(x:torch.Tensor) -> torch.Tensor:
    # 前面保持,只对最后一维切片
    x1 = x[...,:x.shape[-1] // 2]
    x2 = x[...,x.shape[-1] // 2 :]
    return torch.cat((-x2,x1),dim=-1)

def build_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    base: float = 10000.0,
    dtype=torch.float32,
):
    """
    position_ids: [B,T]
    return:
        cos: [B,T,D]
        sin: [B,T,D]
    """
    device = position_ids.device

    # 取从0到head_dim以step为2的tensor , 然后归一化范围 , 然后作为base的指数
    # 最后得到 [1/10000^0 , 1/10000^0.25 , 1/10000^0.5 , 1/10000^0.75]
    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2 , device = device, dtype = torch.float32)
            / head_dim
        )
    ) # [D/2]

    # 每个pos、每组的θ = 每pos ✖️ 每组的inv_freq_i
    # [B,T,D/2] = [B,T,1] ✖️ [1,1,D/2]
    # -1 表示这一维大小自动推断
    freqs = position_ids.float().unsqueeze(-1) * inv_freq.view(1,1,-1)
    #[B,T,D/2] cat [B,T,D/2] = [B,T,D]
    emb = torch.cat([freqs,freqs],dim=-1)
    # 把 θ 变成 cosθ、sinθ
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos,sin

def apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    """
    q : [B,Nh,T,D]
    k : [B,Nkv,T,D]
    cos : [B,T,D]
    sin : [B,T,D]

    return:
        q_embed same shape
        k_embed same shape
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed
```

#### 三、GQA 等等

- MHA、MQA、GQA
    - MHA:
        - num_q_heads= num_k_heads = num_v_heads
    - MQA:
        - 所有 q heads 共享一组 k/v heads
    - GQA:
        - q heads 分组 , 每个组的 q_heads 共享一组 k/v heads
- repeat_kv
    - 把 [B,Nkv,T,D] 扩展成 [B,Nq,T,D]
- 代码:

```python
def repeat_kv(hidden_states: torch.Tensor,n_rep: int) -> torch.Tensor:
    """
    hidden_states: [B,Nkv,T,D]
    n_rep: Nh // Nkv
    return:
        [B,Nh,T,D]
    """
    if n_rep == 1:
        return hidden_states

    B,Nkv,T,D = hidden_states.shape

    hidden_states = hidden_states[:,:,None,:,:] #新增一维 , 因为expand只能把大小为1的维度扩展成更大的
    hidden_states = hidden_states.expand(B,Nkv,n_rep,T,D)

    # expand 本身不copy；后续 reshape 是否复制取决于内存布局
    return hidden_states.reshape(B,Nkv*n_rep,T,D)
```

- 通用类Attention:

```python
class MiniLlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        # MHA、MQA、GQA 对应着只用改 num_key_value_heads
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        #缩放因子 和 dropout比例
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias = attention_bias,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias = attention_bias,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias = attention_bias,
        )

        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            bias = attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        """
        hidden_states: [B,T,H]
        attention_mask: None or [B,1,T,T]
        return:
            attn_output: [B,T,H]
            attn_weights: [B,Nh,T,T]
        """
        B,T,H = hidden_states.shape
        #投影得到 q/k/v
        query_states = self.q_proj(hidden_states) # [B,T,Nh*D]
        key_states = self.k_proj(hidden_states) # [B,T,Nkv*D]
        value_states = self.v_proj(hidden_states) # [B,T,Nkv*D]
        #拆 head, 并把 head 维移到前
        query_states = query_states.view(
            B,T,self.num_attention_heads,self.head_dim
        ).transpose(1,2) # [B,T,Nh,D] -> [B,Nh,T,D]
        key_states = key_states.view(
            B,T,self.num_key_value_heads,self.head_dim
        ).transpose(1,2) # [B,T,Nkv,D] -> [B,Nkv,T,D]
        value_states = value_states.view(
            B,T,self.num_key_value_heads,self.head_dim
        ).transpose(1,2) # [B,T,Nkv,D] -> [B,Nkv,T,D]

        # 在这里加 RoPE (todo)

        key_states = repeat_kv(key_states,self.num_key_value_groups) # [B,Nh,T,D]
        value_states = repeat_kv(value_states,self.num_key_value_groups) # [B,Nh,T,D]
        # [B,Nh,T,D] @ [B,Nh,D,T] = [B,Nh,T,T]
        attn_weights = torch.matmul(
            query_states,
            key_states.transpose(2,3),
        ) * self.scaling

        #加 causal/padding mask , 先加再做 softmax , 因为 mask 位置通常是 -inf or 很大负数, softmax后概率就为 0
        #在这里加 causal_mask (todo)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        #softmax 得到注意力权重
        attn_weights = F.softmax(
            attn_weights,
            dim = -1,
            dtype = torch.float32,
        ).to(query_states.dtype)
        #dropout
        #一般是以概率p把部分attn_weights元素置为0 , 其余元素一般除以 1-p 做一个缩放
        attn_weights = F.dropout(
            attn_weights,
            p = self.attention_dropout,
            training = self.training, # nn.Module自带的bool属性,True表示训练模式,False推理
        )
        # [B,Nh,T,T] @ [B,Nh,T,D] = [B,Nh,T,D]
        attn_output = torch.matmul(attn_weights,value_states)
        # 合并 heads
        attn_output = attn_output.transpose(1,2).contiguous().view(B,T,H)
        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output,attn_weights
```

#### 四、causal mask

- 原理:
    - 第 t 个 token 不能看未来 token
    - mask 矩阵就为

```text
[[0, -inf, -inf, -inf],
 [0,    0, -inf, -inf],
 [0,    0,    0, -inf],
 [0,    0,    0,    0]]
```

- 链路:
    - HF中 LlamaModel.forward 使用 create_causal_mask(...) 生成 causal mask , 然后传给每一层 decoder layer
- 代码:

```python
def make_causal_mask(
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    return:
        causal_mask: [B,1,T,T] (和head无关 , 为 1 即可)
        允许位置为0,被 mask 的位置为 dtype 最小值
    """
    # finfo(dtype) 返回某个浮点类型的数值范围信息, 取其中的min属性
    neg_inf = torch.finfo(dtype).min
    #初始化
    mask = torch.zeros((seq_len,seq_len), dtype = dtype, device = device)
    #上三角的未来token位置设置为 -inf
    # triu(..., diagonal=1) 取上三角,表示从主对角线的上一条对角线开始保留,其余位置置为0
    # torch.ones(...) 设置一个形状为T*T的bool类型的矩阵,都设置为True
    # mask.masked_fill(...) 把mask 在 future_mask为True的地方设置为neg_inf, False位置保持mask原有的 0
    future_mask = torch.triu(
        torch.ones((seq_len,seq_len), dtype = torch.bool, device = device),
        diagonal=1,
    )
    mask = mask.masked_fill(future_mask,neg_inf)

    # [1,1,T,T] -> [B,1,T,T]
    mask = mask[None,None,:,:].expand(batch_size,1,seq_len,seq_len)
    # 在 causal_mask 基础上,把 padding token 也mask掉
    # attention_mask : 1 表示真实 token，0 表示 padding token
    if attention_mask is not None:
        padding_mask = attention_mask[:,None,None,:].eq(0) # [B,T] -> [B,1,1,T] , 并转成bool矩阵
        mask = mask.masked_fill(padding_mask,neg_inf)

    return mask
```

#### 五、Self-Attention

- 原理:
    - Q = x @ Wq 、K = x @ Wk、V = x @ Wv
    - score = Q @ K^T  / sqrt (head_dim)
    - score = score + mask
    - weights = softmax( score )
    - output = weights @ V
    - output = output @ Wo
- shape变化:
    - Q : x [B,T,H] ->
- 代码:

```python
class LlamaAttentionMini(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads

        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = attention_dropout

        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * self.head_dim,
            bias = attention_bias
        )

        self.k_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias = attention_bias,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            num_key_value_heads * self.head_dim,
            bias = attention_bias,
        )

        self.o_proj = nn.Linear(
            num_attention_heads * self.head_dim,
            hidden_size,
            bias = attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor,torch.Tensor],
    ):
        """
        hidden_states: [B,T,H]
        attention_mask: [B,1,T,T]
        position_embeddings: (cos,sin) each[B,T,D]

        return:
            attn_output: [B,T,H]
        """
        B,T,H = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B,T,self.num_attention_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.num_key_value_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_key_value_heads,self.head_dim).transpose(1,2)
        # RoPE
        cos,sin = position_embeddings
        q,k = apply_rotary_pos_emb(q,k,cos,sin)
        # repeatKV
        k = repeat_kv(k,self.num_key_value_groups)
        v = repeat_kv(v,self.num_key_value_groups)
        # score -> mask -> softmax -> dropout -> weights
        attn_scores = torch.matmul(q,k.transpose(-2,-1)) * self.scaling
        attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(
            attn_scores,
            dim = -1,
            dtype = torch.float32,
        ).to(q.dtype)

        attn_weights = F.dropout(
            attn_weights,
            p = self.attention_dropout,
            training = self.training,
        )

        attn_output = torch.matmul(attn_weights,v)
        attn_output = attn_output.transpose(1,2).contiguous().view(B,T,H)
        attn_output = self.o_proj(attn_output)

        return attn_output
```

- tips:
    - RoPE只作用于 q、k , 不包括 v
    - causal_mask 在 softmax 之前
    - GQA、MQA 需要 repeatKV 让 Nkv -> Nh

#### 六、MLP ( SwiGLU )

-  MLP
    - 原理:
        - 对每个 token 自己的 hidden 向量做非线性特征变换 , 不负责交互 , 逐 token 独立处理
        - 每个 token 用同一套 MLP 参数
    - 公式:
        - MLP ( x ) = W_down ( activation ( W_up  ( x )  ))
        - 先升维 , 然后做非线性变换 , 然后降维
    - 形状变化:
        - shape
            - x [B,T,H]
            - -> W_up -> W_up x [B,T,I]
            - -> activation φ(h)
            - -> W_down -> out [B,T,H]
        - 过程
            - 对每个 token 独立做 : x[b,t, : ] -> [I] -> activation -> [H]
    -  activation:
        - 原理:
            -  hidden 向量里可能已经包含 1️⃣语法特征2️⃣实体特征3️⃣推理状态4️⃣格式特征5️⃣上下文依赖特征 , up_proj 后生成一组更加丰富的中间特征 , activation 对这些 feature 进行非线性调制 , 压小 / 放大 / 置 0 / 平滑放缩
        - 常见:
            - 普通形式 :
                - 原理:
                    - 对应改动 MLP (x) = W_down ( φ (W_up x)) 里的 φ ,  activation 作用在 W_up x 上   (h = W_up x)
                - ReLU (硬开关)
                    - ReLU ( h )  =  max ( 0 , h )
                    - 1️⃣h > 0 保留  2️⃣h <= 0 变成 0
                    - 优点 : 简单、快、稀疏、正通过负关闭
                    - 缺点 : 负区间梯度为 0 ,可能出现 deadReLU
                    - 场景 : 早期 CNN、普通MLP、非大模型场景
                - GELU (平滑筛选)
                    - 原笔记图片占位：`Pasted image 20260711163349.png`
                    - GELU (h)  =  h  *  Φ(h)
                    - 1️⃣ h 很大 -> 接近 h 2️⃣ h 很小 -> 接近 0 3️⃣ h 在 0 附近 -> 平滑过渡
                    - 优点: 梯度更连续
                    - 缺点: 稍贵
                    - 场景: BERT、GPT早期
                - SiLU (自门控激活)
                    - 自门控 , 门控系数由 h 产生
                    - SiLU (h)  =  h  *  sigmoid (h)  , 其中 sigmoid (h) =  1 / ( 1 + exp ( -h ) )
                    - 1️⃣ h 很大 -> sig ≈ 1 , 即 接近 x 2️⃣ h 接近 0  ->  sig ≈ 0.5  3️⃣ h 很负 -> sig ≈ 0
                    - 优点: 输入自己生成门控 , 平滑、可自调
                    - 缺点: 不产生强稀疏
                    - 场景: 现代LLM的MLP子层
            - 变种形式:
                - 原理:
                    - W_up x 既要生成候选特征、又要被激活筛选 , 内容生成和门控判断耦合 , 不如解耦 , 解耦有如下好处:
                    -  1️⃣职责解耦
                    -   2️⃣(W_up x ) * φ(W_gate x) 乘性交互后 , 出现各种二阶交互项 , 表达能力远强于 φ(W_gate x)
                    -   3️⃣虽然所有 token 共享一套 W_gate参数 , 但是因为输入的 x_bt 不同 , 经过W_gate得到的 gate_bt 也不同 , 所以不同的 token 通过由它自己决定的不同的门控信号
                - GLU ( 显式门控 )
                    - GLU (x) = A (x) * sigmoid(B(x))
                    - 完整 : MLP_GLU(x)  =  W_down ( (W_up x) * sigmoid(W_gate x) )
                    - 优点: 动态门控(指的是每个token的gate不同)、0~1相对稳定
                    - 缺点:  1️⃣sigmoid 易饱和 , 极大/极小时梯度变小 (导数在极大极小时均为0 , W_gate就不容易继续学习); 2️⃣0~1决定压制/通过 , 无法放大 ; 3️⃣参数量更大
                    - 场景: 稳定场景
                - SwiGLU
                    - SwiGLU (x) = A (x) * SiLU(B(x))
                    - 完整 : MLP_SwiGLU(x) = W_down ((W_up x) * SiLU(W_gate x))
                    - 优点: 平滑自门控、sigmoid软门控+W_gate x 的幅度信息
                    - 缺点: 计算更多、数值调制更自由、对训练稳定性要求高
                    - 场景: 现代大模型 decoder block、追求 FFN 表达能力的场景
                - GEGLU
                    - GEGLU (x) = A (x) * GELU(B(x))
                    - 完整 : MLP_GEGLU(x) = W_down((W_up x) * GELU(W_gate x))
                    - 优点: 更平滑、gate不局限于0~1
                    - 缺点: 计算复杂、直觉上不干净
                    - 场景: 早一批 gated FFN 变体中常见
    - 代码

```python
class LlamaMLPMini(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mlp_bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size,intermediate_size,bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size,intermediate_size,bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size,hidden_size,bias=mlp_bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B,T,H]
        return: [B,T,H]
        """
        gated = F.silu(self.gate_proj(x)) #[B,T,I]
        up = self.up_proj(x) #[B,T,I]

        x = gated * up #[B,T,I] 逐元素乘
        x = self.down_proj(x) #[B,T,H]
        return x
```

#### 七、LlamaDecoderLayer

- 链路位置
    - pre-norm residual :  先 norm , 再子层 , 再残差连接
        - residual = hidden_states
        - hidden_states = input_layernorm(hidden_states)
        - hidden_states = self_attn(hidden_states)
        - hidden_states = hidden_states + residual
    - 两种主流结构:
        - 链路位置
            - Post-Norm  「 Norm ( Sublayer(x) + x )」  残差后
                - x -> sublayer -> residual add -> norm
                - 优点: 输出尺度规整
                - 缺点: 深层模型里 , 要穿过很多 norm 和 sublayer , 稳定性一般
                - 场景: 现在不怎么用
            - Pre-Norm  「x + Sublayer( Norm (x))」 子层前
                - x -> norm -> sublayer -> residual add
                - 优点: 显然求导更简单 , 更稳定
                - 缺点: residual主干得到的产物 , 其实一直是没有norm的 , norm的一直是attn 和 mlp的输入 , 所以在最后送入 lm_head 前 , 要做一次 Final-Norm
                - 场景: 主流
            - Final-Norm   「Norm ( last_hidden_states )」 所有层之后 , lm_head之前
                - x -> sublayer -> residual add -> ... -> norm
                - 优点: 无
                - 缺点: 无
                - 场景: 其实通常结合 Pre-Norm 来用
        - 维度划分
            - BatchNorm
                - 对某个特征 / channel , 在batch维度上统计均值和方差 , 常见于 CNN
                - LLM里 , batch size、seq len、padding、生成方式变化很大 , 不方便依赖batch统计
            - LayerNorm
                - 在每一个layer中 , 对每个 token 的 hidden 向量做 norm , 沿 hidden 维度来统计均值和方差 , 不跨batch , 也不跨token
            - RMSNorm
                - LayerNorm 一种去中心化的变体
    - tips
        - 1️⃣Encoder-Decoder 架构里往往有 self-attn 和 cross-attn , 但是Llama / Qwen / GPT这类主流的Decoder-only架构中 , 每层主要是 masked-self-attn
        - 2️⃣Cross-attn (这里可以深入 todo)
            - Decoder 在生成当前某个 token 时 , 要去看 Encoder 编码好的信息
            - 即: decoder token 作为 query , 去 encoder 输出里计算 query @ key ^T
    - 代码
        - 一些点:
            - 1️⃣这些子模块一般定义在类外自己的类中 , DecoderLayer类只负责把这些组装起来

```python
class LlamaDecoderLayerMini(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        # 第一个 norm , 给 Attention 用 , 让 Attention 看到尺寸稳定的输入
        self.input_layernorm = LlamaRMSNormMini(hidden_size,eps=rms_norm_eps)

        self.self_attn = LlamaAttentionMini(
            hidden_size = hidden_size,
            num_attention_heads = num_attention_heads,
            num_key_value_heads = num_key_value_heads,
            attention_dropout = attention_dropout,
        )

        #在 Attention 残差连接之后 , 尺寸又变了 , 进入 MLP 前再 norm 一次
        self.post_attention_layernorm = LlamaRMSNormMini(
            hidden_size = hidden_size,
            eps = rms_norm_eps,
        )

        self.mlp = LlamaMLPMini(
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor,torch.Tensor],
    ) -> torch.Tensor:
        """
        做的事:
        input_layernorm
        -> self_attn
        -> post_attention_layernorm
        -> mlp
        -> output

        hidden_states : [B,T,H]
        return : [B,T,H]
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_embeddings = position_embeddings,
        )

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp(hidden_states)

        hidden_states = residual + mlp_output

        return hidden_states

```

#### 八、LlamaModel

- 流程串讲
    - 学到这了 , 还是先巩固一下认知 :D , 过一下完整的 forward 流程
        - (前提: input_ids 由 tokenizer 得到)
        - input_ids / attention_mask / position_ids / labels
        - -> Embedding -> input_ids [B,T] -> hidden_states [B,T,H]
        - -> 提前构造: causal & padding mask  +  RoPE 的 cos & sin
        - -> N 个 DecoderLayer
        - -> Final RMSNorm
        - -> lm_head [B,T,H] -> [B,T,V]
        - -> logits
        - -> with labels compute cross entropy -> scalar loss
    - tips
        - 1️⃣mask 构造 & RoPE 并不需要 hidden_states , 但为了能对齐 device 和 dtype , HF Llama中通常是在Embedding input_ids 后再构造这两者
        - 2️⃣forward 做: Embedding、DecoderLayer (attn、mlp、norm、residual add)、lm_head、compute loss、scalar loss
        - backward 做: 在CE -> logits -> lm_head -> Final Norm -> each DecoderLayer 过程中 , 计算每个可训练参数的梯度 , 并累积到 param.grad
        - 3️⃣embed_tokens.weight 和 lm_head.weight 的形状是一致的 , 并且可选权重共享 , 默认不选 , 选了后的好处 :
            - (1)  少一张矩阵, 参数量减少
            - (2)  输入token表示和输出token表示共用一套语义基底 细品!!!!!!!!!!!!!!!!!!!!!!!
            -  ( token 的输入 embedding 同时充当输出分类器中的 token 向量 , 训练中 : 前缀表示 h_t 与正确的下一个 token 对应的 E[y] 点积增大、与错误token的点积减小 , 从而让输入表示空间和输出预测空间使用同一套学习到的几何结构 )
            - (3)  同一个权重同时处理输入和输出 , 有一定正则化效果
        - 但其实两个矩阵的任务不同
            - Embedding -> 取第 i 行作为输入向量
            - lm_head -> hidden_state @ lm_heads.weight.T  -> 每个token的logit
        - 为什么默认 false
            - (1) 要兼容pre-train , 两张矩阵在预训练里是分别训练的 , 如果突然绑定 , 会丢弃其中一张
            - (2)可以独立优化
    - 代码

```python
class LlamaModelMini(nn.Module):
    def __init__(
        self,
        vocab_size : int,
        hidden_size : int,
        intermediate_size : int,
        num_attention_heads : int,
        num_key_value_heads : int,
        num_hidden_layers : int,
        rms_norm_eps : float = 1e-6,
        rope_theta : float = 10000.0,
        pad_token_id: int | None = None,
    ):
        """
        只对应backbone , 不包含 lm_head
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.rope_theta = rope_theta
        ...

        self.embed_tokens = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx = pad_token_id,
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayerMini(
                hidden_size = hidden_size,
                intermediate_size = intermediate_size,
                num_attention_heads = num_attention_heads,
                num_key_value_heads = num_key_value_heads,
                rms_norm_eps = rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ])

        # 独立子模块 , 不是Layer中的 , 负责 lm_head 前 norm
        self.norm = LlamaRMSNormMini(hidden_size,eps = rms_norm_eps)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """
        input_ids: [B,T]
        attention_mask: [B,T]
        return:
            hidden_states: [B,T,H]
        """
        B,T = input_ids.shape
        device = input_ids.device

        hidden_states = self.embed_tokens(input_ids)
        # position_ids 为 None 时,生成一个[B,T],每个[b,:] = [0,1,2,...,T-1]
        if position_ids is None:
            position_ids = torch.arange(T,device = device).unsqueeze(0).expand(B,T)

        causal_mask = make_causal_mask(
            batch_size = B,
            seq_len = T,
            attention_mask = attention_mask,
            dtype = hidden_states.dtype,
            device = device,
        )

        cos,sin = build_rope_cos_sin(
            position_ids = position_ids,
            head_dim = self.head_dim,
            base = self.rope_theta,
            dtype = hidden_states.dtype,
        )
        position_embeddings = (cos,sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states = hidden_states,
                attention_mask = causal_mask,
                position_embeddings = position_embeddings,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states
```

#### 九、LlamaForCausalLM

- 链路位置
    - 这里主要是讲清楚 3 类 mask 的作用时期
    - self.model = LlamaModel(config)
    - self.lm_head = Linear(hidden_size,vocab_size,bias=False)
    - forward:
        - outputs = self.model()  #这里经过了causal_mask和attention_padding_mask(两者都会作用于注意力分数,一个负责因果约束、一个负责padding部分)
        - hidden_states = output.last_hidden_state
        - logits = self.lm_head(hidden_states[:,slice_indices,:]) #这里是可选优化(推理只要最后一位的logits、RLlogprob重算,只需要completion位置、训练时都需要)
        - if labels is not None: #这里经过了completion/padding_mask(不参与损失计算)
            - loss = self.loss_function(...)
        - return CausalLMOutputWithPast(...)
- 我们走一个具体例子 , 来深刻认知一下三类mask

```python
"""
假设序列为 input_ids = [P0, P1, A0 , A1, PAD]
(其中 P 代表 prompt、A代表 assistant)

一、Causal mask
位置: Attention内, score -> weight期间
规则: query 位置 i 不能关注 key 位置 j>i , 本质由triu获得, 为bool矩阵
矩阵:
[[0,-∞,-∞,-∞,-∞],
 [0, 0,-∞,-∞,-∞],
 [0, 0, 0,-∞,-∞],
...]
控制: 每个 token 能看到哪些历史位置

二、Padding attention mask
位置: 同一 , 会和Causal mask相加 , 一般是 0/1形式 , 通过eq(0)转换成 bool 矩阵
...

三、Completion Loss mask
位置: lm_head 得到 logits 后, 计算 loss 时
规则: prompt、padding 不参与 loss 计算 , 一般为 -100

tips、slice before lm_head
位置: lm_head 前
规则: 不同的场景,需要计算logit的位置不同

一个小思考: 为什么要在计算出 score 后再进行 masked_fill ?
答: 我们最开始拿到[B,T], 然后Embedding + QK投影后, 得到[B,Nh,T,D]、[B,Nkv,T,D] , 这时候没有相关性, 只有 Q @ K^T 后得到[B,Nh,T,T], 才有 q-k关系 , 而 mask 屏蔽的正是 q-k 关系, 只有在[T_query , T_key] 的矩阵上逐格屏蔽
"""
```

- 代码

```python
class LlamaCausalLMMini(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        pad_token_ids
    ):
        super().__init__()

        self.model = LlamaModelMini(
            vocab_size,
            hidden_size,
            ...
        )

        self.lm_head = nn.Linear(hidden_size,vocab_size,bias=False)

    def forward(
        self,
        input_ids,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        #这里self.model , 会先调用子模块LlamaModelMini的各类hooks方法,再执行子模块的forward
        hidden_states = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.causal_lm_loss(logits,labels)

        return {
            "loss" : loss,
            "logits" : logits,
        }

    @staticmethod
    def causal_lm_loss(logits,labels,ignore_index=-100):
        B,T,V = logits.shape
        shift_labels = labels[:,1:].contiguous()
        shift_logits = logits[:,:-1,:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1,V),
            shift_labels.view(-1),
            ignore_index = ignore_index,
        )

        return loss
```

#### 十、我们这次练习, 忽略了哪些?

- 真实推理时，会使用past_key_values、logits_to_keep
- 没有配置sdpa、flash attention、flex attention
- 没有写Dynamic RoPE scaling

## 一些经典问题

这部分不只是面试题，也可以用来做自测。回答时不要只给一句定义，最好能同时讲清楚：它在完整训练链路中的位置、输入输出 Shape、对梯度或显存的影响，以及写代码时容易出现的问题。

### 前向传播与 Loss

1. **logits、prob、logprob 的区别是什么？**
    - 需要说明三者之间通过 `softmax`、`log_softmax` 建立的关系；
    - 需要说明它们在词表维度上的 Shape，以及为什么 logits 还不是概率；
    - 进一步说明为什么训练和后训练通常更喜欢直接使用 logprob。
2. **cross entropy 为什么等价于目标 token 的负 logprob？**
    - 从单个位置的 $-\log p_y$ 讲到 Batch 内有效 Token 的聚合；
    - 能解释目标 Token 和非目标 Token 对应的 Logit 梯度方向为什么不同；
    - 能说明 `ignore_index=-100` 和 Loss Mask 会在哪一步生效。
3. **CausalLM 为什么需要 Labels Shift？**
    - 说明位置 $t$ 的 Hidden State 为什么应该预测位置 $t+1$ 的 Token；
    - 写出 `shift_logits` 与 `shift_labels` 的切片方式和 Shape；
    - 能排查没有 Shift、Shift 方向写反或 Mask 没有同步移动的问题。
4. **为什么获取目标 Token 的 logprob 时需要 `gather`？**
    - 说明 `[B,T,V]` 的 Log Probability Tensor 如何根据 Token ID 变成 `[B,T]`；
    - 说明 `gather` 只是选择目标位置，并不重新计算或改变概率分布；
    - 能解释 `selective_log_softmax` 为什么可以降低峰值显存。

### 反向传播与参数更新

1. **`loss.backward()` 后到底更新了什么？**
    - 要明确 `backward()` 只负责计算并累积梯度，不会直接更新参数；
    - 能沿着 `loss -> logits -> lm_head -> decoder layers -> embedding` 说明梯度如何传播；
    - 能解释为什么某些参数的 `.grad` 仍然可能是 `None`。
2. **`requires_grad=False` 的参数会发生什么？**
    - 说明该参数本身不会积累梯度，也不会通过普通梯度更新发生变化；
    - 区分“这个参数被冻结”和“整条计算图被 `detach()` 切断”；
    - 能联系到冻结 Backbone、只训练 LoRA 或只训练部分模块的场景。
3. **为什么训练前要执行 `optimizer.zero_grad()`？**
    - 说明 PyTorch 的梯度默认执行累积，而不是覆盖；
    - 区分正常的 Gradient Accumulation 和忘记清梯度导致的意外累积；
    - 说明 `set_to_none=True` 与把梯度清零之间的区别。
4. **Adam 和 AdamW 的区别是什么？**
    - 能说明一阶矩、二阶矩和 Bias Correction 分别在做什么；
    - 重点解释 AdamW 为什么要让 Weight Decay 与梯度更新解耦；
    - 能指出 `optimizer.step()` 消费的是参数 `.grad` 和 Optimizer State。
5. **Gradient Accumulation 会怎样改变一次 `optimizer.step()`？**
    - 说明多个 Micro-Batch 的梯度如何累积到同一份 `.grad`；
    - 说明为什么通常要按 Accumulation Steps 缩放 Loss；
    - 能区分 Micro-Step、Optimizer Step 和 Learning Rate Scheduler Step。
6. **梯度裁剪裁剪的是什么，为什么 RL 后训练经常需要它？**
    - 区分按值裁剪和按全局范数裁剪，重点解释 `clip_grad_norm_`；
    - 说明裁剪发生在 `backward()` 之后、`optimizer.step()` 之前；
    - 能解释 Reward、Advantage 或 Ratio 波动为什么可能产生异常梯度。

### CausalLM 与 Transformer 结构

1. **Causal Mask、Padding Attention Mask 和 Loss Mask 有什么区别？**
    - 分别指出它们作用在 Attention Score、Padding Key 和 Loss 聚合的哪个阶段；
    - 说明“某个 Token 不参与 Loss”不等于“它不参与前向计算”；
    - 能根据 Prompt、Completion、Padding 的具体序列手画三种 Mask。
2. **RoPE 为什么只作用于 Query 和 Key，而不直接作用于 Value？**
    - 从 $QK^T$ 决定注意力权重这一点解释位置关系应该注入哪里；
    - 能说明旋转后内积如何显式包含相对位置差；
    - 能结合 `[B,Nh,T,D]` 说明 RoPE 实际作用的维度。
3. **MHA、MQA 和 GQA 的区别是什么？**
    - 比较 Query Head 与 Key/Value Head 的数量关系；
    - 说明 `repeat_kv` 为什么只扩展 K/V Head；
    - 能分析 GQA 在表达能力、KV Cache 和推理效率之间的折中。
4. **RMSNorm、Pre-Norm、Residual Connection 分别解决什么问题？**
    - 说明 RMSNorm 沿 Hidden Size 维度做什么；
    - 能手画一层 Decoder 中 Norm、Attention、MLP 和残差连接的顺序；
    - 说明为什么 Pre-Norm 架构通常还需要 Final Norm。

### 精度、显存与工程排查

1. **bf16 和 fp16 的训练稳定性差异是什么？**
    - 比较两者的指数位、尾数位和可表示范围；
    - 解释为什么 FP16 更容易 Overflow/Underflow，以及 Loss Scaling 的作用；
    - 同时说明 BF16 动态范围更大不代表所有计算都不会出现数值问题。
2. **为什么 Loss 正常下降，生成效果却可能没有变好？**
    - 检查 Loss 是否只统计了正确的 Completion Token；
    - 检查数据、Mask、截断、过拟合和训练目标是否与实际评测目标一致；
    - 区分 Token-Level Loss 下降与完整 Sequence 质量提升。
3. **训练出现 NaN、Gradient Norm 突增或 OOM 时，应该怎样排查？**
    - 从数据 Batch、有效 Token 数、Dtype、Learning Rate、Logits 和 Loss 开始逐层定位；
    - 检查是否错误保留计算图、物化完整 `[B,T,V]` Tensor 或遗漏 `detach()`；
    - 能使用日志中的 Loss、Grad Norm、Tokens、显存和吞吐变化缩小问题范围。

## 吃透标准

“吃透”不是把这一章看完，也不是能够复述几个名词，而是能够在没有现成答案的情况下，把概念、公式、Shape、代码和排查思路连起来。可以按下面几个层次检查。

### 1、主链路能够独立讲清楚

- [ ] 能脱离笔记手画 `input_ids -> embeddings -> hidden_states -> logits -> logprobs -> loss -> grad -> step`；
- [ ] 能给每个关键 Tensor 标出 Shape，并解释每次切片、转置、`gather` 和聚合改变了什么；
- [ ] 能明确区分 Forward、`backward()` 和 `optimizer.step()` 各自负责的事情；
- [ ] 能把 SFT、DPO、PPO、GRPO 放回这条主链路，说明它们主要改变了数据、Loss 或采样方式，而不是改变 Autograd 的基本机制。

### 2、核心概念能够解释到原理

- [ ] 能解释 logits、probability、logprob 和 cross entropy 之间的数学关系；
- [ ] 能解释 loss 对 logits 求导后，目标 Token 和非目标 Token 的更新方向为什么不同；
- [ ] 能区分 `requires_grad`、`grad_fn`、`is_leaf`、`.grad`、`detach()` 和 `torch.no_grad()`；
- [ ] 能解释 AdamW 的一阶矩、二阶矩、Bias Correction 和 Decoupled Weight Decay；
- [ ] 能区分 Causal Mask、Padding Attention Mask、Completion Mask 与 `labels=-100` 的作用阶段。

### 3、关键代码能够独立手写

- [ ] 能手写 `masked_mean`，并处理 Mask 全 0、Broadcast 和不同 Reduction Dimension；
- [ ] 能手写 `selective_log_softmax`，正确处理 Token ID、`ignore_index` 和输出 Shape；
- [ ] 能手写 Sequence/Response Log Probability，正确完成 Shift、Mask 和 Token Logprob 求和；
- [ ] 能手写最小 `train_step`，保证 `zero_grad -> forward -> loss -> backward -> clip -> step` 顺序正确；
- [ ] 能不照抄源码写出 RMSNorm、RoPE、GQA Attention、SwiGLU MLP 和 Decoder Layer 的主结构。

### 4、源码能够找到关键路径

- [ ] 在 PyTorch AdamW 源码中，能够找到参数组、`.grad`、Optimizer State 和实际更新函数之间的关系；
- [ ] 在 `LlamaForCausalLM.forward()` 中，能够找到 Backbone、`lm_head`、`logits_to_keep` 和 Loss Function；
- [ ] 能找到 CausalLM Labels Shift 的真正实现位置，而不是只停留在模型 Forward 的表面调用；
- [ ] 在 TRL 代码中看到 Current/Old/Reference Logprob 时，能够判断它们的来源、Shape 和是否需要梯度。

### 5、出现问题时能够定位

- [ ] Shape 报错时，能从 `[B,T,H]`、`[B,T,V]`、`[B,T-1]` 逐步检查，而不是只尝试随机 `reshape`；
- [ ] Loss 异常时，会检查 Labels Shift、`ignore_index`、Completion Mask、有效 Token 数和 Reduction；
- [ ] Gradient Norm 异常时，会检查 Learning Rate、Dtype、Loss Scale、异常 Batch 和梯度累积；
- [ ] OOM 时，能够区分参数、Optimizer State、Activation、Gradient 和临时 Log Probability Tensor 的显存来源。

### 6、能够把知识讲给别人

- [ ] 能在 5～10 分钟内，不看代码讲清楚“一次 CausalLM 训练更新发生了什么”；
- [ ] 面对文末经典问题，不只给结论，还能补充 Shape、公式或一段最小代码；
- [ ] 对暂时不确定的源码细节，知道应该去哪个文件、哪个函数继续确认，而不是依赖模糊记忆；
- [ ] 能把 `p.grad`、AdamW 参数更新与大模型后训练中的 Policy Update 联系起来。

如果上面的内容还不能一次全部完成，也不需要急着否定自己的学习效果。优先保证主链路、Labels Shift、Mask、Gradient 和 AdamW 五部分能够讲清楚，再逐步补齐源码与工程细节。当你能够发现自己的错误、定位它发生在哪个阶段，并知道下一步去哪里验证时，就已经不只是“看懂”，而是在真正建立训练能力。
