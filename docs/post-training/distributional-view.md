# 在分布视角下理解语言模型后训练

> 学习来源：[SFT, RL, and On-Policy Distillation Through a Distributional Lens](https://nrehiew.github.io/blog/sft_rl_opd/)

本文尝试使用“自回归概率树”统一理解 SFT、RL 与 On-Policy Distillation（OPD）：它们分别在什么节点上训练、使用什么信号更新节点的出边概率，以及这些局部更新最终如何重塑完整回答的序列分布。

## 一、基础认知：语言模型本质上是一个序列分布

### 1. 从 Next-Token Distribution 到 Sequence Distribution

给定 Prompt $x$ 和已经生成的前缀 $y_{<t}$，语言模型会进行如下计算：

1. Tokenizer 将文本映射为 Token IDs；
2. Decoder Layers 根据 Prompt 和当前前缀计算隐藏状态；
3. LM Head 将最后一层隐藏状态映射为全词表 Logits；
4. Softmax 将 Logits 转换为下一个 Token 的条件概率分布；
5. Greedy、Temperature、Top-p 等解码策略再从该分布中选择或采样一个 Token。

模型在第 $t$ 步输出的不是一个确定 Token，而是全词表上的条件概率分布：

$$
\pi_\theta(\cdot\mid x,y_{<t}).
$$

其中，$\theta$ 表示模型全部可训练参数，$\mathcal V$ 表示词表。对任意 Token $a\in\mathcal V$：

$$
\pi_\theta(a\mid x,y_{<t})
$$

表示在当前 Prompt 和前缀下，下一个 Token 为 $a$ 的概率。

模型不断重复这一过程：生成一个 Token，将它拼接到前缀中，再计算新的 Next-Token Distribution。于是，完整回答

$$
y=(y_1,y_2,\ldots,y_T)
$$

的概率由概率链式法则给出：

$$
\pi_\theta(y\mid x) = \prod_{t=1}^{T} \pi_\theta(y_t\mid x,y_{<t}).
$$

对应的 Log Probability 为：

$$
\log\pi_\theta(y\mid x) = \sum_{t=1}^{T} \log\pi_\theta(y_t\mid x,y_{<t}).
$$

这就是所谓的“对条件分布进行自回归分解”：

> 将一个完整序列的联合概率，分解为一系列“给定已有前缀时，下一个 Token 的条件概率”的连乘。

因此，语言模型并没有显式保存一张“所有完整回答及其概率”的表。它通过同一组参数 $\theta$，在任意前缀上动态计算下一 Token 分布，从而隐式定义了所有可能完整序列上的概率分布。

换句话说：

> Tokenizer、Decoder Layers 和 LM Head 共同参数化了一个条件序列分布，而“不断预测下一个 Token”只是从该序列分布中生成样本的具体过程。

### 2. 什么是概率质量

固定 Prompt $x$ 后，所有可能回答的概率之和为 $1$：

$$
\sum_y\pi_\theta(y\mid x)=1.
$$

这个总量 $1$ 可以理解为模型拥有的全部概率质量。模型参数决定这份概率质量如何分配给不同回答。

例如，模型可能把较多概率质量分配给：

- 正确回答；
- 符合指令的回答；
- 常见表达方式；
- 模型已经熟悉的推理路径。

也可能把一部分概率质量分配给：

- 错误答案；
- 格式不合法的回答；
- 冗长或偏题的回答；
- 看似合理但无法通过验证的回答。

单条完整序列的概率通常非常小，因此只观察某一条回答往往不够。更有意义的是观察某一类回答占据的总概率质量。

设正确回答集合为：

$$
\mathcal R_{\mathrm{correct}} = \left\{ y:y\text{ 能正确完成任务} \right\}.
$$

模型为正确回答区域分配的概率质量为：

$$
P_\theta(\mathcal R_{\mathrm{correct}}\mid x) = \sum_{y\in\mathcal R_{\mathrm{correct}}} \pi_\theta(y\mid x).
$$

因此，“参数更新改变了不同序列的概率质量分配”指的是：

> 更新模型参数以后，一些回答或回答集合的总概率上升，另一些回答或回答集合的总概率下降；总概率仍然为 $1$，变化的是这份概率质量在回答空间中的分布方式。

> **本节要点：** 语言模型的直接输出是每个前缀下的条件 Token 分布；这些条件分布经过自回归分解，共同定义完整回答上的序列分布。后训练真正改变的是这个序列分布。

---

## 二、我的理解视角：自回归概率树

“序列分布”在数学上准确，但不够直观。固定 Prompt 后，可以把它展开成一棵自回归概率树，用来观察不同训练方法到底改了什么。

### 1. 节点、出边与完整路径

固定 Prompt $x$ 后：

- 根节点表示 Prompt $x$；
- 每个中间节点表示 Prompt 与当前生成前缀；
- 每条出边表示一个可能的下一个 Token；
- 出边权重表示该 Token 的条件概率；
- 从根节点走到 EOS 的一条路径表示一个完整回答。

第 $t$ 步的状态或节点可以写成：

$$
s_t=(x,y_{<t}).
$$

从节点 $s_t$ 选择 Token $a$ 的出边权重为：

$$
w_\theta(s_t,a) = \pi_\theta(a\mid s_t) = \pi_\theta(a\mid x,y_{<t}).
$$

同一节点的所有出边构成一个概率分布：

$$
\sum_{a\in\mathcal V} w_\theta(s_t,a) =1.
$$

一条完整路径 $y=(y_1,\ldots,y_T)$ 的权重，就是沿途出边权重的连乘：

$$
\pi_\theta(y\mid x) = \prod_{t=1}^{T} w_\theta(s_t,y_t).
$$

所以，可以用三个简单对象表示整棵树：

$$
\boxed{ \text{节点 }s_t=(x,y_{<t}), \qquad \text{出边 }w_\theta(s_t,a)=\pi_\theta(a\mid s_t), \qquad \text{路径 }\pi_\theta(y\mid x)=\prod_t w_\theta(s_t,y_t) }
$$

需要注意：这棵树并没有被真实地完整存储在显存中。节点数量随序列长度呈指数增长，模型只是用共享参数在“当前被访问的节点”上临时计算出边概率。

### 2. 子树与回答区域

#### 2.1 什么是子树

给定一个生成前缀 $u$，所有以 $u$ 开头的回答构成一棵前缀子树：

$$
\mathcal T_u = \{y:u\preceq y\},
$$

其中，$u\preceq y$ 表示 $u$ 是完整回答 $y$ 的前缀。

模型进入该子树的概率，就是生成此前缀的概率：

$$
P_\theta(\mathcal T_u\mid x) = \pi_\theta(u\mid x).
$$

例如，某个 Prompt 下以 “We need to prove” 开头的全部回答，可以看作一棵共享该前缀的子树。

#### 2.2 什么是回答区域

“回答区域”通常不是由共同前缀定义，而是由某种性质定义。例如：

$$
\mathcal R_{\mathrm{correct}} = \{y:y\text{ 能正确完成任务}\}.
$$

还可以定义：

- 最小代码修改区域；
- 风格简洁的回答区域；
- 能通过单元测试的代码区域；
- 包含错误推理但答案碰巧正确的区域。

某一类回答可能使用完全不同的开头、推理路径和表达方式，因此它不一定对应一棵连续子树，而可能由很多子树和离散路径共同组成。

该区域的概率质量为：

$$
P_\theta(\mathcal R\mid x) = \sum_{y\in\mathcal R} \pi_\theta(y\mid x).
$$

所以：

- **子树**由共享前缀定义；
- **回答区域**由正确性、风格、长度、代码修改量等性质定义；
- **策略 $\pi_\theta$** 为每条路径、每棵子树和每个回答区域分配概率质量。

从直观上说，“原模型在所有可能回答构成的完整空间中，为某一类回答分配了一块区域”基本是对的；但更严谨地说，这里的“区域”是回答集合，不一定是几何连续区域，也不一定是单棵子树。

### 3. 后训练如何改变概率树

从概率树角度，后训练可以理解为：

> 在特定来源的前缀节点上，根据特定监督信号更新共享参数 $\theta$，从而改变节点的出边概率，并进一步重塑完整回答的序列分布。

参数更新写成：

$$
\theta' = \theta-\eta\nabla_\theta L.
$$

更新以后，任意节点的出边分布都可能改变：

$$
\pi_{\theta'}(\cdot\mid s) \neq \pi_\theta(\cdot\mid s).
$$

但是，“后训练在调整整棵概率树”不等于“优化器逐个找到所有节点并直接修改它们的权重”。实际过程是：

1. 数据或 Rollout 只访问有限数量的前缀节点；
2. 损失只在这些节点上被构造；
3. 反向传播更新共享参数；
4. 共享参数改变后，所有节点的 Logits 计算函数一起改变；
5. 因此，未被访问的节点也可能发生间接变化。

所以必须区分：

$$
\boxed{ \text{直接梯度来自哪些节点} \quad\neq\quad \text{参数更新最终会影响哪些节点} }
$$

SFT、RL 和 OPD 的核心差异可以先压缩成下表：

| 方法 | 训练节点从哪里来 | 更新信号从哪里来 | 概率树上的直接作用 |
|---|---|---|---|
| SFT | 固定外部数据路径 | 示范 Token | 沿示范路径提高目标出边概率 |
| RL | 当前策略采样路径 | Reward / Advantage | 强化高优势路径，抑制低优势路径 |
| OPD | 当前 Student 采样路径 | Teacher Token Distribution | 在 Student 访问的节点上匹配 Teacher 出边分布 |

这构成本文后续分析的统一视角：

$$
\boxed{ \text{训练节点的来源} + \text{更新信号的来源} }
$$

> **本节要点：** 后训练确实是在重新分配整棵自回归概率树的概率质量，但直接监督只落在有限节点上；整棵树的联动变化来自 Transformer 参数共享，而不是对所有节点逐一训练。

---

## 三、分布视角下看待后训练

### 1. SFT：固定的外部目标分布

#### 1.1 概率树视角下的本质

一条 SFT 数据：

$$
(x,y^*)
$$

在自回归概率树上指定了一条固定的外部示范路径：

$$
y^*=(y_1^*,y_2^*,\ldots,y_T^*).
$$

SFT 在这条路径经过的每一个前缀节点：

$$
s_t^*=(x,y_{<t}^*)
$$

提高示范 Token 对应出边的概率：

$$
\pi_\theta(y_t^*\mid x,y_{<t}^*)\uparrow.
$$

其负对数似然损失为：

$$
L_{\mathrm{SFT}} = -\sum_{t=1}^{T} \log\pi_\theta(y_t^*\mid x,y_{<t}^*).
$$

因此，从概率树角度看：

> SFT 使用外部数据预先选定一条路径，然后逐节点提高示范出边的概率，将模型的序列分布拉向数据集分布。

#### 1.2 数据集分布与模型分布

对于同一个 Prompt，如果数据集中只给出一个示范答案 $y^*$，那么经验条件分布可以写成：

$$
\hat p_D(y\mid x) = \mathbf 1[y=y^*].
$$

也就是：

$$
\hat p_D(y^*\mid x)=1.
$$

模型则通过当前参数计算出一个完整的条件序列分布：

$$
\pi_\theta(y\mid x).
$$

SFT 最小化数据分布与模型分布之间的交叉熵：

$$
H(\hat p_D,\pi_\theta) = -\mathbb E_{y\sim\hat p_D} \left[ \log\pi_\theta(y\mid x) \right].
$$

交叉熵可以分解为：

$$
H(\hat p_D,\pi_\theta) = H(\hat p_D) + D_{\mathrm{KL}} \left( \hat p_D\|\pi_\theta \right).
$$

因为 $H(\hat p_D)$ 与模型参数 $\theta$ 无关，所以最小化交叉熵等价于最小化 Forward KL：

$$
D_{\mathrm{KL}} \left( \hat p_D\|\pi_\theta \right).
$$

所以，“SFT 是将模型分布拉近数据集分布的过程”这个理解是对的。但要补充两点：

1. 数据集分布是有限样本形成的经验分布，并不是真实任务分布本身；
2. 对某个 Prompt 只有一个答案时，数据集的确把经验概率 $1$ 放在这个答案上，但整个训练集上的分布还包含不同 Prompt 和不同示范答案的采样频率。

#### 1.3 “初始分布不重要”的准确含义

初始模型当然会影响：

- 初始损失；
- 梯度大小；
- 收敛速度；
- 优化路径；
- 最终找到的参数解。

因此，“初始分布不重要”不能按字面理解为初始模型完全没有影响。

更准确的含义是：

> 标准 SFT 目标中没有显式项要求训练后的策略继续接近初始策略。

纯 SFT 优化：

$$
L_{\mathrm{SFT}} = -\mathbb E_{(x,y)\sim D} \left[ \log\pi_\theta(y\mid x) \right].
$$

目标中通常没有：

$$
D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid x) \|\pi_{\theta_0}(\cdot\mid x) \right),
$$

也没有其他显式项表示“不要偏离初始模型太远”。

因此，初始策略 $\pi_{\theta_0}$ 决定模型从哪里出发，但标准 SFT 的目标函数本身不惩罚模型离开这个出发点。

#### 1.4 为什么不会优先选择“附近的解”

假设对于同一个 Prompt，回答 A 和回答 B 都能正确完成任务：

- A 是数据集展示的标准答案；
- B 是模型原本就会生成的另一种正确答案；
- B 在分布上更接近初始模型；
- 数据集中只出现了 A。

那么 SFT 直接优化的是：

$$
-\log\pi_\theta(A\mid x).
$$

它并不知道：

- B 与 A 在任务结果上等价；
- B 更接近模型原本的生成习惯；
- 保留 B 可能更有利于已有能力。

只要 B 的 Token 与 A 不同，SFT 就不会因为 B “也正确”而保护它。模型只会收到“提高 A 这条示范路径概率”的信号。

所以，“模型没有内在理由优先选择附近的解，而只是被拉向已展示的标记”指的是：

> SFT 的目标定义在示范 Token 是否匹配上，而不是定义在任务是否完成、答案是否等价或新策略是否接近初始策略上。

#### 1.5 密集 Token 监督与数据集伪影

SFT 会对示范回答中的每个目标 Token 计算交叉熵。因此，每个目标 Token 都会参与损失：

$$
L_{\mathrm{SFT}} = \sum_t L_t.
$$

这里的“密集”不是说每个 Token 的梯度数值完全相等，而是：

> 每个示范 Token 都是监督目标；损失函数本身没有机制判断某个 Token 是否真正决定任务成功。

任务关键 Token 可能是：

- 正确的运算符；
- 关键推理步骤；
- 最终答案；
- 正确的工具参数；
- 修复 Bug 所必需的代码改动。

偶然的数据集伪影可能是：

- 固定的注释风格；
- “我们不难看出”等惯用表达；
- 单引号或双引号偏好；
- 某个标注者的冗长口癖；
- 数据生成模型固定使用的思考过渡词；
- 与任务无关的变量重命名或格式重排。

交叉熵对二者采用同一种判断标准：

$$
\text{目标 Token 是否与示范一致}.
$$

它并不知道哪个 Token 对最终任务成功具有因果作用。因此，模型既会学习真正关键的模式，也会学习数据集中高频但非关键的偶然模式。

这就是“交叉熵没有内置机制区分任务关键标记与偶然数据集伪影”的含义。

#### 1.6 为什么可能造成灾难性遗忘

假设基模原本具有“最小代码修改”的能力，只修改真正出错的部分：

~~~diff
- if score > 60:
+ if score >= 60:
~~~

但用于 SFT 的代码数据经常：

- 重写整个函数；
- 修改变量名；
- 加入固定风格的注释；
- 改变无关代码格式；
- 使用与基模不同的实现习惯。

SFT 会在这些外部示范路径上，对所有目标 Token 施加拟合压力。由于 Transformer 的参数在不同 Prompt、任务和前缀节点之间共享，拟合新数据的更新可能同时改变原有行为对应节点的出边分布。

更完整的因果链为：

$$
\text{密集拟合示范 Token} \longrightarrow \text{共享参数发生改变} \longrightarrow \text{其他前缀节点的 Logits 被间接改变} \longrightarrow \text{原有回答区域的概率质量下降} \longrightarrow \text{已有能力退化}.
$$

这不意味着 SFT 一定造成灾难性遗忘，也不意味着交叉熵本身“错误”。问题在于：当外部数据分布离初始模型较远，且目标中缺少能力保持约束时，SFT 会直接朝示范分布移动，而没有内置机制保护与新任务无关的原有行为。

> **SFT 小结：** 数据集指定“走哪条路”，交叉熵逐 Token 提高这条路径的出边概率。它的优点是直接、稳定、适合冷启动；风险是把任务关键模式和数据集伪影一起拟合，并可能把整棵概率树拉离初始分布。

---

### 2. RL：沿预期奖励增大的方向移动

#### 2.1 概率树视角下的本质

RL 不使用固定外部答案预先指定唯一训练路径，而是让当前策略从自己的概率树中采样：

$$
y\sim\pi_\theta(\cdot\mid x).
$$

奖励函数再对采样到的完整轨迹打分：

$$
r=R(x,y).
$$

随后，训练提高高优势轨迹的概率，降低低优势轨迹的概率。

因此，从概率树角度看：

> RL 先按照当前模型的出边概率实际走出一条路径，再根据这条路径的任务结果，强化或抑制沿途出边。

#### 2.2 为什么 RL 没有固定的目标分布

SFT 直接展示一个答案 $y^*$，而 RL 的奖励函数通常只规定“哪些回答更好”。

可以定义高奖励回答区域：

$$
\mathcal R_{\mathrm{good}} = \{y:R(x,y)\text{ 较高}\}.
$$

该区域中的回答通常不唯一。不同表达方式、推理路径和代码实现都可能获得相同奖励。

RL 的目标是最大化当前策略的期望奖励：

$$
J(\theta) = \mathbb E_{y\sim\pi_\theta(\cdot\mid x)} \left[ R(x,y) \right].
$$

如果奖励是二元验证结果：

$$
R(x,y) = \mathbf 1[y\in\mathcal R_{\mathrm{correct}}],
$$

那么：

$$
J(\theta) = P_\theta(\mathcal R_{\mathrm{correct}}\mid x).
$$

此时，最大化期望奖励就等价于提高整个正确回答区域的总概率质量，而不是强制模型拟合某一条唯一答案。

所以，Blog 所说 RL 的“目标分布很难定义”，不是说 RL 没有优化目标，而是说：

> 奖励函数定义了偏好方向或高奖励集合，却通常没有预先给出一个完整、固定且唯一的目标序列分布。

#### 2.3 RL 从采样到训练的完整过程

一次简化的 RL 训练循环可以分成八步：

1. 从训练集采样 Prompt $x$；
2. 当前 Rollout Policy 生成一条或多条回答 $y$；
3. 保存 Token IDs、旧策略 Log Probability、Attention Mask 和工具轨迹；
4. Reward Function 或 Reward Model 为回答打分；
5. 将 Reward 转换成 Return 或 Advantage；
6. 使用采样序列重新 Forward，构建带梯度的计算图；
7. 计算 REINFORCE、PPO、GRPO 等策略损失并反向传播；
8. Optimizer 更新参数，新的策略继续下一轮采样。

简化的策略梯度可以写成：

$$
\nabla_\theta J = \mathbb E_{y\sim\pi_\theta} \left[ A(x,y) \nabla_\theta\log\pi_\theta(y\mid x) \right].
$$

由于一条完整回答的 Log Probability 可以分解为：

$$
\log\pi_\theta(y\mid x) = \sum_{t=1}^{T} \log\pi_\theta(y_t\mid x,y_{<t}),
$$

所以：

$$
\nabla_\theta J = \mathbb E_{y\sim\pi_\theta} \left[ A(x,y) \sum_{t=1}^{T} \nabla_\theta \log\pi_\theta(y_t\mid x,y_{<t}) \right].
$$

直观上：

- $A>0$：提高这条采样轨迹以及沿途动作的概率；
- $A<0$：降低这条采样轨迹以及沿途动作的概率；
- $A\approx 0$：更新较弱。

这里需要区分“采样时的 Forward”和“训练时的 Forward”：

- 采样时可以不保留完整计算图，只需要生成回答并保存必要信息；
- 训练时通常会用采样到的 Token 再次 Forward，计算当前 Log Probability 和损失，然后 Backward。

#### 2.4 什么是策略的状态访问分布

Blog 中的“区域”可以进一步用状态访问分布表示：

$$
d_x^{\pi_\theta}(s,a) = \sum_{t\ge 1} P_{\tau\sim\pi_\theta} (s_t=s,a_t=a\mid x).
$$

它表示：固定 Prompt $x$ 后，按照当前策略生成轨迹时，状态—动作对 $(s,a)$ 在轨迹中被访问的期望次数或访问质量。

这里不是在累加一条已采样序列的 Log Probability，而是在对当前策略可能产生的所有轨迹取期望：

- 某个前缀越容易被当前策略生成，对应状态的访问概率越大；
- 某个 Token 在该前缀下概率越高，对应状态—动作对的访问概率越大；
- 低概率前缀和低概率动作的访问质量更小。

因此，策略 $\pi_\theta$ 与“区域”的关系是：

> 策略在每个节点定义出边概率；这些局部概率共同决定哪些子树和回答区域更容易被访问，也决定 RL 的训练样本主要来自哪里。

#### 2.5 为什么说 RL 把直接梯度集中在高概率区域

RL 的训练轨迹来自：

$$
y\sim\pi_\theta(\cdot\mid x).
$$

当前概率质量较大的路径更容易被采样，因此也更频繁地进入训练 Batch 并提供直接策略梯度。

需要把两个量分开：

- **采样概率**决定某条路径多频繁地出现；
- **Advantage**决定该路径出现后，更新方向和强度是什么。

所以：

- 高概率、高奖励轨迹会被频繁采样并强化；
- 高概率、低奖励轨迹会被频繁采样并压低；
- 极低概率轨迹即使潜在奖励很高，也可能因为几乎采不到而没有直接训练信号。

这就是“RL 的直接梯度集中在当前策略的高概率区域”的具体含义。

它并不是说每个 Layer 只有局部参数参与训练。对一条采样轨迹进行反向传播时，通常所有未冻结的 Decoder Layers 和 LM Head 参数都可以收到梯度。所谓“局部”指的是：

> 构造损失所使用的状态和动作，来自当前采样轨迹；不是说只有模型中的局部 Layer 或局部参数被计算。

#### 2.6 为什么这种局部性不绝对

局部的是直接梯度的来源，不是参数更新后的影响范围。

第一，在一个已访问节点上，Softmax 中所有出边互相竞争。设 Logits 为 $z$，概率为：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

即使损失主要提高采样 Token 的 Logit，该 Token 概率上升时，其他 Token 的相对概率也会下降。因此，从同一节点出发的未采样子树，其入口概率也可能发生变化。

第二，所有节点共享同一组 Transformer 参数。某个参数更新为：

$$
\theta'= \theta-\eta\nabla_\theta L_{\mathrm{sampled}}.
$$

对于另一个未访问节点 $s'$：

$$
z_{\theta'}(s') \neq z_\theta(s')
$$

完全可能成立，因为 $s'$ 的 Logits 也由同一组参数计算。

所以：

$$
\text{未采样区域没有直接梯度} \not\Rightarrow \text{未采样区域完全不变化}.
$$

更准确的说法是：

- 采样轨迹决定直接优化压力落在哪里；
- Softmax 耦合让同一节点的其他出边跟随变化；
- 参数共享让其他节点和其他 Prompt 也可能被间接影响；
- 但未采样区域没有来自其自身行为结果的直接监督。

#### 2.7 RLVR 与 RLHF

**RLVR（Reinforcement Learning with Verifiable Rewards）** 的奖励可以由规则、程序或环境直接验证，例如：

- 数学最终答案是否正确；
- 代码是否通过单元测试；
- 工具调用参数是否合法；
- 环境任务是否达到目标状态。

它的特点是：Reward 通常较客观、低噪声，但只适用于可验证任务。

**RLHF（Reinforcement Learning from Human Feedback）** 的奖励来自人类偏好，或来自在人类偏好数据上训练的 Reward Model，例如：

- 回答是否有帮助；
- 写作风格是否自然；
- 内容是否安全；
- 两个开放式回答中哪个更好。

它能处理主观目标，但 Reward Model 可能存在偏差、漏洞和过度优化问题。

二者并不互斥。实际训练可以组合多个奖励：

$$
R_{\mathrm{total}} = R_{\mathrm{verifiable}} + \alpha R_{\mathrm{preference}} - \beta R_{\mathrm{constraint}}.
$$

> **RL 小结：** 当前策略先决定“会走到哪里”，Reward / Advantage 再决定“采样路径应该被强化还是抑制”。RL 的直接训练区域因此天然受当前模型分布限制，但共享参数仍会使影响扩散到未采样区域。

---

### 3. OPD：在 Student 的状态上学习 Teacher

#### 3.1 概率树视角下的本质

设 Student 为 $\pi_\theta$，Teacher 为 $q_\phi$。

OPD 首先让 Student 从自己的概率树中采样：

$$
y\sim\pi_\theta(\cdot\mid x).
$$

对于 Student 实际访问的每一个前缀节点：

$$
s_t=(x,y_{<t}),
$$

Teacher 都给出一个下一 Token 分布：

$$
q_\phi(\cdot\mid s_t).
$$

然后，Student 在这些节点上调整自身的出边分布，使其接近 Teacher：

$$
L_{\mathrm{OPD}} = \mathbb E_{s\sim d^{\pi_\theta}} \left[ D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid s) \|q_\phi(\cdot\mid s) \right) \right].
$$

所以，OPD 可以压缩为一句话：

> Student 决定在哪里学习，Teacher 决定在这些位置上应该如何重新分配出边概率。

这与 SFT 的差异非常关键：

- SFT 在数据集或 Teacher 已经走过的前缀上训练；
- OPD 在 Student 自己真实会访问的前缀上训练。

因此，“OPD 本质是 Student 已经探索到的子树，再通过 Teacher 更新子树权重”这个理解大体正确。更严谨地说，OPD 不一定一次覆盖完整子树，而是从 Student 分布中采样有限路径，并在这些路径经过的节点上匹配 Teacher 分布；多轮采样后，训练节点才逐渐覆盖 Student 的高概率子树与回答区域。

#### 3.2 为什么叫 Pseudo RL

单个状态上的 Reverse KL 可以展开为：

$$
D_{\mathrm{KL}}(\pi_\theta\|q_\phi) = \mathbb E_{a\sim\pi_\theta} \left[ \log\pi_\theta(a\mid s) - \log q_\phi(a\mid s) \right].
$$

将符号反过来，可以定义一个类似 Advantage 的信号：

$$
A_{\mathrm{OPD}}(s,a) = \log q_\phi(a\mid s) - \log\pi_\theta(a\mid s).
$$

其含义是：

- Teacher 比 Student 更认可某个 Token，$A_{\mathrm{OPD}}>0$，提高其概率；
- Student 比 Teacher 更偏好某个 Token，$A_{\mathrm{OPD}}<0$，降低其概率；
- 二者接近，更新较小。

这种形式与 Policy Gradient 很像：都在 Student 自己采样的状态或动作上，根据一个标量信号调整概率。

但它又不是真正的环境 RL，因为这个“Advantage”并非来自任务执行结果，而是来自 Teacher 和 Student 的 Log Probability 差。因此 Blog 将其称为 **Pseudo RL**。

#### 3.3 On-Policy 到底体现在哪里

OPD 中的 On-Policy 首先指状态分布来自当前 Student：

$$
s\sim d^{\pi_\theta}.
$$

这意味着训练数据会随 Student 更新而变化：

$$
\pi_{\theta_0} \rightarrow d^{\pi_{\theta_0}}, \qquad \pi_{\theta_1} \rightarrow d^{\pi_{\theta_1}}, \qquad \ldots
$$

Student 学会新行为后，会进入新的前缀状态；Teacher 随后又在这些新状态上提供指导。训练数据不是一个始终固定的外部集合，而是由 Student 当前行为持续产生。

还要区分两种实现：

1. 若在已访问状态上计算完整词表 KL，那么该节点的全部出边都会参与损失；
2. 若只用采样 Token 构造 Monte Carlo 估计，那么直接信号集中在采样动作上。

因此，“On-Policy”不等于“永远只更新实际采样的一个 Token”；它首先描述训练状态来自谁。

#### 3.4 OPD 与 SFT、RL 的统一对比

从概率树视角看：

| 方法 | 谁选择训练路径或状态 | 谁提供方向 | 监督粒度 |
|---|---|---|---|
| SFT | 外部数据 | 示范 Token | 密集 Token 级 |
| RL | Student / 当前策略 | Reward / Advantage | 通常是轨迹级或经过估计的 Token 级 |
| OPD | Student / 当前策略 | Teacher Distribution | 密集 Token Distribution 级 |

因此：

- SFT：外部路径 + 外部 Token 标签；
- RL：Student 路径 + 环境结果信号；
- OPD：Student 路径 + Teacher 分布信号。

OPD 正好处在 SFT 和 RL 之间：

- 它像 RL 一样使用 On-Policy Data；
- 它像 Distillation 一样获得密集的 Token 级监督。

#### 3.5 On-Policy Self-Distillation（OPSD）

OPSD 是 OPD 的一种特殊形式：Teacher 和 Student 来自同一个模型，但两条分支获得的信息不同。

- Student 只能看到正常输入；
- Teacher 额外看到参考答案、正确轨迹或其他 Privileged Information；
- Teacher 分支 Stop Gradient，在当前训练步中作为固定目标；
- Student 学习 Teacher 在相同前缀状态上的下一 Token 分布。

可以写成：

$$
q_\phi(\cdot\mid s,z),
$$

其中 $z$ 表示 Teacher 额外获得的信息。

其目标仍然是：

$$
L_{\mathrm{OPSD}} = \mathbb E_{s\sim d^{\pi_\theta}} \left[ D_{\mathrm{KL}} \left( \pi_\theta(\cdot\mid s) \|q_\phi(\cdot\mid s,z) \right) \right].
$$

Teacher 因为知道额外答案信息，理论上能够在 Student 的当前状态上给出更好的 continuation 分布。

#### 3.6 OPSD 的问题：高 KL 不等于任务关键

Blog 的 Token 级分析发现，Teacher 与 Student 差异最大的 Token 可能集中在：

- wait；
- alright；
- therefore；
- 其他思考转折词或风格 Token。

而真正决定数学任务的 Token，例如 power、exponent、logarithm，Teacher 与 Student 的 KL 差异反而可能较小。

因此：

$$
\text{Teacher 与 Student 分歧大} \not\Rightarrow \text{该 Token 对任务最关键}.
$$

底层原因是：Teacher 分布提供的是“Teacher 会如何说”的密集信息，它不会自动区分：

- 任务正确性差异；
- 表达风格差异；
- 思考口癖差异；
- Teacher 自身的不稳定偏好。

如果少数高 KL Token 主导训练，Student 可能快速降低熵，过度集中到某些固定表达模式，最终出现 Mode Collapse。

一种直接处理方式是对 Token 级信号进行 Clipping：

$$
\widetilde A_t = \operatorname{clip}(A_t,-c,c).
$$

Clipping 的目的不是否定 Teacher，而是限制单个高偏差信号对参数更新的支配程度。

#### 3.7 为什么 OPSD 更像 RLHF，而不是 RLVR

RLVR 的结果奖励通常具有以下特点：

- 信号稀疏；
- Credit Assignment 困难；
- 但奖励与任务是否完成通常直接相关，偏差较低。

OPSD / OPD 的 Teacher Logits 则具有以下特点：

- 每个 Token 都可以获得密集信号；
- 优化更容易；
- 但信号可能夹杂风格、口癖和 Teacher 偏差。

这与 RLHF 中 Reward Model 的问题更相似：信号可用、密集，但可能被模型过度优化。因此更需要：

- KL Constraint；
- Ratio Clipping；
- Token-Level Clipping；
- Trust Region；
- 熵控制。

> **OPD 小结：** OPD 让 Student 决定训练状态，让 Teacher 决定局部出边方向。它保留了 On-Policy 的状态分布，又获得了 Distillation 的密集监督；代价是 Teacher 的偏差也会被密集传递。

---

## 四、使用不同 Teacher 进行 OPD

### 1. 实验任务：Minimal Code Editing

Blog 使用 Minimal Code Editing 任务比较不同 Teacher。

任务要求模型：

- 修复代码中的 Bug；
- 尽量只修改必要部分；
- 不重写无关代码；
- 不引入无关注释和格式变化。

这个任务同时考察两个方面：

1. **泛化能力：** 能否从训练中出现的错误类型泛化到新的代码错误；
2. **能力保持：** 学会最小修改后，通用代码生成能力是否下降。

### 2. 实验设计

作者先训练两种 Teacher：

- SFT Teacher；
- RL Teacher。

再分别使用它们训练两个 OPD Student：

- OPD + SFT Teacher；
- OPD + RL Teacher。

原本的直觉是：RL Teacher 泛化更好、遗忘更少，因此由 RL Teacher 蒸馏出的 Student 应该明显优于由 SFT Teacher 蒸馏出的 Student。

### 3. 实验结果

| 模型 | Pass@1 ↑ | Norm. Levenshtein ↓ | Added CC ↓ | LiveCodeBench v6 ↑ |
|---|---:|---:|---:|---:|
| SFT Teacher | 0.775 | 0.450 | 0.450 | 0.286 |
| RL Teacher | 0.792 | 0.063 | 0.206 | 0.320 |
| OPD + SFT Teacher | 0.800 | 0.059 | 0.206 | 0.297 |
| OPD + RL Teacher | 0.787 | 0.055 | 0.228 | 0.314 |

可以得到几个直接观察：

1. RL Teacher 相比 SFT Teacher，最小修改行为和 LiveCodeBench 表现更好；
2. 两个 OPD Student 的核心任务表现非常接近；
3. 两个 OPD Student 都显著优于 SFT Teacher 的编辑风格；
4. 即使 Teacher 是已经出现明显遗忘的 SFT 模型，OPD Student 的遗忘仍然较轻；
5. Teacher 本身的差异，没有原本预期中那么决定性。

### 4. 该实验真正说明了什么

如果 Student 只是复制 Teacher，那么 SFT Teacher 的缺陷应该被完整传递给 Student。但实验中并没有发生这种情况。

这说明 OPD 不是简单复制 Teacher 的输出行为。Teacher 决定局部指导信号，但训练节点来自 Student 自身：

$$
s\sim d^{\pi_{\mathrm{student}}}.
$$

因此，Student 的 On-Policy State Distribution 可能比 Teacher 的来源更重要。

更直观地说：

> Teacher 告诉 Student“在你当前走到的位置，下一步哪些方向更值得提高”；但 Student 不需要先变成 Teacher，也不必完整重走 Teacher 的行为分布。

这也带来一个实用思路：

1. 可以先通过 SFT 或 RL 训练多个领域专家；
2. 专家即使为了能力提升而变得偏科，仍可作为指导模型；
3. 最终模型通过 OPD 在自身状态分布上吸收专家能力；
4. 这样可能减少直接合并专家参数或直接复制专家行为带来的附带损伤。

> **本节要点：** Teacher 决定“方向”，Student 的 On-Policy 分布决定“方向在哪些状态上生效”。实验提示，后者可能是 OPD 保持原有能力的核心因素。

---

## 五、为什么 RL 遗忘得更少

Blog 讨论了多种解释。它们都能解释一部分现象，但作者最终更倾向于 On-Policy Data。

### 1. Forward KL 与 Reverse KL

#### 1.1 SFT 的 Forward KL

SFT 的交叉熵等价于：

$$
D_{\mathrm{KL}}(p_D\|\pi_\theta).
$$

Forward KL 从数据分布出发，要求模型为数据出现的模式分配足够概率。若数据覆盖多个模式，模型需要尽量覆盖这些外部模式。

#### 1.2 RL 的 Reverse KL 视角

在某些带参考策略和 KL Regularization 的推导中，RL 可以被理解为接近一个奖励加权目标分布：

$$
p^*(y\mid x) \propto \pi_{\mathrm{ref}}(y\mid x) \exp\left(\frac{R(x,y)}{\beta}\right).
$$

对应目标可写成某种 Reverse KL：

$$
D_{\mathrm{KL}}(\pi_\theta\|p^*).
$$

Reverse KL 的期望从当前策略 $\pi_\theta$ 取样，因此更关注当前模型已有概率质量的区域，具有 Mode-Seeking 倾向。

这提供了一个直觉：RL 更容易从当前模型已经会生成的解法中，寻找奖励更高的模式，而不是全面覆盖一个外部数据分布。

#### 1.3 为什么该解释不完整

不能简单得到：

$$
\text{Forward KL 会遗忘，Reverse KL 不会遗忘}.
$$

原因包括：

1. RL 的 Reverse KL 解释依赖具体目标形式，并非所有 RL 算法都能直接等同为一个固定 Reverse KL；
2. 现实中的 RLVR 即使弱化或移除显式 Reference KL，也仍可能表现出较强的能力保持；
3. OPD 同样具有 On-Policy 性质，却使用 Teacher 分布而不是环境 Reward，也能减少遗忘。

因此，KL 方向提供了有用的几何直觉，但还不是最底层、最统一的解释。

### 2. 密集 Token 梯度与稀疏参数更新

#### 2.1 SFT 的梯度为什么更密集

SFT 的每个示范 Token 都产生交叉熵监督：

$$
L_{\mathrm{SFT}} = -\sum_t\log\pi_\theta(y_t^*\mid s_t^*).
$$

无论一个 Token 是：

- 关键推理步骤；
- 最终答案；
- 标点符号；
- 固定套话；
- 无关格式；

它都会进入损失。

特别是当数据目标 Token 在初始模型下概率很低，而模型对另一个 Token 极其确信时，交叉熵会产生较强的纠正压力。这种外部分布上的密集纠正，更可能扰动原有表示。

#### 2.2 RL 的更新为何具有数据依赖性

RL 中，Token 是否产生显著更新还取决于：

- 轨迹是否被当前策略采样；
- Reward 大小；
- Baseline；
- Advantage；
- 组内奖励均值与方差；
- PPO / GRPO Ratio 和 Clipping。

例如，一个简化损失可写成：

$$
L_{\mathrm{RL}} = -\sum_t A_t\log\pi_\theta(y_t\mid s_t).
$$

当 $A_t$ 很小或样本没有被采到时，直接更新也很小。梯度强度不只取决于“Token 是否与标签匹配”，而会随 Student 当前行为和奖励统计变化。

Blog 将这种性质理解为一种 **Data-Dependent Regularization**：优化压力由当前采样数据与奖励共同决定，而不是对所有外部目标 Token 施加同样类型的拟合压力。

#### 2.3 RL 的稀疏参数更新

一些经验工作观察到：

- RL 的显著参数变化往往集中在较小的参数子网络；
- SFT 的参数变化更广泛、更冗余；
- 限制可更新参数数量时，RL 性能可能更快下降，说明它使用的有效更新更集中。

这可以解释部分抗遗忘现象：如果新能力只需要改变较少的功能方向，那么原有能力受到的参数干扰可能更小。

但这里的“稀疏”是经验现象，不是所有 RL 算法都必然满足的数学定理。并且，它仍然不能单独解释为什么 OPD 也能表现出较好的能力保持。

### 3. 更核心的解释：On-Policy Data

#### 3.1 Reward 可以看成轨迹过滤器

考虑最简单的二元奖励：

$$
R(x,y)\in\{0,1\}.
$$

在一种简化理解中：

- Reward $=1$ 的 Student 轨迹被强化；
- Reward $=0$ 的轨迹被忽略或压低。

这时 Reward 很像一个 Rejection Sampling Filter：从 Student 自己生成的回答中筛选成功轨迹。

训练数据不是由外部专家完全指定，而是来自：

$$
y\sim\pi_\theta(\cdot\mid x),
$$

再由 Reward 决定哪些样本值得保留或强化。

#### 3.2 为什么它倾向于寻找“附近的任务解”

设所有能完成任务的策略构成集合：

$$
\mathcal P^* = \{\pi:J(\pi)\text{ 达到任务要求}\}.
$$

满足任务的策略可能有很多。On-Policy 训练的数据来自当前策略，因此它更容易发现：

$$
\text{当前策略已经能以非零概率访问的成功解}.
$$

训练过程表现为：

$$
\pi_{\theta_0} \rightarrow \pi_{\theta_1} \rightarrow \pi_{\theta_2} \rightarrow \cdots \rightarrow \pi^*.
$$

每一轮都从当前策略重新采样，再沿当前能访问到的成功方向移动。它不会像固定外部 SFT 数据那样，直接要求模型跳到一条可能离当前分布很远的示范路径。

这不是显式距离约束，也不保证参数变化一定很小。它是一种由数据生成机制带来的隐式约束：

> 当前模型不会生成的状态，很难直接进入训练；当前模型已经会生成的有效行为，更容易成为新策略的基础。

#### 3.3 为什么这能统一解释 RL 和 OPD

RL 与 OPD 使用的监督信号不同：

- RL 使用 Reward / Advantage；
- OPD 使用 Teacher Distribution。

但它们的训练状态都来自 Student 当前策略：

$$
s\sim d^{\pi_\theta}.
$$

这解释了为什么：

- RL 往往比固定外部 SFT 遗忘更少；
- OPD 即使使用一个已经发生遗忘的 SFT Teacher，也可能训练出遗忘较轻的 Student；
- Teacher 的质量重要，但 Student 的状态分布同样关键。

因此，Blog 更偏好的结论是：

> 抗遗忘的关键不只在于优化的是 Forward KL 还是 Reverse KL，也不只在于梯度是否稀疏，而在于训练数据是否来自模型自身当前的状态分布。

### 4. 为什么 Student 可以超过 Teacher

#### 4.1 Student 与 Teacher 访问的状态不同

如果只训练 Teacher 的轨迹，监督可能集中在 Teacher 会访问的状态上。但 Student 的错误会把它带入另一批状态。

OPD 让 Teacher 在 Student 状态上给出指导：

$$
s\sim d^{\pi_{\mathrm{student}}}, \qquad q_\phi(\cdot\mid s).
$$

因此，Teacher 的能力被用于修正 Student 真正面对的问题，而不是只重复 Teacher 已经能够正确处理的轨迹。

这与 Imitation Learning 中的 Distribution Shift 问题非常接近：Student 推理时会遇到自身错误产生的状态，训练也必须覆盖这些状态。

#### 4.2 学习分布不等于复制 Greedy Output

Teacher 的 Logits 中不仅包含最高概率 Token，还包含：

- 不同 continuation 的相对偏好；
- 不确定性；
- 备选推理分支；
- 风格与语义结构。

Student 学习的是：

$$
q_\phi(\cdot\mid s),
$$

而不是 Teacher 单次 Greedy Decode 得到的一条序列。

因此，Student 可能重新组合 Teacher 分布中的信息，在自己的状态分布上形成比 Teacher 单次生成更好的行为。

#### 4.3 分布集中与 Mode Concentration

Blog 还提到一个反直觉现象：某些工作即使使用未经正确性过滤、甚至高温采样得到的模型生成数据进行自蒸馏，性能仍然可能提升。

一种推测是：OPD 不仅传递知识，还会使分布围绕新能力进一步集中。由于 OPD 的熵下降可能比 RL 更剧烈，收益有时来自：

$$
\text{原本分散的有效概率质量} \longrightarrow \text{更加集中到少数可用模式}.
$$

但这仍然是解释性猜测，而不是已经完全验证的定论。过度集中也可能变成 Mode Collapse，因此必须结合熵、KL 和生成多样性一起判断。

> **本节要点：** 多种机制都可能降低遗忘，但 On-Policy Data 能同时解释 RL 和 OPD。它限制的不是参数影响范围，而是直接训练状态从哪里产生。

---

## 六、为什么 RL 和 OPD 泛化得更好

### 1. 任务成功不等于复现唯一序列

SFT 的监督目标是某个固定 Token 序列。只要模型没有为该序列分配足够概率，就会受到惩罚，即使它生成了另一条同样正确的回答。

假设两个回答不同，但都能完成任务：

$$
y_1\neq y_2, \qquad R(x,y_1)=R(x,y_2)=1.
$$

对于 RLVR，它们都可以获得相同正奖励。RL 因此优化的是正确回答区域的概率质量，而不是唯一参考答案的表面形式。

OPD 虽然学习 Teacher 分布，但由于训练状态来自 Student，也不要求 Student 完整复现 Teacher 的固定轨迹。

所以，RL 和 OPD 都比“逐 Token 复现一条外部示范路径”保留了更大的可行解空间。

### 2. 状态分布失配与误差累积

SFT 使用 Teacher Forcing。训练第 $t$ 个 Token 时，模型看到的是正确前缀：

$$
s_t^*=(x,y_{<t}^*).
$$

但推理时，前缀来自 Student 自己：

$$
s_t=(x,\hat y_{<t}).
$$

一旦 Student 在前面生成了不同 Token，就可能进入训练数据从未覆盖的状态。此后每一步都建立在偏离后的前缀上，误差会继续累积。

这就是状态分布失配：

$$
d^{\pi_{\mathrm{data}}}(s) \neq d^{\pi_\theta}(s).
$$

### 3. RL 与 OPD 如何缓解该问题

RL 和 OPD 都直接在 Student 当前会访问的状态上训练：

- RL：对 Student 轨迹使用 Reward 修正；
- OPD：在 Student 前缀上查询 Teacher 分布并修正。

于是，当 Student 进入自己的偏离状态时，这些状态仍有机会进入训练，而不再完全依赖外部正确前缀。

这带来两种泛化优势：

1. **解空间更宽：** 不要求绑定到唯一示范序列；
2. **状态覆盖更匹配：** 训练覆盖 Student 推理时真实会访问的状态。

### 4. 需要避免过度概括

这不意味着 RL 或 OPD 在所有任务上一定比 SFT 泛化更好。实际结果仍取决于：

- Reward 是否准确；
- Teacher 是否可靠；
- Rollout 是否覆盖到有效区域；
- 探索是否充分；
- Clipping 和 KL 是否合理；
- 数据与评测是否匹配。

更准确的结论是：

> 从目标定义和状态分布来看，RL 与 OPD 具有减少表面序列过拟合和状态分布失配的结构性条件，但最终效果仍由训练信号质量决定。

---

## 七、完整的后训练流水线

### 1. 为什么不同阶段不能简单互相替代

很多模型采用类似的后训练流水线：

$$
\text{Pretrain} \rightarrow \text{SFT} \rightarrow \text{RL} \rightarrow \text{OPD}.
$$

每个阶段解决的问题不同。

#### 1.1 Pretrain：建立基础分布

预训练建立：

- 语言建模能力；
- 世界知识；
- 通用代码与数学模式；
- 基础上下文学习能力。

从概率树视角看，Pretrain 构造了一棵覆盖范围极广、但不一定严格服从指令或任务奖励的初始概率树。

#### 1.2 SFT：完成冷启动

SFT 适合快速建立：

- 指令遵循格式；
- Chat Template；
- 工具调用协议；
- Chain-of-Thought 的基本输出结构；
- 某类任务的标准输入输出形式。

没有这些基础行为，模型可能连 RL 环境要求的输出格式都无法稳定产生，因此 SFT 很难被完全跳过。

#### 1.3 RL：训练可验证专家能力

当任务存在可靠 Reward 时，RL 可以让模型从自身分布中发现并强化有效策略。例如：

- 数学；
- 代码；
- 工具使用；
- Agent 环境交互。

从概率树视角看，RL 让正确回答区域的概率质量在 On-Policy Rollout 中逐步扩大。

#### 1.4 OPD：合并专家能力

OPD 可以让最终模型在自己的状态分布上吸收一个或多个专家的 Token Distribution。

最终 Checkpoint 不一定需要亲自经历每一种 RL 训练。可以先构造不同专家，再将能力蒸馏回最终 Student：

$$
\{ q_{\mathrm{math}}, q_{\mathrm{code}}, q_{\mathrm{agent}} \} \xrightarrow{\mathrm{OPD}} \pi_{\mathrm{final}}.
$$

### 2. 不同领域适合不同训练信号

#### 2.1 Math 与 Code

数学答案和代码执行结果容易验证，因此更适合 RLVR：

- Reward 与正确性直接相关；
- 噪声较低；
- 可以进行大规模自动评估。

#### 2.2 Creative Writing 与开放式知识任务

这些任务缺少唯一标准答案，Reward 往往更主观、更噪声化。此时可以更多依赖：

- 高质量 SFT；
- Teacher Distillation；
- Self-Distillation；
- 人类偏好数据；
- 多维 Reward Model。

#### 2.3 最终模型的能力合并

不同领域专家可能各自发生偏科或遗忘。OPD 提供了一种思路：最终模型保持自己的状态分布，再分别接受专家指导，从而减少直接复制专家全部分布造成的冲突。

### 3. 从概率树统一理解整个流水线

可以将完整流水线理解为：

1. Pretrain 建立宽广的初始概率树；
2. SFT 用外部路径修剪出基本可用的交互结构；
3. RL 在当前树的可访问区域内扩大高奖励路径；
4. OPD 在最终 Student 的访问节点上吸收不同 Teacher 的局部方向。

所以，问题不应该只是“SFT、RL、OPD 谁更好”，而应该是：

> 当前阶段缺少的是基本行为、可靠奖励下的能力探索，还是多专家能力的稳定合并？

---

## 八、什么才是更理想的算法

### 1. Blog 的核心判断

Blog 最后的核心判断是：

> RL 可能不是唯一特殊的算法；真正承担关键作用的，可能是 On-Policy Data。

显式 KL 约束不是全部原因。On-Policy Sampling 使模型主要在当前能够访问的状态上接收训练信号，从而在学习新能力时，减少与当前分布无关的大范围移动。

### 2. 现有方法的核心矛盾：Credit Assignment

#### 2.1 RLVR：低偏差，但信号稀疏

RLVR 通常只在完整轨迹结束后得到一个结果奖励：

$$
R(x,y).
$$

它可以准确告诉模型“整条轨迹成功还是失败”，却很难告诉模型：

- 哪一步推理真正关键；
- 哪个 Token 导致失败；
- 哪些中间步骤虽然看似错误但后来被修正；
- 成功轨迹中的哪些 Token 只是偶然表达。

因此，它的 Reward 偏差较低，但 Credit Assignment 很稀疏。

#### 2.2 Process Reward：更细，但难以扩展

Process Reward Model 希望为中间步骤提供奖励：

$$
r_t=R(s_t,a_t).
$$

它能缓解轨迹级奖励过于稀疏的问题，但需要可靠的过程标注或过程验证。对开放式推理而言，这通常成本高、误差大，也难以稳定扩展。

#### 2.3 Teacher Logits：信号密集，但偏差更高

Distillation 可以在每个 Token 上得到完整 Teacher Distribution：

$$
q_\phi(\cdot\mid s_t).
$$

它的优点是信号密集、优化稳定；问题是 Teacher 分歧未必等于任务关键性，高 KL Token 可能只是风格差异。

因此：

| 信号 | 密度 | 与任务结果的直接相关性 | 主要问题 |
|---|---|---|---|
| Outcome Reward | 低 | 高 | Credit Assignment 稀疏 |
| Process Reward | 中到高 | 取决于标注质量 | 难以低成本扩展 |
| Teacher Logits | 高 | 可能有偏 | 容易放大风格与 Teacher 偏差 |

### 3. 理想算法需要同时具备什么

理想方法希望同时拥有：

1. Distillation 的密集 Token 级信号；
2. RLVR 与真实任务结果直接对应的低偏差；
3. RL 和 OPD 的 On-Policy State Distribution；
4. 能够控制过度集中与 Mode Collapse 的 Trust Region；
5. 更准确的 Token / Step Credit Assignment。

可以压缩成：

$$
\boxed{ \text{Dense Signal} + \text{Low Bias} + \text{On-Policy Data} + \text{Reliable Credit Assignment} }
$$

### 4. 从概率树看这个问题

更理想的算法需要回答四个问题：

1. **访问哪些节点？** —— 由 On-Policy Sampling 决定；
2. **哪些路径真正成功？** —— 由低偏差 Outcome Reward 判断；
3. **沿途哪些出边应该承担贡献？** —— 由更可靠的 Credit Assignment 判断；
4. **每个节点的完整出边分布应如何调整？** —— 由密集但受约束的 Token 信号提供。

也就是说，真正缺少的不只是一个新的 Loss 名称，而是把“正确的训练区域”和“正确的局部更新方向”同时解决的机制。

目前，问题的结构已经比较清楚，但具体最优算法仍没有确定答案。

---

## 总结

### 1. 三种方法的概率树本质

- **SFT：** 外部数据指定路径，逐 Token 提高示范出边概率；
- **RL：** 当前模型采样路径，Reward / Advantage 强化成功路径、抑制失败路径；
- **OPD：** Student 产生训练状态，Teacher 调整这些状态上的出边分布。

### 2. 后训练的统一表达

无论使用哪种方法，底层都在更新同一组共享参数：

$$
\theta' = \theta-\eta\nabla_\theta L.
$$

共享参数的变化会重新定义各个前缀节点的出边概率，最终改变完整序列的分布：

$$
\pi_\theta(y\mid x) = \prod_t\pi_\theta(y_t\mid x,y_{<t}).
$$

因此，将后训练理解为“不断调整整棵自回归概率树，使其更贴近期望行为”是成立的。但必须加上一个限定：

> 优化器不会逐节点修改整棵树；它只在有限训练节点上构造损失，再通过共享参数使整棵树发生联动变化。

### 3. 最关键的区分

三种方法真正不同的不是“都在更新参数”，而是：

$$
\boxed{ \text{训练节点从哪里来} + \text{更新信号从哪里来} }
$$

SFT 的节点和方向都主要来自外部数据；RL 的节点来自 Student、方向来自 Reward；OPD 的节点来自 Student、方向来自 Teacher。

Blog 最值得保留的观点是：

> 当训练节点来自模型自身的 On-Policy 分布时，模型更可能在当前已经能够访问的行为基础上学习新能力，而不是被直接拉向一个任意远的外部分布。这可能是 RL 和 OPD 更少遗忘、泛化更好的共同原因。
