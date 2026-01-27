# 介绍
## RLHF干什么的？
Pretrain：让LLM学习到各种各样的知识

SFT：让LLM具有指令遵循的能力，按照我们规定的范式进行回答。But，LLM无法对回答的好坏做出判断。他只能是机械的的你问我答。

RLHF：在SFT的基础上，引入RL，让模型能够判断什么样的回答是好的，什么样的回答是不好的。



# DPO
**DPO（Direct Preference Optimization），**一种**不需要**显式强化学习（RL）过程，直接使用我们标注好的偏好数据`**训练大模型**`LLM的方法。

## 解决问题
传统基于RLHF的训练流程：

+ 预训练语言模型（Pretrain）
+ SFT阶段，让LLM拥有指令遵循的能力
+ RLHF阶段
    - 使用我们标注好的偏好数据 → 训练奖励模型（Reward Model）
    - 用PPO等RL算法，优化语言模型

## 痛点
+ reward Model  + PPO ，系统很重
+ 训练不稳定（PPO对参数的调整，会影响SFT阶段训练的参数）

## DPO核心思想
既然拥有了标注好的偏好数据，为什么不直接用来训练LLM，而需要绕一个圈子，用PPO的方式来优化模型。

## DPO训练数据长什么样子？
使用偏好数据（Preference Data）：

```plain
(prompt, chosen, rejected)
```

示例：

```plain
Prompt: Explain DPO in simple terms.

Chosen: DPO directly trains the model to prefer better answers using human feedback.

Rejected: DPO uses reinforcement learning with a reward model.
```

## DPO核心数学思想
在RLHF中，目标的最大化：

$ E_{y ~ \pi_\theta(y|x)}[r(x,y)] $

而Reward Model通常来自偏好比较：

$ P(y^+ > y^-) = \sigma(r(y^+) - r(y^-)) $

## DPO的工程实现（高层流程）
```plain
1. 训练 SFT 模型（reference model）
2. 收集偏好数据 (prompt, chosen, rejected)
3. 冻结 reference model
4. 用 DPO loss 微调 SFT
5. 得到对齐后的模型
```



# GRPO
GRPO（Group Relative Policy Optimization）是一种“无价值函数（no value model）的 PPO 变体”，通过“组内相对奖励”来训练策略模型。

## 解决问题
### PPO在LLM中训练的问题
+ reward模型不好训练，方差大
+ 价值函数value function不稳定

结果：

+ value loss震荡
+ advantage偏差大
+ PPO发散

### reward的绝对值不可靠
对于reward的输出，对于不同prompt，在不同prompt下是不可比较的。

但是，排序我们认为是可靠的。

## GRPO的核心
不再使用reward model输出的标量直接进行计算。而是使用相对的reward进行计算。

## 训练方法
### step 1 使用当前policy进行rollout
对同一个 prompt `x`：

```plain
πθ → 采样 K 个回答
y₁, y₂, ..., y_K
```

### Step 2使用reward model进行打分
```plain
r₁ = RM(x, y₁)
r₂ = RM(x, y₂)
...
r_K = RM(x, y_K)

```

### Step 3构造“组内相对 Advantage”
$ A_i = r_i - mean(r_1, ..., r_k) $

可以看出，GRPO不再依赖value model来计算advantage，减少了value model的训练。

## GRPO的loss是什么？
与PPO相同，只是换了一个A罢了



