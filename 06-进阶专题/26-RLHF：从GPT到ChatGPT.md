# 26-RLHF：从GPT到ChatGPT

当你第一次使用 ChatGPT 时，可能会惊叹于它的"善解人意"——它不仅能理解你的问题，还能按照你的要求调整回答风格：要简洁就简洁，要详细就详细，还能承认自己的错误。

但你知道吗？这些能力并非来自预训练。基础版 GPT 模型（如 GPT-3）其实是个"话痨"：你问它一个问题，它可能会自顾自地续写下去，根本不理会你的指令；你说"不要解释，直接给答案"，它依然长篇大论；甚至有时会生成有害、偏见或错误的内容。

**是什么让 GPT-3 进化成了 ChatGPT？答案是 RLHF（Reinforcement Learning from Human Feedback，人类反馈强化学习）。**

这项技术的核心思想很简单：既然模型的目标是预测下一个 token，而非满足人类需求，那我们就引入人类的反馈来告诉它"什么是好的回答"。通过训练一个奖励模型来评估回复质量，再用强化学习优化语言模型，最终让模型学会"听话"——理解指令、遵循约束、提供有用且无害的回复。

2022 年，OpenAI 凭借 RLHF 技术推出了 ChatGPT，开启了对话 AI 的新纪元。从 GPT-3 到 ChatGPT，从 Llama-2 到 Llama-2-Chat，RLHF 已成为实现大模型对齐（Alignment）的行业标准。本章将深入讲解 RLHF 的三阶段训练流程、数学原理和实践技巧。

---

## 1. 从文本续写到对话助手

### 1.1 GPT 的局限性：它并不"听话"

GPT本质上是"鹦鹉学舌"——它只是预测下一个最可能的token，而非真正理解人类想要什么。

**问题表现**：

```
用户：请帮我写一首关于春天的诗
GPT续写：春天到了，万物复苏，花儿开了，太阳出来了...
       （可能继续生成无关内容，或者写出奇怪的诗）

用户：不要解释，直接告诉我答案
GPT续写：当然，我可以帮你...
       （忽略"不要解释"的指令）
```

**根本原因**：
- 预训练目标：最大化 token 预测概率
- 优化目标与人类意图之间存在巨大鸿沟

### 1.2 RLHF的核心思想

**如何让模型理解"人类想要什么"？**

传统方法：
- 人工设计奖励函数（hard to define）
- 收集成对偏好数据（pairwise comparison）

RLHF的解决方案：
```
Step 1: 收集人类反馈 → 训练奖励模型
Step 2: 用奖励模型指导微调 → 优化策略
Step 3: 迭代改进 → 持续对齐人类偏好
```

**类比**：

想象训练一只导盲犬：
- **传统有监督**：给导盲犬看"这是红灯，停"、"这是绿灯，走"的例子
- **RLHF**：让驯犬师根据导盲犬的表现给反馈（好/不好），导盲犬逐渐学会判断

### 1.3 从GPT-3到ChatGPT的进化

| 版本 | 训练方式 | 能力 | 对话质量 |
|------|---------|------|----------|
| GPT-3 | 纯预训练 | 续写文本 | 很差 |
| GPT-3.5 | +有监督微调(SFT) | 遵循指令 | 一般 |
| GPT-4 | +RLHF | 对话助手 | 优秀 |

**关键洞察**：
- SFT让模型"知道"该做什么（能力）
- RLHF让模型"知道"人类想要什么（对齐）

---

## 2. 数学原理

### 2.1 整体框架

RLHF训练分为三个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│  阶段1: 有监督微调 (SFT)                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  人类标注   │ →  │   SFT模型   │ →  │  遵循指令   │        │
│  │  (问答对)   │    │  (微调GPT)  │    │  的基础模型 │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                           ↓                                    │
│  阶段2: 奖励模型训练 (RM)                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  比较数据   │ →  │   奖励模型  │ →  │  评估质量   │        │
│  │  (A vs B)   │    │  (回归模型) │    │  的打分器   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                           ↓                                    │
│  阶段3: 强化学习微调 (PPO)                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  生成回复   │ →  │  奖励评估   │ →  │  策略优化   │        │
│  │  → 环境     │    │  → 奖励模型 │    │  → PPO更新  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 阶段1：有监督微调 (SFT)

**目标**：让模型学会回答问题

```
数据格式：(prompt, response)
Loss = -Σ log P(response_i | prompt, response_{<i})

这是标准的语言模型训练，只是数据变成了"指令-回答"对
```

**数据量**：OpenAI使用约1万-10万条高质量问答对

### 2.3 阶段2：奖励模型 (RM)

**目标**：学习评估"回答的好坏"

**训练数据**：人类比较数据

```
Prompt: "解释量子纠缠"
Response A: "量子纠缠是两个粒子之间的神秘联系..."
Response B: "量子纠缠就是..."
         ↓
人类选择：A更好
         ↓
训练目标：奖励模型给A更高分，B更低分
```

**模型架构**：
- 输入：prompt + response
- 输出：单个标量分数

```
rθ(prompt, response) → R
```

**训练目标（Bradley-Terry模型）**：

```
给定prompt的两个响应 (response_1, response_2)，人类偏好 response_1

P(prefer_1) = sigmoid(rθ(prompt, response_1) - rθ(prompt, response_2))

Loss = -log σ(r_A - r_B)
     = -log σ(Δr)

其中 σ 是 sigmoid 函数
```

**直觉**：如果r_A > r_B，则Δr > 0，σ(Δr) → 1，loss → 0

### 2.4 阶段3：PPO强化学习

**目标**：用奖励模型作为"伪人类反馈"，优化语言模型

**核心目标函数**：

```
L(θ) = E[ rθ(prompt, response) ] - β * KL(πθ || πSFT)
               ↑                              ↑
            奖励项                         KL惩罚项
```

**两项的直观理解**：

1. **奖励项**：让模型生成奖励模型打分高的回复
2. **KL惩罚项**：防止模型偏离SFT模型太远，避免"灾难性遗忘"

**PPO的具体实现**：

```
PPO目标（剪切版本）：
L(θ) = E[ min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) ]

其中：
- r(θ) = πθ(a|s) / πold(a|s)  (重要性比率)
- A = advantage (优势函数) = rθ - baseline
- ε = 0.2 (裁剪边界)

作用：防止策略更新过大导致性能崩溃
```

**完整PPO训练流程**：

```
for each batch:
    1. 用当前策略πθ生成 responses
    2. 用奖励模型rϕ计算奖励
    3. 计算KL散度
    4. 组合总奖励：r_total = r - β*KL
    5. 用PPO目标更新策略网络
    6. (可选) 更新价值网络
```

### 2.5 完整的RLHF目标

```
L_total = E[rθ(x,y)] - β * KL(πθ(y|x) || πSFT(y|x)) + γ * E[entropy(πθ(·|x))]
           ↑               ↑                              ↑
        奖励模型        KL散度约束                      熵奖励(鼓励探索)
```

**参数说明**：
- β：通常0.1-0.5，控制与SFT的偏离程度
- γ：通常很小(0.01)，防止模型"collapse"

---

## 3. 代码实现

### 3.1 奖励模型训练

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    """
    奖励模型：评估 (prompt, response) 对的质量
    
    架构：将 prompt 和 response 拼接后，用一个模型编码，
    然后用一个线性层输出单个分数
    """
    def __init__(self, base_model_name="gpt2", hidden_dropout=0.1):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # 奖励输出头
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch, seq_len] - 包含 prompt 和 response
            attention_mask: [batch, seq_len]
        
        Returns:
            rewards: [batch, 1] - 每个样本的奖励分数
        """
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # 取最后一个token的表示（类似CLS）
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled = last_hidden[:, -1, :]  # [batch, hidden]
        
        reward = self.reward_head(pooled)  # [batch, 1]
        
        return reward


class PairwiseLoss(nn.Module):
    """
    成对比较损失
    
    给定同一个prompt的两个response，训练模型给更好的response更高分
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, rewards_chosen, rewards_rejected):
        """
        Args:
            rewards_chosen: [batch] - 被选中的response的奖励
            rewards_rejected: [batch] - 被拒绝的response的奖励
        
        Returns:
            loss: 标量损失
        """
        # 使用 hinge loss
        losses = torch.relu(self.margin - (rewards_chosen - rewards_rejected))
        return losses.mean()


def train_reward_model(
    train_dataset,
    model_name="gpt2",
    lr=1e-5,
    epochs=3,
    batch_size=4,
):
    """
    训练奖励模型的完整流程
    """
    # 初始化模型
    model = RewardModel(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 损失函数
    criterion = PairwiseLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in train_dataset:
            # batch 包含: prompt, chosen_response, rejected_response
            prompt = batch["prompt"]
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            
            # 构建输入：prompt + response
            # 格式: [EOS]prompt[A]response[EOS]
            inputs_chosen = tokenizer(
                prompt + "\n" + chosen,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs_rejected = tokenizer(
                prompt + "\n" + rejected,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # 前向传播
            rewards_chosen = model(
                inputs_chosen["input_ids"],
                inputs_chosen["attention_mask"]
            ).squeeze()
            
            rewards_rejected = model(
                inputs_rejected["input_ids"],
                inputs_rejected["attention_mask"]
            ).squeeze()
            
            # 计算损失
            loss = criterion(rewards_chosen, rewards_rejected)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataset):.4f}")
    
    return model
```

### 3.2 PPO训练循环

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque
import random

class PPOTrainer:
    """
    PPO训练器：用于RLHF的第三阶段
    
    核心组件：
    - 策略模型 (policy): 要训练的语言模型
    - 参考模型 (ref_model): SFT模型，用于KL约束
    - 奖励模型 (reward_model): 评估response质量
    - 价值模型 (value_model): 估计状态价值
    """
    def __init__(
        self,
        policy_model_name="gpt2",
        ref_model_name="gpt2", 
        reward_model=None,
        lr=1e-5,
        clip_eps=0.2,
        kl_coef=0.1,
    ):
        # 策略模型（可训练）
        self.policy = AutoModelForCausalLM.from_pretrained(policy_model_name)
        self.policy_ref = AutoModelForCausalLM.from_pretrained(ref_model_name)
        
        # 冻结参考模型
        for param in self.policy_ref.parameters():
            param.requires_grad = False
        
        # 奖励模型（冻结）
        self.reward_model = reward_model
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=lr
        )
        
        # 超参数
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        
        self.tokenizer = AutoTokenizer.from_pretrained(policy_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompts, max_length=100):
        """
        用当前策略生成responses
        """
        self.policy.eval()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            outputs = self.policy.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token,
            )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses
    
    def compute_rewards(self, prompts, responses):
        """
        用奖励模型计算每个response的分数
        """
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # 构建输入
            text = prompt + "\n" + response
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                reward = self.reward_model(
                    inputs["input_ids"],
                    inputs["attention_mask"]
                ).item()
            
            rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def compute_kl(self, prompts, responses):
        """
        计算策略和参考模型之间的KL散度
        """
        # 编码prompts和responses
        inputs = self.tokenizer(
            [p + "\n" + r for p, r in zip(prompts, responses)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 策略模型的logits
        outputs_policy = self.policy(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        logits_policy = outputs_policy.logits
        
        # 参考模型的logits（不计算梯度）
        with torch.no_grad():
            outputs_ref = self.policy_ref(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )
            logits_ref = outputs_ref.logits
        
        # 计算log概率
        log_probs_policy = torch.log_softmax(logits_policy, dim=-1)
        log_probs_ref = torch.log_softmax(logits_ref, dim=-1)
        
        # KL散度
        kl = torch.sum(
            torch.exp(log_probs_ref) * (log_probs_ref - log_probs_policy),
            dim=-1
        )
        
        return kl
    
    def ppo_update(self, prompts, responses, rewards, old_log_probs=None):
        """
        PPO更新步骤
        """
        # 计算KL惩罚
        kl = self.compute_kl(prompts, responses)
        
        # 总奖励 = 原始奖励 - β * KL
        total_rewards = rewards - self.kl_coef * kl
        
        # 这里简化处理，实际实现需要：
        # 1. 计算 advantage (需要value model)
        # 2. 用PPO目标更新策略
        
        # 简化的策略梯度更新
        self.policy.train()
        
        # 重新计算log概率（用于策略梯度）
        inputs = self.tokenizer(
            [p + "\n" + r for p, r in zip(prompts, responses)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        outputs = self.policy(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        
        # 简化的损失：最大化奖励
        loss = -total_rewards.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "reward": rewards.mean().item(),
            "kl": kl.mean().item(),
            "loss": loss.item(),
        }
    
    def train_step(self, prompts, num_rollouts=4):
        """
        完整的训练步骤
        
        1. 用当前策略生成responses
        2. 计算奖励
        3. 用PPO更新策略
        """
        all_metrics = []
        
        for _ in range(num_rollouts):
            # 生成
            responses = self.generate(prompts)
            
            # 计算奖励
            rewards = self.compute_rewards(prompts, responses)
            
            # 更新
            metrics = self.ppo_update(prompts, responses, rewards)
            all_metrics.append(metrics)
        
        # 返回平均指标
        avg_metrics = {
            k: sum(m[k] for m in all_metrics) / len(all_metrics)
            for k in all_metrics[0].keys()
        }
        
        return avg_metrics


def full_rlhf_pipeline(
    sft_model_path="./sft_model",
    output_dir="./rlhf_output",
    num_epochs=3,
    prompts_dataset=None,
):
    """
    完整的RLHF训练流程
    
    假设：
    - 已有SFT微调后的模型
    - 已训练好奖励模型
    """
    # 加载SFT模型
    policy_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
    
    # 加载奖励模型
    reward_model = RewardModel()
    reward_model.load_state_dict(torch.load("./reward_model.pt"))
    reward_model.eval()
    
    # 创建PPO训练器
    trainer = PPOTrainer(
        policy_model_name=sft_model_path,
        ref_model_name=sft_model_path,
        reward_model=reward_model,
        lr=1e-6,
        kl_coef=0.1,
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        
        for batch_idx, prompts in enumerate(prompts_dataset):
            # 训练步骤
            metrics = trainer.train_step(prompts, num_rollouts=4)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}:")
                print(f"  Reward: {metrics['reward']:.4f}")
                print(f"  KL: {metrics['kl']:.4f}")
                print(f"  Loss: {metrics['loss']:.4f}")
        
        # 保存检查点
        trainer.policy.save_pretrained(f"{output_dir}/epoch_{epoch}")
    
    return trainer.policy
```

### 3.3 DPO：直接偏好优化

```python
class DPO Loss:
    """
    DPO (Direct Preference Optimization)
    
    核心思想：绕过奖励模型，直接用偏好数据优化策略
    
    优势：
    - 不需要单独的奖励模型训练
    - 训练更稳定，速度更快
    - 效果与PPO+RM相当
    """
    def __init__(self, beta=0.1):
        """
        Args:
            beta: KL惩罚系数，类似于PPO中的clip
        """
        self.beta = beta
    
    def forward(
        self,
        policy_chosen_logprobs,      # 选中response的log概率
        policy_rejected_logprobs,    # 拒绝response的log概率
        ref_chosen_logprobs,         # 参考模型的log概率
        ref_rejected_logprobs,       # 参考模型的log概率
    ):
        """
        计算DPO损失
        
        公式：
        L = -log σ( (log_π(y_w) - log_π(y_l)) / β - (log_π_ref(y_w) - log_π_ref(y_l)) / β )
        
        其中 y_w 是被选中的response，y_l 是被拒绝的response
        """
        # 策略的优势
        policy_advantage = policy_chosen_logprobs - policy_rejected_logprobs
        
        # 参考模型的优势
        ref_advantage = ref_chosen_logprobs - ref_rejected_logprobs
        
        # 归一化优势
        normalized_advantage = (policy_advantage - ref_advantage) / self.beta
        
        # 损失 = -log σ(归一化优势)
        loss = -F.logsigmoid(normalized_advantage).mean()
        
        return loss


def dpo_train_step(
    policy_model,
    ref_model,
    chosen_batch,
    rejected_batch,
    beta=0.1,
):
    """
    DPO训练步骤
    """
    # 编码数据
    chosen_inputs = tokenizer(chosen_batch, return_tensors="pt", padding=True)
    rejected_inputs = tokenizer(rejected_batch, return_tensors="pt", padding=True)
    
    # 策略模型的log概率
    chosen_outputs = policy_model(**chosen_inputs, labels=chosen_inputs["input_ids"])
    rejected_outputs = policy_model(**rejected_inputs, labels=rejected_inputs["input_ids"])
    
    # 简化的log概率计算（实际需要更复杂的处理）
    policy_chosen_logprobs = chosen_outputs.logits.log_softmax(dim=-1).mean(dim=-1)
    policy_rejected_logprobs = rejected_outputs.logits.log_softmax(dim=-1).mean(dim=-1)
    
    # 参考模型（不计算梯度）
    with torch.no_grad():
        ref_chosen_outputs = ref_model(**chosen_inputs, labels=chosen_inputs["input_ids"])
        ref_rejected_outputs = ref_model(**rejected_inputs, labels=rejected_inputs["input_ids"])
        
        ref_chosen_logprobs = ref_chosen_outputs.logits.log_softmax(dim=-1).mean(dim=-1)
        ref_rejected_logprobs = ref_rejected_outputs.logits.log_softmax(dim=-1).mean(dim=-1)
    
    # 计算DPO损失
    criterion = DPOLoss(beta=beta)
    loss = criterion(
        policy_chosen_logprobs,
        policy_rejected_logprobs,
        ref_chosen_logprobs,
        ref_rejected_logprobs,
    )
    
    return loss


"""
DPO vs PPO 对比：

| 方面           | PPO+RM              | DPO                  |
|----------------|---------------------|----------------------|
| 训练阶段       | 3个（RM + PPO）     | 1个                  |
| 样本效率       | 较低                | 较高                 |
| 稳定性         | 较复杂              | 较简单               |
| 显存需求       | 高（需要value网络） | 低                   |
| 训练速度       | 慢                  | 快                   |
| 效果           | 最优                | 与PPO相当            |

实际选择：
- 计算资源充足 → PPO+RM
- 资源有限 → DPO
- 追求最佳效果 → PPO+RM
"""
```

---

## 4. 实践指南

### 4.1 RLHF数据收集

```python
# RLHF数据收集最佳实践
data_collection_guide = {
    "数据来源": {
        "人工标注": "付费标注员，质量和一致性较高",
        "众包": "成本低，但质量参差不齐",
        "AI辅助": "用GPT-4生成候选，人工筛选，降低成本",
    },
    "数据量级": {
        "SFT数据": "1万-10万条高质量问答对",
        "偏好数据": "10万-100万对比较数据",
        "经验": "更多偏好数据通常更好",
    },
    "质量控制": {
        "标注指南": "详细说明什么是'好'的回答",
        "一致性检查": "让多个标注员标同一样本",
        "数据清洗": "移除低质量、矛盾的数据",
    },
    "多样性": {
        "prompt多样性": "覆盖不同主题、格式、难度",
        "response多样性": "确保有positive和negative examples",
    },
}
```

### 4.2 超参数选择

```python
# RLHF超参数调优指南
rlhf_hyperparams = {
    "奖励模型": {
        "学习率": "1e-5 - 5e-5",
        "batch_size": "4 - 16",
        "epochs": "1-3（容易过拟合）",
        "margin": "0.5 - 1.0",
    },
    "PPO训练": {
        "学习率": "1e-6 - 1e-5",
        "clip_eps": "0.1 - 0.3（常用0.2）",
        "kl_coef": "0.05 - 0.2",
        "entropy_coef": "0 - 0.01",
        "batch_size": "64 - 512（生成数量）",
        "ppo_epochs": "1 - 4",
    },
    "DPO训练": {
        "beta": "0.1 - 0.5",
        "learning_rate": "1e-6 - 1e-5",
        "batch_size": "8 - 32",
    },
}
```

### 4.3 常见训练问题

```python
# RLHF训练问题排查
troubleshooting = {
    "问题": "模型生成变得保守，只生成短回复",
    "原因": "KL惩罚系数太大，限制了探索",
    "解决": "降低kl_coef，或增加entropy_coef",
    
    "问题": "奖励不再提升",
    "原因": "可能奖励模型过拟合，或探索不足",
    "解决": "检查reward model质量，增加生成多样性",
    
    "问题": "模型开始生成乱码",
    "原因": "策略偏离参考模型太远",
    "解决": "增加KL惩罚，或减少学习率",
    
    "问题": "训练不稳定，loss震荡",
    "原因": "学习率太高，或batch size太小",
    "解决": "降低学习率，增加batch size",
}
```

### 4.4 评估指标

```python
# RLHF模型评估
def evaluate_rlhf_model(model, test_prompts, reward_model):
    """
    评估RLHF训练后的模型
    """
    results = {
        "生成质量": [],
        "奖励分数": [],
        "长度": [],
        "多样性": [],
    }
    
    for prompt in test_prompts:
        # 生成
        response = generate(model, prompt)
        
        # 奖励
        reward = compute_reward(reward_model, prompt, response)
        
        # 记录
        results["生成质量"].append(quality_score(response))
        results["奖励分数"].append(reward)
        results["长度"].append(len(response))
        results["多样性"].append(diversity_score(response))
    
    # 打印平均
    for key, values in results.items():
        print(f"{key}: {sum(values)/len(values):.4f}")
    
    return results


# 常用评估指标
evaluation_metrics = {
    "自动化指标": {
        "奖励分数": "用RM评估，越高越好",
        "Perplexity": "语言模型质量，越低越好",
        "Diversity": "生成多样性，n-gram diversity",
    },
    "人工评估": {
        "帮助性": "回答是否有帮助",
        "诚实性": "是否承认不知道",
        "无害性": "是否生成有害内容",
        "指令遵循": "是否遵循用户指令",
    },
    "基准测试": {
        "MT-Bench": "多轮对话能力",
        "ChatArena": " Elo评分",
        "HH-RLHF": " 对齐数据集",
    },
}
```

---

## 5. 进阶技术

### 5.1 PPO-ptx：加入预训练数据

```python
"""
PPO-ptx：在PPO目标中加入预训练损失

目的：防止模型在RLHF过程中遗忘语言能力

公式：
L_total = L_PPO + α * L_pretrain

其中 L_pretrain 是标准语言模型损失（预测下一个token）
"""
class PPO_PTX:
    def __init__(self, ppo_coef=1.0, pretrain_coef=0.1):
        self.ppo_coef = ppo_coef
        self.pretrain_coef = pretrain_coef
    
    def compute_loss(self, ppo_loss, policy_logprobs, pretrain_ids, pretrain_labels):
        """
        Args:
            ppo_loss: PPO策略损失
            policy_logprobs: 策略模型的log概率
            pretrain_ids: 预训练数据的input_ids
            pretrain_labels: 预训练数据的labels
        """
        # 预训练损失
        pretrain_logits = self.policy(pretrain_ids).logits
        pretrain_loss = F.cross_entropy(
            pretrain_logits.view(-1, pretrain_logits.size(-1)),
            pretrain_labels.view(-1),
            reduction="mean"
        )
        
        # 总损失
        total_loss = self.ppo_coef * ppo_loss + self.pretrain_coef * pretrain_loss
        
        return total_loss
```

### 5.2 迭代式RLHF

```python
"""
迭代式RLHF：多轮RLHF，每轮用最新模型收集新数据

优点：
- 逐步改进，数据分布更接近当前策略
- 可以针对特定问题进行多轮优化
"""
def iterative_rlhf(
    base_model,
    num_iterations=3,
    prompts_per_iteration=1000,
):
    """
    迭代式RLHF流程
    """
    current_model = base_model
    
    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")
        
        # 1. 用当前模型生成responses
        prompts = load_prompts(prompts_per_iteration)
        responses = generate_responses(current_model, prompts)
        
        # 2. 收集人类反馈（这里用RM模拟）
        preferences = collect_preferences(prompts, responses)
        
        # 3. 训练新的奖励模型（可选）
        if iteration > 0:
            reward_model = update_reward_model(preferences)
        
        # 4. 用PPO/DPO微调当前模型
        current_model = train_with_rlhf(
            current_model, 
            preferences,
            method="dpo"  # 或 "ppo"
        )
        
        # 5. 评估当前模型
        eval_score = evaluate(current_model)
        print(f"Evaluation score: {eval_score:.4f}")
    
    return current_model
```

### 5.3 KTO：卡内曼-特沃斯基优化

```python
"""
KTO (Kahneman-Tversky Optimization)

基于行为经济学的 prospect theory：
- 人们对于损失的感受强于收益（loss aversion）
- 用 sigmoid 替代 log-sigmoid，更符合人类偏好
"""
class KTOLoss:
    """
    KTO损失函数
    
    不需要成对偏好，只需要每个样本的"人类判断"（好/坏）
    
    优势：
    - 数据标注更简单（不需要成对比较）
    - 可以利用更多单边反馈数据
    """
    def __init__(self, beta=0.1, reference_point=0.0):
        self.beta = beta
        self.reference_point = reference_point
    
    def forward(self, logprobs, labels):
        """
        Args:
            logprobs: 模型对每个样本的log概率
            labels: 1表示被人类认为好，0表示不好
        
        Returns:
            loss: KTO损失
        """
        # 相对于参考点的偏离
        deviation = (logprobs - self.reference_point) / self.beta
        
        # 使用 prospect theory 的权重函数
        if labels == 1:
            # 收益侧：使用凸函数（不太敏感）
            weight = torch.sigmoid(deviation)
        else:
            # 损失侧：使用凹函数（更敏感）
            weight = torch.sigmoid(-deviation)
        
        # 损失：最大化收益权重，最小化损失权重
        loss = -torch.log(weight * labels + (1 - weight) * (1 - labels))
        
        return loss.mean()
```

---

## 6. 常见陷阱与 FAQ

### Q1: RLHF需要多少数据？

```
经验法则：
- SFT数据：1万-10万条（质量>数量）
- 偏好数据：10万-100万对

但取决于：
- 任务复杂度：复杂任务需要更多数据
- 模型大小：大模型需要更多数据才能充分训练
- 预算：数据收集成本很高

建议：从1万对开始，逐步增加
```

### Q2: RLHF效果不如预期怎么办？

**排查清单**：
```
1. 检查SFT基座模型是否足够好
   - RLHF无法弥补SFT的不足
   - 确保SFT模型已经能生成合理回复

2. 检查奖励模型质量
   - 在测试集上评估RM准确率
   - 低于70%准确率的RM会影响效果

3. 检查数据质量
   - 偏好数据是否一致
   - 是否存在标注偏见

4. 调整超参数
   - 尝试降低KL惩罚
   - 增加PPO epochs

5. 考虑使用DPO
   - 有时比PPO更稳定
```

### Q3: PPO训练崩溃怎么办？

```
常见原因和解决方案：

1. 奖励突然下降
   - 原因：生成质量崩溃，RM给出都是低分
   - 解决：检查RM，检查KL惩罚

2. NaN loss
   - 原因：log概率计算溢出
   - 解决：梯度裁剪，降低学习率

3. 策略collapse（只生成短回复）
   - 原因：KL惩罚太重或entropy太低
   - 解决：降低KL系数，增加entropy coefficient

4. 训练不稳定
   - 原因：batch size太小
   - 解决：增加batch size，使用gradient accumulation
```

### Q4: DPO和PPO怎么选？

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 计算资源有限 | DPO | 不需要value网络，显存需求低 |
| 数据量有限 | PPO+RM | RM可以学习更复杂的偏好 |
| 追求最佳效果 | PPO+RM | 经典方法，经过更多验证 |
| 快速迭代 | DPO | 训练更稳定，速度更快 |
| 多任务泛化 | PPO+RM | RM可以单独调整 |

### Q5: RLHF会让模型变笨吗？

```
风险：是的，可能发生

原因：
- KL惩罚限制了策略变化
- 奖励模型可能过于保守
- 过度优化特定目标

解决方案：
- 使用PPO-ptx加入预训练损失
- 定期在通用数据上评估
- 监控模型的通用能力（如MMLU）
- 迭代式RLHF，逐步改进
```

### Q6: 如何评估对齐效果？

```
多维度评估：

1. 任务特定指标
   - 问答准确率
   - 摘要质量（ROUGE）
   - 代码生成（Pass@k）

2. 对齐指标
   - HH-RLHF数据集准确率
   - 帮助性/无害性人工评估
   - 指令遵循基准（IFEval）

3. 通用能力
   - MMLU（多任务理解）
   - GSM8K（数学推理）
   - HumanEval（代码）

4. 生成多样性
   - n-gram diversity
   - unique ratio
   - Repetition率
```

---

## 7. 核心要点总结

```
RLHF = 让语言模型学会"人类想要什么"

┌─────────────────────────────────────────────────────────────┐
│  训练三阶段                                                   │
│  1. SFT：有监督微调，让模型学会回答问题                      │
│  2. RM：训练奖励模型，学习评估质量                           │
│  3. PPO：用奖励模型优化策略                                  │
├─────────────────────────────────────────────────────────────┤
│  核心目标                                                     │
│  L = E[r] - β * KL(π || π_ref) + γ * H(π)                  │
│  └── 最大化奖励  └── 约束偏离       └── 鼓励探索             │
├─────────────────────────────────────────────────────────────┤
│  关键组件                                                     │
│  - 奖励模型：学习人类偏好                                    │
│  - PPO：稳定策略更新                                         │
│  - KL惩罚：防止灾难性遗忘                                    │
├─────────────────────────────────────────────────────────────┤
│  进阶方法                                                     │
│  - DPO：绕过RM，直接用偏好优化                               │
│  - PPO-ptx：加入预训练损失防止遗忘                           │
│  - 迭代式RLHF：多轮迭代逐步改进                              │
├─────────────────────────────────────────────────────────────┤
│  训练技巧                                                     │
│  ✓ 从高质量SFT开始                                           │
│  ✓ 奖励模型要足够大且训练充分                                │
│  ✓ KL惩罚要适中（0.1-0.2）                                   │
│  ✓ 监控通用能力，防止灾难性遗忘                              │
├─────────────────────────────────────────────────────────────┤
│  应用                                                         │
│  - ChatGPT, Claude, GPT-4                                    │
│  - 开源模型：InstructGPT, LLaMA2-chat                        │
│  - 对话系统、代码助手、写作辅助                              │
└─────────────────────────────────────────────────────────────┘

下一步：27-Diffusion：从噪声中创造
```

---

## 8. 延伸阅读

- **论文**：
  - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022) - InstructGPT/RLHF原论文
  - [Fine-tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (2019) - RLHF早期工作
  - [Direct Preference Optimization: Your Language Model is a Reward Model](https://arxiv.org/abs/2305.18290) (2023) - DPO论文
  - [Kahneman-Tversky Optimization](https://arxiv.org/abs/2306.17476) (2023) - KTO论文

- **代码与工具**：
  - [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) - Hugging Face的RLHF库
  - [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed) - 微软的高效RLHF训练
  - [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) - 开源RLHF数据

- **博客与教程**：
  - [Hugging Face RLHF Guide](https://huggingface.co/blog/rlhf)
  - [Chip Huyen的RLHF介绍](https://huyenchip.com/2023/05/02/rlhf.html)
  - [LLaMA2论文解读](https://ai.meta.com/llama/)

- **上一章**：[25-LoRA：参数高效微调](25-LoRA：参数高效微调.md)
- **下一章**：[27-Diffusion：从噪声中创造](27-Diffusion：从噪声中创造.md)