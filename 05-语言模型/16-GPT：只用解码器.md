# 16-GPT：只用解码器

## 一句话理解

**GPT 把 Transformer 解码器堆叠到极致，用"预测下一个词"这一个任务学会了所有任务。**

---

## 1. 直觉解释

### GPT 是什么？

**G**enerative **P**re-trained **T**ransformer

生成式预训练 Transformer：

- **Generative**：能生成文本
- **Pre-trained**：在大规模文本上预训练
- **Transformer**：基于 Transformer 架构

### 为什么"只用解码器"？

编码器-解码器适合：输入→输出（翻译、摘要）

只用解码器适合：输入→延续（生成、对话）

```
编码器-解码器：
输入: "翻译这段话"
输出: "Translate this paragraph"

只用解码器：
输入: "从前有座山，"
输出: "山里有座庙，庙里有个老和尚..."
```

---

## 2. GPT 架构

### 2.1 整体结构

```
输入文本
    ↓
[Token Embedding + Position Embedding]
    ↓
┌─────────────────────────────────────┐
│  Transformer Decoder Block × N      │
│  ┌───────────────────────────────┐  │
│  │ Masked Multi-Head Attention   │  │
│  │ + Add & Norm                  │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │ Feed Forward Network          │  │
│  │ + Add & Norm                  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    ↓
[Layer Norm]
    ↓
[Linear → Softmax]
    ↓
下一个词的概率分布
```

### 2.2 核心组件

```python
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm: LayerNorm 在子层之前
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=1024):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.head.weight = self.token_emb.weight
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        
        # 嵌入
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb
        
        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=input_ids.device)).view(1, 1, T, T)
        
        # Transformer 块
        for block in self.blocks:
            x = block(x, mask)
        
        # 输出
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
```

### 2.3 与 BERT 的区别

| 特性 | GPT | BERT |
|------|-----|------|
| 架构 | Decoder-only | Encoder-only |
| 注意力 | 单向（因果） | 双向 |
| 预训练 | LM（预测下一个） | MLM（预测中间） |
| 任务 | 生成 | 理解 |
| 下游 | 微调或 Prompt | 微调 |

---

## 3. GPT-1：预训练+微调

### 3.1 论文信息

**Improving Language Understanding by Generative Pre-Training**

时间：2018 年

核心思想：先无监督预训练，再有监督微调

### 3.2 两阶段训练

**阶段一：无监督预训练**

```python
# 在大规模文本上学习语言规律
for batch in unlabeled_data:
    input_ids, targets = batch
    logits = model(input_ids)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
```

**阶段二：有监督微调**

```python
# 在特定任务上微调
for batch in labeled_data:
    input_ids, labels = batch
    features = model.extract_features(input_ids)
    logits = classifier(features)
    loss = task_loss(logits, labels)
    loss.backward()
    optimizer.step()
```

### 3.3 任务适配

不同任务如何适配？

```
分类：
[Start] 文本 [Extract] → 分类器 → 类别

相似度：
[Start] 文本1 [Delim] 文本2 [Extract] → 分类器 → 相似度

问答：
[Start] 上下文 [Delim] 问题 [Delim] 答案 → 分类器 → 得分
```

### 3.4 GPT-1 规格

| 参数 | 值 |
|------|-----|
| 层数 | 12 |
| d_model | 768 |
| 注意力头 | 12 |
| 参数量 | 117M |
| 训练数据 | BookCorpus |

---

## 4. GPT-2：零样本学习

### 4.1 论文信息

**Language Models are Unsupervised Multitask Learners**

时间：2019 年

核心思想：模型足够大，就能零样本完成任务

### 4.2 零样本学习

不需要微调，直接用 Prompt：

```python
# 翻译
prompt = "Translate to French: Hello, world"
output = gpt2.generate(prompt)

# 问答
prompt = "Q: What is the capital of France? A:"
output = gpt2.generate(prompt)

# 摘要
prompt = "Article: [长文本] TL;DR:"
output = gpt2.generate(prompt)
```

### 4.3 任务即格式

所有任务都变成文本生成：

```
翻译任务：
输入: "Translate English to French: Hello"
输出: "Bonjour"

分类任务：
输入: "Review: This movie is great. Sentiment:"
输出: "Positive"

问答任务：
输入: "Q: What is 2+2? A:"
输出: "4"
```

### 4.4 GPT-2 规格

| 版本 | 参数量 | 层数 | d_model | 头数 |
|------|--------|------|---------|------|
| Small | 117M | 12 | 768 | 12 |
| Medium | 345M | 24 | 1024 | 16 |
| Large | 762M | 36 | 1280 | 20 |
| XL | 1.5B | 48 | 1600 | 25 |

训练数据：WebText（800万网页）

---

## 5. GPT-3：少样本学习

### 5.1 论文信息

**Language Models are Few-Shot Learners**

时间：2020 年

核心思想：规模带来质变，1750 亿参数的模型能从少量示例中学习

### 5.2 三种学习范式

**Zero-shot**：不给示例

```
输入: "Translate to French: Hello"
输出: "Bonjour"
```

**One-shot**：给一个示例

```
输入: "Translate to French: Hello → Bonjour
      Translate to French: Goodbye →"
输出: "Au revoir"
```

**Few-shot**：给多个示例

```
输入: "Translate to French:
      Hello → Bonjour
      Goodbye → Au revoir
      Thanks → Merci
      Please →"
输出: "S'il vous plaît"
```

### 5.3 In-Context Learning

**上下文学习**：不需要梯度更新，在上下文中学习

```python
def in_context_learning(model, task_examples, query):
    prompt = ""
    
    # 添加示例
    for example in task_examples:
        prompt += f"{example.input} → {example.output}\n"
    
    # 添加查询
    prompt += f"{query} →"
    
    # 生成答案
    answer = model.generate(prompt)
    return answer
```

特点：
- 不更新参数
- 推理时学习
- 灵活适应任务

### 5.4 GPT-3 规格

| 参数 | 值 |
|------|-----|
| 层数 | 96 |
| d_model | 12288 |
| 注意力头 | 96 |
| 参数量 | 175B |
| 训练数据 | 300B tokens |

### 5.5 规模定律（Scaling Laws）

参数量、数据量、计算量的关系：

```
Loss ∝ N^{-0.076}  (N: 参数量)
Loss ∝ D^{-0.095}  (D: 数据量)
Loss ∝ C^{-0.050}  (C: 计算量)
```

结论：
- 更大的模型效果更好
- 还没有看到饱和
- 投入更多计算能持续改进

---

## 6. 涌现能力

### 6.1 什么是涌现？

**涌现**：模型规模达到某个阈值后，突然展现出的新能力。

```
小模型：不能完成复杂任务
中等模型：开始表现
大模型：突然变得很好
```

### 6.2 典型涌现能力

**思维链（Chain-of-Thought）**

```
问题：Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在有多少网球？

普通回答（错误）：11

思维链回答：
Roger 开始有 5 个网球。
买了 2 罐，每罐 3 个，所以买了 2×3=6 个。
总共 5+6=11 个网球。
答案：11
```

**代码生成**

```
输入: "写一个 Python 函数计算斐波那契数列"
输出: 完整可运行的代码
```

**数学推理**

```
输入: "如果 x+5=12，求 x"
输出: "x = 12-5 = 7"
```

**指令遵循**

```
输入: "用三句话解释量子力学，每句不超过10个字"
输出: 
"微观粒子具有波粒二象性。
叠加态可同时处于多种状态。
观测导致波函数坍缩。"
```

### 6.3 涌现的阈值

| 能力 | 涌现规模 |
|------|----------|
| 简单推理 | ~1B |
| 上下文学习 | ~10B |
| 思维链 | ~100B |
| 复杂推理 | ~500B+ |

### 6.4 为什么会涌现？

假设：
- 规模带来组合能力
- 学会了"如何学习"
- 模式匹配能力增强

---

## 7. GPT 系列演进总结

### 7.1 规模对比

| 模型 | 参数量 | 训练数据 | 年份 |
|------|--------|----------|------|
| GPT-1 | 117M | BookCorpus | 2018 |
| GPT-2 | 1.5B | WebText | 2019 |
| GPT-3 | 175B | 300B tokens | 2020 |
| GPT-3.5 | 未公开 | 未公开 | 2022 |
| GPT-4 | 未公开 | 未公开 | 2023 |

### 7.2 能力演进

```
GPT-1: 预训练+微调
  └─ 证明了预训练的价值

GPT-2: 零样本学习
  └─ 任务即文本生成

GPT-3: 少样本学习
  └─ In-Context Learning

GPT-3.5/4: 涌现能力
  └─ 思维链、指令遵循、多模态
```

### 7.3 关键创新

| 模型 | 关键创新 |
|------|----------|
| GPT-1 | 生成式预训练 |
| GPT-2 | 零样本学习 |
| GPT-3 | In-Context Learning |
| GPT-4 | 涌现能力、多模态 |

---

## 8. 训练技巧

### 8.1 数据质量

GPT 系列的训练数据越来越"干净"：

- GPT-1: BookCorpus
- GPT-2: WebText（过滤后）
- GPT-3: 高质量混合数据

```python
def filter_data(texts):
    filtered = []
    for text in texts:
        # 过滤太短的
        if len(text) < 100:
            continue
        # 过滤重复的
        if is_duplicate(text):
            continue
        # 过滤低质量的
        if not is_high_quality(text):
            continue
        filtered.append(text)
    return filtered
```

### 8.2 训练稳定性

大模型训练容易不稳定，需要：

```python
# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 学习率预热
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2000,
    num_training_steps=total_steps
)

# 权重初始化
def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
```

### 8.3 高效训练

**混合精度**：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**梯度累积**：

```python
accumulation_steps = 8

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. GPT 的局限

### 9.1 幻觉（Hallucination）

生成不真实的内容：

```
Q: 谁发明了电话？
A: 爱迪生在 1876 年发明了电话。（错误，应该是贝尔）
```

原因：
- 训练数据有噪声
- 模型记忆不准确
- 缺乏事实检验

### 9.2 上下文长度有限

GPT-3: 2048 tokens
GPT-4: 8K/32K tokens

长文档处理困难。

### 9.3 缺乏真正的理解

模式匹配而非真正理解：

```
Q: 我有3个苹果，吃了2个，还剩几个？
A: 还剩1个。（正确）

Q: 我有3个苹果，吃了2个，买了5个，吃了1个，还剩几个？
A: 可能会算错
```

### 9.4 偏见和有害内容

会放大训练数据中的偏见。

---

## 10. 核心要点总结

```
GPT = Transformer Decoder × N

架构特点：
- 单向注意力（因果掩码）
- Pre-Norm
- 权重共享

演进路径：
GPT-1: 预训练+微调
GPT-2: 零样本学习
GPT-3: 少样本学习、In-Context Learning
GPT-4: 涌现能力、多模态

核心能力：
- 文本生成
- 上下文学习
- 思维链推理
- 指令遵循

规模定律：
更大 = 更好
量变引起质变（涌现）

局限性：
- 幻觉
- 上下文长度
- 缺乏真正理解
- 偏见
```

---

## 11. 延伸阅读

- **论文**：[GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-3](https://arxiv.org/abs/2005.14165)
- **博客**：[OpenAI Blog](https://openai.com/blog/)
- **代码**：[minGPT](https://github.com/karpathy/minGPT)
- **上一章**：[15-语言模型](15-语言模型.md)
- **下一章**：[17-分词与训练准备](17-分词与训练准备.md)
