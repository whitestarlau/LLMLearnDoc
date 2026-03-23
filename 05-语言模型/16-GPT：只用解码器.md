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

### 3.0 GPT 之前的 NLP 范式

在 GPT 出现之前，NLP 领域主要采用**任务特定模型**的范式。让我们通过几个典型任务来理解传统方法的处理方式。

#### 典型 NLP 任务示例

**1. 情感分类（Sentiment Classification）**

```
任务：判断句子的情感倾向
输入："这家餐厅的菜非常好吃，服务也很周到！"
输出：正面 / 负面 / 中性

传统方法流程：
┌─────────────────────────────────────────────────────────┐
│  1. 人工设计特征                                          │
│     - 词袋模型（Bag of Words）                            │
│     - TF-IDF 特征                                        │
│     - 情感词典匹配                                        │
│     - n-gram 特征                                        │
│                                                         │
│  2. 训练分类器                                            │
│     - 逻辑回归（Logistic Regression）                     │
│     - 支持向量机（SVM）                                   │
│     - 随机森林（Random Forest）                           │
│                                                         │
│  3. 在标注数据上训练                                      │
│     - 需要大量人工标注的情感数据集                         │
│     - 每个任务独立训练一个模型                             │
└─────────────────────────────────────────────────────────┘
```

**2. 命名实体识别（Named Entity Recognition）**

```
任务：识别文本中的实体（人名、地名、组织名等）
输入："马云在杭州创办了阿里巴巴"
输出："[马云]人名 在[杭州]地名 创办了[阿里巴巴]组织名"

传统方法流程：
┌─────────────────────────────────────────────────────────┐
│  1. 特征工程                                              │
│     - 词性标注（POS Tagging）                             │
│     - 词边界特征                                          │
│     - 上下文窗口特征                                      │
│     - 词典特征（地名词典、人名词典）                       │
│                                                         │
│  2. 序列标注模型                                          │
│     - 隐马尔可夫模型（HMM）                               │
│     - 条件随机场（CRF）                                   │
│     - 最大熵模型                                          │
│                                                         │
│  3. 需要大量标注数据                                      │
│     - 每个实体类型需要单独标注                             │
│     - 标注成本极高                                        │
└─────────────────────────────────────────────────────────┘
```

**3. 文本分类（Text Classification）**

```
任务：将文本分到预定义的类别
输入："苹果公司发布了新款iPhone手机"
输出：科技 / 财经 / 体育 / 娱乐

传统方法流程：
┌─────────────────────────────────────────────────────────┐
│  1. 文本表示                                              │
│     - 词袋模型                                            │
│     - TF-IDF 向量                                        │
│     - LDA 主题模型                                        │
│                                                         │
│  2. 分类算法                                              │
│     - 朴素贝叶斯（Naive Bayes）                           │
│     - SVM                                                │
│     - 神经网络（浅层）                                    │
│                                                         │
│  3. 问题                                                  │
│     - 特征需要人工设计                                    │
│     - 无法捕捉语义相似性                                  │
│     - 每个领域需要重新训练                                │
└─────────────────────────────────────────────────────────┘
```

**4. 机器翻译（Machine Translation）**

```
任务：将一种语言翻译成另一种语言
输入："Hello, how are you?"
输出："你好，你好吗？"

传统方法流程：
┌─────────────────────────────────────────────────────────┐
│  1. 基于规则（Rule-based）                                │
│     - 人工编写语法规则                                    │
│     - 双语词典                                          │
│     - 规则冲突难以解决                                    │
│                                                         │
│  2. 统计机器翻译（SMT）                                   │
│     - 平行语料库（人工翻译的句子对）                       │
│     - 词对齐模型                                          │
│     - 短语翻译表                                          │
│     - 语言模型                                          │
│                                                         │
│  3. 问题                                                  │
│     - 需要大量平行语料                                    │
│     - 特征工程复杂                                        │
│     - 长距离依赖难以处理                                  │
└─────────────────────────────────────────────────────────┘
```

#### 传统 NLP 的核心问题

```
传统 NLP 范式的三大痛点：

┌─────────────────────────────────────────────────────────┐
│  1. 特征工程依赖人工                                      │
│     ───────────────────────────────────────────────────  │
│     • 需要领域专家设计特征                                │
│     • 特征质量直接影响模型性能                            │
│     • 不同任务需要不同的特征                              │
│     • 特征设计耗时耗力                                    │
│                                                         │
│  2. 标注数据需求量大                                      │
│     ───────────────────────────────────────────────────  │
│     • 每个任务需要大量标注数据                            │
│     • 人工标注成本高昂                                    │
│     • 标注质量难以保证                                    │
│     • 小数据集容易过拟合                                  │
│                                                         │
│  3. 任务间知识无法迁移                                    │
│     ───────────────────────────────────────────────────  │
│     • 每个任务独立训练模型                                │
│     • 学到的知识无法复用                                  │
│     • 新任务需要从头开始                                  │
│     • 计算资源浪费严重                                    │
└─────────────────────────────────────────────────────────┘
```

#### 传统方法 vs GPT 范式对比

```
┌─────────────────────────────────────────────────────────┐
│                    传统 NLP 范式                          │
├─────────────────────────────────────────────────────────┤
│  情感分类任务：                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ 原始文本 │ →  │ 特征提取 │ →  │ 分类器  │ → 输出      │
│  └─────────┘    └─────────┘    └─────────┘             │
│       ↓              ↓              ↓                   │
│  "这家餐厅..."  词袋/TF-IDF    SVM/逻辑回归   "正面"      │
│                                                         │
│  特点：                                                  │
│  • 特征需要人工设计                                       │
│  • 每个任务独立训练                                       │
│  • 需要大量标注数据                                       │
│  • 知识无法迁移                                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    GPT 范式                              │
├─────────────────────────────────────────────────────────┤
│  情感分类任务：                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │              预训练 GPT 模型                      │   │
│  │  (已学习通用语言表示)                              │   │
│  └─────────────────────────────────────────────────┘   │
│       ↓                                                 │
│  ┌─────────┐    ┌─────────┐                            │
│  │ 原始文本 │ →  │  GPT    │ → 输出                    │
│  └─────────┘    └─────────┘                            │
│                    ↓                                    │
│              "这家餐厅..." → "正面"                      │
│                                                         │
│  特点：                                                  │
│  • 自动学习特征表示                                       │
│  • 预训练模型可复用                                       │
│  • 只需少量标注数据                                       │
│  • 知识可跨任务迁移                                       │
└─────────────────────────────────────────────────────────┘
```

#### 传统方法的具体例子

**情感分类的传统实现**：

```python
# 传统方法：特征工程 + 分类器
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# 1. 人工设计特征：TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,      # 限制特征数量
    ngram_range=(1, 2),      # 使用 unigram 和 bigram
    stop_words='english'     # 去除停用词
)

# 2. 训练分类器
classifier = SVC(kernel='linear')

# 3. 构建管道
pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', classifier)
])

# 4. 需要大量标注数据训练
pipeline.fit(train_texts, train_labels)

# 问题：
# - 特征需要人工设计
# - 无法理解语义（"好吃"和"美味"被视为不同特征）
# - 每个任务需要独立训练
```

**GPT 的实现**：

```python
# GPT 方法：预训练 + 微调
# 1. 预训练阶段（只需一次）
# 在海量无标注文本上训练语言模型
# 学习通用的语言表示

# 2. 微调阶段（每个任务）
# 添加任务特定的输出层
model = GPTWithClassifier(pretrained_gpt, num_classes=2)

# 3. 只需少量标注数据微调
for batch in small_labeled_data:
    outputs = model(batch.input_ids)
    loss = cross_entropy(outputs, batch.labels)
    loss.backward()
    optimizer.step()

# 优势：
# - 自动学习特征
# - 理解语义相似性
# - 预训练模型可复用
```

#### 为什么传统方法难以扩展？

```
传统方法的扩展性问题：

任务1：情感分类
├── 特征工程：情感词典 + n-gram
├── 标注数据：10,000 条
└── 训练时间：2 小时

任务2：新闻分类
├── 特征工程：主题词 + TF-IDF
├── 标注数据：15,000 条
└── 训练时间：3 小时

任务3：垃圾邮件检测
├── 特征工程：关键词 + 统计特征
├── 标注数据：20,000 条
└── 训练时间：4 小时

总计：
• 特征工程：3 套不同的特征系统
• 标注数据：45,000 条
• 训练时间：9 小时
• 知识迁移：无

GPT 方法：
• 预训练：1 次（使用海量无标注数据）
• 微调：每个任务只需少量数据
• 知识迁移：自动实现
```

### 3.1 论文信息

**Improving Language Understanding by Generative Pre-Training**

时间：2018 年

核心思想：先无监督预训练，再有监督微调

### 3.2 GPT-1 的设计动机

**GPT-1 最初设计的目的**：解决 NLP 领域的一个核心矛盾——**标注数据稀缺 vs 语言知识丰富**。

#### 问题背景

在 GPT-1 之前，NLP 模型面临两大挑战：

1. **标注数据极其稀缺**
   - 高质量的人工标注成本高昂
   - 不同任务需要不同的标注格式
   - 小数据集容易过拟合

2. **语言知识分散在海量文本中**
   - 互联网上有海量无标注文本
   - 这些文本蕴含丰富的语言规律、世界知识、推理模式
   - 但传统方法无法有效利用

#### GPT-1 的解决方案

```
┌─────────────────────────────────────────────────────────┐
│                    GPT-1 的两阶段范式                      │
├─────────────────────────────────────────────────────────┤
│  阶段一：无监督预训练（Unsupervised Pre-training）         │
│  ─────────────────────────────────────────────────────  │
│  目标：从海量无标注文本中学习通用语言表示                    │
│  数据：BookCorpus（~8000 本书，~5GB 文本）                 │
│  任务：语言建模（预测下一个词）                            │
│  效果：模型学会语法、语义、常识、推理等通用能力              │
│                                                         │
│  阶段二：有监督微调（Supervised Fine-tuning）              │
│  ─────────────────────────────────────────────────────  │
│  目标：将通用语言能力适配到具体任务                         │
│  数据：少量标注数据（几百到几千条）                        │
│  任务：分类、相似度、问答等                                │
│  效果：在具体任务上达到 SOTA 性能                          │
└─────────────────────────────────────────────────────────┘
```

#### 为什么需要无监督预训练？

**核心原因：语言知识的层次性与可迁移性**

```
语言知识的层次结构：
┌─────────────────────────────────────┐
│  表层：词汇、语法、句法              │  ← 预训练学习
├─────────────────────────────────────┤
│  中层：语义、指代、逻辑关系          │  ← 预训练学习
├─────────────────────────────────────┤
│  深层：常识、世界知识、推理模式      │  ← 预训练学习
├─────────────────────────────────────┤
│  任务层：具体任务的决策边界          │  ← 微调学习
└─────────────────────────────────────┘
```

**无监督预训练的优势**：

1. **数据无限**：互联网文本几乎无限，无需人工标注
2. **知识通用**：学到的语言表示可迁移到各种下游任务
3. **表示丰富**：模型内部形成了多层次的语言理解能力
4. **成本低廉**：只需一次预训练，多次微调复用

**"预测下一个词"为什么有效？**

```python
# 预测下一个词需要理解：
input = "今天天气很好，我决定去"

# 模型需要知道：
# 1. 语法："决定去"后面通常接地点或活动
# 2. 常识：天气好时人们喜欢户外活动
# 3. 世界知识：常见的户外活动有哪些
# 4. 推理：根据上下文推断最可能的续写

# 预测下一个词 = 压缩所有相关知识
```

#### 为什么需要有监督微调？

**核心原因：任务特异性与泛化能力的平衡**

```
预训练模型的能力分布：
┌────────────────────────────────────────┐
│  通用语言能力 ████████████████ 80%     │  ← 预训练获得
│  任务特定能力 ████ 20%                 │  ← 微调获得
└────────────────────────────────────────┘
```

**有监督微调的必要性**：

1. **任务对齐**：将通用表示映射到具体任务的输出空间
2. **决策边界**：学习任务特定的分类/生成边界
3. **少样本学习**：用少量标注数据激活预训练中的相关知识
4. **性能提升**：在具体任务上超越从头训练的模型

**微调的效率优势**：

```
从头训练 vs 预训练+微调：

从头训练：
- 需要大量标注数据（10万+）
- 训练时间长（几天到几周）
- 容易过拟合
- 每个任务独立训练

预训练+微调：
- 只需少量标注数据（几百到几千）
- 训练时间短（几小时到几天）
- 泛化能力强
- 多任务共享预训练模型
```

#### 两阶段的协同效应

```
无监督预训练 ──────────────────────────────────────┐
    │                                              │
    │  学到：语法、语义、常识、推理                  │
    │  形式：通用的语言表示空间                      │
    │                                              │
    ▼                                              │
┌─────────────────┐                                │
│  预训练模型      │◄───────────────────────────────┘
│  (通用语言能力)  │
└─────────────────┘
    │
    │  微调：用少量标注数据调整
    │  方式：添加任务头 + 端到端训练
    │
    ▼
┌─────────────────┐
│  任务特定模型    │
│  (分类/问答/...) │
└─────────────────┘
```

#### GPT-1 的历史意义

GPT-1 验证了一个重要假设：

> **"无监督预训练 + 有监督微调" 是 NLP 的正确范式**

这一范式后来成为：
- BERT（2018）：双向预训练
- GPT-2（2019）：零样本学习
- GPT-3（2020）：少样本学习
- GPT-4（2023）：多模态大模型

的核心基础。

### 3.3 两阶段训练

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

## 7. 压缩即智能：语言模型的本质

### 7.1 Ilya Sutskever 的洞见

**Ilya Sutskever**（OpenAI 联合创始人、前首席科学家）提出了一个深刻的观点：

> **"压缩即智能"（Compression is Intelligence）**

这个观点的核心是：**为了准确预测下一个词，模型必须理解世界的结构**。

### 7.2 为什么预测下一个词需要理解世界？

考虑这个例子：

```
输入："今天天气很好，我决定去"
```

为了准确预测下一个词，模型需要知道：
- 天气好时人们通常做什么（常识）
- "决定去"后面通常接地点（语法）
- 常见的户外活动有哪些（世界知识）
- 上下文的逻辑关系（推理）

**预测下一个词 = 压缩所有相关知识**

### 7.3 压缩的层次

语言模型在多个层次上进行压缩：

| 层次 | 压缩内容 | 例子 |
|------|----------|------|
| 语法 | 语言规则 | 主谓宾结构 |
| 语义 | 词义关系 | 同义词、反义词 |
| 常识 | 世界规律 | 重力、因果关系 |
| 推理 | 逻辑关系 | 如果...那么... |
| 元知识 | 如何学习 | 上下文学习 |

### 7.4 压缩与智能的关系

**传统观点**：智能需要专门设计的推理模块

**压缩观点**：智能是压缩的副产品

```
更好的压缩 → 更好的预测 → 更强的智能
```

这解释了为什么：
- 模型越大，压缩能力越强
- 压缩能力越强，涌现能力越多
- 预测下一个词能学会翻译、编程、推理

### 7.5 信息论视角

从信息论角度看：

```
H(X) = -Σ P(x) log P(x)  # 熵
H(X|Y) = -Σ P(x,y) log P(x|y)  # 条件熵
I(X;Y) = H(X) - H(X|Y)  # 互信息
```

**语言模型的目标**：最小化条件熵 H(下一个词|上下文)

**这意味着**：模型必须最大化上下文与下一个词之间的互信息

### 7.6 压缩的极限

**Kolmogorov 复杂度**：描述一个对象所需的最短程序长度

```
K(x) = min{|p| : U(p) = x}
```

语言模型在逼近这个极限：
- 更好的压缩 = 更短的描述 = 更深的理解

### 7.7 实际意义

**为什么这个观点重要？**

1. **解释了规模定律**：更大模型能压缩更多信息
2. **解释了涌现能力**：压缩达到阈值后出现新能力
3. **指导了训练方向**：追求更好的压缩
4. **预测了未来**：压缩能力决定智能上限

**对开发者的启示**：

```python
# 好的训练策略应该追求更好的压缩
def train_for_compression(model, data):
    # 使用更长的上下文
    # 使用更多样化的数据
    # 使用更难的预测任务
    # 这些都能提升压缩能力
    pass
```

---

## 8. GPT 系列演进总结

### 8.1 规模对比

| 模型 | 参数量 | 训练数据 | 年份 |
|------|--------|----------|------|
| GPT-1 | 117M | BookCorpus | 2018 |
| GPT-2 | 1.5B | WebText | 2019 |
| GPT-3 | 175B | 300B tokens | 2020 |
| GPT-3.5 | 未公开 | 未公开 | 2022 |
| GPT-4 | 未公开 | 未公开 | 2023 |

### 8.2 能力演进

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

### 8.3 关键创新

| 模型 | 关键创新 |
|------|----------|
| GPT-1 | 生成式预训练 |
| GPT-2 | 零样本学习 |
| GPT-3 | In-Context Learning |
| GPT-4 | 涌现能力、多模态 |

---

## 9. 训练技巧

### 9.1 数据质量

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

### 9.2 训练稳定性

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

### 9.3 高效训练

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

## 10. GPT 的局限

### 10.1 幻觉（Hallucination）

生成不真实的内容：

```
Q: 谁发明了电话？
A: 爱迪生在 1876 年发明了电话。（错误，应该是贝尔）
```

原因：
- 训练数据有噪声
- 模型记忆不准确
- 缺乏事实检验

### 10.2 上下文长度有限

GPT-3: 2048 tokens
GPT-4: 8K/32K tokens

长文档处理困难。

### 10.3 缺乏真正的理解

模式匹配而非真正理解：

```
Q: 我有3个苹果，吃了2个，还剩几个？
A: 还剩1个。（正确）

Q: 我有3个苹果，吃了2个，买了5个，吃了1个，还剩几个？
A: 可能会算错
```

### 10.4 偏见和有害内容

会放大训练数据中的偏见。

---

## 11. 核心要点总结

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

## 12. 延伸阅读

- **论文**：[GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-3](https://arxiv.org/abs/2005.14165)
- **博客**：[OpenAI Blog](https://openai.com/blog/)
- **代码**：[minGPT](https://github.com/karpathy/minGPT)
- **上一章**：[15-语言模型](15-语言模型.md)
- **下一章**：[17-分词与训练准备](17-分词与训练准备.md)
