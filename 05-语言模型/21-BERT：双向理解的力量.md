# 21-BERT：双向理解的力量

我们已经深入学习了 GPT 系列——从架构到训练范式，再到涌现能力。现在，让我们看看另一条同样重要的技术路线：**BERT**。

2018年，就在 GPT-1 发布的几个月后，Google 推出了 BERT。它采用了与 GPT 截然不同的策略：**双向注意力 + 编码器架构**。这一选择让 BERT 在理解任务上大放异彩，成为当时 NLP 领域的新标杆。

如果说 GPT 擅长"写作"，那 BERT 则擅长"阅读"。理解这两种范式的差异，对于把握现代 NLP 的全貌至关重要。

---

## 1. BERT 是什么？

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers

双向编码器表示 Transformer：
- **Bidirectional**：同时看左边和右边
- **Encoder**：只用编码器
- **Representations**：学习文本表示

### GPT vs BERT

```
GPT（单向）：
我 喜欢 吃 苹果
↓  ↓   ↓  ↓
只看左边预测右边

BERT（双向）：
我 喜欢 [MASK] 苹果
↓  ↓     ↓    ↓
同时看左右预测中间
```

类比：
- GPT 像写作文：一个字一个字往后写
- BERT 像做填空题：看前后文填空

### BERT 能做什么？

- **文本分类**：情感分析、主题分类
- **命名实体识别**：识别姓名、地点
- **问答系统**：从文章中找答案
- **语义相似度**：判断两句是否相关

---

## 2. BERT 架构

### 2.1 整体结构

```
输入文本
    ↓
[CLS] 词1 词2 ... 词N [SEP]
    ↓
[Token Embedding + Segment Embedding + Position Embedding]
    ↓
┌─────────────────────────────────────┐
│  Transformer Encoder Block × N      │
│  ┌───────────────────────────────┐  │
│  │ Multi-Head Attention          │  │
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
每个 token 的上下文表示
```

### 2.2 输入表示

BERT 的输入是三个 embedding 的和：

```python
class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size):
        super().__init__()
        
        # Token Embedding
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Position Embedding
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Segment Embedding（区分句子 A 和 B）
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        # Token Embedding
        word_emb = self.word_embeddings(input_ids)
        
        # Position Embedding
        position_ids = torch.arange(seq_length, device=input_ids.device)
        position_emb = self.position_embeddings(position_ids)
        
        # Segment Embedding
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        # 求和
        embeddings = word_emb + position_emb + token_type_emb
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

### 2.3 特殊标记

```
[CLS]  - 句子开头，用于分类任务
[SEP]  - 句子分隔
[MASK] - 遮盖标记，用于预训练
[UNK]  - 未知词
[PAD]  - 填充
```

示例：

```
单句：[CLS] 我喜欢自然语言处理 [SEP]
双句：[CLS] 我喜欢苹果 [SEP] 我也喜欢 [SEP]
```

---

## 3. 预训练任务

### 3.1 掩码语言模型（MLM）

**Masked Language Model**：随机遮盖一些词，预测这些词

```
输入：我 喜欢 [MASK] 语言 处理
目标：自然

输入：我 [MASK] 自然语言处理
目标：喜欢
```

#### 遮盖策略

```python
def mask_tokens(tokens, tokenizer, mlm_probability=0.15):
    """
    15% 的 token 被选中处理：
    - 80% 替换为 [MASK]
    - 10% 替换为随机词
    - 10% 保持不变
    """
    labels = tokens.clone()
    
    # 随机选择 15% 的位置
    probability_matrix = torch.full(tokens.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
        for val in tokens.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 不计算损失的位置
    
    # 80% 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full(tokens.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10% 替换为随机词
    indices_random = torch.bernoulli(torch.full(tokens.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), tokens.shape, dtype=torch.long)
    tokens[indices_random] = random_words[indices_random]
    
    # 10% 保持不变（已经隐含在上述逻辑中）
    
    return tokens, labels
```

#### 为什么要这样设计？

```
原始：我 喜欢 自然 语言 处理

80% [MASK]：我 喜欢 [MASK] 语言 处理
      → 避免 [MASK] 只在预训练时出现

10% 随机：我 喜欢 苹果 语言 处理
      → 强迫模型利用上下文

10% 不变：我 喜欢 自然 语言 处理
      → 让模型学习真实分布
```

### 3.2 下一句预测（NSP）

**Next Sentence Prediction**：判断两个句子是否连续

```
正例：
句子A：我喜欢自然语言处理
句子B：它是一个有趣的领域
标签：IsNext

负例：
句子A：我喜欢自然语言处理
句子B：今天天气很不错
标签：NotNext
```

```python
def create_nsp_examples(sentences):
    examples = []
    for i in range(len(sentences) - 1):
        # 50% 选择下一句
        if random.random() < 0.5:
            sentence_a = sentences[i]
            sentence_b = sentences[i + 1]
            label = 1  # IsNext
        # 50% 随机选择其他句子
        else:
            sentence_a = sentences[i]
            sentence_b = sentences[random.randint(0, len(sentences) - 1)]
            label = 0  # NotNext
        
        examples.append((sentence_a, sentence_b, label))
    
    return examples
```

输入格式：

```
[CLS] 句子A [SEP] 句子B [SEP]
      ↑________________↑
         Segment=0      Segment=1
```

### 3.3 为什么需要两个任务？

```
MLM：学习词层面的理解
    - 根据上下文推断词
    - 学习词汇语义

NSP：学习句层面的理解
    - 判断句子关系
    - 学习篇章结构
```

---

## 4. BERT 模型实现

### 4.1 BERT 编码器

```python
class BertEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, 
                 num_hidden_layers, dropout=0.1):
        super().__init__()
        
        self.layer = nn.ModuleList([
            BertLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
```

### 4.2 注意力层

```python
class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        
        self.self = BertSelfAttention(hidden_size, num_attention_heads, dropout)
        self.output = BertSelfOutput(hidden_size, dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout=0.1):
        super().__init__()
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # (B, H, T, D)
    
    def forward(self, hidden_states, attention_mask=None):
        # Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 注意力掩码（处理 padding）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        return context_layer
```

### 4.3 完整 BERT 模型

```python
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embeddings = BertEmbeddings(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size
        )
        
        self.encoder = BertEncoder(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers
        )
        
        self.pooler = BertPooler(config.hidden_size)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Embeddings
        embeddings = self.embeddings(input_ids, token_type_ids)
        
        # 扩展 attention mask
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Encoder
        encoder_output = self.encoder(embeddings, extended_attention_mask)
        
        # Pooler（取 [CLS] 的表示）
        pooled_output = self.pooler(encoder_output)
        
        return encoder_output, pooled_output

class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # 取 [CLS] token（第一个）的表示
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

---

## 5. 预训练 BERT

### 5.1 预训练模型

```python
class BertForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.bert = BertModel(config)
        
        # MLM 头
        self.cls = BertOnlyMLMHead(config)
        
        # NSP 头
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                masked_lm_labels=None, next_sentence_label=None):
        
        # BERT 编码
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask, token_type_ids
        )
        
        # MLM 预测
        prediction_scores = self.cls(sequence_output)
        
        # NSP 预测
        seq_relationship_score = self.seq_relationship(pooled_output)
        
        # 计算损失
        total_loss = None
        if masked_lm_labels is not None and next_sentence_label is not None:
            # MLM 损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.bert.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            
            # NSP 损失
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1)
            )
            
            total_loss = masked_lm_loss + next_sentence_loss
        
        return total_loss, prediction_scores, seq_relationship_score
```

### 5.2 训练循环

```python
def pretrain_bert(model, train_dataloader, optimizer, scheduler, epochs, device):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            masked_lm_labels = batch['masked_lm_labels'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            
            # 前向传播
            loss, _, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                masked_lm_labels=masked_lm_labels,
                next_sentence_label=next_sentence_label
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Average Loss {avg_loss:.4f}")
    
    return model
```

---

## 6. 微调范式

### 6.1 预训练-微调范式

```
预训练阶段：
大规模无标注数据 → 学习通用语言表示

微调阶段：
小规模有标注数据 → 适应下游任务
```

```python
# 预训练：学习通用表示
pretrained_model = BertForPreTraining(config)
pretrained_model = pretrain(pretrained_model, large_corpus)

# 微调：适应特定任务
task_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
task_model = finetune(task_model, task_dataset)
```

### 6.2 文本分类

```python
class BertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT 编码
        _, pooled_output = self.bert(input_ids, attention_mask, token_type_ids)
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return loss, logits

# 使用示例
model = BertForSequenceClassification(config, num_labels=2)

# 输入
input_ids = tokenizer.encode("This movie is great!", return_tensors='pt')
loss, logits = model(input_ids, labels=torch.tensor([1]))  # 1: positive
```

### 6.3 命名实体识别

```python
class BertForTokenClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # BERT 编码
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        # 每个位置的分类
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        
        return loss, logits
```

### 6.4 问答系统

```python
class BertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start + end
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):
        # BERT 编码
        sequence_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        # 预测开始和结束位置
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        # 计算损失
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        
        return total_loss, start_logits, end_logits
```

### 6.5 微调策略

```python
def finetune(model, train_dataloader, num_epochs, learning_rate, device):
    model.to(device)
    
    # 优化器（不同层使用不同学习率）
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # 训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs[0]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss {total_loss / len(train_dataloader):.4f}")
    
    return model
```

---

## 7. BERT 变体

### 7.1 RoBERTa

**改进**：
- 更大训练数据
- 更长训练时间
- 去掉 NSP 任务
- 更大 batch size
- 动态掩码（每次训练不同遮盖）

```python
# 静态掩码 vs 动态掩码
# 静态：预处理时固定掩码
# 动态：每次前向传播时随机掩码
def dynamic_masking(tokens, mlm_probability=0.15):
    return mask_tokens(tokens, mlm_probability)  # 每次调用都不同
```

### 7.2 ALBERT

**改进**：
- 参数共享（所有层共享参数）
- 因式分解嵌入

```python
# 因式分解嵌入
# 原始：vocab_size × hidden_size
# 改进：vocab_size × embedding_size + embedding_size × hidden_size

class AlbertEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embedding_hidden_mapping = nn.Linear(embedding_size, hidden_size)
    
    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.embedding_hidden_mapping(embeddings)
        return embeddings
```

### 7.3 DistilBERT

**改进**：
- 知识蒸馏
- 模型更小更快

```python
# 蒸馏损失
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    # 软标签
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL 散度
    loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return loss
```

### 7.4 对比

| 模型 | 改进 | 特点 |
|------|------|------|
| BERT | 基础版本 | 双向理解 |
| RoBERTa | 更大更强 | 去掉 NSP、动态掩码 |
| ALBERT | 参数共享 | 更小更快 |
| DistilBERT | 知识蒸馏 | 轻量级 |

---

## 8. 实践建议

### 8.1 模型选择

```python
# 英文
'bert-base-uncased'    # 12层, 110M参数
'bert-large-uncased'   # 24层, 340M参数

# 中文
'bert-base-chinese'    # 中文预训练

# 多语言
'bert-base-multilingual-cased'  # 104种语言
```

### 8.2 微调技巧

```python
# 1. 学习率
# 推荐值：2e-5 到 5e-5

# 2. Epochs
# 推荐：2-4 个 epoch（避免过拟合）

# 3. Batch Size
# 推荐：16 或 32

# 4. 最大序列长度
# 推荐：根据任务，通常 128 或 512

# 5. 梯度累积（显存不够时）
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(**batch)
    (loss / accumulation_steps).backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 8.3 常见问题

**Q1: 微调效果差？**

检查：
- 学习率是否合适
- 数据是否足够
- 是否过拟合（减少 epoch）
- 预处理是否正确

**Q2: 显存不够？**

解决：
- 减小 batch size
- 使用梯度累积
- 使用更小的模型
- 使用梯度检查点

**Q3: 推理速度慢？**

解决：
- 使用 DistilBERT
- 模型量化
- ONNX 导出
- 批量推理

```python
# 模型量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

---

## 9. 核心要点总结

```
BERT 核心思想：
- 双向注意力：同时看左右上下文
- 预训练：在大规模数据上学习通用表示
- 微调：适应下游任务

预训练任务：
- MLM（掩码语言模型）：预测被遮盖的词
- NSP（下一句预测）：判断两句是否连续

预训练-微调范式：
1. 预训练：大规模无标注数据
2. 微调：小规模有标注数据

BERT 变体：
- RoBERTa：更大更强
- ALBERT：参数共享，更小
- DistilBERT：知识蒸馏，更快

应用场景：
- 文本分类
- 命名实体识别
- 问答系统
- 语义相似度
```

---

## 10. 延伸阅读

- **论文**：[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **论文**：[RoBERTa: A Robustly Optimized BERT](https://arxiv.org/abs/1907.11692)
- **代码**：[Hugging Face Transformers](https://github.com/huggingface/transformers)
- **博客**：[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)
- **上一章**：[20-从零训练GPT](20-从零训练GPT.md)
- **下一章**：[20-采样策略与生成质量](../06-进阶专题/20-采样策略与生成质量.md)
