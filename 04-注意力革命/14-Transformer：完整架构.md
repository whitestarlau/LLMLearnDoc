# 14-Transformer：完整架构

## 一句话理解

**把前面学的所有组件——多头注意力、位置编码、残差连接、层归一化——组装成一台完整的机器。**

---

## 1. 直觉解释

### 变形金刚？不是

Transformer 不是机器人，而是一种**序列建模架构**。

名字的含义：它能把一种序列"变换"成另一种序列：
- 中文 → 英文
- 问题 → 答案
- 文本摘要 → 原文

### 为什么 Transformer 如此重要？

它是现代大模型的**地基**：
- GPT 系列
- BERT
- LLaMA
- Claude
- ...

几乎所有现代语言模型都基于 Transformer。

---

## 2. 整体架构

### 2.1 编码器-解码器结构

原始 Transformer 用于机器翻译，采用编码器-解码器设计：

```
输入序列 → [Encoder] → 编码向量 → [Decoder] → 输出序列
```

类比：
- **编码器**：听懂问题（理解输入）
- **解码器**：给出回答（生成输出）

### 2.2 完整架构图

```
Encoder (×N 层)                    Decoder (×N 层)
┌─────────────────┐               ┌─────────────────────────────┐
│ 输入嵌入        │               │ 输出嵌入（右移）            │
│ + 位置编码      │               │ + 位置编码                  │
└────────┬────────┘               └─────────────┬───────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐               ┌─────────────────────────────┐
│ 多头自注意力    │               │ 掩码多头自注意力             │
│ (Self-Attn)     │               │ (Masked Self-Attn)          │
│ + Add & Norm    │               │ + Add & Norm                │
└────────┬────────┘               └─────────────┬───────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────┐               ┌─────────────────────────────┐
│ 前馈网络 (FFN)  │◄──────────────│ 交叉注意力 (Cross-Attn)      │
│ + Add & Norm    │   编码器输出  │ + Add & Norm                │
└────────┬────────┘               └─────────────┬───────────────┘
         │                                      │
         │                                      ▼
         │                        ┌─────────────────────────────┐
         │                        │ 前馈网络 (FFN)              │
         │                        │ + Add & Norm                │
         │                        └─────────────┬───────────────┘
         │                                      │
         ▼                                      ▼
    编码器输出                           ┌─────────────┐
                                         │ 线性 + Softmax │
                                         └─────────────┘
```

---

## 3. 核心组件详解

### 3.1 输入嵌入 + 位置编码

```python
class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = self._create_position_encoding(d_model, max_len)
    
    def _create_position_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        embed = self.token_embedding(x)
        return embed + self.position_encoding[:, :seq_len]
```

### 3.2 残差连接（Add）

**问题**：网络越深，梯度越难传递。

**解决**：加一条"捷径"，让梯度直接流过。

```python
output = x + sublayer(x)  # 残差连接
```

直觉：
- 如果 sublayer 没用，网络可以学成 x + 0 = x
- 不会因为加深而变差

### 3.3 层归一化（LayerNorm）

**为什么不用 BatchNorm？**

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | 跨样本 | 跨特征 |
| 批次依赖 | 依赖 | 不依赖 |
| 序列长度 | 固定 | 任意 |
| 小批次 | 不稳定 | 稳定 |

Transformer 用 LayerNorm，因为：
- 序列长度可变
- 批次大小可能很小
- 每个位置独立归一化

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

### 3.4 前馈网络（FFN）

两层全连接 + 激活函数：

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

**典型配置**：d_ff = 4 × d_model
- d_model = 512 → d_ff = 2048
- d_model = 768 → d_ff = 3072

### 3.5 Add & Norm 组合

```python
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
```

---

## 4. 编码器层

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + Add & Norm
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
```

---

## 5. 解码器层

解码器多了一个**交叉注意力**：

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_out, tgt_mask=None, src_mask=None):
        # 掩码自注意力（不能看未来）
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 交叉注意力（查编码器）
        attn_out, _ = self.cross_attn(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x
```

---

## 6. 掩码机制

### 6.1 填充掩码（Padding Mask）

忽略填充位置：

```python
def create_padding_mask(seq, pad_token=0):
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)
    # [batch, 1, 1, seq_len]
```

### 6.2 因果掩码（Causal Mask）

解码器不能"偷看"未来：

```python
def create_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0  # 上三角为 False
```

```
因果掩码可视化（True = 可见）：
位置  0   1   2   3
0   [T,  F,  F,  F]  ← 位置0只能看自己
1   [T,  T,  F,  F]  ← 位置1能看0,1
2   [T,  T,  T,  F]  ← 位置2能看0,1,2
3   [T,  T,  T,  T]  ← 位置3能看所有
```

---

## 7. 完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = EmbeddingWithPosition(vocab_size, d_model, max_len)
        self.decoder_embedding = EmbeddingWithPosition(vocab_size, d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def encode(self, src, src_mask=None):
        x = self.encoder_embedding(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x
    
    def decode(self, tgt, encoder_out, tgt_mask=None, src_mask=None):
        x = self.decoder_embedding(tgt)
        for layer in self.decoder_layers:
            x = layer(x, encoder_out, tgt_mask, src_mask)
        return self.final_norm(x)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(tgt, encoder_out, tgt_mask, src_mask)
        logits = self.output_proj(decoder_out)
        return logits
```

---

## 8. Transformer 变体

### 8.1 Encoder-Only（BERT 风格）

只用编码器，适合**理解任务**：
- 文本分类
- 命名实体识别
- 问答

```python
class BERTStyleEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = EmbeddingWithPosition(vocab_size, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

### 8.2 Decoder-Only（GPT 风格）

只用解码器，适合**生成任务**：
- 文本生成
- 代码补全
- 对话

```python
class GPTBlock(nn.Module):
    """GPT 风格的解码器块（只有自注意力，无交叉注意力）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm: 自注意力 + 残差
        x = x + self.dropout(self.attn(self.ln1(x), x, x, mask))
        # FFN + 残差
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class GPTStyleDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.embedding = EmbeddingWithPosition(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff)  # 使用 GPTBlock，无交叉注意力
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        causal_mask = create_causal_mask(x.size(1))
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, causal_mask)  # 只传 mask，不需要 encoder_out
        return self.output_proj(self.norm(x))
```

### 8.3 对比

| 架构 | 代表模型 | 特点 | 典型应用 |
|------|----------|------|----------|
| Encoder-Only | BERT | 双向理解 | 分类、抽取 |
| Decoder-Only | GPT | 单向生成 | 生成任务 |
| Encoder-Decoder | T5 | 理解+生成 | 翻译、摘要 |

---

## 9. 训练与推理

### 9.1 训练：Teacher Forcing

```python
# 输入：<s> I love you
# 目标：I love you </s>

logits = model(src, tgt_input)  # tgt_input 是目标左移
loss = F.cross_entropy(logits.view(-1, vocab_size), tgt_output.view(-1))
```

### 9.2 推理：自回归生成

```python
def generate(model, src, max_len=100):
    model.eval()
    with torch.no_grad():
        encoder_out = model.encode(src)
        
        # 从 <s> 开始
        generated = torch.tensor([[BOS_TOKEN]])
        
        for _ in range(max_len):
            logits = model.decode(generated, encoder_out)
            next_token = logits[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == EOS_TOKEN:
                break
        
        return generated
```

---

## 10. 典型超参数

### Transformer Base

| 参数 | 值 |
|------|-----|
| d_model | 512 |
| num_heads | 8 |
| d_ff | 2048 |
| num_layers | 6 |
| dropout | 0.1 |
| 参数量 | ~65M |

### Transformer Big

| 参数 | 值 |
|------|-----|
| d_model | 1024 |
| num_heads | 16 |
| d_ff | 4096 |
| num_layers | 6 |
| dropout | 0.3 |
| 参数量 | ~213M |

---

## 11. 常见问题

### Q1: Pre-Norm vs Post-Norm？

原始 Transformer 用 **Post-Norm**：
```python
x = norm(x + sublayer(x))
```

现代大模型用 **Pre-Norm**：
```python
x = x + sublayer(norm(x))
```

Pre-Norm 训练更稳定，梯度流更顺畅。

### Q2: 为什么用 ReLU 而不是 GELU？

原始论文用 ReLU，但现代模型多用 GELU：
- GELU 在零点更平滑
- 训练更稳定
- BERT、GPT 都用 GELU

### Q3: 参数量怎么算？

```
词嵌入: vocab_size × d_model
位置编码: 0 (固定)

每层:
- 注意力: 4 × d_model² (Q/K/V/O)
- FFN: 2 × d_model × d_ff
- LayerNorm: 2 × 2 × d_model (可忽略)

总计 ≈ num_layers × (4d² + 2d×d_ff) + vocab×d
```

### Q4: Transformer 能处理多长的序列？

理论无限，但实际受限于：
- 显存：O(n²) 的注意力矩阵
- 计算量：O(n²d)

解决：稀疏注意力、线性注意力、FlashAttention

---

## 12. 核心要点总结

```
Transformer = 注意力 + 残差 + LayerNorm + FFN

编码器：自注意力 → FFN（×N 层）
解码器：掩码自注意力 → 交叉注意力 → FFN（×N 层）

关键设计：
- 残差：梯度直通，深层训练稳定
- LayerNorm：不依赖批次，适合序列
- 掩码：因果约束，防止偷看未来
- 多头：多视角理解

变体：
- Encoder-Only (BERT)：理解任务
- Decoder-Only (GPT)：生成任务
- Encoder-Decoder (T5)：翻译、摘要
```

---

## 13. 延伸阅读

- **论文**：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **图解**：[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- **代码**：[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- **上一章**：[13-位置编码：让模型知道顺序](13-位置编码：让模型知道顺序.md)
- **下一章**：[15-语言模型](../05-语言模型/15-语言模型.md)
