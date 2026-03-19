# Seq2Seq：从RNN到Attention的桥梁

## 1. 从分类到生成

前一章我们用 RNN/LSTM 做**序列分类**：输入一段文字，输出一个标签。

但很多任务需要**输入一个序列，输出另一个序列**：

| 任务 | 输入序列 | 输出序列 |
|------|----------|----------|
| 机器翻译 | "I love you" | "我爱你" |
| 文本摘要 | 长文章 | 摘要 |
| 对话系统 | 用户问题 | 回答 |
| 语音识别 | 音频序列 | 文字序列 |

这就是 **Seq2Seq（Sequence-to-Sequence）** 要解决的问题。

---

## 2. 编码器-解码器架构

### 2.1 核心思想

Seq2Seq 的核心思想很简单：**压缩 → 解压**

```
编码器 (Encoder)          解码器 (Decoder)
"I love you"  →  [上下文向量 c]  →  "我爱你"
```

1. **编码器**：把输入序列压缩成一个固定长度的向量 $c$
2. **解码器**：从这个向量展开，生成输出序列

### 2.2 数学定义

**编码器**：逐词读取输入，更新隐藏状态

$$
h_t = \text{LSTM}(x_t, h_{t-1})
$$

最终隐藏状态 $h_T$ 就是上下文向量：

$$
c = h_T
$$

**解码器**：用 $c$ 作为初始状态，逐步生成输出

$$
s_t = \text{LSTM}(y_{t-1}, s_{t-1})
$$

$$
P(y_t | y_{<t}, x) = \text{softmax}(W_s s_t + b_s)
$$

### 2.3 直觉理解

想象翻译任务：

**编码阶段**（理解中文）：
```
读入: "我 爱 你"
处理: [我] → [我, 爱] → [我, 爱, 你]
最终: 一个向量，包含了整句话的意思
```

**解码阶段**（生成英文）：
```
初始: <START> + 上下文向量
第一步: 上下文 → "I"
第二步: "I" + 上下文 → "love"  
第三步: "love" + 上下文 → "you"
第四步: "you" + 上下文 → <END>
```

---

## 3. Teacher Forcing

### 3.1 问题

训练解码器时，有一个问题：**下一步的输入应该用什么？**

两种选择：
1. 用**上一步的真实标签**（Teacher Forcing）
2. 用**上一步的预测结果**（Free Running）

### 3.2 Teacher Forcing

```
训练时:
输入: <START> → 预测: "I"    (用真实标签"I"作为下一步输入)
输入: "I"     → 预测: "love" (用真实标签"love"作为下一步输入)
输入: "love"  → 预测: "you"  (用真实标签"you"作为下一步输入)
```

**优点**：训练稳定、收敛快
**缺点**：训练和推理不一致（Exposure Bias）

### 3.3 Exposure Bias 问题

```
训练时: 用真实标签 → 模型"被喂"正确答案
推理时: 用自己预测 → 模型从没见过自己的错误
```

如果某一步预测错了，错误会累积，导致后面全部错乱。

**解决方案**：
- Scheduled Sampling：逐渐从 Teacher Forcing 过渡到 Free Running
- 强化学习：直接优化最终指标（如 BLEU）

---

## 4. 注意力机制的引入

### 4.1 瓶颈问题

基础 Seq2Seq 有一个致命缺陷：**信息瓶颈**。

```
输入: 20个词 → 压缩成1个向量 → 展开成15个词
```

一个向量能装下20个词的全部信息吗？

**实验表明**：短句子可以，长句子效果急剧下降。

### 4.2 Attention：让解码器回头看

Bahdanau et al. (2014) 提出了一个优雅的解决方案：

> **每一步解码时，让解码器"回头看"编码器的所有隐藏状态，决定重点关注哪些部分。**

```
编码器: h₁, h₂, h₃, ..., h_T

解码器第一步: 看 h₁, h₂, ..., h_T，发现 h₂ 最重要 → 聚焦 h₂
解码器第二步: 看 h₁, h₂, ..., h_T，发现 h₅ 最重要 → 聚焦 h₅
...
```

### 4.3 数学定义

**步骤 1：计算注意力分数**

$$
e_{t,s} = \text{score}(s_{t-1}, h_s)
$$

其中 $s_{t-1}$ 是解码器上一步的隐藏状态，$h_s$ 是编码器第 s 步的隐藏状态。

常用 score 函数：
- **加性注意力**（Bahdanau）：$e_{t,s} = v^T \tanh(W_1 s_{t-1} + W_2 h_s)$
- **乘性注意力**（Luong）：$e_{t,s} = s_{t-1}^T W h_s$

**步骤 2：归一化为权重**

$$
\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{k=1}^{T} \exp(e_{t,k})}
$$

**步骤 3：加权求和得到上下文向量**

$$
c_t = \sum_{s=1}^{T} \alpha_{t,s} h_s
$$

**步骤 4：用上下文向量生成输出**

$$
s_t = \text{LSTM}([y_{t-1}; c_t], s_{t-1})
$$

$$
P(y_t) = \text{softmax}(W_o s_t + b_o)
$$

### 4.4 直觉理解

翻译 "猫 坐在 垫子 上" → "The cat sat on the mat"

```
生成 "The":  注意力 → [猫: 0.7, 坐在: 0.1, 垫子: 0.1, 上: 0.1]
生成 "cat":  注意力 → [猫: 0.9, 坐在: 0.05, 垫子: 0.03, 上: 0.02]
生成 "sat":  注意力 → [猫: 0.1, 坐在: 0.8, 垫子: 0.05, 上: 0.05]
生成 "on":   注意力 → [猫: 0.05, 坐在: 0.1, 垫子: 0.3, 上: 0.55]
生成 "the":  注意力 → [猫: 0.05, 坐在: 0.05, 垫子: 0.6, 上: 0.3]
生成 "mat":  注意力 → [猫: 0.05, 坐在: 0.05, 垫子: 0.85, 上: 0.05]
```

每一步都专注于源句中最相关的词！

---

## 5. Seq2Seq + Attention 的实现

### 5.1 编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        """
        x: (batch, src_len)
        """
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # outputs: (batch, src_len, hidden_dim * 2)
        # hidden: (2, batch, hidden_dim) - 双向
        
        # 合并双向隐藏状态
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden_dim*2)
        hidden = torch.tanh(self.fc(hidden))  # (batch, hidden_dim)
        
        # cell 也需要处理
        cell = torch.cat([cell[0], cell[1]], dim=1)
        cell = torch.tanh(self.fc(cell))
        
        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0)
```

### 5.2 注意力机制

```python
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: (1, batch, dec_hidden_dim)
        encoder_outputs: (batch, src_len, enc_hidden_dim*2)
        """
        src_len = encoder_outputs.shape[1]
        
        # 重复隐藏状态 src_len 次
        hidden = hidden.permute(1, 0, 2)  # (batch, 1, dec_hidden_dim)
        hidden = hidden.repeat(1, src_len, 1)  # (batch, src_len, dec_hidden_dim)
        
        # 拼接
        energy = torch.cat([hidden, encoder_outputs], dim=2)
        energy = torch.tanh(self.attn(energy))
        
        # 计算注意力分数
        attention = self.v(energy).squeeze(2)  # (batch, src_len)
        
        return F.softmax(attention, dim=1)
```

### 5.3 解码器

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.lstm = nn.LSTM(embed_dim + enc_hidden_dim * 2, dec_hidden_dim, batch_first=True)
        self.fc = nn.Linear(dec_hidden_dim + enc_hidden_dim * 2 + embed_dim, vocab_size)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        """
        x: (batch, 1) - 当前时间步的输入
        hidden: (1, batch, dec_hidden_dim)
        cell: (1, batch, dec_hidden_dim)
        encoder_outputs: (batch, src_len, enc_hidden_dim*2)
        """
        embedded = self.embedding(x)  # (batch, 1, embed_dim)
        
        # 计算注意力
        attn_weights = self.attention(hidden, encoder_outputs)  # (batch, src_len)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, src_len)
        
        # 加权求和
        context = torch.bmm(attn_weights, encoder_outputs)  # (batch, 1, enc_hidden_dim*2)
        
        # 拼接输入和上下文
        lstm_input = torch.cat([embedded, context], dim=2)
        
        # LSTM 前向
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 预测下一个词
        prediction = self.fc(torch.cat([output, context, embedded], dim=2))
        
        return prediction, hidden, cell, attn_weights.squeeze(1)
```

---

## 6. 完整 Seq2Seq 模型

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        # 存储输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # 第一个输入是 <sos>
        x = trg[:, 0:1]  # (batch, 1)
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(x, hidden, cell, encoder_outputs)
            outputs[:, t:t+1, :] = output
            
            # Teacher Forcing
            top1 = output.argmax(2)
            x = trg[:, t:t+1] if torch.rand(1) < teacher_forcing_ratio else top1
            
        return outputs
```

---

## 7. Attention 的可视化

注意力机制的一个巨大优势：**可解释性**。

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(source_words, target_words, attention):
    """
    attention: (trg_len, src_len)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=source_words, yticklabels=target_words, cmap='YlOrRd')
    plt.xlabel('Source')
    plt.ylabel('Target')
    plt.title('Attention Weights')
    plt.show()

# 示例
src = ['我', '爱', '你']
trg = ['<sos>', 'I', 'love', 'you', '<eos>']
# attention 矩阵通过模型计算得到
```

通过可视化，我们可以看到模型是否正确学习了词对齐。

---

## 8. Seq2Seq 的历史地位

Seq2Seq + Attention 是深度学习历史上的重要里程碑：

```
2014: Seq2Seq (Sutskever et al.)
2014: Attention (Bahdanau et al.)  
2015: 更好的 Attention (Luong et al.)
2017: Transformer (Vaswani et al.) ← 把 Attention 发挥到极致
```

**Attention 机制的启示**：

> 不需要把所有信息压缩到一个向量里。让模型自己决定在每一步应该关注输入的哪些部分。

这个思想直接启发了 **Transformer** 的诞生。在 Transformer 中，Attention 取代了 RNN，成为序列建模的核心。

---

## 9. 常见问题

**Q: 为什么用双向 LSTM 做编码器？**

单向 LSTM 只能看到"过去"的信息。双向 LSTM 同时看"过去"和"未来"，对每个词的理解更完整。

例如："我 昨天 去了 北京"，理解"去了"时，双向能同时看到"我"和"北京"。

**Q: Attention 的计算复杂度是多少？**

$O(T \times S)$，其中 T 是输出长度，S 是输入长度。这在长序列上会成为瓶颈，也是后来 Transformer 需要优化的地方。

**Q: 为什么解码器只用单向 LSTM？**

因为解码是自回归的：生成第 t 个词时，只能看到前 t-1 个词，不能"偷看"未来的词。

---

## 10. 小结

| 概念 | 要点 |
|------|------|
| Seq2Seq | 编码器-解码器架构，输入序列→输出序列 |
| 上下文向量 | 编码器的最终隐藏状态，压缩了整个输入 |
| Teacher Forcing | 训练时用真实标签作为解码器输入 |
| Attention | 让解码器每一步都能"回头看"编码器的所有状态 |
| 注意力权重 | 可解释，显示输入和输出的对齐关系 |

**关键洞察**：

Seq2Seq + Attention 解决了序列到序列的转换问题，但 Attention 机制的价值远不止于此。它揭示了一个更深刻的思想：

> **不是所有信息都需要压缩，让模型自己选择关注什么。**

这个思想在下一章"注意力革命"中将得到充分发展，最终诞生 Transformer。

---

## 延伸阅读

- Sutskever et al. (2014) "Sequence to Sequence Learning with Neural Networks" - Seq2Seq 原始论文
- Bahdanau et al. (2014) "Neural Machine Translation by Jointly Learning to Align and Translate" - Attention 原始论文
- Luong et al. (2015) "Effective Approaches to Attention-based Neural Machine Translation" - Attention 改进