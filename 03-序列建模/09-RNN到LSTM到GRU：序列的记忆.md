# RNN到LSTM到GRU：序列的记忆

## 1. 为什么需要序列模型？

前几章的神经网络有一个共同点：**输入之间没有顺序关系**。

打乱图片的像素顺序，CNN 依然能识别猫；打乱句子的词序，MLP 可能就懵了。

> "今天天气真好" 和 "天气今天好真" 是完全不同的意思。

这就是**序列数据**的特点：**顺序很重要**。

| 数据类型 | 示例 | 顺序重要吗？ |
|----------|------|--------------|
| 图像 | 猫的照片 | ❌ |
| 文本 | "我爱你" | ✅ |
| 语音 | 录音 | ✅ |
| 股票 | 每日价格 | ✅ |
| DNA | 基因序列 | ✅ |

---

## 2. RNN：让网络有记忆

### 2.1 核心思想

普通神经网络每次处理一个输入，处理完就忘了。RNN 的想法很简单：**把上一步的输出，作为下一步的输入之一**。

```
x₁ → [RNN] → h₁
         ↓
x₂ → [RNN] → h₂
         ↓
x₃ → [RNN] → h₃
```

每个时刻 t，RNN 做两件事：
1. 接收当前输入 $x_t$
2. 结合上一时刻的隐藏状态 $h_{t-1}$

### 2.2 数学定义

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
y_t = W_{hy} h_t + b_y
$$

其中：
- $h_t$：t 时刻的隐藏状态（记忆）
- $W_{hh}$：隐藏状态到隐藏状态的权重
- $W_{xh}$：输入到隐藏状态的权重
- $W_{hy}$：隐藏状态到输出的权重

**关键点**：所有时刻共享同一组参数 $(W_{hh}, W_{xh}, W_{hy})$！

### 2.3 直觉理解

把 RNN 想象成一个人读句子：

```
句子: "我 昨天 去了 北京"

时刻1: 读到"我"      → 记忆: [主语是"我"]
时刻2: 读到"昨天"    → 记忆: [主语是"我", 时间是"昨天"]
时刻3: 读到"去了"    → 记忆: [主语是"我", 时间是"昨天", 动作是"去"]
时刻4: 读到"北京"    → 记忆: [..., 地点是"北京"]
```

每一步都在更新"记忆"，这个记忆就是隐藏状态 $h_t$。

---

## 3. 梯度消失：RNN 的致命弱点

### 3.1 问题描述

训练 RNN 时，我们需要通过时间反向传播（BPTT）。问题来了：

```
h₅ → h₄ → h₃ → h₂ → h₁
```

梯度要从 $h_5$ 一路传回 $h_1$，中间经过 4 个 tanh 和 4 次矩阵乘法。

**链式法则**告诉我们：

$$
\frac{\partial h_5}{\partial h_1} = \frac{\partial h_5}{\partial h_4} \cdot \frac{\partial h_4}{\partial h_3} \cdot \frac{\partial h_3}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1}
$$

每个 $\frac{\partial h_t}{\partial h_{t-1}}$ 都涉及 $W_{hh}$。如果 $W_{hh}$ 的特征值小于 1，多次相乘后梯度会趋近于 0。

### 3.2 数学解释

假设 $W_{hh}$ 的最大特征值为 $\lambda_{max}$：

- 如果 $\lambda_{max} < 1$：梯度指数衰减 → **梯度消失**
- 如果 $\lambda_{max} > 1$：梯度指数爆炸 → **梯度爆炸**

**实际中，梯度消失更常见**，因为 tanh 的导数最大值是 1，通常小于 1。

### 3.3 后果

梯度消失意味着：**网络记不住长期依赖**。

```python
# RNN 很难学到这种模式
"The cat, which I saw yesterday in the park near my house, __is__ sleeping."
#                 ↑                                        ↑
#             主语"cat"                                  动词"is"
#             相隔很远，RNN 记不住
```

---

## 4. LSTM：精心设计的记忆

### 4.1 核心思想

LSTM（Long Short-Term Memory）的核心思想：**把记忆分成两部分**。

1. **隐藏状态 $h_t$**：短期记忆，每个时刻都变
2. **细胞状态 $c_t$**：长期记忆，通过"门"控制哪些信息保留、哪些遗忘

```
        ┌─────────────────────────────┐
        │  细胞状态 c_t (高速公路)      │
        └─────────────────────────────┘
               ↑         ↑         ↑
            遗忘门     输入门     输出门
```

### 4.2 三个门

**遗忘门（Forget Gate）**：决定丢弃哪些旧信息

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

**输入门（Input Gate）**：决定写入哪些新信息

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

**输出门（Output Gate）**：决定输出哪些信息

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

### 4.3 完整公式

更新细胞状态：

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

更新隐藏状态：

$$
h_t = o_t \odot \tanh(c_t)
$$

### 4.4 直觉理解

想象一个笔记本（细胞状态）：

1. **遗忘门**：看旧笔记，决定擦掉哪些内容
2. **输入门**：看新信息，决定写入哪些内容
3. **输出门**：看当前需要，决定展示哪些内容

**关键优势**：细胞状态 $c_t$ 的梯度路径是**加法**，不是乘法！

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t
$$

当 $f_t \approx 1$ 时，梯度几乎不衰减，信息可以长距离传递。

---

## 5. GRU：简化的 LSTM

### 5.1 动机

LSTM 有三个门，参数多、计算慢。GRU（Gated Recurrent Unit）把遗忘门和输入门合并成一个"更新门"。

### 5.2 两个门

**更新门（Update Gate）**：决定保留多少旧信息、接受多少新信息

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
$$

**重置门（Reset Gate）**：决定遗忘多少历史信息

$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
$$

### 5.3 完整公式

候选隐藏状态：

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t] + b)
$$

更新隐藏状态：

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

### 5.4 直觉理解

- 当 $z_t \approx 0$：保留大部分旧状态（记忆模式）
- 当 $z_t \approx 1$：用新状态替换旧状态（更新模式）
- $r_t$ 控制：在生成新状态时，参考多少历史信息

---

## 6. 三者对比

| 特性 | RNN | LSTM | GRU |
|------|-----|------|-----|
| 门控机制 | 无 | 3个门 | 2个门 |
| 长期记忆 | 弱 | 强 | 中等 |
| 参数量 | 最少 | 最多 | 中等 |
| 训练速度 | 最快 | 最慢 | 中等 |
| 长序列表现 | 差 | 好 | 较好 |
| 短序列表现 | 一般 | 好 | 好 |

**经验法则**：
- 数据量小、序列短 → GRU
- 数据量大、序列长 → LSTM
- 需要快速实验 → GRU

---

## 7. 代码示例

### 7.1 手写 RNN

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 权重矩阵
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hy = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size)
        
        for t in range(seq_len):
            h = torch.tanh(self.W_xh(x[:, t, :]) + self.W_hh(h))
        
        return self.W_hy(h)

# 使用
model = SimpleRNN(input_size=10, hidden_size=20, output_size=5)
x = torch.randn(32, 100, 10)  # 32个样本，100个时间步，10维特征
output = model(x)  # (32, 5)
```

### 7.2 使用 PyTorch 内置 LSTM

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        x: (batch, seq_len) 词索引序列
        """
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM 前向传播
        # output: 所有时间步的隐藏状态
        # (h_n, c_n): 最后时间步的隐藏状态和细胞状态
        output, (h_n, c_n) = self.lstm(embedded)
        
        # 取最后一个时间步的输出
        last_hidden = output[:, -1, :]  # (batch, hidden_dim)
        
        # 分类
        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits

# 使用
model = LSTMClassifier(
    vocab_size=10000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=5
)

# 输入: 批量句子，每个句子是词索引序列
x = torch.randint(0, 10000, (32, 50))  # 32个句子，长度50
logits = model(x)  # (32, 5)
```

### 7.3 双向 LSTM

```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向，所以是 hidden_dim * 2
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)
        
        # 合并双向的最后隐藏状态
        # h_n 形状: (2, batch, hidden_dim) - 第一维是方向
        forward_hidden = h_n[0]   # (batch, hidden_dim)
        backward_hidden = h_n[1]  # (batch, hidden_dim)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.fc(combined)
```

---

## 8. 常见问题

**Q: 为什么用 tanh 而不是 ReLU？**

| 函数 | 值域 | 梯度 |
|------|------|------|
| tanh | (-1, 1) | 有界，稳定 |
| ReLU | [0, ∞) | 无界，可能爆炸 |

序列模型中，ReLU 的无界性会导致梯度爆炸。tanh 把输出压缩到 (-1, 1)，更稳定。

**Q: LSTM 真的能解决梯度消失吗？**

部分解决。细胞状态的梯度路径是加法，但：
- 如果 $f_t$ 经常接近 0，长期记忆还是会丢失
- 实际中 LSTM 比普通 RNN 好很多，但不是完美的

**Q: GRU 和 LSTM 选哪个？**

没有绝对答案。通常：
- GRU 参数少 33%，训练快 20-30%
- LSTM 在长序列任务上通常略好
- 建议两个都试，看验证集表现

---

## 9. 小结

| 模型 | 核心思想 | 解决的问题 |
|------|----------|------------|
| RNN | 隐藏状态传递记忆 | 序列建模 |
| LSTM | 三个门 + 细胞状态 | 长期依赖 |
| GRU | 两个门，简化版 LSTM | 效率与效果平衡 |

**关键洞察**：

LSTM 的设计哲学是：**信息应该有选择地流动**。不是所有历史信息都有用，门控机制让网络学会"记住重要的，忘记无关的"。

这个思想后来被 **Attention 机制** 继承并发扬光大，成为现代 Transformer 的核心。

---

## 延伸阅读

- Hochreiter & Schmidhuber (1997) "Long Short-Term Memory" - LSTM 原始论文
- Cho et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder" - GRU 论文
- Christopher Olah 的博客 "Understanding LSTM Networks" - 经典的图解 LSTM