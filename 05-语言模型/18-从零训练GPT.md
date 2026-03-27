# 18-从零训练GPT

## 一句话理解

**用 PyTorch 从零实现 GPT，理解数据准备、模型构建、训练循环、生成推理的完整流程。**

---

## 1. 直觉解释

### 为什么要从零实现？

理解大模型的最好方式：
- 看论文：了解原理
- 用 API：知道能做什么
- **从零实现：真正理解细节**

从零实现让你明白：
- 数据怎么流动
- 参数怎么更新
- 模型怎么生成

### 实现路线图

```
数据准备 → 模型构建 → 训练循环 → 保存/加载 → 生成推理
    ↓          ↓          ↓          ↓          ↓
  tokenize   GPT架构    loss计算   checkpoint  temperature
```

---

## 2. 数据准备

### 2.1 文本数据集

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 将整个文本编码
        self.tokens = tokenizer.encode(text)
        
        # 计算可以切分多少个样本
        self.n_samples = len(self.tokens) // block_size
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 获取一个块
        start = idx * self.block_size
        end = start + self.block_size + 1
        
        chunk = self.tokens[start:end]
        
        # 输入和目标错位一位
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y
```

### 2.2 字符级分词器

```python
class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        
    def train(self, text):
        # 获取所有唯一字符
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # 建立映射
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        
    def encode(self, text):
        return [self.char2idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices])
    
    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump({'char2idx': self.char2idx}, f)
    
    def load(self, path):
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.char2idx = data['char2idx']
            self.idx2char = {v: k for k, v in self.char2idx.items()}
            self.vocab_size = len(self.char2idx)
```

### 2.3 数据加载

```python
def get_dataloader(text_path, batch_size=32, block_size=128):
    # 读取文本
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 训练分词器
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    
    # 创建数据集
    dataset = TextDataset(text, tokenizer, block_size)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader, tokenizer
```

---

## 3. GPT 模型实现

### 3.1 多头注意力

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # 投影并拆分多头
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax + Dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        out = self.out_proj(out)
        
        return out
```

### 3.2 前馈网络

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
```

### 3.3 Transformer 块

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Pre-Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 注意力 + 残差
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        
        # FFN + 残差
        x = x + self.dropout(self.ffn(self.ln2(x)))
        
        return x
```

### 3.4 完整 GPT 模型

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_len=512, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Position Embedding
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的 Layer Norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出头
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.head.weight = self.token_emb.weight
        
        self.dropout = nn.Dropout(dropout)
        
> 🔍 **深入理解：权重共享（Weight Tying）**
> 
> 上面的代码 `self.head.weight = self.token_emb.weight` 做了**权重共享**（Weight Tying），让输入嵌入矩阵和输出投影矩阵使用**同一组参数**。这意味着：
> - 输入时：Token ID → 查表得到嵌入向量（Embedding 查表）
> - 输出时：隐藏状态 → 矩阵乘法得到词表概率分布（Linear 投影）
> 
> **为什么这样做？**
> 
> 1. **减少参数量**：对于大词表效果显著。假设词表大小 50000，嵌入维度 1024，单独的嵌入矩阵和输出矩阵各需要 `50000 × 1024 ≈ 51M` 参数。共享后直接减半，节省约 51M 参数。对于 GPT-3 这样词表 50257 的模型，这是不小的节省。
> 
> 2. **语义对齐**：输入嵌入学习的是「词的含义」，输出投影也应该映射到同样的语义空间。共享权重强制模型在同一个空间中表示词，让「apple」的输入向量和输出向量指向相同的位置，语义更加一致。
> 
> 3. **训练稳定性**：共享权重相当于给输出层施加了一个强正则化，减少过拟合风险，同时梯度可以从输出层直接流回嵌入层，有利于端到端学习。
> 
> **类比理解**：想象一本双语词典，输入时你查词找到释义（嵌入），输出时你根据释义写出对应的词（投影）。如果查词和写词用的是**同一本词典**，前后一致，不容易出错。反之，用两本不同的词典，可能出现「查出来是 apple，写回去变成了 orange」的矛盾。
> 
> **数学视角**：Embedding 层本质上是一个查表操作，等价于 One-Hot 向量乘以嵌入矩阵 `W`；而 Linear 层是 `x @ W.T`（转置）。所以共享权重时，输出投影正好是嵌入矩阵的转置，两者数学上自然对齐。
> 
> **实践意义**：这是 GPT、BERT、Transformer 等现代语言模型的**标准做法**。原始 Transformer 论文就采用了权重共享，后续几乎所有主流 LLM 都继承了这一设计。
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids):
        B, T = input_ids.shape
        
        # 检查序列长度
        assert T <= self.max_len, f"Sequence length {T} > max_len {self.max_len}"
        
        # Token + Position Embedding
        tok_emb = self.token_emb(input_ids)  # (B, T, d_model)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)  # (T, d_model)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # 因果掩码
        mask = torch.tril(torch.ones(T, T, device=input_ids.device))
        mask = mask.view(1, 1, T, T)
        
        # Transformer 块
        for block in self.blocks:
            x = block(x, mask)
        
        # 最终 Layer Norm
        x = self.ln_f(x)
        
        # 输出 logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
```

---

## 4. 训练流程

### 4.1 训练配置

```python
@dataclass
class GPTConfig:
    vocab_size: int = 65
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    max_len: int = 256
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_epochs: int = 100
    warmup_steps: int = 100
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### 4.2 学习率调度

```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        # Warmup 阶段
        if current_step < num_warmup_steps:
            return current_step / num_warmup_steps
        
        # Cosine 衰减
        progress = (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 4.3 训练循环

```python
def train(model, dataloader, config):
    model.to(config.device)
    model.train()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # 学习率调度
    total_steps = len(dataloader) * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    for epoch in range(config.max_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(config.device)
            y = y.to(config.device)
            
            # 前向传播
            logits = model(x)
            
            # 计算损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        # 打印 epoch 信息
        avg_loss = total_loss / num_batches
        ppl = math.exp(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, PPL = {ppl:.2f}")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, config, f"checkpoint_epoch_{epoch+1}.pt")
    
    return model
```

### 4.4 混合精度训练

```python
def train_with_amp(model, dataloader, config):
    from torch.cuda.amp import autocast, GradScaler
    
    model.to(config.device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    
    for epoch in range(config.max_epochs):
        for x, y in dataloader:
            x, y = x.to(config.device), y.to(config.device)
            
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
    return model
```

### 4.5 梯度累积

```python
def train_with_gradient_accumulation(model, dataloader, config, accumulation_steps=4):
    model.to(config.device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    for epoch in range(config.max_epochs):
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(config.device), y.to(config.device)
            
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            ) / accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
    
    return model
```

---

## 5. 保存和加载

### 5.1 保存检查点

```python
def save_checkpoint(model, optimizer, epoch, config, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__ if hasattr(config, '__dict__') else config
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
```

### 5.2 加载检查点

```python
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, epoch {epoch}")
    return model, optimizer, epoch
```

### 5.3 只保存模型权重

```python
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model
```

---

## 6. 文本生成

### 6.1 贪心生成

```python
@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, max_tokens=100):
    model.eval()
    
    # 编码提示
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    
    for _ in range(max_tokens):
        # 前向传播
        logits = model(input_ids)
        
        # 取最后一个位置的预测
        next_logits = logits[0, -1, :]
        next_token = next_logits.argmax().unsqueeze(0).unsqueeze(0)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 超过最大长度就停止
        if input_ids.size(1) >= model.max_len:
            break
    
    # 解码
    output = tokenizer.decode(input_ids[0].tolist())
    return output
```

### 6.2 温度采样

```python
@torch.no_grad()
def generate_with_temperature(model, tokenizer, prompt, max_tokens=100, temperature=1.0):
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    
    for _ in range(max_tokens):
        # 只取最后 block_size 个 token（避免超长）
        if input_ids.size(1) > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        
        # 采样
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())
```

### 6.3 Top-k 采样

```python
@torch.no_grad()
def generate_top_k(model, tokenizer, prompt, max_tokens=100, temperature=1.0, k=50):
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    
    for _ in range(max_tokens):
        if input_ids.size(1) > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        
        # Top-k 过滤
        values, indices = torch.topk(next_logits, k)
        probs = F.softmax(values, dim=-1)
        
        # 采样
        idx = torch.multinomial(probs, num_samples=1)
        next_token = indices[idx].unsqueeze(0).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())
```

### 6.4 Top-p（Nucleus）采样

```python
@torch.no_grad()
def generate_top_p(model, tokenizer, prompt, max_tokens=100, temperature=1.0, p=0.9):
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=next(model.parameters()).device)
    
    for _ in range(max_tokens):
        if input_ids.size(1) > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        
        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature
        
        # 排序
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 找到超过 p 的位置
        sorted_indices_to_remove = cum_probs > p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # 过滤
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # 采样
        probs = F.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_token = sorted_indices[idx].unsqueeze(0).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return tokenizer.decode(input_ids[0].tolist())
```

---

## 7. 完整训练脚本

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import math
import json
from tqdm import tqdm
import numpy as np

# 配置
@dataclass
class GPTConfig:
    vocab_size: int = 65
    d_model: int = 256
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 1024
    max_len: int = 256
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 3e-4
    max_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # 1. 准备数据
    print("Loading data...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    
    dataset = TextDataset(text, tokenizer, config.max_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 2. 创建模型
    print("Creating model...")
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # 3. 训练
    print("Training...")
    model = train(model, dataloader, config)
    
    # 4. 生成
    print("Generating...")
    output = generate_top_p(model, tokenizer, "The ", max_tokens=500, temperature=0.8)
    print(output)
    
    # 5. 保存
    save_model(model, 'gpt_model.pt')
    tokenizer.save('tokenizer.json')

if __name__ == "__main__":
    main()
```

---

## 8. 训练技巧总结

### 8.1 稳定性技巧

```python
# 1. 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 2. 权重初始化
def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

# 3. 学习率预热
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=10000
)

# 4. 权重衰减
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
```

### 8.2 效率技巧

```python
# 1. 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 2. 梯度累积（模拟大 batch）
accumulation_steps = 4

# 3. 编译优化（PyTorch 2.0+）
model = torch.compile(model)

# 4. 数据预取
dataloader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
```

### 8.3 监控技巧

```python
# 1. 打印梯度范数
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
print(f"Gradient norm: {grad_norm:.4f}")

# 2. 打印参数统计
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

# 3. 计算困惑度
ppl = math.exp(avg_loss)
print(f"Perplexity: {ppl:.2f}")
```

---

## 9. 常见问题

### Q1: 训练不收敛怎么办？

检查：
- 学习率是否太大（试试 1e-4）
- 梯度是否爆炸（添加梯度裁剪）
- 数据是否正确（检查输入输出）
- 模型是否太大（减少层数）

### Q2: 生成的文本重复怎么办？

尝试：
- 降低温度（temperature < 1.0）
- 使用 Top-p 采样
- 增加 repetition penalty

```python
# Repetition penalty
def apply_repetition_penalty(logits, tokens, penalty=1.2):
    for token in tokens:
        logits[token] /= penalty
    return logits
```

### Q3: 如何加速训练？

方法：
- 使用 GPU
- 混合精度训练
- 增大 batch_size
- 使用 torch.compile
- 减少打印频率

### Q4: 显存不够怎么办？

解决：
- 减小 batch_size
- 减小 max_len
- 使用梯度累积
- 使用梯度检查点

```python
# 梯度检查点
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

---

## 10. 核心要点总结

```
从零训练 GPT 的流程：

1. 数据准备
   - 文本 → Token ID
   - 构建 Dataset 和 DataLoader
   - 输入和目标错位一位

2. 模型构建
   - Token Embedding + Position Embedding
   - Transformer Block × N
   - LayerNorm + Linear Head
   - 权重共享

3. 训练循环
   - Forward: logits = model(x)
   - Loss: CrossEntropy(logits, y)
   - Backward: loss.backward()
   - Update: optimizer.step()

4. 生成推理
   - 贪心：每次选概率最高的
   - 温度采样：控制随机性
   - Top-k：只从 top-k 中采样
   - Top-p：只从累积概率 p 中采样

5. 训练技巧
   - 梯度裁剪
   - 学习率预热 + 衰减
   - 混合精度训练
   - 梯度累积
```

---

## 11. 延伸阅读

- **代码**：[minGPT](https://github.com/karpathy/minGPT), [nanoGPT](https://github.com/karpathy/nanoGPT)
- **教程**：[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **论文**：[GPT-3 Paper](https://arxiv.org/abs/2005.14165)
- **上一章**：[17-分词与训练准备](17-分词与训练准备.md)
- **下一章**：[19-BERT：双向理解的力量](19-BERT：双向理解的力量.md)
