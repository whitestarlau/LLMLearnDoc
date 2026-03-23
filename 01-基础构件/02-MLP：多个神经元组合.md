# MLP：多个神经元组合

## 历史背景：从寒冬到复苏

### 1969年的打击

1969年，Minsky和Papert在《Perceptrons》一书中严格证明了单层感知机的致命缺陷——无法学习XOR问题。这本书直接导致了神经网络研究的**第一次寒冬**，资金和兴趣骤降。

当时的计算机学者们面临一个核心问题：

> **多层网络是否真的能解决单层感知机解决不了的问题？如何训练这样的网络？**

### 多层网络的曙光

1970年代，一些研究者开始探索多层感知机（MLP）。他们直觉上相信：**多个神经元组合起来应该能表达更复杂的函数**。但面临两个关键障碍：

1. **训练问题**：如何调整多层网络的权重？（反向传播算法尚未普及）
2. **理论问题**：多层网络是否真的比单层更强大？（缺乏严格证明）

### 反向传播的复兴

1986年，**David Rumelhart**、**Geoffrey Hinton** 和 **Ronald Williams** 发表了里程碑式的论文：

> *"Learning representations by back-propagating errors"*

他们重新发现并推广了**反向传播算法**，使得训练多层网络成为可能。这篇论文直接引发了神经网络研究的**第二次热潮**。

### 万能逼近定理的证明

1989年，**Kurt Hornik** 等人发表了著名论文：

> *"Multilayer feedforward networks are universal approximators"*

他们严格证明了**万能逼近定理**：一个包含足够多隐藏神经元的单隐层前馈网络，可以以任意精度逼近任意连续函数。

这个定理为多层网络提供了坚实的理论基础，证明了MLP确实比单层感知机更强大。

---

## 1. 从感知机到网络

单个感知机只能画直线。但如果我们把多个感知机组合起来呢？

答案：可以逼近**任意函数**。

---

## 2. 万能逼近定理

**定理**（Hornik et al., 1989）：一个包含足够多隐藏神经元的单隐层前馈网络，可以以任意精度逼近任意连续函数。

这个定理为多层网络提供了坚实的理论基础，证明了MLP确实比单层感知机更强大。

直觉理解：
- 每个神经元画一条线（超平面）
- 多条线组合起来，可以逼近任意曲线

```
单个神经元：     多个神经元组合：
    ───              ╭──╮
                      ╰──╯
```

但这并不意味着"一个隐藏层就够用"——深层网络通常更高效。

---

## 3. MLP 的结构

### 3.1 网络架构

```
输入层     隐藏层1    隐藏层2    输出层
  x₁  ───→  h₁⁽¹⁾ ───→  h₁⁽²⁾ ───→  y₁
     ╲    ╱ ╲    ╱ ╲    ╱
      ╲  ╱   ╲  ╱   ╲  ╱
      ╱  ╲   ╱  ╲   ╱  ╲
     ╱    ╲ ╱    ╲ ╱    ╲
  x₂  ───→  h₂⁽¹⁾ ───→  h₂⁽²⁾ ───→  y₂
```

### 3.2 数学表达

每一层的操作：

$$
z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}
$$

$$
a^{[l]} = g(z^{[l]})
$$

其中：
- $W^{[l]}$：第 $l$ 层的权重矩阵
- $b^{[l]}$：第 $l$ 层的偏置向量
- $g(\cdot)$：激活函数
- $a^{[0]} = x$：输入

### 3.3 矩阵维度

假设第 $l-1$ 层有 $n_{l-1}$ 个神经元，第 $l$ 层有 $n_l$ 个神经元：

| 符号 | 形状 |
|------|------|
| $W^{[l]}$ | $(n_l, n_{l-1})$ |
| $b^{[l]}$ | $(n_l, 1)$ |
| $a^{[l-1]}$ | $(n_{l-1}, m)$ |
| $z^{[l]}$ | $(n_l, m)$ |

其中 $m$ 是 batch size。

---

## 4. 为什么深层比浅层好

### 4.1 效率问题

假设要逼近一个有 1000 个"片段"的函数：

| 网络 | 参数量 |
|------|--------|
| 1 层，1000 个神经元 | ~$1000 \times n_{in}$ |
| 10 层，每层 10 个神经元 | ~$10 \times 10 \times n_{in}$ |

深层网络可以用**指数级更少的参数**表达同样的函数。

### 4.2 组合性

深度网络天然具有**层次化特征提取**的能力：

```
图像识别示例：
Layer 1: 边缘
Layer 2: 角点、纹理
Layer 3: 局部模式（眼睛、轮子）
Layer 4: 物体部件
Layer 5: 完整物体
```

---

## 5. PyTorch 实现

### 5.1 简单实现

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super().__init__()
        
        # 激活函数选择
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU()
        }
        act_fn = activations[activation]
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn,
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# 使用示例
model = MLP(
    input_dim=784,           # 例如 MNIST
    hidden_dims=[256, 128],  # 两个隐藏层
    output_dim=10,           # 10 个类别
    activation='relu'
)

# 打印模型结构
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

### 5.2 训练完整流程

> 下面代码使用了 `loss.backward()` 进行反向传播，这是下一章的核心内容。这里先展示整体流程，细节留到下一章。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 模型、损失函数、优化器
model = MLP(784, [256, 128], 10, 'relu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.view(data.size(0), -1)  # 展平
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)

# 测试循环
def test(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return total_loss / len(loader), correct / len(loader.dataset)

# 运行训练
for epoch in range(10):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = test(model, test_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
```

---

## 6. 设计 MLP 的经验法则

### 6.1 隐藏层大小

| 策略 | 说明 |
|------|------|
| 漏斗型 | 逐层递减：512 → 256 → 128 |
| 矩形型 | 每层相同：256 → 256 → 256 |
| 瓶颈型 | 中间小两头大：128 → 64 → 128 |

### 6.2 层数选择

- **简单任务**（如 MNIST）：1-2 层足够
- **中等任务**（如表格数据）：2-4 层
- **复杂任务**：使用专门架构（CNN、Transformer）

### 6.3 实用建议

```
1. 先用小网络验证流程正确
2. 逐步增加容量直到过拟合
3. 加入正则化对抗过拟合
4. 使用验证集监控泛化能力
```

---

## 7. MLP 的局限性

| 问题 | 说明 |
|------|------|
| 参数量大 | 全连接层参数是 $O(n^2)$ |
| 缺乏结构先知 | 不知道输入有空间/时序结构 |
| 难以处理长序列 | 参数量和序列长度成正比 |

这就是为什么图像用 CNN，序列用 RNN/Transformer。

---

## 8. 小结

| 概念 | 要点 |
|------|------|
| MLP | 多层感知机，全连接网络 |
| 万能逼近定理 | 单隐层即可逼近任意函数 |
| 深度优势 | 更少参数表达更复杂函数 |
| 实现 | `nn.Linear` + 激活函数堆叠 |

---

## 延伸阅读

- Hornik et al. (1989) "Multilayer feedforward networks are universal approximators"
- Rumelhart, Hinton & Williams (1986) "Learning representations by back-propagating errors" - 反向传播算法的里程碑论文
- Minsky & Papert (1969) "Perceptrons" - 导致第一次AI寒冬的著作
- 神经网络可视化工具：[TensorFlow Playground](https://playground.tensorflow.org)
