# 深度学习常用 Python 库

## 为什么需要这些库？

深度学习涉及大量的矩阵运算和梯度计算。如果从零实现，你需要：
- 手动编写矩阵乘法（且要高效）
- 实现反向传播算法
- 管理 GPU 内存

这些库让我们能**专注于模型设计**，而不是底层实现。

---

## 1. NumPy：张量运算的基础

### 它是什么？

NumPy 是 Python 科学计算的基础库，提供了高性能的多维数组（ndarray）和数学函数。

### 为什么重要？

深度学习的核心操作是**张量运算**（多维矩阵运算）。NumPy 是所有深度学习框架的设计原型：
- PyTorch 的 Tensor API 几乎照搬 NumPy
- 理解 NumPy 就理解了深度学习的计算本质

### 核心概念

```python
import numpy as np

# 创建数组
x = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 矩阵
print(x.shape)  # (2, 3)

# 矩阵运算
y = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2 矩阵
z = np.dot(x, y)  # 矩阵乘法 → 2x2 矩阵

# 广播（自动扩展维度）
a = np.array([[1], [2], [3]])  # 3x1
b = np.array([10, 20, 30])     # 3,
c = a + b  # 自动广播为 3x3
```

### 深度学习中的应用

```python
# 权重初始化
w = np.random.randn(784, 128) * 0.01  # He 初始化

# 前向传播
def relu(x):
    return np.maximum(0, x)

def forward(x, w, b):
    return relu(np.dot(x, w) + b)

# 批量处理
batch = np.random.randn(32, 784)  # 32 个样本
output = forward(batch, w, b)     # 一次处理整个 batch
```

---

## 2. CUDA：GPU 并行计算

### 它是什么？

CUDA 是 NVIDIA 的并行计算平台，让 GPU 可以做通用计算（不仅仅是图形渲染）。

### 为什么重要？

深度学习的核心是**大规模矩阵乘法**，而 GPU 天生擅长并行计算：
- 训练时间从几天 → 几小时
- 现代大模型（GPT、BERT）在 CPU 上根本无法训练

### 原理直觉

```
CPU: 1 个工人，很聪明，串行处理复杂任务
GPU: 10000 个工人，只会简单计算，但可以并行工作

矩阵乘法 = 大量独立的乘法和加法 → 适合 GPU
```

### 检查和使用

```python
import torch

# 检查 GPU 是否可用
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# 将数据移到 GPU
x = torch.randn(1000, 1000)
x_gpu = x.cuda()  # 或 x.to('cuda')

# 模型移到 GPU
model = model.cuda()
```

---

## 3. PyTorch：主流深度学习框架

### 它是什么？

PyTorch 是 Facebook（Meta）开发的深度学习框架，以**动态计算图**和**Pythonic 设计**著称。

### 为什么是主流？

- **学术界标准**：顶会论文 90%+ 使用 PyTorch
- **调试方便**：动态图，可以像普通 Python 一样打断点
- **API 直观**：NumPy 用户几乎零成本迁移

### 核心概念

#### 3.1 Tensor：多维数组 + GPU 支持

```python
import torch

# 创建张量
x = torch.randn(3, 4)        # 随机张量
y = torch.zeros(3, 4)        # 全零
z = torch.ones(3, 4)         # 全一

# 与 NumPy 互转
import numpy as np
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)  # NumPy → Tensor
back = tensor.numpy()            # Tensor → NumPy

# GPU 加速
x_gpu = x.cuda()
y_gpu = y.cuda()
z_gpu = x_gpu + y_gpu  # 在 GPU 上计算
```

#### 3.2 Autograd：自动求导

```python
# requires_grad=True 表示需要计算梯度
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x² + 3x + 1

y.backward()  # 反向传播，自动计算 dy/dx
print(x.grad)  # tensor([7.]) = 2*2 + 3
```

#### 3.3 nn.Module：定义模型

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)   # 全连接层
        self.fc2 = nn.Linear(128, 10)    # 输出层
        self.relu = nn.ReLU()            # 激活函数
    
    def forward(self, x):
        x = self.relu(self.fc1(x))       # 隐藏层 + 激活
        return self.fc2(x)               # 输出（未归一化）

model = SimpleNet()
print(model)  # 打印网络结构
```

#### 3.4 训练循环

```python
import torch.optim as optim

# 准备
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        # 1. 前向传播
        output = model(batch_x)
        
        # 2. 计算损失
        loss = criterion(output, batch_y)
        
        # 3. 反向传播
        optimizer.zero_grad()  # 清除旧梯度
        loss.backward()        # 计算新梯度
        
        # 4. 更新参数
        optimizer.step()
```

### PyTorch 生态

| 库 | 用途 |
|---|------|
| `torchvision` | 图像数据集、预训练模型（ResNet、VGG）、图像变换 |
| `torchaudio` | 音频处理 |
| `torchtext` | 文本处理（逐渐被 Hugging Face 取代） |
| `torch.nn` | 神经网络层和损失函数 |
| `torch.optim` | 优化器（SGD、Adam） |
| `torch.utils.data` | 数据加载（Dataset、DataLoader） |

---

## 4. Hugging Face Transformers：预训练模型库

### 它是什么？

Hugging Face 提供了数千个预训练模型（BERT、GPT、T5、LLaMA 等），是现代 LLM 生态的核心。

### 为什么重要？

- **不用从头训练**：加载预训练模型，微调即可使用
- **统一 API**：所有模型用相同的接口
- **社区驱动**：最新的模型几乎都会第一时间上架

### 核心用法

```python
from transformers import AutoModel, AutoTokenizer

# 加载 BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 文本编码
text = "Hello, world!"
inputs = tokenizer(text, return_tensors='pt')

# 推理
outputs = model(**inputs)
last_hidden = outputs.last_hidden_state  # [batch, seq_len, 768]
```

### 快速上手 Pipeline

```python
from transformers import pipeline

# 情感分析
classifier = pipeline('sentiment-analysis')
print(classifier("I love PyTorch!"))
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 文本生成
generator = pipeline('text-generation', model='gpt2')
print(generator("Once upon a time", max_length=50))
```

---

## 5. torchvision：计算机视觉工具箱

### 它是什么？

PyTorch 官方的计算机视觉库，提供数据集、预训练模型和图像变换。

### 核心用法

```python
import torchvision
from torchvision import transforms, datasets, models

# 图像预处理（Compose 组合多个变换）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)

# 加载预训练模型
resnet = models.resnet50(pretrained=True)
```

---

## 6. Matplotlib：数据可视化

### 它是什么？

Matplotlib 是 Python 最基础的绑图库，用于创建各种图表。

### 为什么重要？

深度学习中可视化无处不在：
- 绘制激活函数曲线
- 可视化注意力权重
- 监控训练过程（损失曲线）

### 核心用法

```python
import matplotlib.pyplot as plt
import numpy as np

# 折线图
x = np.linspace(-5, 5, 100)
y = 1 / (1 + np.exp(-x))  # Sigmoid
plt.plot(x, y, label='Sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 子图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(x, np.maximum(0, x))  # ReLU
axes[0].set_title('ReLU')
axes[1].plot(x, 1 / (1 + np.exp(-x)))  # Sigmoid
axes[1].set_title('Sigmoid')
axes[2].plot(x, np.tanh(x))  # Tanh
axes[2].set_title('Tanh')
plt.tight_layout()
plt.show()
```

### 深度学习中的应用

```python
# 绘制训练曲线
train_losses = [0.5, 0.3, 0.2, 0.15, 0.1]
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化注意力权重
attention_weights = model.get_attention(inputs)  # [heads, seq, seq]
plt.imshow(attention_weights[0].detach().numpy(), cmap='hot')
plt.colorbar()
plt.title('Attention Weights')
plt.show()
```

---

## 总结：核心工具链

```
计算基础 → NumPy
    ↓
模型构建 → PyTorch
    ↓
预训练模型 → Hugging Face Transformers
    ↓
GPU 加速 → CUDA
    ↓
可视化 → Matplotlib
```

| 工具 | 角色 | 重要性 |
|------|------|--------|
| NumPy | 计算基础 | ★★★★★ |
| CUDA | GPU 加速 | ★★★★★ |
| PyTorch | 深度学习框架 | ★★★★★ |
| Matplotlib | 数据可视化 | ★★★★☆ |
| Transformers | 预训练模型 | ★★★★☆ |
| torchvision | 视觉工具 | ★★★★☆ |

---

## 下一步

掌握 PyTorch 基本用法后，我们从最简单的神经元开始，逐步构建完整的深度学习模型。