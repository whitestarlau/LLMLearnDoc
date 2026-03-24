# ResNet：残差连接突破深度极限

## 1. 为什么需要 ResNet

### 1.1 深度网络的困境

2014 年，VGGNet 证明了更深的网络（16-19 层）通常更好。但当人们尝试继续加深网络时，遇到了两个严重问题：

**问题一：梯度消失/爆炸**
- 随着层数增加，梯度在反向传播时不断衰减或放大
- 深层网络的参数难以更新

**问题二：退化问题（Degradation Problem）**
- 不是过拟合，而是训练误差本身反而增大
- 56 层网络的训练误差比 20 层网络更高

```
层数增加 → 训练误差上升（反直觉！）
这不是过拟合，而是优化困难
```

### 1.2 直觉理解

想象你要学习一个函数 H(x)，让网络直接学 H(x) 很难。

**残差学习的核心想法**：
- 把 H(x) 拆成 F(x) + x
- 让网络只学习残差 F(x) = H(x) - x
- 恒等映射 x 直接"跳过"连接

为什么这样更容易？
- 如果最优解接近恒等映射（H(x) ≈ x），那么 F(x) ≈ 0
- 学习"接近 0"比学习"复杂函数"容易得多
- 网络只需学习"微小的调整"

---

## 2. 残差连接原理

### 2.1 核心公式

```
标准网络：y = F(x)
残差网络：y = F(x) + x
```

其中：
- `x`：输入（跳跃连接）
- `F(x)`：残差函数（需要学习的部分）
- `y`：输出

**图示**：

```
        x ─────────────────┐
        │                  │（恒等映射）
        ↓                  │
    ┌───────────────┐      │
    │  Conv → BN →  │      │
    │  ReLU → Conv  │      │
    │  → BN         │      │
    └───────────────┘      │
        │                  │
        ↓                  │
       (+) ←───────────────┘
        │
        ↓
       ReLU
        │
        y = ReLU(F(x) + x)
```

### 2.2 维度匹配

当输入和输出维度不同时（比如通道数从 64 变成 128），需要处理维度不匹配：

**方案一：零填充**
```python
# 给 x 补零，使其维度与 F(x) 匹配
# 简单但不优雅
```

**方案二：投影快捷连接**
```python
# 用 1×1 卷积调整维度
shortcut = Conv1x1(x)  # 将 64 通道 → 128 通道
y = F(x) + shortcut
```

### 2.3 梯度流分析

残差连接如何解决梯度消失？

**标准网络的梯度**：
```
∂Loss/∂x = ∂Loss/∂y × ∂y/∂x = ∂Loss/∂y × ∂F(x)/∂x
```
梯度需要穿过所有层，容易消失。

**残差网络的梯度**：
```
∂Loss/∂x = ∂Loss/∂y × (∂F(x)/∂x + 1)
```
多了一个 "+1"，梯度可以直接通过跳跃连接回传，不会消失。

---

## 3. ResNet 架构

### 3.1 基本模块

**BasicBlock**（用于 ResNet-18/34）：
```
Conv 3×3 → BN → ReLU → Conv 3×3 → BN → (+) → ReLU
                                    ↑
                               shortcut (x)
```

**Bottleneck**（用于 ResNet-50/101/152）：
```
Conv 1×1 → BN → ReLU → Conv 3×3 → BN → ReLU → Conv 1×1 → BN → (+) → ReLU
                                                      ↑
                                                 shortcut (x)
```

Bottleneck 用 1×1 卷积先降维再升维，减少计算量：
- 1×1：256 → 64（降维）
- 3×3：64 → 64（计算）
- 1×1：64 → 256（升维）

### 3.2 网络配置

| 模型 | 层数 | 输出层配置 | 参数量 |
|------|------|-----------|--------|
| ResNet-18 | 18 | [2, 2, 2, 2] | 11M |
| ResNet-34 | 34 | [3, 4, 6, 3] | 21M |
| ResNet-50 | 50 | [3, 4, 6, 3] | 25M |
| ResNet-101 | 101 | [3, 4, 23, 3] | 44M |
| ResNet-152 | 152 | [3, 8, 36, 3] | 60M |

### 3.3 详细结构（以 ResNet-50 为例）

```
输入：224 × 224 × 3

Conv 7×7, 64 filters, stride 2 → BN → ReLU → MaxPool 3×3, stride 2
│
├─ Stage 1: 3 × Bottleneck (64→256)
│
├─ Stage 2: 4 × Bottleneck (128→512), stride 2
│
├─ Stage 3: 6 × Bottleneck (256→1024), stride 2
│
├─ Stage 4: 3 × Bottleneck (512→2048), stride 2
│
Global Average Pooling → FC 1000 → Softmax
```

---

## 4. PyTorch 实现

### 4.1 BasicBlock

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # 残差连接
        out = self.relu(out)
        
        return out
```

### 4.2 完整 ResNet

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# 创建不同深度的 ResNet
def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
```

### 4.3 使用预训练模型

```python
import torchvision.models as models

# 加载预训练的 ResNet-50
model = models.resnet50(pretrained=True)

# 修改最后一层，适配自己的任务（比如 10 类分类）
model.fc = nn.Linear(2048, 10)

# 冻结前面的层，只训练最后一层
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
```

---

## 5. ResNet 的影响与演进

### 5.1 ImageNet 成绩

| 模型 | 年份 | Top-5 错误率 | 层数 |
|------|------|-------------|------|
| AlexNet | 2012 | 15.3% | 8 |
| VGGNet | 2014 | 6.7% | 19 |
| GoogLeNet | 2014 | 6.7% | 22 |
| **ResNet-152** | **2015** | **3.57%** | **152** |
| ResNeXt | 2016 | 3.03% | 101 |

### 5.2 后续发展

```
ResNet (2015)
    ↓
ResNeXt (2016) - 分组卷积，更强
    ↓
DenseNet (2017) - 密集连接，每层连所有层
    ↓
EfficientNet (2019) - 复合缩放，效率最优
    ↓
ConvNeXt (2022) - 现代化 CNN，对标 Transformer
```

### 5.3 残差连接的广泛应用

残差连接已经成为深度学习的**标准组件**：
- **Transformer**：每个子层都有残差连接
- **BERT/GPT**：基于 Transformer，自然继承残差连接
- **U-Net**：跳跃连接 + 残差连接
- **Diffusion Models**：U-Net 骨架，大量使用残差连接

---

## 6. 常见问题

**Q: 残差连接和跳跃连接是一回事吗？**

不完全一样。跳跃连接（Skip Connection）是更广泛的概念，包括：
- 恒等映射（ResNet 的做法）
- 拼接（DenseNet 的做法）
- 门控连接（Highway Network 的做法）

残差连接特指"恒等映射 + 逐元素相加"。

**Q: 为什么 ResNet 能训练 152 层，而 VGG 只能训练 19 层？**

两个原因：
1. **梯度流**：残差连接让梯度可以直接回传，避免消失
2. **优化地形**：残差学习简化了优化目标，损失曲面更平滑

**Q: 什么时候用 BasicBlock，什么时候用 Bottleneck？**

- **BasicBlock**：层数较少（18/34），参数少，速度快
- **Bottleneck**：层数较多（50/101/152），用 1×1 卷积减少计算量

实际应用中，ResNet-50（Bottleneck）是性价比最高的选择。

**Q: 残差连接只适用于 CNN 吗？**

不是。残差连接是通用的深度学习技巧：
- **RNN/LSTM**：本质上也是门控残差连接
- **Transformer**：每个子层都有残差连接
- **MLP**：也常用残差连接提升训练稳定性

---

## 7. 小结

| 要点 | 内容 |
|------|------|
| 核心思想 | 学习残差 F(x) = H(x) - x，而非直接学习 H(x) |
| 关键创新 | 恒等跳跃连接，解决梯度消失和退化问题 |
| 架构变体 | BasicBlock（浅层）/ Bottleneck（深层） |
| 历史意义 | 2015 ImageNet 冠军，突破 100 层深度限制 |
| 持续影响 | 残差连接成为现代深度网络的标准组件 |

---

## 8. 延伸阅读

- **原始论文**：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., 2015)
- **后续改进**：[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (He et al., 2016)
- **理论分析**：[ResNet 原理的数学证明](https://arxiv.org/abs/1605.06431)
- **现代应用**：[ConvNeXt](https://arxiv.org/abs/2201.03545) (2022) - 用 ResNet 思想改进的现代 CNN
