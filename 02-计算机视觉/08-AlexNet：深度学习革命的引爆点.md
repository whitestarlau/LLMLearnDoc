# AlexNet：深度学习革命的引爆点

## 1. 历史背景：ImageNet 竞赛

### 1.1 ImageNet 挑战赛

ImageNet 是一个包含 1400 万张图像、21841 个类别的视觉数据集。

**ILSVRC（ImageNet Large Scale Visual Recognition Challenge）**：
- 1000 个类别
- 120 万张训练图像
- 5 万张验证图像
- 10 万张测试图像

评价指标：**Top-5 错误率**（前 5 个预测中包含正确答案即可）

### 1.2 2012 年的突破

| 年份 | 方法 | Top-5 错误率 |
|------|------|-------------|
| 2011 | 传统机器学习（手工特征 + SVM） | ~25% |
| **2012** | **AlexNet** | **15.3%** |
| 2013 | ZFNet | 11.2% |
| 2014 | GoogLeNet | 6.7% |
| 2015 | ResNet | 3.57% |

**AlexNet 将错误率直接降低了 10 个百分点**，这是深度学习的历史性时刻。

---

## 2. AlexNet 架构

### 2.1 网络结构

```
输入：227 × 227 × 3

Layer 1: Conv 11×11, stride 4, 96 filters → ReLU → MaxPool 3×3, stride 2
Layer 2: Conv 5×5, 256 filters → ReLU → MaxPool 3×3, stride 2
Layer 3: Conv 3×3, 384 filters → ReLU
Layer 4: Conv 3×3, 384 filters → ReLU
Layer 5: Conv 3×3, 256 filters → ReLU → MaxPool 3×3, stride 2

Flatten

FC1: 4096 neurons → ReLU → Dropout
FC2: 4096 neurons → ReLU → Dropout
FC3: 1000 neurons → Softmax
```

### 2.2 关键数字

| 组件 | 数量 |
|------|------|
| 参数量 | ~6000 万 |
| 层数 | 8 层（5 卷积 + 3 全连接） |
| 训练时间 | 5-6 天（2 个 GTX 580 GPU） |

---

## 3. AlexNet 的关键创新

### 3.1 ReLU 激活函数

**之前**：普遍使用 Sigmoid 或 Tanh

**问题**：
- 梯度消失：深层网络的梯度在反向传播时指数衰减
- 计算慢：涉及指数运算

**ReLU**：

$$
f(x) = \max(0, x)
$$

**优势**：
- 计算简单（只需判断正负）
- 缓解梯度消失（正区间梯度恒为 1）
- 稀疏激活（约 50% 的神经元输出为 0）

### 3.2 Dropout 正则化

```python
# 训练时：随机将 50% 的神经元置零
class Dropout(nn.Module):
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(x) * 0.5)
            return x * mask / 0.5  # 缩放保持期望不变
        return x
```

**直觉**：
- 强迫网络不依赖任何单个神经元
- 相当于同时训练多个子网络，取平均效果
- 降低过拟合

### 3.3 数据增强

AlexNet 使用了多种数据增强技术：

| 方法 | 说明 |
|------|------|
| 随机裁剪 | 从 256×256 裁剪出 224×224 |
| 水平翻转 | 50% 概率翻转 |
| 颜色抖动 | PCA 颜色扰动 |

**效果**：训练数据量隐式扩大了 2048 倍。

### 3.4 GPU 并行训练

AlexNet 使用 **2 个 GTX 580**（每个 3GB 显存）进行训练。

**策略**：
- 将网络分成两半，分别放在两个 GPU 上
- 只在特定层进行 GPU 间通信

```
GPU 0: Conv1 (48 filters) → Conv2 → Conv3 → Conv4 → Conv5
                                          ↕ (跨 GPU 连接)
GPU 1: Conv1 (48 filters) → Conv2 → Conv3 → Conv4 → Conv5
```

---

## 4. 代码实现

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 使用预训练模型
from torchvision import models

model = models.alexnet(pretrained=True)
print(model)
```

---

## 5. AlexNet 的影响

### 5.1 开启深度学习时代

| 影响 | 说明 |
|------|------|
| 学术界 | 深度学习成为主流研究方向 |
| 工业界 | 各大公司开始投入 AI 研发 |
| GPU 计算 | GPU 从游戏转向 AI 训练 |

### 5.2 后续发展

```
AlexNet (2012)
    ↓
ZFNet (2013) - 更好的超参数
    ↓
VGGNet (2014) - 更深（16-19层），全用 3×3 卷积
    ↓
GoogLeNet (2014) - Inception 模块，更宽
    ↓
ResNet (2015) - 残差连接，突破 100 层
```

### 5.3 今天的地位

AlexNet 今天已经很少直接使用，但它的重要性在于：
- 证明了深度学习的可行性
- 确立了 CNN 在视觉任务中的主导地位
- ReLU、Dropout、数据增强等技术至今仍在使用

---

## 6. 常见问题

**Q: 为什么 AlexNet 的第一层卷积核这么大（11×11）？**

早期网络倾向于使用大卷积核来快速扩大感受野。后来的研究（如 VGGNet）发现，堆叠多个 3×3 卷积效果更好。

**Q: AlexNet 能用在今天的任务上吗？**

可以，但有更好的选择（如 ResNet、EfficientNet）。AlexNet 主要用于教学和理解 CNN 发展史。

**Q: 为什么 AlexNet 需要两个 GPU？**

当时的 GPU 显存只有 3GB，放不下整个网络。今天的 GPU 显存通常在 16-80GB，单卡就能训练。

---

## 7. 小结

| 要点 | 内容 |
|------|------|
| 历史意义 | 2012 ImageNet 冠军，深度学习元年 |
| 关键创新 | ReLU、Dropout、数据增强、GPU 训练 |
| 架构 | 5 卷积 + 3 全连接，6000 万参数 |
| 影响 | 开启了深度学习的黄金时代 |

---

## 延伸阅读

- 原论文：Krizhevsky et al. (2012) "ImageNet Classification with Deep Convolutional Neural Networks"
- ImageNet 官网：[https://www.image-net.org/](https://www.image-net.org/)
- 可视化工具：[https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)
