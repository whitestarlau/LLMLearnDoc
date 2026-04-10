# 07-06 Diffusion：从噪声中创造

当我们谈论 AI 生成内容时，大多数人的第一反应可能是 ChatGPT 这样的文本模型。但就在 ChatGPT 引发全球关注的同一时期，另一项生成技术正在悄然改变视觉创作领域——这就是 Diffusion（扩散）模型。

2022 年，Midjourney 生成的"太空歌剧院"画作获奖，Stable Diffusion 开源让任何人都能在本地生成高质量图像，DALL·E 2 让"用文字画图像"成为现实。这些工具的背后，都是 Diffusion 模型在驱动。

**与语言模型"预测下一个 token"的生成方式不同，Diffusion 采用了一种完全相反的思路：它先学会如何"毁掉"一张图像（逐步添加噪声），再学会如何"重建"它（逐步去除噪声）。** 当模型完全掌握这套"先破坏再修复"的流程后，给它纯粹的噪声，它就能"修复"出从未存在过的新图像。

这种"逆向工程"的生成方式不仅训练稳定、生成质量高，而且为条件生成（如文本引导图像生成）提供了优雅的解决方案。从图像生成到视频合成，从 3D 建模到音频合成，Diffusion 正在重新定义 AI 内容创作的边界。

本章将深入讲解 Diffusion 模型的核心原理、数学形式和关键变体。

---

## 1. 从墨水扩散到图像生成

### 1.1 物理直觉：扩散过程的启示

Diffusion（扩散）这个词来自物理学：当你在一滴墨水里滴入水，墨水会逐渐扩散到整个水面，最终变得均匀。这个过程是**不可逆的**——墨水不会自动回到那小小的一滴。

AI研究者想到一个绝妙的主意：**如果把这个过程反过来呢？**

```
正常图像 → 加噪声 → 加噪声 → ... → 纯噪声  (前向扩散)
纯噪声 → 去噪声 → 去噪声 → ... → 新图像    (逆向生成)
```

### 1.2  Diffusion的核心思想

**两步走战略**：

**第一步：破坏（Forward Diffusion）**
- 准备一堆清晰的图片
- 逐步给它们添加噪声，直到变成纯噪声
- 这个过程可以精确控制（每一步加多少噪声是固定的）

**第二步：重建（Reverse Diffusion）**
- 训练一个神经网络
- 学会"逆转"这个过程：从噪声图片逐步还原出清晰图片
- 训练完成后，从纯噪声开始，就能"生成"新图片了！

**类比**：就像教一个人修复老照片
- 先把新照片故意弄脏、弄模糊
- 让他学习如何把脏照片变回干净的
- 学会后，他就能把一张完全脏的照片修复干净（甚至创造新图像）

### 1.3 为什么Diffusion这么火？

| 特点 | 优势 |
|------|------|
| **训练稳定** | 不像GAN那样容易训练崩溃 |
| **生成质量高** | 细节丰富，纹理自然 |
| **多样性好** | 不容易陷入模式collapse |
| **可解释性强** | 每个步骤清晰可控 |

**缺点**：采样慢（需要多步迭代），但近年通过各种加速技术已经大大改善。

---

## 2. 数学原理

### 2.1 前向扩散过程

前向过程是给数据逐步添加噪声。设原始数据为 $x_0$，经过 $T$ 步扩散后变成 $x_T$（近似纯噪声）。

**关键设计**：每一步的噪声水平由超参数 $\beta_t$ 控制：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \cdot x_{t-1}, \beta_t \cdot \mathbf{I})$$

**直观理解**：
- $\beta_t$ 是第 $t$ 步的噪声方差
- 系数 $\sqrt{1-\beta_t}$ 保证每一步后 $x_t$ 的方差不会爆炸
- 最终 $x_T \approx \mathcal{N}(0, \mathbf{I})$（纯噪声）

**重要性质**：我们可以直接计算任意时刻 $t$ 的 $x_t$，无需逐步计算：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$，$\epsilon \sim \mathcal{N}(0, \mathbf{I})$

### 2.2 逆向扩散过程

逆向过程是前向的逆：我们想从噪声 $x_T$ 恢复出 $x_0$。

**核心思想**：训练一个神经网络 $p_\theta(x_{t-1} | x_t)$ 来预测每一步应该如何去噪。

**条件概率**：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

**训练目标**：让模型预测出每一步被添加的噪声 $\epsilon$

$$\text{Loss} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

这个简单的 $L_2$ 损失函数就是Diffusion训练的全部！

### 2.3 采样过程

训练好后，采样（生成新图像）的过程：

```
输入：纯噪声 x_T ~ N(0, I)
for t = T to 1:
    z = random_noise() if t > 1 else 0
    x_{t-1} = (x_t - predicted_noise) / sqrt(alpha_t) + z * sigma_t
输出：x_0 (生成的图像)
```

**简化理解**：
- 每一步，模型预测"这里有多少噪声"
- 把噪声减掉，就恢复了一点图像
- 加一点随机性（防止结果太死板）
- 重复直到完全恢复

---

## 3. 重要变体与改进

### 3.1 DDPM → DDIM：加速采样

原始DDPM需要1000步采样，太慢了。DDIM提出一种更高效的采样策略：

**核心思想**：跳过一些中间步骤，采样速度提升10倍不止，质量几乎不变。

### 3.2 Classifier-Free Guidance：更好的条件生成

如何在Diffusion中实现"根据文本生成图像"？

**旧方法**：训练一个额外的分类器引导生成

**新方法（无分类器引导）**：
- 训练一个同时支持条件和非条件的模型
- 生成时：用条件预测 - 非条件预测 × 引导系数
- 效果：文本控制力更强，生成质量更高

### 3.3 Stable Diffusion：潜空间Diffusion

**问题**：直接在像素空间做Diffusion太慢（512×512图像，每步要处理海量像素）

**解决方案**：先压缩到"潜空间"再Diffusion

```
图像 → VAE编码器 → 潜向量 → Diffusion → 新潜向量 → VAE解码器 → 新图像
```

**效果**：速度提升100倍，显存占用大幅降低！

---

## 4. 代码实现

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x_0, t, noise):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).reshape(-1, 1, 1, 1)
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    def sample(self, model, shape, device):
        x_t = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x_t, t_batch)
            
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            
            x_t = (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
            x_t = x_t + torch.sqrt(beta_t) * noise
        
        return x_t

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(1, time_emb_dim)
        self.fc = nn.Linear(time_emb_dim, out_channels)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x, t):
        h = self.act(self.norm(self.conv1(x)))
        t_emb = self.act(self.time_mlp(t.float().unsqueeze(-1)))
        h = h + self.fc(t_emb)[:, :, None, None]
        h = self.act(self.norm(self.conv2(h)))
        return h

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.down1 = UNetBlock(in_channels, base_channels)
        self.down2 = UNetBlock(base_channels, base_channels * 2)
        self.up2 = UNetBlock(base_channels * 2, base_channels)
        self.up1 = UNetBlock(base_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(out_channels, out_channels, 1)
    
    def forward(self, x, t):
        x1 = self.down1(x, t)
        x2 = self.down2(self.pool(x1), t)
        x = self.up2(self.upsample(x2), t)
        x = self.up1(self.upsample(x), t)
        return self.final(x)

diffusion = SimpleDiffusion()
model = SimpleUNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    for batch in dataloader:
        x_0 = batch.cuda()
        t = torch.randint(0, 1000, (x_0.shape[0],)).cuda()
        noise = torch.randn_like(x_0)
        x_t = diffusion.add_noise(x_0, t, noise)
        
        predicted_noise = model(x_t, t)
        loss = nn.functional.mse_loss(noise, predicted_noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. 总结

Diffusion模型为什么重要：

1. **生成范式的革新**：从GAN的直接生成 → 逐步迭代优化
2. **训练稳定性**：不再需要对抗训练的复杂博弈
3. **生成质量**：在图像生成领域达到SOTA
4. **应用广泛**：文本到图像、图像编辑、视频生成...

**代表作品**：DALL·E 2、Stable Diffusion、Midjourney、Sora...

Diffusion已经彻底改变了AI生成内容的格局！