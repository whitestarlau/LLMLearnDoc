# 优化器：SGD到Adam

## 1. 为什么需要优化器

神经网络的训练本质上是一个**优化问题**：找到使损失函数最小的参数。

```
参数空间：
    L(w)
    ^
    |  * 起点（随机初始化）
    |   \
    |    \  ← 如何找到下降方向？
    |     \
    |      * 终点（最优参数）
    +---------> w
```

**梯度下降**的基本思想：沿着梯度的反方向更新参数：

$$
w_{t+1} = w_t - \eta \cdot \nabla L(w_t)
$$

但原始梯度下降存在很多问题，优化器就是为了解决这些问题而设计的。

---

## 2. SGD：最基础的优化器

### 2.1 批量梯度下降（BGD）

每次用**全部数据**计算梯度：

$$
w_{t+1} = w_t - \eta \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla L_i(w_t)
$$

**问题**：
- 每次更新需要遍历全部数据，计算量大
- 无法处理超出内存的数据
- 容易陷入局部最优

### 2.2 随机梯度下降（SGD）

每次只用**一个样本**计算梯度：

$$
w_{t+1} = w_t - \eta \cdot \nabla L_i(w_t)
$$

**优点**：
- 更新频繁，收敛快
- 随机性有助于跳出局部最优

**问题**：
- 更新方向噪声大，收敛不稳定
- 无法利用向量化加速

### 2.3 小批量梯度下降（Mini-batch SGD）

折中方案：每次用一小批数据（如32、64个样本）：

$$
w_{t+1} = w_t - \eta \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(w_t)
$$

**这是实践中最常用的形式**，通常直接称为"SGD"。

```python
# PyTorch 实现
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

---

## 3. 动量：加速与稳定

### 3.1 直觉理解

想象一个球在山坡上滚动：

```
无动量：              有动量：
    *                    *
   /                    /
  /     震荡大         /      累积方向，平稳前进
 /                   ------>
------>
```

- **没有动量**：每一步只看当前梯度，在沟壑中来回震荡
- **有动量**：累积历史梯度方向，抑制震荡，加速收敛

### 3.2 数学形式

引入速度变量 $v$，累积梯度的指数移动平均：

$$
v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla L(w_t)
$$

$$
w_{t+1} = w_t - v_t
$$

其中 $\beta$ 是动量系数，通常取 0.9。

**展开形式**：

$$
v_t = \eta \sum_{k=0}^{t} \beta^{t-k} \cdot \nabla L(w_k)
$$

距离当前时刻越远的梯度，权重越小（指数衰减）。

### 3.3 Nesterov 动量

先根据动量"预测"下一步位置，再计算梯度：

$$
v_t = \beta \cdot v_{t-1} + \eta \cdot \nabla L(w_t - \beta \cdot v_{t-1})
$$

$$
w_{t+1} = w_t - v_t
$$

**效果**：在接近最优时减少震荡，收敛更快。

```python
# PyTorch 实现
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

---

## 4. 自适应学习率方法

### 4.1 为什么需要自适应学习率

不同参数的重要性不同：

```
参数 w₁：梯度一直很大  → 需要小学习率
参数 w₂：梯度一直很小  → 需要大学习率
```

统一的学习率无法满足所有参数的需求。

### 4.2 AdaGrad

**思想**：累积历史梯度的平方，对频繁更新的参数减小学习率。

$$
G_t = G_{t-1} + (\nabla L(w_t))^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot \nabla L(w_t)
$$

**优点**：自动调整每个参数的学习率。

**问题**：$G_t$ 只增不减，学习率会越来越小，最终停止学习。

### 4.3 RMSProp

**改进**：用指数移动平均代替累积和，解决学习率过早衰减的问题。

$$
E[g^2]_t = \beta \cdot E[g^2]_{t-1} + (1-\beta) \cdot (\nabla L(w_t))^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot \nabla L(w_t)
$$

通常 $\beta = 0.9$，$\epsilon = 10^{-8}$。

---

## 5. Adam：集大成者

### 5.1 核心思想

Adam = **Adaptive Moment Estimation**

结合了：
- **动量**（一阶矩）：累积梯度方向
- **RMSProp**（二阶矩）：自适应学习率

### 5.2 数学形式

**一阶矩（动量）**：

$$
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot \nabla L(w_t)
$$

**二阶矩（学习率缩放）**：

$$
v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot (\nabla L(w_t))^2
$$

**偏差修正**（解决初始时刻估计偏小的问题）：

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

**参数更新**：

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

### 5.3 默认超参数

- $\beta_1 = 0.9$（动量系数）
- $\beta_2 = 0.999$（二阶矩系数）
- $\epsilon = 10^{-8}$
- $\eta = 0.001$（学习率）

### 5.4 代码实现

```python
# PyTorch 实现
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)

# 带权重衰减的 Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 5.5 AdamW：修正的权重衰减

Adam 中直接使用 `weight_decay` 会与自适应学习率产生冲突。AdamW 将权重衰减与梯度更新解耦：

```python
# Adam: 权重衰减在梯度更新中
grad = grad + weight_decay * w
m = beta1 * m + (1-beta1) * grad

# AdamW: 权重衰减独立应用
m = beta1 * m + (1-beta1) * grad
w = w - lr * (m / sqrt(v) + weight_decay * w)
```

**现代大模型（GPT、BERT）通常使用 AdamW。**

---

## 6. 学习率调度

### 6.1 为什么需要学习率调度

```
训练初期：需要大学习率快速下降
训练后期：需要小学习率精细调优
```

```
Loss
  ^
  |\
  | \
  |  \
  |   \_____
  |         \___  ← 学习率太大，在最优解附近震荡
  +-------------> Epoch
```

### 6.2 常见调度策略

#### Step Decay

每 $s$ 个 epoch 学习率乘以 $\gamma$：

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

#### Cosine Annealing

学习率按余弦函数从初始值衰减到 0：

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))
$$

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

#### Warmup + Decay

先线性增加学习率（Warmup），再衰减：

```
学习率
  ^
  |    /\
  |   /  \
  |  /    \___________
  | /                  \___
  +-------------------------> Step
```

```python
# Transformer 常用：Warmup + Cosine Decay
def get_lr(step, d_model, warmup_steps):
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, 512, 4000))
```

#### ReduceLROnPlateau

监控指标，停滞时降低学习率：

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)
```

### 6.3 完整训练循环示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 模型、优化器、调度器
model = MyModel()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    # 更新学习率
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")
```

---

## 7. 优化器对比与选择

### 7.1 收敛速度对比

```
Loss
  ^
  |\
  | \ SGD（慢，震荡）
  |  \
  |   \___ SGD+Momentum（较快）
  |       \
  |        \__ Adam（最快收敛）
  |           \____
  +-----------------> Epoch
```

### 7.2 选择建议

| 场景 | 推荐优化器 | 说明 |
|------|-----------|------|
| CV 任务（ResNet等） | SGD + Momentum | 泛化性能更好 |
| NLP / Transformer | AdamW | 默认选择 |
| GAN | Adam | 生成器和判别器都用 |
| 快速实验 | Adam | 收敛快，调参简单 |
| 稀疏数据（推荐系统） | AdaGrad / Adam | 自适应学习率有优势 |

### 7.3 泛化性能的争议

有研究表明，SGD（带动量）在某些任务上的泛化性能优于 Adam：

- Adam 收敛到尖锐的最小值
- SGD 收敛到平坦的最小值，泛化更好

```
尖锐最小值（Adam）：     平坦最小值（SGD）：
Loss                     Loss
  ^                        ^
  |   /\                   |
  |  /  \                  |  ___
  | /    \                 | /   \
  |/      \_____           |/     \_____
  +---------> w            +---------> w
  训练好，测试差           训练测试都好
```

---

## 8. 常见问题

**Q: Adam 的学习率应该设多少？**

常用范围：$10^{-4}$ 到 $10^{-3}$。比 SGD 的学习率小一个数量级。

**Q: 什么时候用 AdamW 而不是 Adam？**

需要权重衰减时（几乎所有场景），用 AdamW。AdamW 是 Adam 的改进版，现代大模型默认使用。

**Q: 学习率 Warmup 有必要吗？**

对于 Transformer 架构，Warmup 几乎是必须的。否则训练初期可能不稳定甚至发散。

**Q: 梯度裁剪设多少合适？**

常用值：`max_norm=1.0` 或 `max_norm=5.0`。用于防止梯度爆炸。

**Q: 如何选择 batch size？**

- 大 batch：训练快，但泛化可能变差
- 小 batch：训练慢，但有正则化效果
- 常用：32、64、128、256

---

## 9. 小结

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD | 简单，泛化好 | CNN、图像分类 |
| SGD + Momentum | 加速收敛，抑制震荡 | 大多数 CV 任务 |
| AdaGrad | 自适应学习率 | 稀疏数据 |
| RMSProp | AdaGrad 的改进 | RNN |
| Adam | 动量 + 自适应 | 通用，快速实验 |
| AdamW | Adam + 解耦权重衰减 | Transformer、大模型 |

**核心演进路线**：
```
SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW
```

---

## 延伸阅读

- Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
- Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization"（AdamW 原论文）
- Ruder (2016) "An overview of gradient descent optimization algorithms"
- Smith (2017) "Cyclical Learning Rates for Training Neural Networks"