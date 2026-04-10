# 07-02 FlashAttention：IO感知的优化

在上一章中，我们学习了如何通过采样策略控制生成文本的质量——从贪心解码到温度采样，再到 Top-p 采样，这些技术让我们能在确定性和创造性之间找到平衡。

但这些控制策略的应用有一个前提：模型必须能够**高效地处理长序列**。当你使用 Top-p 采样生成一篇长文章，或者使用束搜索探索多个候选序列时，序列长度的增加会急剧放大计算开销。问题的核心在于注意力机制：标准实现需要存储一个与序列长度平方成正比的注意力矩阵，这使得长序列生成很快耗尽显存。

**本章将深入探讨 FlashAttention**——一种革命性的注意力算法。它通过重新组织计算流程，在不改变数学结果的前提下，将内存复杂度从 O(N²) 降到 O(N)，使模型能够处理 4-8 倍更长的序列。这项技术自 2022 年提出以来，已成为现代大模型训练和推理的事实标准。

---

## 1. 注意力计算的内存困境

### 1.1 标准注意力的内存瓶颈

回顾注意力机制的计算：

```python
# 标准注意力
scores = Q @ K.T / sqrt(d_k)    # [N, N] 注意力分数矩阵
attn = softmax(scores, dim=-1)   # [N, N] 注意力权重矩阵
output = attn @ V                # [N, d]
```

**问题在哪里？**

当序列长度 N = 4096，批次大小 B = 32，头数 H = 12 时：

```
注意力矩阵大小：B × H × N × N × 4 bytes (float32)
                = 32 × 12 × 4096 × 4096 × 4
                ≈ 24 GB
```

这还只是**一层**的注意力矩阵！一个 24 层的 GPT 模型，中间激活值需要近 **600GB** 显存——显然不现实。

### 1.2 内存墙：计算的隐藏成本

GPU 内存层次：

```
┌─────────────────────────────────────┐
│  HBM (High Bandwidth Memory)        │ ← 容量大 (40GB+)，速度慢 (~1TB/s)
│  存储模型参数和大部分激活值           │
├─────────────────────────────────────┤
│  SRAM (Static RAM / Shared Memory)  │ ← 容量小 (~100KB)，速度快 (~19TB/s)
│  用于 CUDA 核的计算                  │
└─────────────────────────────────────┘
```

**关键洞察**：
- 标准注意力需要频繁读写 HBM（存储 Q、K、V，以及中间结果 S、P）
- HBM 带宽是瓶颈，计算单元大部分时间都在等待数据
- **内存读写比计算更慢、更耗能**

### 1.3 FlashAttention 的核心思想

**类比**：

想象你要计算一个大表格的统计值：

```
标准方法：
1. 把整个表格从仓库（HBM）搬到办公室（SRAM）
2. 计算一部分结果
3. 把中间结果存回仓库
4. 重复 N 次

FlashAttention：
1. 把表格切成小块
2. 一次只拿一小块到办公室
3. 在当前块内完成所有计算
4. 只把最终结果存回仓库
```

**核心创新**：
1. **分块计算**：将 Q、K、V 切分成小块，逐块加载到 SRAM
2. **在线 Softmax**：不存储完整的注意力矩阵，边计算边归一化
3. **重计算（Recomputation）**：反向传播时不存储前向激活值，重新计算

---

## 2. 算法原理

### 2.1 Softmax 的分块计算

标准 Softmax 需要看到所有数值才能归一化：

```
softmax(x_i) = exp(x_i) / Σ(exp(x_j))
```

**挑战**：如果我们只能一次看一部分数据，如何计算正确的 softmax？

#### 在线 Softmax 技巧

假设我们有两批数据：
- 第一批：`[x1, x2]`，当前最大值 `m1 = max(x1, x2)`，和 `s1 = exp(x1-m1) + exp(x2-m1)`
- 第二批：`[x3, x4]`，当前最大值 `m2 = max(x3, x4)`，和 `s2 = exp(x3-m2) + exp(x4-m2)`

**合并两批**：

```
m_new = max(m1, m2)                          # 新的全局最大值

# 调整第一批
s1' = s1 × exp(m1 - m_new)                   # 按新最大值缩放

# 调整第二批  
s2' = s2 × exp(m2 - m_new)                   # 按新最大值缩放

s_new = s1' + s2'                            # 新的全局和

softmax(x_i) = exp(x_i - m_new) / s_new      # 正确的归一化
```

**关键**：只维护 `m`（最大值）和 `s`（缩放后的和），不需要存储所有中间值。

### 2.2 FlashAttention 算法流程

```
输入：Q, K, V ∈ R^(N×d)，分块大小 B_r, B_c
输出：O ∈ R^(N×d)（注意力输出），L ∈ R^N（logsumexp，用于反向传播）

1. 初始化：
   - O = 0 ∈ R^(N×d)          # 输出矩阵
   - ℓ = 0 ∈ R^N              # 每行的指数和
   - m = -∞ ∈ R^N             # 每行的最大值

2. 将 Q 分成 T_r = ⌈N/B_r⌉ 块：Q_1, Q_2, ..., Q_Tr
   将 K, V 分成 T_c = ⌈N/B_c⌉ 块：K_1, ..., K_Tc, V_1, ..., V_Tc

3. 对于每个 Q 块 i = 1 to T_r：
   
   加载 Q_i 到 SRAM
   
   对于每个 K, V 块 j = 1 to T_c：
     加载 K_j, V_j 到 SRAM
     
     # 计算当前块的注意力分数
     S_ij = Q_i @ K_j^T          # [B_r, B_c]
     
     # 在线更新 softmax 统计量
     m̃_ij = rowmax(S_ij)         # 当前块的行最大值
     P̃_ij = exp(S_ij - m̃_ij)     # 当前块的指数（未归一化）
     ℓ̃_ij = rowsum(P̃_ij)         # 当前块的指数和
     
     # 更新全局统计量
     m_new = max(m_i, m̃_ij)
     ℓ_new = exp(m_i - m_new) × ℓ_i + exp(m̃_ij - m_new) × ℓ̃_ij
     
     # 更新输出（核心：在线累加）
     O_i = (ℓ_i × exp(m_i - m_new) × O_i + exp(m̃_ij - m_new) × P̃_ij @ V_j) / ℓ_new
     
     # 保存新的统计量
     m_i = m_new
     ℓ_i = ℓ_new
   
   将 O_i 写回 HBM
   将 L_i = m_i + log(ℓ_i) 写回 HBM（用于反向传播）
```

**关键洞察**：
- 每次只加载小块 Q、K、V 到 SRAM
- 不存储完整的 S 和 P 矩阵
- 通过维护 `m` 和 `ℓ`，在线计算归一化的输出

### 2.3 复杂度对比

| 指标 | 标准注意力 | FlashAttention |
|------|-----------|----------------|
| **FLOPs** | O(N²d) | O(N²d) （计算量相同） |
| **HBM 访问量** | O(N²) | O(N²/B_c) + O(N) ≈ **O(N)** |
| **内存占用** | O(N²) | **O(N)** |
| **可处理序列长度** | 受显存限制 | 提升 4-8 倍 |

**注意**：FlashAttention 没有减少计算量（FLOPs），而是减少了**内存读写**（HBM 访问）。

---

## 3. 代码实现

### 3.1 基础版：理解算法核心

```python
import torch
import torch.nn.functional as F
import math

def flash_attention_basic(Q, K, V, block_size=64):
    """
    FlashAttention 简化实现（用于理解算法）
    
    Args:
        Q, K, V: [batch, num_heads, seq_len, head_dim]
        block_size: 分块大小
    
    Returns:
        O: [batch, num_heads, seq_len, head_dim]
        L: [batch, num_heads, seq_len] (logsumexp，用于反向传播)
    """
    B, H, N, d = Q.shape
    
    # 初始化输出和统计量
    O = torch.zeros_like(Q)
    L = torch.zeros(B, H, N, device=Q.device)  # logsumexp
    
    # 计算块数
    num_blocks = (N + block_size - 1) // block_size
    
    # 分块遍历
    for b in range(B):
        for h in range(H):
            for i in range(num_blocks):
                # 当前 Q 块的范围
                q_start = i * block_size
                q_end = min((i + 1) * block_size, N)
                q_len = q_end - q_start
                
                # 加载 Q_i
                Q_i = Q[b, h, q_start:q_end]  # [q_len, d]
                
                # 初始化当前块的统计量
                m_i = torch.full((q_len,), float('-inf'), device=Q.device)  # 最大值
                l_i = torch.zeros(q_len, device=Q.device)                    # 指数和
                O_i = torch.zeros(q_len, d, device=Q.device)                 # 输出累加器
                
                # 遍历所有 K, V 块
                for j in range(num_blocks):
                    k_start = j * block_size
                    k_end = min((j + 1) * block_size, N)
                    k_len = k_end - k_start
                    
                    # 加载 K_j, V_j
                    K_j = K[b, h, k_start:k_end]  # [k_len, d]
                    V_j = V[b, h, k_start:k_end]  # [k_len, d]
                    
                    # 计算 S_ij = Q_i @ K_j^T
                    S_ij = Q_i @ K_j.T  # [q_len, k_len]
                    S_ij = S_ij / math.sqrt(d)
                    
                    # 计算当前块的 softmax 统计量
                    m_ij = S_ij.max(dim=-1).values  # [q_len]
                    P_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))  # [q_len, k_len]
                    l_ij = P_ij.sum(dim=-1)  # [q_len]
                    
                    # 更新全局统计量（在线 softmax 的核心）
                    m_new = torch.maximum(m_i, m_ij)
                    
                    # 调整旧值
                    alpha = torch.exp(m_i - m_new)  # [q_len]
                    beta = torch.exp(m_ij - m_new)  # [q_len]
                    
                    l_new = alpha * l_i + beta * l_ij
                    
                    # 更新输出（先缩放旧值，再加新值）
                    O_i = alpha.unsqueeze(-1) * O_i + beta.unsqueeze(-1) * (P_ij @ V_j)
                    O_i = O_i / l_new.unsqueeze(-1)
                    
                    # 保存新的统计量
                    m_i = m_new
                    l_i = l_new
                
                # 写回 HBM
                O[b, h, q_start:q_end] = O_i
                L[b, h, q_start:q_end] = m_i + torch.log(l_i)
    
    return O, L
```

### 3.2 向量化优化版

```python
def flash_attention_vectorized(Q, K, V, block_q=64, block_kv=64):
    """
    向量化优化的 FlashAttention
    注意：这是教学实现，实际生产环境应使用 CUDA 核
    """
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    
    O = torch.zeros_like(Q)
    M = torch.full((B, H, N), float('-inf'), device=Q.device)  # 行最大值
    L = torch.zeros(B, H, N, device=Q.device)                   # 缩放后的行和
    
    num_q_blocks = (N + block_q - 1) // block_q
    num_kv_blocks = (N + block_kv - 1) // block_kv
    
    for i in range(num_q_blocks):
        q_start = i * block_q
        q_end = min((i + 1) * block_q, N)
        
        # 加载当前 Q 块 [B, H, block_q, d]
        Q_i = Q[:, :, q_start:q_end]
        q_len = Q_i.size(2)
        
        # 初始化当前块的累加器
        O_i = torch.zeros(B, H, q_len, d, device=Q.device)
        m_i = torch.full((B, H, q_len), float('-inf'), device=Q.device)
        l_i = torch.zeros(B, H, q_len, device=Q.device)
        
        for j in range(num_kv_blocks):
            k_start = j * block_kv
            k_end = min((j + 1) * block_kv, N)
            
            # 加载 K_j, V_j
            K_j = K[:, :, k_start:k_end]
            V_j = V[:, :, k_start:k_end]
            
            # 计算注意力分数 [B, H, block_q, block_kv]
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale
            
            # 在线 softmax 更新
            m_ij = S_ij.max(dim=-1, keepdim=True).values  # [B, H, block_q, 1]
            P_ij = torch.exp(S_ij - m_ij)                  # [B, H, block_q, block_kv]
            l_ij = P_ij.sum(dim=-1, keepdim=True)          # [B, H, block_q, 1]
            
            # 更新全局统计量
            m_new = torch.maximum(m_i.unsqueeze(-1), m_ij.squeeze(-1)).squeeze(-1)
            
            # 计算调整系数
            alpha = torch.exp(m_i - m_new)  # [B, H, block_q]
            beta = torch.exp(m_ij.squeeze(-1) - m_new)  # [B, H, block_q]
            
            # 更新 l_i
            l_new = alpha * l_i + beta * l_ij.squeeze(-1)
            
            # 更新输出：先缩放旧输出，再加新贡献
            O_i = alpha.unsqueeze(-1) * O_i + torch.matmul(P_ij, V_j)
            O_i = O_i / l_new.unsqueeze(-1)
            
            # 保存统计量
            m_i = m_new
            l_i = l_new
        
        # 写回
        O[:, :, q_start:q_end] = O_i
    
    return O
```

### 3.3 使用 FlashAttention 库

生产环境推荐使用官方实现：

```python
# 安装: pip install flash-attn

try:
    from flash_attn import flash_attn_func
    
    # 使用 FlashAttention-2
    def efficient_attention(Q, K, V, causal=False, softmax_scale=None):
        """
        使用 FlashAttention-2 的高效实现
        
        Args:
            Q, K, V: [batch, seq_len, num_heads, head_dim]
            causal: 是否使用因果掩码（用于自回归模型）
            softmax_scale: 缩放因子，默认 1/sqrt(d)
        """
        # FlashAttention 期望的输入格式: [batch, seqlen, nheads, headdim]
        output = flash_attn_func(
            Q, K, V,
            causal=causal,
            softmax_scale=softmax_scale
        )
        return output

except ImportError:
    print("FlashAttention not installed, falling back to standard attention")
    
    def efficient_attention(Q, K, V, causal=False, softmax_scale=None):
        """Fallback to standard scaled dot-product attention"""
        d_k = Q.size(-1)
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d_k)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
        
        if causal:
            # 因果掩码
            seq_len = Q.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output
```

### 3.4 内存使用对比实验

```python
import torch
import tracemalloc

def memory_profile(func, *args, **kwargs):
    """分析函数内存使用"""
    tracemalloc.start()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    cuda_peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return {
        'peak_memory_mb': peak / 1024**2,
        'cuda_peak_mb': cuda_peak
    }


def compare_memory(seq_len, d_model=64, num_heads=8, batch_size=2):
    """对比标准注意力和 FlashAttention 的内存使用"""
    head_dim = d_model // num_heads
    
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    K = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    V = torch.randn(batch_size, seq_len, num_heads, head_dim, device='cuda')
    
    # 标准注意力
    def standard_attn():
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
    
    # FlashAttention
    def flash_attn():
        return flash_attention_vectorized(
            Q.transpose(1, 2), 
            K.transpose(1, 2), 
            V.transpose(1, 2),
            block_q=64, 
            block_kv=64
        ).transpose(1, 2)
    
    print(f"\n序列长度: {seq_len}")
    print("-" * 50)
    
    mem_std = memory_profile(standard_attn)
    print(f"标准注意力 - CUDA 峰值内存: {mem_std['cuda_peak_mb']:.1f} MB")
    
    mem_flash = memory_profile(flash_attn)
    print(f"FlashAttention - CUDA 峰值内存: {mem_flash['cuda_peak_mb']:.1f} MB")
    
    reduction = mem_std['cuda_peak_mb'] / mem_flash['cuda_peak_mb']
    print(f"内存减少倍数: {reduction:.2f}x")

# 运行对比
# for seq_len in [512, 1024, 2048, 4096]:
#     compare_memory(seq_len)
```

---

## 4. FlashAttention 版本演进

### 4.1 FlashAttention-1 → FlashAttention-2

| 特性 | FlashAttention-1 | FlashAttention-2 |
|------|------------------|------------------|
| 并行策略 | 在 batch/head 维度并行 | 额外在 seq 维度并行 |
| softmax 计算 | 需要额外 kernel 调用 | 融合到单个 kernel |
| 速度提升 | 2-4x | 2-3x（比 FA-1） |
| 支持特性 | 基础 attention | 支持 GQA、ALiBi 等 |

### 4.2 FlashAttention-3（Hopper GPU 优化）

针对 NVIDIA H100 (Hopper 架构) 的新特性：

```python
# FlashAttention-3 利用的新硬件特性
features = {
    "WGMMA": "Warp Group Matrix Multiply-Accumulate",  # 更快的矩阵乘法
    "TMA": "Tensor Memory Accelerator",                 # 异步数据加载
    "FP8": "8-bit floating point",                      # 更低精度、更快计算
    "Softmax": "硬件加速 softmax"
}

# 使用示例（需要 Hopper GPU）
# from flash_attn import flash_attn_func
# out = flash_attn_func(q, k, v, use_fp8=True)  # 使用 FP8 精度
```

**性能提升**：在 H100 上比 FlashAttention-2 再快 1.5-2 倍。

### 4.3 变体：PagedAttention

用于**推理阶段**的 KV Cache 管理：

```
问题：自回归生成时，每次都要存储所有位置的 K 和 V
      长序列 + 大 batch → KV Cache 占用巨量显存

PagedAttention 解决方案：
1. 将 KV Cache 分成固定大小的 "块"（pages）
2. 像操作系统管理内存一样动态分配块
3. 支持块共享（例如 beam search 中的多个候选）
```

```python
# vLLM 使用 PagedAttention 实现高效推理
from vllm import LLM, SamplingParams

# PagedAttention 自动管理 KV Cache
llm = LLM(model="meta-llama/Llama-2-7b")
outputs = llm.generate(prompts, sampling_params)
```

---

## 5. 实际应用配置

### 5.1 不同场景的推荐设置

```python
# 场景 1：训练大模型（长序列）
def training_config():
    """训练时使用 FlashAttention 减少激活值内存"""
    return {
        "attn_implementation": "flash_attention_2",
        "block_size": 128,  # 根据 GPU SRAM 调整
        "causal": True,     # 自回归模型
    }

# 场景 2：推理加速（vLLM + PagedAttention）
def inference_config():
    """推理时使用 PagedAttention 高效管理 KV Cache"""
    return {
        "framework": "vllm",
        "block_size": 16,           # KV cache 块大小
        "gpu_memory_utilization": 0.9,  # 显存使用率
        "max_num_seqs": 256,        # 最大并发序列数
    }

# 场景 3：变长序列处理
def variable_length_config():
    """处理不同长度的序列批次"""
    return {
        "padding_side": "left",     # 左填充，便于生成
        "flash_attn_varlen": True,  # 使用变长版本
    }
```

### 5.2 与 HuggingFace Transformers 集成

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 自动检测并使用 FlashAttention
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2",  # 启用 FlashAttention
    torch_dtype=torch.float16,
    device_map="auto"
)

# 或者使用 SDPA (PyTorch 原生，包含 FlashAttention 内核)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="sdpa",  # Scaled Dot Product Attention
)
```

---

## 6. 常见陷阱与 FAQ

### Q1: FlashAttention 支持所有 GPU 吗？

**不支持**。FlashAttention 需要：
- NVIDIA GPU (Turing 架构及以上，即 RTX 20 系列、V100、A100、H100)
- CUDA 11.6+ / CUDA 12.x
- 足够的共享内存（SRAM）

**不支持**：
- AMD GPU（有社区版本但不完整）
- CPU
- 老旧的 NVIDIA GPU（如 GTX 1080）

```python
def check_flash_attention_support():
    """检查是否支持 FlashAttention"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    capability = torch.cuda.get_device_capability()
    major, minor = capability
    
    if major < 7:  # Turing 是 7.5
        return False, f"GPU compute capability {major}.{minor} < 7.0"
    
    return True, "FlashAttention supported"
```

### Q2: 为什么 FlashAttention 有时反而更慢？

**可能原因**：
1. **序列太短**：短序列时，分块开销 > 内存节省收益
2. **分块大小不合适**：块太小导致 kernel 启动开销大
3. **头维度不匹配**：某些 head_dim 未被优化

**建议**：
```python
# 只有当序列长度 > 512 时才使用 FlashAttention
use_flash = seq_len > 512

# 推荐分块大小（根据 GPU 调整）
block_sizes = {
    "A100": 128,
    "V100": 64,
    "RTX3090": 64,
}
```

### Q3: FlashAttention 和梯度检查点能一起用吗？

**可以**，且推荐组合使用：

```python
# 组合使用：FlashAttention + 梯度检查点
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2",
)

# 启用梯度检查点（以计算换内存）
model.gradient_checkpointing_enable()

# 这种组合可以训练极长的序列
```

### Q4: 因果注意力（Causal）在 FlashAttention 中如何实现？

FlashAttention 原生支持因果掩码，**不需要显式构造 mask 矩阵**：

```python
# 标准实现：需要存储 [N, N] 的 mask
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()

# FlashAttention：通过算法避免存储 mask
# 只计算上三角部分，下三角直接忽略
output = flash_attn_func(Q, K, V, causal=True)  # 不增加内存开销！
```

### Q5: 多头注意力（MHA）vs 分组查询注意力（GQA）

FlashAttention-2+ 支持 GQA 和 MQA：

```python
# 标准 MHA：num_heads_k == num_heads_q
# GQA：num_heads_k < num_heads_q（多个 query 共享一组 KV）

# FlashAttention 自动处理 GQA
from flash_attn import flash_attn_func

# Q: [batch, seqlen, num_heads_q, head_dim]
# K: [batch, seqlen, num_heads_k, head_dim]  # num_heads_k 可以是 1 或 8
# V: [batch, seqlen, num_heads_k, head_dim]

output = flash_attn_func(Q, K, V)  # 自动广播 KV
```

### Q6: 调试 FlashAttention 问题

```python
# 启用调试模式
debug_flash_attention = True

if debug_flash_attention:
    # 1. 检查输入维度
    assert Q.dtype in [torch.float16, torch.bfloat16], "FA2 requires fp16/bf16"
    assert Q.dim() == 4, f"Expected 4D input, got {Q.dim()}D"
    
    # 2. 回退到标准实现验证结果
    standard_out = standard_attention(Q, K, V)
    flash_out = flash_attn_func(Q, K, V)
    
    max_diff = (standard_out - flash_out).abs().max()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-3, "FlashAttention output mismatch!"
```

---

## 7. 核心要点总结

```
FlashAttention = 分块 + 在线 Softmax + 减少 HBM 访问

┌─────────────────────────────────────────────────────────────┐
│  问题：标准注意力的 O(N²) 内存瓶颈                           │
│  └── 注意力矩阵 S, P ∈ R^(N×N) 占用巨量显存                  │
├─────────────────────────────────────────────────────────────┤
│  核心洞察：GPU 内存层次                                     │
│  ├── HBM：容量大、速度慢 (瓶颈)                             │
│  └── SRAM：容量小、速度快 (充分利用)                        │
├─────────────────────────────────────────────────────────────┤
│  解决方案：分块计算                                         │
│  1. 将 Q, K, V 切分为小块                                   │
│  2. 每次加载一小块到 SRAM                                   │
│  3. 在线计算 softmax（不存储中间矩阵）                      │
│  4. 只写回最终结果                                          │
├─────────────────────────────────────────────────────────────┤
│  关键算法：在线 Softmax                                     │
│  m_new = max(m_old, m_new)                                  │
│  l_new = exp(m_old - m_new) * l_old + exp(m_new - m_new) * l_new │
├─────────────────────────────────────────────────────────────┤
│  效果                                                       │
│  ├── 内存：O(N²) → O(N)                                     │
│  ├── 速度：2-4x 提升（受限于内存带宽）                      │
│  └── 序列长度：提升 4-8 倍                                  │
├─────────────────────────────────────────────────────────────┤
│  版本演进                                                   │
│  ├── FA-1：基础分块算法                                     │
│  ├── FA-2：更多并行、更好性能                               │
│  └── FA-3：Hopper 架构优化 (FP8, TMA)                       │
├─────────────────────────────────────────────────────────────┤
│  相关技术                                                   │
│  ├── PagedAttention：推理阶段的 KV Cache 管理               │
│  └── GQA/MQA：减少 KV Cache 大小                           │
└─────────────────────────────────────────────────────────────┘

使用建议：
• 训练长序列模型时必开
• 配合梯度检查点使用效果更好
• 需要 Ampere 架构及以上 GPU
```

---

## 8. 延伸阅读

- **论文**：
  - [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
  - [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
  - [FlashAttention-3: Fast and Accurate Attention on Hopper GPUs](https://arxiv.org/abs/2407.08608) (2024)
  - [PagedAttention: vLLM's approach to efficient KV cache management](https://arxiv.org/abs/2309.06180)

- **代码**：
  - [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)（官方实现）
  - [vLLM Project](https://github.com/vllm-project/vllm)（PagedAttention + 推理优化）
  - [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)（PyTorch 原生）

- **博客与教程**：
  - [Tri Dao's Blog on FlashAttention](https://princeton-nlp.github.io/flash-attention/)
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)（注意力基础）

- **上一章**：[采样策略与生成质量](07-01-采样策略与生成质量.md)
- **下一章**：[缩放定律与计算最优](07-03-缩放定律与计算最优.md)
