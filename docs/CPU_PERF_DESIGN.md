# CPU RTFx — 源头设计分析

## 1. 正确性现状（先对齐预期）

| 指标 | 结论 |
|------|------|
| 文本序列 | 与 golden **逐词一致**（3m 597、15m 2848） |
| 时间戳 | **帧级对齐**，不是 float bit-exact |
| 3m | **597/597 within 20ms**（max=20ms = 1×stride） |
| 15m | **2844/2848 within 20ms**，极少数边界 max≈440ms |
| logits | vs Python CPU maxabs≈0.0027，corr≈1.0 |
| CTC path | 同一 emissions 上 path **完全一致** |

**不是**每个 f32 与 Python 逐位相同；**是**对齐路径与时间戳在实用阈值内与原版一致。  
20ms 来自模型 stride（约 20ms/帧）的边界取整，不是乱漂。

---

## 2. RTFx 慢的真正来源（设计层，不是“少写了两行 SIMD”）

### 2.1 算量本身极大

对 3m 窗口推理（~7× 34s chunk，T≈1700，24 层）：

| 组件 | 约 GFLOP / 层 | 占比 |
|------|---------------|------|
| FFN (H=1024→4096→1024) | ~28.5 | **~52%** |
| QKV+O proj | ~14.3 | ~26% |
| Attn QK+AV (16 heads) | ~11.8 | ~22% |
| **24 层 × 7 窗** | **~9000+ GFLOP** | |

官方 Python CPU 15s ≈ 有效 **~600 GFLOP/s** 量级（MKL/oneDNN）。  
我们若只跑到 ~150 GFLOP/s，就会是 **~60s**，与实测一致。

→ **主矛盾：矩阵乘吞吐 + 调度/内存**，不是 CTC/文本。

### 2.2 架构反模式（当前实现的设计债）

```
每层 × 每窗:
  分配 q,k,v,scores,ctx,inter,ff   ← 分配器 + 缺页
  3 次 Linear gemm (Q/K/V 分开)  ← 重复 pack 权重
  pack heads → gemm → unpack       ← 多余带宽
  再分配 FFN 中间张量
CLI 每次进程: 重新 mmap + 解析 423 tensors  ← bench 口径被污染
Rayon 嵌套: 窗并行 + head 并行 + gemm 内并行  ← 过订阅
```

官方 torch 的优势不在“Python 快”，而在：

1. **持久 workspace**（不每层 malloc）
2. **MKL/oneDNN 权重预打包 + 大块 sgemm**
3. **融合算子**（bias、激活）
4. **进程内多次 forward 不重载模型**

### 2.3 优化方向（按 ROI）

| 优先级 | 设计手段 | 预期 |
|--------|----------|------|
| P0 | **Scratch/Arena**：一次按 max_T 分配，热路径零 alloc | 1.3–2× |
| P0 | **融合 QKV**：一次 gemm 出 3H | 1.1–1.3× |
| P0 | **并行策略写死**：单序列多线程 gemm；禁止三重 rayon | 稳定满核 |
| P1 | **权重布局**：load 时固定行主序 + 可选预转置；减少反复 pack | 1.1–1.5× |
| P1 | **库式 API bench**：load 一次，多次 align（真 RTFx） | 口径正确 |
| P2 | flash/online attn（减 T² 写带宽） | 中等 |
| P2 | 链接 OpenBLAS/MKL 或 faer 顶级 matmul | 冲官方 CPU |
| P3 | CUDA 手写（最终目标 40×+） | 数量级 |

**Rust 新特性/最佳实践对应：**

- `Scratch` + 复用 `Vec` 容量（或 `bumpalo`/自定义 arena）
- 明确 `Send+Sync` 只读权重 + 每线程/每调用可变 scratch
- `rayon` **单层**并行（要么 B，要么 heads，要么 gemm 内，三选一）
- `#[inline]`、避免热路径 `Result` 分支炸栈（已 load 后的 shape 用 assert）
- 特征 flag：`cpu` 路径纯计算；profiling 用 `CTC_PROFILE`

---

## 3. 成功标准

- **正确性红线不变**：3m 仍 597/597 within 20ms；15m 不显著劣化  
- **RTFx**：先冲 **> 官方 Python CPU（~13×）**；再上 CUDA 冲官方 GPU（~40×）

## 4. 实施顺序

1. Scratch + 融合 QKV + 单层并行策略（本轮）
2. 常驻进程 bench（load once）
3. matmul 后端评估（gemm vs faer vs BLAS）
4. CUDA 引擎
