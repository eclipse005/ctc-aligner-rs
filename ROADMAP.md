# ROADMAP — ctc-aligner-rs

> Golden：**现跑** 原版 Python `ctc_forced_aligner`（同 flags：`romanize + edges + eng + window 30/2`）。  
> 引擎：CPU = 纯 Rust `gemm` + `rayon`；CUDA = cudarc + 手写 kernel + cuBLAS（**进行中**）。  
> 约束：**不接 MKL**；不绑 `target-cpu=native` 发布；音频一律 ffmpeg → 16 kHz mono。

## 目标模型

`Wav2Vec2ForCTC` / MMS-300m：

- Feature extractor：7× 1D conv（512 ch，stride 5/2/2/2/2/2/2）
- Encoder：24 层 transformer，`hidden=1024`，`heads=16`，`ffn=4096`，stable LayerNorm
- CTC head → log_softmax → Viterbi → spans / timestamps

---

## 里程碑

### M0 — 脚手架 ✅

- [x] 独立 crate + 独立 git 仓库
- [x] `DeviceRequest` / features / CLI
- [x] CTC Viterbi
- [x] safetensors mmap

### M1 — CPU 端到端 ✅（**冻结优化**）

- [x] 完整 forward + 窗口推理 + 文本/postprocess
- [x] im2col+gemm FE、融合 QKV、scratch、ffmpeg 16k
- [x] 与 **现跑 Python** 逐帧对齐

**正确性（vs 现跑 Python golden）**

| Fixture | 词数 | 逐帧一致 | max \|Δt\| |
|---------|------|----------|------------|
| 3m | 597 | **597/597** | &lt;0.01 ms |
| 15m | 2848 | **2848/2848** | &lt;0.03 ms |

**RTFx（load once，3m，CPU f32）**

| 实现 | median | RTFx |
|------|--------|------|
| 官方 Python CPU | ~15.4s | **~13.4×** |
| **Rust CPU（现）** | **~15.6s** | **~13.3×** |

**3m 内部 profile（约）**

| 阶段 | 时间 | 占比 |
|------|------|------|
| frontend (conv+pos) | ~2.0s | ~13% |
| encoder attn | ~7.1s | ~46% |
| encoder FFN | ~5.9s | ~38% |
| **total forward** | **~15.4s** | 100% |

**CPU 再榨收益评估（无 MKL、可移植）**

| 方向 | 预估 | 结论 |
|------|------|------|
| flash/online attn | 5–15% | 收益有限，风险改数值 |
| 再抠 pack/alloc | 5–10% | 边际 |
| faer 换 gemm | 未知，可能 ±10% | 可选实验，非必须 |
| MKL | 可能 1.5–2× | **不做**（破坏通用性） |
| **CUDA** | **数倍～数量级** | **下一刀** |

→ **CPU 优化冻结**：已与官方 torch CPU 同级 + 帧级对齐。后续只修正确性回归，不主动堆 CPU 微优化。

### M2 — CUDA 引擎 ⬅ **当前**

对标 `qwen-aligner-rs`（`cudarc_engine` + `kernels.cu` + 预编译 PTX）。

- [ ] `CudaState` 持有 context / stream / cuBLAS（现仅 probe）
- [ ] safetensors → device 权重（f16/bf16 存储 + f32 累加，与 qwen 类似）
- [ ] GPU feature extractor（1D conv 栈 + LN + GELU）
- [ ] GPU encoder：24 层 stable LN，MHA，FFN
- [ ] lm_head → host emissions（或 GPU log_softmax）
- [ ] 复用现有窗口拼接 / CTC / postprocess（可先 host CTC）
- [ ] 与 **CPU / 现跑 Python** 帧级对齐（3m 必过，15m 必过）
- [ ] RTFx：目标 **≥ 官方 Python CUDA**（约 40× on 3m/15m 量级）

**建议实现顺序**

1. 权重上传 + Linear (cuBLAS GEMM) + LN/GELU kernel  
2. FE conv（im2col on GPU 或自定义 1D conv）  
3. Attention（grouped QKV + scaled matmul + softmax）  
4. 端到端 forward_logits → 接现有 `generate_emissions`  
5. 压测 + 对齐 golden  

### M3 — CUDA RTFx 压榨

- [ ] 多窗 batch on GPU
- [ ] kernel fuse（bias+act、residual+LN）
- [ ] 减少 H2D/D2H；长音频流水线
- [ ] 预编译 PTX（sm_61+ 按需）

### M4 — 交付

- [ ] 库 API 稳定
- [ ] 整目录迁出 monorepo

---

## 精度红线

- 时间戳与 **现跑 Python** 逐帧一致（20 ms 网格）
- 浮点允许 ULP 噪声；**不允许**系统性帧偏移
- 改 fused/online 算子后必须重跑 3m + 15m golden

## 可移植性红线

- 默认无 MKL / 无绑定本机 CPU
- `cuda` / `cpu` feature 可裁剪
- 音频：ffmpeg → 16 kHz mono（`FFMPEG` 或 PATH）
