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

### M2 — CUDA 引擎 ✅（端到端可用）

对标 `qwen-aligner-rs` + `cohere-transcribe-native`（voxtrans 同一套 PTX scheme B）。

**已完成**

- [x] `CudaState`：context / stream / cuBLAS workspace / prebuilt PTX
- [x] `src/kernels/kernels.cu` + `scripts/compile-ptx.ps1` + `ptx/kernels_sm{61,70,75,80,86,89,90}.ptx`
- [x] `prebuilt_ptx.rs` 运行时选 SM
- [x] 全模型权重 **f32** 上传 + 融合 QKV；Linear / LN / GELU / softmax / head pack
- [x] GPU FE：im2col + SGEMM + bias/LN/GELU（7× conv）
- [x] GPU pos_conv（weight_norm 展开 + grouped im2col）
- [x] GPU encoder：24 层 MHA（strided-batched QK/AV）+ FFN + residual fuse
- [x] encoder scratch 双缓冲 + 多窗 batch（B≤8）
- [x] lm_head → host emissions → 现有窗口/CTC/postprocess
- [x] `forward_logits` 端到端 + `--device cuda`
- [x] 与 **现跑 Python golden** 帧级对齐（3m + 15m）

**正确性（CUDA f32 vs 现跑 Python CPU float32 golden）**

| Fixture | 词数 | 逐帧一致 | 文本 |
|---------|------|----------|------|
| 3m | 597 | **597/597** | 597/597 |
| 15m | 2848 | **2848/2848** | 2848/2848 |

**RTFx（load once，本机 sm_61 / Pascal，f32 路径）**

| 实现 | fixture | median total | RTFx |
|------|---------|--------------|------|
| 官方 Python CUDA f16 | 3m | ~4.35s | ~47× |
| 官方 Python CUDA f32 | 3m | ~4.57s | ~45× |
| **Rust CUDA f32** | 3m | **~4.5s** | **~46×**（首枪 ~58×） |
| 官方 Python CUDA f16 | 15m | ~23.2s | ~40× |
| **Rust CUDA f32** | 15m | **~21–22s** | **~42–44×** |
| Rust CPU f32 | 3m | ~15.6s | ~13.3× |

→ **M2 冻结交付**：Pascal 上与官方 CUDA 同级；帧级 f32 对齐。再榨 RTFx 见 M3（可选，非本机必须）。

### M3 — CUDA RTFx 压榨（可选 / 换卡后再做）

Pascal 无 TC，与 Python 稳态差 &lt;5%；不值得在 P104 上继续微优化。

- [ ] 可选 f16/TC 快路径（Ampere+；默认仍 f32 保帧级）
- [ ] FE+encoder 跨窗流水；workspace 复用
- [ ] kernel fuse 加深
- [x] 预编译 PTX（sm_61+）
- [x] 多窗 batch encoder

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
