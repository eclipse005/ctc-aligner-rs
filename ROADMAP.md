# ROADMAP — ctc-aligner-rs

> Golden：**原版 Python** `ctc_forced_aligner`（transformers Wav2Vec2ForCTC + forced_align_impl）。
> 引擎形态对标 `qwen-aligner-rs`：cudarc 手写 CUDA + gemm/rayon CPU。

## 目标模型

`Wav2Vec2ForCTC` / MMS-300m：

- Feature extractor：7× 1D conv（512 ch，stride 5/2/2/2/2/2/2）
- Encoder：24 层 transformer，`hidden=1024`，`heads=16`，`ffn=4096`，stable LayerNorm
- CTC head → log_softmax → Viterbi → spans / timestamps

## 里程碑

### M0 — 脚手架 ✅

- [x] 独立 crate + 独立 git 仓库（本目录，便于整目录迁走）
- [x] `DeviceRequest` / features / CLI 骨架
- [x] CTC Viterbi（对齐 Python `forced_align_impl.py`）
- [x] safetensors mmap 加载骨架

### M1 — CPU 端到端（正确性）

- [ ] 解析 `config.json` → `Wav2Vec2Config`
- [ ] CPU：feature extractor + pos conv + 24 encoder + CTC head
- [ ] 窗口/context batch 推理（对齐 Python `generate_emissions`）
- [ ] 文本 normalize / uroman / token→index（对齐 Python）
- [ ] spans + postprocess timestamps
- [ ] 与 Python 时间戳 diff（固定 fixture）

### M2 — CUDA 引擎

- [ ] cudarc + cuBLAS HGEMM / f16 路径
- [ ] 手写 kernels：LayerNorm、GELU、attention、1D conv 等
- [ ] 预编译 PTX（可选，多 sm）
- [ ] 与 Python / CPU 结果对齐

### M3 — RTFx

- [ ] CUDA：batch window、kernel fuse、减少 H2D
- [ ] CPU：f32 权重预转换、attn SIMD / flash-style、rayon
- [ ] 子阶段 profile 环境变量

### M4 — 交付

- [ ] 库 API 稳定，可供 GUI / voxtrans 等调用
- [ ] 整目录迁出本 monorepo

## 精度红线

- 时间戳与 Python 一致（允许极小边界阈值；不允许路径/argmax 系统性偏移）
- 激进 fuse 前先锁 golden
