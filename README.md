# ctc-aligner-rs

[CTC forced aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的 Rust 实现。

为音频 + 文稿生成字/词级时间戳。声学模型为 **Wav2Vec2ForCTC**（默认 `MahmoudAshraf/mms-300m-1130-forced-aligner`）。

- **正确性 golden**：原版 Python `ctc_forced_aligner`（transformers + torch），**不是**任何旧 Rust/ONNX 路径。
- **推理引擎**：零深度学习框架。CUDA = cudarc + 手写 kernel + cuBLAS；CPU = gemm + rayon。
- **架构参考**：`qwen-aligner-rs`（DeviceRequest / RawTensor / Engine 双后端）。

## 状态

**CPU 端到端已跑通**，与原版 Python 时间戳对齐（见下方验证）。CUDA 引擎仍为 stub（M2）。

### 可移植性

- **默认纯 Rust**：`gemm` + `rayon`，**不依赖 MKL / 厂商 BLAS**
- 发布构建用普通 `cargo build --release` 即可；**不**要求 `target-cpu=native`（那会绑死编译机）
- SIMD 由 matmul 库在**运行时**探测（有 AVX2/NEON 就用，没有就标量/更弱路径）
- CUDA 为 **optional feature**，无卡机器自动可走 GPU，无卡仍走 CPU

### 音频预处理

一律经 **ffmpeg** 转为 16 kHz mono pcm_s16le（`FFMPEG` 环境变量 / 仓库内 ffmpeg / PATH）。

### 验证（golden = 现跑原版 Python，同 flags）

| Fixture | 词数 | 逐帧一致 | max dt |
|---------|------|----------|--------|
| `tests/3m` | 597 | **597/597** | less than 0.01 ms |
| `tests/15m` | 2848 | **2848/2848** | less than 0.03 ms |

旧 `tests/*.json` 可能与当前 Python 参数不一致；以现跑 Python 为准。

```bash
# CPU-only（通用发布）
cargo build --release --no-default-features --features cpu

cargo run --release --no-default-features --features cpu -- align \
  --model path/to/mms-300m-1130-forced-aligner \
  --audio tests/3m.wav --text tests/3m.txt \
  --language eng --device cpu --output out.json
```

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端，需要 CUDA 12.8+ |
| `cpu`（默认） | CPU 后端 |
| `uroman` | 启用 uroman 文本罗马化（与 Python 一致） |

CPU-only：

```bash
cargo build --release --no-default-features --features cpu
```

## CLI（规划）

```bash
cargo run --release -- align \
  --model path/to/model \
  --audio audio.wav \
  --text transcript.txt \
  --language iso \
  --device auto \
  --output out.json
```

## 模型目录

期望本地 Hugging Face 风格目录：

```
model/
├── config.json
├── model.safetensors   # 或 pytorch_model.bin（后续支持）
├── vocab.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## License

MIT
