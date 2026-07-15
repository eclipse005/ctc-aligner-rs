# ctc-aligner-rs

[CTC forced aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的 Rust 实现。

为音频 + 文稿生成字/词级时间戳。声学模型为 **Wav2Vec2ForCTC**（默认 `MahmoudAshraf/mms-300m-1130-forced-aligner`）。

- **正确性 golden**：原版 Python `ctc_forced_aligner`（transformers + torch），**不是**任何旧 Rust/ONNX 路径。
- **推理引擎**：零深度学习框架。CUDA = cudarc + 手写 kernel + cuBLAS；CPU = gemm + rayon。
- **架构参考**：`qwen-aligner-rs`（DeviceRequest / RawTensor / Engine 双后端）。

## 状态

脚手架阶段：CTC Viterbi 已实现并对齐 Python 语义；Wav2Vec2 forward（CPU/CUDA）待实现。

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
