# ctc-aligner-rs

[ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的 Rust 实现：为音频 + 文稿生成**词级时间戳**。

- **模型**：Wav2Vec2ForCTC / [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner)（MMS 多语言字符 CTC）
- **正确性 golden**：原版 Python `ctc_forced_aligner`（transformers + torch），同 flags：`romanize`、`star=edges`、`window=30/2`
- **推理引擎**：**无深度学习框架**。CUDA = cudarc + 手写 kernel + cuBLAS；CPU = gemm + rayon
- **架构参考**：`qwen-aligner-rs` / `cohere-transcribe-native`（`DeviceRequest`、双后端、scheme B 预编译 PTX）
- **目标集成**：[voxtrans](https://github.com/) 等本地转写管线中的 forced-align 环节

## 状态

| 后端 | 状态 |
|------|------|
| **CPU** | 冻结；与 Python torch CPU 帧级对齐，RTFx 同级 |
| **CUDA** | **端到端可用**（f32 全路径）；与 Python golden 帧级对齐；RTFx 与官方 CUDA 同级（Pascal 本机） |

### 可移植性

- **默认纯 Rust CPU**：`gemm` + `rayon`，**不依赖 MKL / 厂商 BLAS**
- 发布：普通 `cargo build --release`；**不**要求 `target-cpu=native`
- CUDA 为 optional feature；无卡可 `--features cpu` / `--device cpu`
- 预编译多架构 PTX（sm_61–90），终端用户**无需**本机 CUDA Toolkit / NVRTC

### 音频

一律归一到 **16 kHz mono**（ffmpeg）。若输入已是 16 kHz mono WAV，则跳过 ffmpeg 直读。

## 安装（库）

```toml
[dependencies]
ctc-forced-aligner-rs = { path = "../ctc-aligner-rs" }
# 或日后 git：
# ctc-forced-aligner-rs = { git = "https://github.com/<you>/ctc-aligner-rs.git" }
```

CPU-only：

```toml
ctc-forced-aligner-rs = { path = "../ctc-aligner-rs", default-features = false, features = ["cpu"] }
```

## 使用

### 作为库（voxtrans / 其它 crate）

```rust
use ctc_forced_aligner_rs::{
    load_model, AlignRequest, DeviceRequest, ModelOptions,
};

let aligner = load_model(
    "models/mms-300m-1130-forced-aligner",
    ModelOptions {
        device: DeviceRequest::Auto, // 或 Cuda(0) / Cpu
    },
)?;

let result = aligner.align(AlignRequest::from_paths(
    "audio.wav",
    "transcript.txt",
    "eng", // ISO 语言码；非拉丁文通常需 romanize（默认开）
))?;

for item in &result.items {
    println!("{:.2}-{:.2} {}", item.start, item.end, item.text);
}
```

### 命令行

```bash
# 默认 cuda + cpu features
cargo build --release

cargo run --release -- align \
  --model models/mms-300m-1130-forced-aligner \
  --audio audio.wav \
  --text transcript.txt \
  --language eng \
  --device auto \
  --output out.json
```

`--device`：`auto` | `cuda` | `cuda:0` | `cpu`

### 冒烟 / 压测

```bash
# CUDA 加载冒烟
cargo run --release --example check_cuda -- models/mms-300m-1130-forced-aligner

# load-once RTFx（第 5 参 device）
cargo run --release --example bench_rtfx -- \
  models/mms-300m-1130-forced-aligner audio.wav text.txt 3 cuda
```

## Features

| Feature | 说明 |
|---------|------|
| `cuda`（默认） | CUDA 后端；需要本机 NVIDIA 驱动 + cudart/cublas（预编译 PTX，无需 nvcc） |
| `cpu`（默认） | CPU 后端 |

```bash
# 仅 CPU
cargo build --release --no-default-features --features cpu
```

## 模型目录

期望本地 Hugging Face 布局（本仓库默认拷贝路径）：

```
models/mms-300m-1130-forced-aligner/
├── config.json
├── model.safetensors    # ~1.2 GB
├── vocab.json
├── tokenizer_config.json
└── special_tokens_map.json
```

下载（示例）：

```bash
# huggingface-cli
huggingface-cli download MahmoudAshraf/mms-300m-1130-forced-aligner \
  --local-dir models/mms-300m-1130-forced-aligner
```

大权重目录 `models/` 已在 `.gitignore` 中，不进 git。

## 验证（golden = 现跑原版 Python）

| Fixture | 词数 | 逐帧一致（20 ms 网格） | 文本 |
|---------|------|------------------------|------|
| 3m | 597 | **597/597**（CPU + CUDA f32） | 全一致 |
| 15m | 2848 | **2848/2848**（CPU + CUDA f32） | 全一致 |

flags：`romanize`、`star_frequency=edges`、`window_size=30`、`context_size=2`、`language=eng`。

### RTFx（参考，本机 P104-100 / sm_61，load-once）

| 实现 | 3m | 15m |
|------|----|-----|
| 官方 Python CUDA | ~47× (f16) / ~45× (f32) | ~40× |
| **Rust CUDA f32** | **~46×**（首枪可 ~58×） | **~42–44×** |
| Rust CPU f32 | ~13× | — |

Pascal 无 Tensor Core；与官方 CUDA 稳态同级为预期。Ampere+ 可再开可选 f16/TC 快路径（见 `ROADMAP.md` M3）。

## 目录结构

```
ctc-aligner-rs/
├── src/
│   ├── cudarc_engine.rs   # CUDA f32 引擎
│   ├── cpu_engine.rs      # CPU gemm/rayon 引擎
│   ├── kernels/kernels.cu # 手写 kernel 源
│   ├── prebuilt_ptx.rs    # scheme B 选 PTX
│   ├── inference.rs       # 窗口 emissions → CTC → spans
│   └── ...
├── ptx/                   # 预编译 sm_61–90
├── models/                # 本地权重（gitignore）
├── scripts/compile-ptx.ps1
└── examples/
```

重编 PTX（开发机需 CUDA Toolkit + VS）：

```powershell
.\scripts\compile-ptx.ps1
```

## voxtrans 接入提示

- 依赖本 crate 的 `load_model` + `Aligner::align` / `AlignRequest`
- 设备用 `DeviceRequest::Auto` 或显式 `Cuda(n)`
- 模型目录指向 `models/mms-300m-1130-forced-aligner`（或你打包时的资源路径）
- 音频可为任意 ffmpeg 可读格式；内部统一 16 kHz mono
- 输出 `ForcedAlignResult.items`：`start` / `end`（秒）+ `text`（+ 可选 `score`）

## License

MIT

## 致谢

- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- 架构参考：`qwen-aligner-rs`、`cohere-transcribe-native`
