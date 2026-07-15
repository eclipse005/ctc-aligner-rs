# ctc-aligner-rs

[ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的 **Rust** 实现：输入**音频 + 文稿**，输出**词级起止时间戳**。

| | |
|--|--|
| 模型 | Wav2Vec2ForCTC / [mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) |
| 引擎 | **无深度学习框架**；CUDA = cudarc + 手写 PTX + cuBLAS；CPU = gemm + rayon |
| Golden | 原版 Python `ctc_forced_aligner`（同 flags 帧级对齐） |
| 本机路径 | `D:\ctc-aligner-rs`（独立 git 仓库，准备接入 voxtrans） |

---

## 给集成方 / AI 的最短路径

```toml
# voxtrans/Cargo.toml（或其它下游）
[dependencies]
ctc-forced-aligner-rs = { path = "D:/ctc-aligner-rs" }
# 仅 CPU：
# ctc-forced-aligner-rs = { path = "D:/ctc-aligner-rs", default-features = false, features = ["cpu"] }
```

```rust
use ctc_forced_aligner_rs::{
    load_model, write_forced_align_items_json, AlignOptions, AlignRequest, DeviceRequest,
    ForcedAlignItem, ForcedAlignResult, ModelOptions,
};

// 1) 加载一次，复用 Aligner（含 GPU 权重）
let aligner = load_model(
    r"D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner",
    ModelOptions {
        device: DeviceRequest::Auto, // 优先 CUDA，失败回落 CPU
        // device: DeviceRequest::Cuda(0),
        // device: DeviceRequest::Cpu,
    },
)?;

// 2) 对齐：音频路径 + 文本路径 + 语言 ISO
let mut req = AlignRequest::from_paths(
    r"D:\path\to\audio.wav",   // 任意 ffmpeg 可读；内部 → 16 kHz mono
    r"D:\path\to\transcript.txt",
    "eng",                     // 中文常用 "cmn"，日语 "jpn" 等
);
// 可选：改默认 options（见下方 AlignOptions）
// req.options.romanize = true;
// req.options.split_size = "word".into();

let result: ForcedAlignResult = aligner.align(req)?;
// result.backend  = "cuda" | "cpu"
// result.stride_ms ≈ 20
// result.items    = 词级时间戳

for w in &result.items {
    println!("{:.3}-{:.3}  {}", w.start, w.end, w.text);
}

// 3) 写 JSON（与 CLI 相同格式）
write_forced_align_items_json(std::path::Path::new("out.json"), &result.items)?;
```

**要点：**

1. **`load_model` 只做一次**；多次 `align` 复用同一 `Aligner`。
2. 文本文件需 UTF-8；`language` 写入 `AlignOptions.language`。
3. 默认 **romanize=true**（MMS 拉丁字符表；多数非拉丁语言需要）。
4. 输出时间单位为 **秒**（`f64`），网格约 **20 ms**。

---

## 公开 API（crate 根 re-export）

| 符号 | 说明 |
|------|------|
| `load_model(model_dir, ModelOptions) -> Aligner` | 加载权重 + 选设备 |
| `Aligner::align(&self, AlignRequest) -> ForcedAlignResult` | 端到端对齐 |
| `Aligner::backend(&self) -> &str` | `"cuda"` / `"cpu"` |
| `write_forced_align_items_json(path, &[ForcedAlignItem])` | 写 JSON 数组 |
| `DeviceRequest` | `Auto` \| `Cpu` \| `Cuda(usize)`（需 feature `cuda`） |
| `ModelOptions { device }` | 默认 `device: Auto` |
| `AlignRequest { audio_path, text_path, options }` | `from_paths(audio, text, language)` 设默认 options + language |
| `AlignOptions` | 见下表 |
| `ForcedAlignResult { items, stride_ms, backend }` | 对齐结果 |
| `ForcedAlignItem { start, end, text, score? }` | 一词（秒） |

### `AlignOptions`（默认与 Python CLI 对齐）

| 字段 | 默认 | 含义 |
|------|------|------|
| `window_size_sec` | `30.0` | 滑窗长度（秒） |
| `context_size_sec` | `2.0` | 窗两侧上下文（秒） |
| `batch_size` | `4` | （CPU 批；CUDA 内部另有窗 batch） |
| `language` | `"eng"` | ISO 语言码（`from_paths` 会覆盖） |
| `split_size` | `"word"` | `word` \| `char` \| `sentence` \| `auto` |
| `star_frequency` | `"edges"` | `edges` \| `segment`（\* 注入） |
| `merge_threshold` | `0.0` | 后处理合并阈值 |
| `romanize` | `true` | uroman；MMS 词表需要时请保持 true |

自定义示例：

```rust
let mut req = AlignRequest::from_paths("a.wav", "a.txt", "cmn");
req.options.split_size = "char".into();
req.options.romanize = true;
let result = aligner.align(req)?;
```

### 输出 JSON 形状

`write_forced_align_items_json` 写入 **JSON 数组**：

```json
[
  { "start": 2.04, "end": 2.12, "text": "All", "score": -14.99 },
  { "start": 2.12, "end": 2.3, "text": "right,", "score": -5.91 }
]
```

`score` 可能省略（serde `skip_serializing_if`）。

---

## 模型目录

本仓库本地默认（已拷贝，~1.2 GB，**gitignore**）：

```
D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner\
├── config.json
├── model.safetensors
├── vocab.json
├── tokenizer_config.json
└── special_tokens_map.json
```

自行下载：

```bash
huggingface-cli download MahmoudAshraf/mms-300m-1130-forced-aligner \
  --local-dir models/mms-300m-1130-forced-aligner
```

---

## Features / 构建

| Feature | 默认 | 说明 |
|---------|------|------|
| `cuda` | ✓ | 需 NVIDIA 驱动 + cudart/cublas；**无需**终端用户装 nvcc（预编译 PTX sm_61–90） |
| `cpu` | ✓ | gemm + rayon，无 MKL |

```bash
cd D:\ctc-aligner-rs
cargo build --release
# 仅 CPU
cargo build --release --no-default-features --features cpu
```

运行时依赖：**ffmpeg** 在 PATH / `FFMPEG` / 常见相对路径（非 16k mono WAV 时会转码）。

---

## CLI

```bash
cargo run --release -- align \
  --model D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner \
  --audio audio.wav \
  --text transcript.txt \
  --language eng \
  --device auto \
  --output out.json
```

`--device`：`auto` | `cuda` | `cuda:0` | `cpu`

```bash
# CUDA 加载冒烟
cargo run --release --example check_cuda -- D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner

# load-once RTFx：args = model audio text n_runs device
cargo run --release --example bench_rtfx -- \
  D:\ctc-aligner-rs\models\mms-300m-1130-forced-aligner audio.wav text.txt 3 cuda
```

---

## 正确性与性能（参考）

| Fixture | 词数 | CPU/CUDA f32 逐帧 vs Python golden |
|---------|------|-------------------------------------|
| 3m | 597 | **597/597** |
| 15m | 2848 | **2848/2848** |

默认 flags：`romanize`、`edges`、`window 30/2`、`eng`。

RTFx（P104-100 sm_61，load-once，约）：Rust CUDA f32 **~46×**（3m）/ **~42–44×**（15m），与官方 Python CUDA 同级；首枪可更高。Pascal 无 TC，不以「远超 Python」为交付目标。

---

## 状态与约束

- **CPU 引擎**：冻结，与 torch CPU 帧级对齐  
- **CUDA 引擎**：f32 全路径端到端；帧级对齐  
- 不绑 MKL / `target-cpu=native`

---

## License

MIT

## 致谢

- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [MMS](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- 架构参考：`qwen-aligner-rs`、`cohere-transcribe-native`
