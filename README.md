# ctc-aligner-rs

[ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的 **Rust** 实现：输入**音频 + 文稿**，输出**词级起止时间戳**。

无深度学习框架依赖。CUDA = cudarc + 预编译 PTX + cuBLAS；CPU = gemm + rayon。正确性对标原版 Python（同 flags 帧级对齐）。

| | |
|--|--|
| 模型 | Wav2Vec2ForCTC / [mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) |
| 仓库 | https://github.com/eclipse005/ctc-aligner-rs |
| 默认 options | `language=eng`，`romanize=true`，`split_size=word`，`star_frequency=edges`，窗 30s / 上下文 2s |

---

## 安装

```toml
[dependencies]
ctc-forced-aligner-rs = { git = "https://github.com/eclipse005/ctc-aligner-rs.git" }

# 仅 CPU：
# ctc-forced-aligner-rs = { git = "https://github.com/eclipse005/ctc-aligner-rs.git", default-features = false, features = ["cpu"] }
```

本地 path 依赖：

```toml
ctc-forced-aligner-rs = { path = "../ctc-aligner-rs" }
```

---

## 模型

权重不在 git 中（~1.2 GB）。目录布局与 Hugging Face 一致：

```text
models/mms-300m-1130-forced-aligner/
├── config.json
├── model.safetensors
├── vocab.json
├── tokenizer_config.json
└── special_tokens_map.json
```

```bash
huggingface-cli download MahmoudAshraf/mms-300m-1130-forced-aligner \
  --local-dir models/mms-300m-1130-forced-aligner
```

---

## 库用法

```rust
use ctc_forced_aligner_rs::{
    load_model, write_forced_align_items_json, AlignRequest, DeviceRequest,
    ForcedAlignResult, ModelOptions,
};

// 1) 加载一次，复用 Aligner
let aligner = load_model(
    "models/mms-300m-1130-forced-aligner",
    ModelOptions {
        device: DeviceRequest::Auto, // 优先 CUDA，失败回落 CPU
        // device: DeviceRequest::Cuda(0),
        // device: DeviceRequest::Cpu,
    },
)?;

// 2) 对齐：音频路径 + 文本路径 + 语言 ISO
let mut req = AlignRequest::from_paths(
    "audio.wav",        // 任意 ffmpeg 可读；内部 → 16 kHz mono
    "transcript.txt",
    "eng",              // 中文常用 "cmn"，日语 "jpn" 等
);
// req.options.split_size = "char".into();
// req.options.romanize = true; // 默认已是 true

let result: ForcedAlignResult = aligner.align(req)?;
// result.backend  = "cuda" | "cpu"
// result.stride_ms ≈ 20
// result.items    = 词级时间戳（秒）

for w in &result.items {
    println!("{:.3}-{:.3}  {}", w.start, w.end, w.text);
}

write_forced_align_items_json(std::path::Path::new("out.json"), &result.items)?;
```

**要点**

1. **`load_model` 只做一次**；多次 `align` 复用同一 `Aligner`。
2. 文本文件 UTF-8；`from_paths` 的 language 写入 `AlignOptions.language`。
3. 默认 **romanize=true**（MMS 拉丁字符表；多数非拉丁语言需要）。
4. 输出时间单位为 **秒**（`f64`），网格约 **20 ms**。

### 公开 API（crate 根 re-export）

| 符号 | 说明 |
|------|------|
| `load_model(model_dir, ModelOptions) -> Aligner` | 加载权重 + 选设备 |
| `Aligner::align(&self, AlignRequest) -> ForcedAlignResult` | 端到端对齐 |
| `Aligner::backend(&self) -> &str` | `"cuda"` / `"cpu"` |
| `write_forced_align_items_json(path, &[ForcedAlignItem])` | 写 JSON 数组 |
| `DeviceRequest` | `Auto` \| `Cpu` \| `Cuda(usize)`（需 feature `cuda`） |
| `ModelOptions { device }` | 默认 `device: Auto` |
| `AlignRequest { audio_path, text_path, options }` | `from_paths(audio, text, language)` |
| `AlignOptions` | 见下表 |
| `ForcedAlignResult { items, stride_ms, backend }` | 对齐结果 |
| `ForcedAlignItem { start, end, text, score? }` | 一词（秒） |

### `AlignOptions`

| 字段 | 默认 | 含义 |
|------|------|------|
| `window_size_sec` | `30.0` | 滑窗长度（秒） |
| `context_size_sec` | `2.0` | 窗两侧上下文（秒） |
| `language` | `"eng"` | ISO 语言码（`from_paths` 会覆盖） |
| `split_size` | `"word"` | **`word` / `char` 正式支持**；`jpn`/`chi` 强制 `char` |
| `star_frequency` | `"edges"` | `edges` \| `segment`（`<star>` 注入） |
| `merge_threshold` | `0.0` | 后处理合并阈值 |
| `romanize` | `true` | uroman；MMS 词表需要时请保持 true |
| `batch_size` | `4` | 保留字段（与 Python 同名）；**当前推理路径未使用** |

### 输出 JSON

`write_forced_align_items_json` 写入 JSON 数组：

```json
[
  { "start": 2.04, "end": 2.12, "text": "All", "score": -14.99 },
  { "start": 2.12, "end": 2.3, "text": "right,", "score": -5.91 }
]
```

`score` 可能省略（`skip_serializing_if`）。

---

## CLI

```bash
cargo run --release -- align \
  --model models/mms-300m-1130-forced-aligner \
  --audio audio.wav \
  --text transcript.txt \
  --language eng \
  --device auto \
  --output out.json
```

`--device`：`auto` | `cuda` | `cuda:0` | `cpu`  
`--language` 默认 `eng`。CLI 不暴露 romanize 开关，固定走库默认（romanize on）。

---

## Features / 构建

| Feature | 默认 | 说明 |
|---------|------|------|
| `cuda` | ✓ | NVIDIA 驱动 + cudart/cublas；**无需**终端用户装 nvcc（预编译 PTX sm_61–90） |
| `cpu` | ✓ | gemm + rayon，无 MKL |

```bash
cargo build --release
cargo build --release --no-default-features --features cpu
```

运行时依赖：**ffmpeg**（PATH / 环境变量 `FFMPEG`；非 16 kHz mono 时会转码）。

---

## 正确性与性能（参考）

| Fixture | 词数 | CPU/CUDA f32 逐帧 vs Python golden |
|---------|------|-------------------------------------|
| 3m | 597 | **597/597** |
| 15m | 2848 | **2848/2848** |

默认 flags：`romanize`、`edges`、`window 30/2`、`eng`。

RTFx（P104-100 sm_61，load-once，约）：Rust CUDA f32 **~46×**（3m）/ **~42–44×**（15m），与官方 Python CUDA 同级。Pascal 无 Tensor Core，不以「远超 Python」为交付目标。

### 约束

- CPU / CUDA 引擎均与 Python golden **帧级**对齐（f32）
- 不绑 MKL / `target-cpu=native`
- 音频统一 **16 kHz mono**

---

## 致谢 / 原版出处

本仓库是 **独立的 Rust 推理实现**，对标 Python [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) 的对齐流程，并加载社区发布的 MMS forced-aligner 权重；**不是** Meta / 原作者官方发行版，与各方无隶属关系。

| 组件 | 原版 | 链接 | 协议（以官方页面为准） |
|------|------|------|------------------------|
| Python 对齐流程 / 工具 | ctc-forced-aligner | [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | 见上游仓库 |
| 模型权重 | mms-300m-1130-forced-aligner | [MahmoudAshraf/mms-300m-1130-forced-aligner](https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner) | 见模型卡 `license` 字段 |
| 底座模型族 | Meta MMS（Massively Multilingual Speech） | [fairseq examples/mms](https://github.com/facebookresearch/fairseq/tree/main/examples/mms) · [facebook/mms-300m](https://huggingface.co/facebook/mms-300m) | 见 Meta / 模型卡（部分 MMS 权重含 **CC-BY-NC** 等限制，请自行核对） |

使用模型权重时请严格遵守对应许可证（含是否允许商用/再分发）；本仓库的 Rust 推理代码以本仓库 License 为准。

工程实现上亦参考了同系列引擎布局：[qwen-aligner-rs](https://github.com/eclipse005/qwen-aligner-rs)、[cohere-transcribe-rs](https://github.com/eclipse005/cohere-transcribe-rs)。

## License

MIT
