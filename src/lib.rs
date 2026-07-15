//! CTC forced aligner — hand-written CUDA / CPU engines.
//!
//! Correctness golden is the original Python `ctc_forced_aligner` package
//! (transformers Wav2Vec2ForCTC + Viterbi forced alignment).

pub mod audio_io;
// re-export for diagnostics
pub mod backend;
pub mod config;
pub mod ctc;
pub mod inference;
pub mod postprocess;
pub mod raw_tensor;
pub mod text;
mod weights;

#[cfg(feature = "cpu")]
pub mod cpu_engine;

#[cfg(feature = "cuda")]
pub(crate) mod cudarc_engine;

pub use backend::DeviceRequest;
pub use config::{AlignOptions, ModelConfig};
pub use inference::{
    load_model, AlignRequest, Aligner, ForcedAlignItem, ForcedAlignResult, ModelOptions,
};
pub use postprocess::write_forced_align_items_json;
