//! CUDA Wav2Vec2ForCTC engine — cudarc + prebuilt multi-arch PTX + cuBLAS.
//!
//! Mirrors `qwen-aligner-rs` / `cohere-transcribe-native` for voxtrans integration:
//! - [`CudaState`]: context, stream, cuBLAS, kernel registry (prebuilt PTX)
//! - f16 storage, f32 accumulate (sm_61+)
//! - Scheme B PTX: no CUDA Toolkit on end-user machines
//!
//! Weights upload + kernels ready. Full GPU FE+encoder forward is next.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::{DevicePtr, PushKernelArg};
use cudarc::nvrtc::Ptx;
use half::f16;

use crate::config::ModelConfig;
use crate::prebuilt_ptx;

// ─── Kernel registry ─────────────────────────────────────────────────────────

pub struct CudaKernels {
    pub layer_norm: CudaFunction,
    pub gelu: CudaFunction,
    pub gelu_inplace: CudaFunction,
    pub add: CudaFunction,
    pub add_inplace: CudaFunction,
    pub add_bias_inplace: CudaFunction,
    pub bias_residual: CudaFunction,
    pub softmax_last_dim: CudaFunction,
    pub scale_inplace: CudaFunction,
    pub split_to_heads: CudaFunction,
    pub merge_heads: CudaFunction,
    pub split_qkv_to_heads: CudaFunction,
    pub cast_f32_to_f16: CudaFunction,
    pub cast_f16_to_f32: CudaFunction,
}

impl CudaKernels {
    pub fn load_all(ctx: &Arc<CudaContext>) -> Result<Self> {
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| anyhow!("compute_capability: {e:?}"))?;
        let (ptx_src, selected_sm) = prebuilt_ptx::resolve_ptx_for_device(major, minor)
            .map_err(|e| anyhow!("{e}"))?;
        log::info!(
            "loading prebuilt CUDA kernels for device sm_{major}{minor} (selected sm_{selected_sm})"
        );
        let module = ctx
            .load_module(Ptx::from_src(ptx_src))
            .context("load prebuilt PTX module")?;

        let load = |name: &str| {
            module
                .load_function(name)
                .with_context(|| format!("load kernel {name}"))
        };

        Ok(Self {
            layer_norm: load("layer_norm_f16")?,
            gelu: load("gelu_f16")?,
            gelu_inplace: load("gelu_inplace_f16")?,
            add: load("add_f16")?,
            add_inplace: load("add_inplace_f16")?,
            add_bias_inplace: load("add_bias_inplace_f16")?,
            bias_residual: load("bias_residual_f16")?,
            softmax_last_dim: load("softmax_last_dim_f16")?,
            scale_inplace: load("scale_inplace_f16")?,
            split_to_heads: load("split_to_heads_f16")?,
            merge_heads: load("merge_heads_f16")?,
            split_qkv_to_heads: load("split_qkv_to_heads_f16")?,
            cast_f32_to_f16: load("cast_f32_to_f16")?,
            cast_f16_to_f32: load("cast_f16_to_f32")?,
        })
    }
}

// ─── CudaState ───────────────────────────────────────────────────────────────

/// Long-lived CUDA context (one per device). Safe to wrap in `Arc` for voxtrans.
pub struct CudaState {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub k: CudaKernels,
    pub ordinal: usize,
    _blas_ws: Option<CudaSlice<u8>>,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal).context("CudaContext::new")?;
        Self::new_with_ctx(ordinal, &ctx)
    }

    pub fn new_with_ctx(ordinal: usize, ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).context("CudaBlas::new")?;

        unsafe {
            sys::cublasSetMathMode(*blas.handle(), sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH);
        }

        let blas_ws = stream.alloc_zeros::<u8>(16 * 1024 * 1024).ok();
        if let Some(ref ws) = blas_ws {
            unsafe {
                let (ptr, _guard) = ws.device_ptr(&stream);
                let _ = sys::cublasSetWorkspace_v2(
                    *blas.handle(),
                    ptr as *mut std::ffi::c_void,
                    16 * 1024 * 1024,
                );
            }
        }

        let k = CudaKernels::load_all(ctx).context("CudaKernels::load_all")?;
        log::info!("CUDA device {ordinal} ready (cuBLAS + prebuilt PTX)");
        Ok(Self {
            ctx: ctx.clone(),
            stream,
            blas,
            k,
            ordinal,
            _blas_ws: blas_ws,
        })
    }

    pub fn upload_f16(&self, data: &[f16]) -> Result<CudaSlice<f16>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn alloc_zeros_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        Ok(self.stream.alloc_zeros::<f16>(n)?)
    }

    pub fn alloc_uninit_f16(&self, n: usize) -> Result<CudaSlice<f16>> {
        Ok(unsafe { self.stream.alloc::<f16>(n)? })
    }

    pub fn download_f16(&self, slice: &CudaSlice<f16>) -> Result<Vec<f16>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }

    pub fn gelu_inplace(&self, x: &mut CudaSlice<f16>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.k.gelu_inplace);
        b.arg(x).arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| anyhow!("gelu_inplace: {e:?}"))?;
        Ok(())
    }

    pub fn add_inplace(&self, a: &mut CudaSlice<f16>, b: &CudaSlice<f16>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_inplace);
        bb.arg(a).arg(b).arg(&n_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("add_inplace: {e:?}"))?;
        Ok(())
    }

    pub fn layer_norm(
        &self,
        x: &CudaSlice<f16>,
        w: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaSlice<f16>> {
        let block = dim.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let mut out = self.alloc_uninit_f16(rows * dim)?;
        let rows_i = rows as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.layer_norm);
        bb.arg(&mut out)
            .arg(x)
            .arg(w)
            .arg(b)
            .arg(&rows_i)
            .arg(&dim_i)
            .arg(&eps);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("layer_norm: {e:?}"))?;
        Ok(out)
    }

    pub fn softmax_last_dim(
        &self,
        x: &CudaSlice<f16>,
        rows: usize,
        dim: usize,
    ) -> Result<CudaSlice<f16>> {
        let block = 256u32.min(dim as u32).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let mut out = self.alloc_uninit_f16(rows * dim)?;
        let rows_i = rows as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.softmax_last_dim);
        bb.arg(&mut out).arg(x).arg(&rows_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("softmax: {e:?}"))?;
        Ok(out)
    }

    /// y = x @ W^T  (row-major). W: [out, in], x: [rows, in], y: [rows, out].
    /// Same cuBLAS layout as cohere `linear_gpu`.
    pub fn linear_raw(
        &self,
        x: &CudaSlice<f16>,
        rows: usize,
        w: &CudaSlice<f16>,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&CudaSlice<f16>>,
    ) -> Result<CudaSlice<f16>> {
        let mut y = self.alloc_uninit_f16(rows * out_dim)?;
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: out_dim as i32,
                        n: rows as i32,
                        k: in_dim as i32,
                        alpha: f16::from_f32(1.0),
                        lda: in_dim as i32,
                        ldb: in_dim as i32,
                        beta: f16::from_f32(0.0),
                        ldc: out_dim as i32,
                    },
                    w,
                    x,
                    &mut y,
                )
                .map_err(|e| anyhow!("cublas HGEMM: {e:?}"))?;
        }
        if let Some(bias) = bias {
            let n = rows * out_dim;
            let cfg = LaunchConfig::for_num_elems(n as u32);
            let n_i = n as i32;
            let cols_i = out_dim as i32;
            let mut bb = self.stream.launch_builder(&self.k.add_bias_inplace);
            bb.arg(&mut y).arg(bias).arg(&n_i).arg(&cols_i);
            unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("add_bias: {e:?}"))?;
        }
        Ok(y)
    }

    pub fn cast_f32_to_f16_slice(&self, x: &[f32]) -> Result<CudaSlice<f16>> {
        let d_x = self.upload_f32(x)?;
        let mut d_y = self.alloc_uninit_f16(x.len())?;
        let n = x.len() as i32;
        let cfg = LaunchConfig::for_num_elems(x.len() as u32);
        let mut bb = self.stream.launch_builder(&self.k.cast_f32_to_f16);
        bb.arg(&mut d_y).arg(&d_x).arg(&n);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("cast_f32_to_f16: {e:?}"))?;
        Ok(d_y)
    }

    pub fn cast_f16_to_f32_host(&self, x: &CudaSlice<f16>) -> Result<Vec<f32>> {
        let n = x.len();
        let mut d_y = self.stream.alloc_zeros::<f32>(n)?;
        let n_i = n as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut bb = self.stream.launch_builder(&self.k.cast_f16_to_f32);
        bb.arg(&mut d_y).arg(x).arg(&n_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("cast_f16_to_f32: {e:?}"))?;
        self.download_f32(&d_y)
    }

    pub fn download_f32(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }
}

// ─── GPU weights ─────────────────────────────────────────────────────────────

#[allow(dead_code)] // used once encoder forward is wired
struct GpuLinear {
    w: CudaSlice<f16>,
    b: CudaSlice<f16>,
    out: usize,
    inn: usize,
}

#[allow(dead_code)]
struct GpuLayerNorm {
    w: CudaSlice<f16>,
    b: CudaSlice<f16>,
    eps: f32,
    dim: usize,
}

#[allow(dead_code)]
struct GpuEncoderLayer {
    ln_attn: GpuLayerNorm,
    qkv: GpuLinear,
    o_proj: GpuLinear,
    ln_ff: GpuLayerNorm,
    ff_inter: GpuLinear,
    ff_out: GpuLinear,
    n_heads: usize,
    head_dim: usize,
    hidden: usize,
}

// ─── Engine ──────────────────────────────────────────────────────────────────

#[allow(dead_code)] // fields consumed by upcoming GPU FE/encoder
pub struct CudaEngine {
    pub config: ModelConfig,
    pub state: Arc<CudaState>,
    layers: Vec<GpuEncoderLayer>,
    feat_proj_ln: GpuLayerNorm,
    feat_proj: GpuLinear,
    encoder_ln: GpuLayerNorm,
    lm_head: GpuLinear,
}

impl CudaEngine {
    pub fn load(
        model_dir: &Path,
        config: ModelConfig,
        state: Arc<CudaState>,
    ) -> Result<Self> {
        let weights = crate::weights::load_weights(model_dir)?;
        log::info!(
            "CUDA engine: uploading {} tensors as f16 to device {}",
            weights.len(),
            state.ordinal
        );
        let eps = config.layer_norm_eps as f32;
        let h = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let head_dim = h / n_heads;

        let upload_linear = |prefix: &str| -> Result<GpuLinear> {
            let wt = weights
                .get(&format!("{prefix}.weight"))
                .ok_or_else(|| anyhow!("missing {prefix}.weight"))?;
            let bt = weights
                .get(&format!("{prefix}.bias"))
                .ok_or_else(|| anyhow!("missing {prefix}.bias"))?;
            if wt.shape.len() != 2 {
                bail!("{prefix}.weight expected rank-2");
            }
            let out = wt.shape[0];
            let inn = wt.shape[1];
            let w = wt.to_f16_vec()?;
            let b = bt.to_f16_vec().or_else(|_| {
                Ok::<_, anyhow::Error>(
                    bt.to_f32_vec()?
                        .into_iter()
                        .map(f16::from_f32)
                        .collect(),
                )
            })?;
            Ok(GpuLinear {
                w: state.upload_f16(&w)?,
                b: state.upload_f16(&b)?,
                out,
                inn,
            })
        };

        let upload_ln = |prefix: &str| -> Result<GpuLayerNorm> {
            let w = weights
                .get(&format!("{prefix}.weight"))
                .ok_or_else(|| anyhow!("missing {prefix}.weight"))?
                .to_f16_vec()?;
            let b = weights
                .get(&format!("{prefix}.bias"))
                .ok_or_else(|| anyhow!("missing {prefix}.bias"))?
                .to_f16_vec()?;
            let dim = w.len();
            Ok(GpuLayerNorm {
                w: state.upload_f16(&w)?,
                b: state.upload_f16(&b)?,
                eps,
                dim,
            })
        };

        let fuse_qkv = |base: &str| -> Result<GpuLinear> {
            let q = weights
                .get(&format!("{base}.attention.q_proj.weight"))
                .ok_or_else(|| anyhow!("q"))?
                .to_f32_vec()?;
            let k = weights
                .get(&format!("{base}.attention.k_proj.weight"))
                .ok_or_else(|| anyhow!("k"))?
                .to_f32_vec()?;
            let v = weights
                .get(&format!("{base}.attention.v_proj.weight"))
                .ok_or_else(|| anyhow!("v"))?
                .to_f32_vec()?;
            let qb = weights
                .get(&format!("{base}.attention.q_proj.bias"))
                .ok_or_else(|| anyhow!("qb"))?
                .to_f32_vec()?;
            let kb = weights
                .get(&format!("{base}.attention.k_proj.bias"))
                .ok_or_else(|| anyhow!("kb"))?
                .to_f32_vec()?;
            let vb = weights
                .get(&format!("{base}.attention.v_proj.bias"))
                .ok_or_else(|| anyhow!("vb"))?
                .to_f32_vec()?;
            let inn = weights
                .get(&format!("{base}.attention.q_proj.weight"))
                .unwrap()
                .shape[1];
            let out = weights
                .get(&format!("{base}.attention.q_proj.weight"))
                .unwrap()
                .shape[0];
            let mut w = Vec::with_capacity(3 * out * inn);
            w.extend_from_slice(&q);
            w.extend_from_slice(&k);
            w.extend_from_slice(&v);
            let mut b = Vec::with_capacity(3 * out);
            b.extend_from_slice(&qb);
            b.extend_from_slice(&kb);
            b.extend_from_slice(&vb);
            let wh: Vec<f16> = w.into_iter().map(f16::from_f32).collect();
            let bh: Vec<f16> = b.into_iter().map(f16::from_f32).collect();
            Ok(GpuLinear {
                w: state.upload_f16(&wh)?,
                b: state.upload_f16(&bh)?,
                out: 3 * out,
                inn,
            })
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let base = format!("wav2vec2.encoder.layers.{i}");
            layers.push(GpuEncoderLayer {
                ln_attn: upload_ln(&format!("{base}.layer_norm"))?,
                qkv: fuse_qkv(&base)?,
                o_proj: upload_linear(&format!("{base}.attention.out_proj"))?,
                ln_ff: upload_ln(&format!("{base}.final_layer_norm"))?,
                ff_inter: upload_linear(&format!("{base}.feed_forward.intermediate_dense"))?,
                ff_out: upload_linear(&format!("{base}.feed_forward.output_dense"))?,
                n_heads,
                head_dim,
                hidden: h,
            });
        }

        let feat_proj_ln = upload_ln("wav2vec2.feature_projection.layer_norm")?;
        let feat_proj = upload_linear("wav2vec2.feature_projection.projection")?;
        let encoder_ln = upload_ln("wav2vec2.encoder.layer_norm")?;
        let lm_head = upload_linear("lm_head")?;
        let ordinal = state.ordinal;
        let n_layers = layers.len();
        log::info!("CUDA engine ready: {n_layers} layers on device {ordinal}");
        Ok(Self {
            config,
            state,
            layers,
            feat_proj_ln,
            feat_proj,
            encoder_ln,
            lm_head,
        })
    }

    /// Full GPU FE is next. Until then, refuse pure CUDA forward so callers
    /// fall back to CPU via `DeviceRequest::Auto` or explicit `--device cpu`.
    pub fn forward_logits(&self, _waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let _ = (
            &self.layers,
            &self.feat_proj,
            &self.feat_proj_ln,
            &self.encoder_ln,
            &self.lm_head,
        );
        bail!(
            "CUDA forward: weights+PTX+cuBLAS ready on device {}; \
             GPU feature-extractor + encoder wiring is next. Use --device cpu for production.",
            self.state.ordinal
        );
    }
}

// Keep fields "used" for compiler when forward incomplete.
#[allow(dead_code)]
fn _use_layer(l: &GpuEncoderLayer) -> usize {
    l.hidden + l.n_heads + l.head_dim + l.qkv.out + l.ln_attn.dim
}
