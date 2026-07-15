//! CUDA Wav2Vec2ForCTC engine — cudarc + prebuilt multi-arch PTX + cuBLAS.
//!
//! **f32 activations + SGEMM** (match Python CPU golden; native path on Pascal sm_61
//! where f16 has no Tensor Cores and hurts both accuracy and speed).
//! Scheme B PTX: no CUDA Toolkit on end-user machines.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use cudarc::cublas::safe::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::cublas::sys;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::config::ModelConfig;
use crate::prebuilt_ptx;

// ─── Kernel registry (f32) ───────────────────────────────────────────────────

pub struct CudaKernels {
    pub layer_norm: CudaFunction,
    pub gelu_inplace: CudaFunction,
    pub add_inplace: CudaFunction,
    pub add_bias_inplace: CudaFunction,
    pub bias_residual: CudaFunction,
    pub softmax_inplace: CudaFunction,
    pub merge_heads: CudaFunction,
    pub split_qkv_to_heads: CudaFunction,
    pub im2col_1d: CudaFunction,
    pub im2col_1d_ch: CudaFunction,
    pub pad_time: CudaFunction,
    pub add_bias_gelu_inplace: CudaFunction,
    pub scatter_group_bias_gelu: CudaFunction,
    pub split_qkv_batched: CudaFunction,
    pub merge_heads_batched: CudaFunction,
}

impl CudaKernels {
    pub fn load_all(ctx: &Arc<CudaContext>) -> Result<Self> {
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|e| anyhow!("compute_capability: {e:?}"))?;
        let (ptx_src, selected_sm) = prebuilt_ptx::resolve_ptx_for_device(major, minor)
            .map_err(|e| anyhow!("{e}"))?;
        log::info!(
            "loading prebuilt CUDA kernels for device sm_{major}{minor} (selected sm_{selected_sm}, f32 path)"
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
            layer_norm: load("layer_norm_f32")?,
            gelu_inplace: load("gelu_inplace_f32")?,
            add_inplace: load("add_inplace_f32")?,
            add_bias_inplace: load("add_bias_inplace_f32")?,
            bias_residual: load("bias_residual_f32")?,
            softmax_inplace: load("softmax_inplace_last_dim_f32")?,
            merge_heads: load("merge_heads_f32")?,
            split_qkv_to_heads: load("split_qkv_to_heads_f32")?,
            im2col_1d: load("im2col_1d_f32")?,
            im2col_1d_ch: load("im2col_1d_ch_f32")?,
            pad_time: load("pad_time_f32")?,
            add_bias_gelu_inplace: load("add_bias_gelu_inplace_f32")?,
            scatter_group_bias_gelu: load("scatter_group_bias_gelu_f32")?,
            split_qkv_batched: load("split_qkv_to_heads_batched_f32")?,
            merge_heads_batched: load("merge_heads_batched_f32")?,
        })
    }
}

// ─── CudaState ───────────────────────────────────────────────────────────────

pub struct CudaState {
    /// Keep the CUDA context alive for the lifetime of stream / kernels / slices.
    _ctx: Arc<CudaContext>,
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

        // Default math mode is fine for SGEMM on Pascal.
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
        log::info!("CUDA device {ordinal} ready (cuBLAS SGEMM + prebuilt PTX f32)");
        Ok(Self {
            _ctx: ctx.clone(),
            stream,
            blas,
            k,
            ordinal,
            _blas_ws: blas_ws,
        })
    }

    pub fn upload_f32(&self, data: &[f32]) -> Result<CudaSlice<f32>> {
        Ok(self.stream.clone_htod(data)?)
    }

    pub fn alloc_zeros_f32(&self, n: usize) -> Result<CudaSlice<f32>> {
        Ok(self.stream.alloc_zeros::<f32>(n)?)
    }

    pub fn alloc_uninit_f32(&self, n: usize) -> Result<CudaSlice<f32>> {
        Ok(unsafe { self.stream.alloc::<f32>(n)? })
    }

    pub fn download_f32(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(slice)?)
    }

    pub fn gelu_inplace(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut b = self.stream.launch_builder(&self.k.gelu_inplace);
        b.arg(x).arg(&n_i);
        unsafe { b.launch(cfg) }.map_err(|e| anyhow!("gelu_inplace: {e:?}"))?;
        Ok(())
    }

    pub fn add_inplace(&self, a: &mut CudaSlice<f32>, b: &CudaSlice<f32>, n: usize) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_inplace);
        bb.arg(a).arg(b).arg(&n_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("add_inplace: {e:?}"))?;
        Ok(())
    }

    fn layer_norm_into(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<()> {
        let block = dim.next_power_of_two().min(1024).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block as u32, 1, 1),
            shared_mem_bytes: (2 * block * 4) as u32,
        };
        let rows_i = rows as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.layer_norm);
        bb.arg(out)
            .arg(x)
            .arg(w)
            .arg(b)
            .arg(&rows_i)
            .arg(&dim_i)
            .arg(&eps);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("layer_norm: {e:?}"))?;
        Ok(())
    }

    pub fn layer_norm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        rows: usize,
        dim: usize,
        eps: f32,
    ) -> Result<CudaSlice<f32>> {
        let mut out = self.alloc_uninit_f32(rows * dim)?;
        self.layer_norm_into(&mut out, x, w, b, rows, dim, eps)?;
        Ok(out)
    }

    fn gemm_nt_into<W: DevicePtr<f32>>(
        &self,
        y: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        rows: usize,
        w: &W,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<()> {
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: sys::cublasOperation_t::CUBLAS_OP_T,
                        transb: sys::cublasOperation_t::CUBLAS_OP_N,
                        m: out_dim as i32,
                        n: rows as i32,
                        k: in_dim as i32,
                        alpha: 1.0f32,
                        lda: in_dim as i32,
                        ldb: in_dim as i32,
                        beta: 0.0f32,
                        ldc: out_dim as i32,
                    },
                    w,
                    x,
                    y,
                )
                .map_err(|e| anyhow!("cublas SGEMM: {e:?}"))?;
        }
        Ok(())
    }

    pub fn gemm_nt<W: DevicePtr<f32>>(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        w: &W,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<CudaSlice<f32>> {
        let mut y = self.alloc_uninit_f32(rows * out_dim)?;
        self.gemm_nt_into(&mut y, x, rows, w, out_dim, in_dim)?;
        Ok(y)
    }

    pub fn linear_raw(
        &self,
        x: &CudaSlice<f32>,
        rows: usize,
        w: &CudaSlice<f32>,
        out_dim: usize,
        in_dim: usize,
        bias: Option<&CudaSlice<f32>>,
    ) -> Result<CudaSlice<f32>> {
        let mut y = self.gemm_nt(x, rows, w, out_dim, in_dim)?;
        if let Some(bias) = bias {
            self.add_bias_into(&mut y, bias, rows * out_dim, out_dim)?;
        }
        Ok(y)
    }

    fn add_bias_into(
        &self,
        y: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        n: usize,
        cols: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_bias_inplace);
        bb.arg(y).arg(bias).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("add_bias: {e:?}"))?;
        Ok(())
    }

    fn bias_residual_into(
        &self,
        y: &mut CudaSlice<f32>,
        gemm_out: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let n = rows * cols;
        let n_i = n as i32;
        let cols_i = cols as i32;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut bb = self.stream.launch_builder(&self.k.bias_residual);
        bb.arg(y)
            .arg(gemm_out)
            .arg(bias)
            .arg(residual)
            .arg(&n_i)
            .arg(&cols_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("bias_residual: {e:?}"))?;
        Ok(())
    }

    fn softmax_inplace(&self, x: &mut CudaSlice<f32>, rows: usize, dim: usize) -> Result<()> {
        let block = 256u32.min(dim as u32).max(32);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        let rows_i = rows as i32;
        let dim_i = dim as i32;
        let mut bb = self.stream.launch_builder(&self.k.softmax_inplace);
        bb.arg(x).arg(&rows_i).arg(&dim_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("softmax_inplace: {e:?}"))?;
        Ok(())
    }

    fn im2col_1d(
        &self,
        x: &CudaSlice<f32>,
        t_in: usize,
        c_in: usize,
        k: usize,
        stride: usize,
        t_out: usize,
    ) -> Result<CudaSlice<f32>> {
        let kk = c_in * k;
        let n = t_out * kk;
        let mut col = self.alloc_uninit_f32(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_in_i = t_in as i32;
        let c_in_i = c_in as i32;
        let k_i = k as i32;
        let stride_i = stride as i32;
        let t_out_i = t_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.im2col_1d);
        bb.arg(&mut col)
            .arg(x)
            .arg(&t_in_i)
            .arg(&c_in_i)
            .arg(&k_i)
            .arg(&stride_i)
            .arg(&t_out_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("im2col: {e:?}"))?;
        Ok(col)
    }

    fn im2col_1d_ch(
        &self,
        x: &CudaSlice<f32>,
        t_in: usize,
        c_full: usize,
        c_start: usize,
        c_in: usize,
        k: usize,
        stride: usize,
        t_out: usize,
    ) -> Result<CudaSlice<f32>> {
        let kk = c_in * k;
        let n = t_out * kk;
        let mut col = self.alloc_uninit_f32(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_in_i = t_in as i32;
        let c_full_i = c_full as i32;
        let c_start_i = c_start as i32;
        let c_in_i = c_in as i32;
        let k_i = k as i32;
        let stride_i = stride as i32;
        let t_out_i = t_out as i32;
        let mut bb = self.stream.launch_builder(&self.k.im2col_1d_ch);
        bb.arg(&mut col)
            .arg(x)
            .arg(&t_in_i)
            .arg(&c_full_i)
            .arg(&c_start_i)
            .arg(&c_in_i)
            .arg(&k_i)
            .arg(&stride_i)
            .arg(&t_out_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("im2col_ch: {e:?}"))?;
        Ok(col)
    }

    fn pad_time(
        &self,
        x: &CudaSlice<f32>,
        t: usize,
        h: usize,
        pad: usize,
    ) -> Result<CudaSlice<f32>> {
        let t_pad = t + 2 * pad;
        let n = t_pad * h;
        let mut out = self.alloc_uninit_f32(n)?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_i = t as i32;
        let h_i = h as i32;
        let pad_i = pad as i32;
        let mut bb = self.stream.launch_builder(&self.k.pad_time);
        bb.arg(&mut out)
            .arg(x)
            .arg(&t_i)
            .arg(&h_i)
            .arg(&pad_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("pad_time: {e:?}"))?;
        Ok(out)
    }

    fn add_bias_gelu_inplace(
        &self,
        x: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        n: usize,
        cols: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let n_i = n as i32;
        let cols_i = cols as i32;
        let mut bb = self.stream.launch_builder(&self.k.add_bias_gelu_inplace);
        bb.arg(x).arg(bias).arg(&n_i).arg(&cols_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("add_bias_gelu: {e:?}"))?;
        Ok(())
    }

    fn scatter_group_bias_gelu(
        &self,
        out: &mut CudaSlice<f32>,
        y_g: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        t_out: usize,
        c_g: usize,
        h: usize,
        c_start: usize,
    ) -> Result<()> {
        let n = t_out * c_g;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_out_i = t_out as i32;
        let c_g_i = c_g as i32;
        let h_i = h as i32;
        let c_start_i = c_start as i32;
        let mut bb = self.stream.launch_builder(&self.k.scatter_group_bias_gelu);
        bb.arg(out)
            .arg(y_g)
            .arg(bias)
            .arg(&t_out_i)
            .arg(&c_g_i)
            .arg(&h_i)
            .arg(&c_start_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("scatter_group: {e:?}"))?;
        Ok(())
    }

    fn split_qkv_into(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        qkv: &CudaSlice<f32>,
        t: usize,
        n_heads: usize,
        hd: usize,
    ) -> Result<()> {
        let n = t * n_heads * hd;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_i = t as i32;
        let nh_i = n_heads as i32;
        let hd_i = hd as i32;
        let mut bb = self.stream.launch_builder(&self.k.split_qkv_to_heads);
        bb.arg(q)
            .arg(k)
            .arg(v)
            .arg(qkv)
            .arg(&t_i)
            .arg(&nh_i)
            .arg(&hd_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("split_qkv: {e:?}"))?;
        Ok(())
    }

    fn merge_heads_into(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        t: usize,
        n_heads: usize,
        hd: usize,
    ) -> Result<()> {
        let n = t * n_heads * hd;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let t_i = t as i32;
        let nh_i = n_heads as i32;
        let hd_i = hd as i32;
        let mut bb = self.stream.launch_builder(&self.k.merge_heads);
        bb.arg(out)
            .arg(x)
            .arg(&t_i)
            .arg(&nh_i)
            .arg(&hd_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("merge_heads: {e:?}"))?;
        Ok(())
    }

    fn split_qkv_batched_into(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        qkv: &CudaSlice<f32>,
        b: usize,
        t: usize,
        n_heads: usize,
        hd: usize,
    ) -> Result<()> {
        let n = b * t * n_heads * hd;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let b_i = b as i32;
        let t_i = t as i32;
        let nh_i = n_heads as i32;
        let hd_i = hd as i32;
        let mut bb = self.stream.launch_builder(&self.k.split_qkv_batched);
        bb.arg(q)
            .arg(k)
            .arg(v)
            .arg(qkv)
            .arg(&b_i)
            .arg(&t_i)
            .arg(&nh_i)
            .arg(&hd_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("split_qkv_batched: {e:?}"))?;
        Ok(())
    }

    fn merge_heads_batched_into(
        &self,
        out: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        b: usize,
        t: usize,
        n_heads: usize,
        hd: usize,
    ) -> Result<()> {
        let n = b * t * n_heads * hd;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let b_i = b as i32;
        let t_i = t as i32;
        let nh_i = n_heads as i32;
        let hd_i = hd as i32;
        let mut bb = self.stream.launch_builder(&self.k.merge_heads_batched);
        bb.arg(out)
            .arg(x)
            .arg(&b_i)
            .arg(&t_i)
            .arg(&nh_i)
            .arg(&hd_i);
        unsafe { bb.launch(cfg) }.map_err(|e| anyhow!("merge_heads_batched: {e:?}"))?;
        Ok(())
    }

    /// Q/K layout: [batch_heads, T, hd], scores: [batch_heads, T, T]
    fn attention_qk_into(
        &self,
        scores: &mut CudaSlice<f32>,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        batch_heads: usize,
        t: usize,
        hd: usize,
        scale: f32,
    ) -> Result<()> {
        self.attention_qk_strided(
            scores,
            q,
            k,
            batch_heads,
            t,
            hd,
            scale,
            (t * hd) as i64,
        )
    }

    /// Strided QK: q/k start at head base; stride between batch items may be n_heads*T*hd.
    fn attention_qk_strided<Q: DevicePtr<f32>, K: DevicePtr<f32>>(
        &self,
        scores: &mut CudaSlice<f32>,
        q: &Q,
        k: &K,
        batch: usize,
        t: usize,
        hd: usize,
        scale: f32,
        stride_qk: i64,
    ) -> Result<()> {
        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_T,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: t as i32,
                            n: t as i32,
                            k: hd as i32,
                            alpha: scale,
                            lda: hd as i32,
                            ldb: hd as i32,
                            beta: 0.0f32,
                            ldc: t as i32,
                        },
                        batch_size: batch as i32,
                        stride_a: stride_qk,
                        stride_b: stride_qk,
                        stride_c: (t * t) as i64,
                    },
                    k,
                    q,
                    scores,
                )
                .map_err(|e| anyhow!("attn QK SGEMM: {e:?}"))?;
        }
        Ok(())
    }

    /// attn: [batch, T, T], V/out strided by `stride_v`
    fn attention_av_strided<O: DevicePtrMut<f32>, V: DevicePtr<f32>>(
        &self,
        out: &mut O,
        attn: &CudaSlice<f32>,
        v: &V,
        batch: usize,
        t: usize,
        hd: usize,
        stride_v: i64,
    ) -> Result<()> {
        unsafe {
            self.blas
                .gemm_strided_batched(
                    StridedBatchedConfig {
                        gemm: GemmConfig {
                            transa: sys::cublasOperation_t::CUBLAS_OP_N,
                            transb: sys::cublasOperation_t::CUBLAS_OP_N,
                            m: hd as i32,
                            n: t as i32,
                            k: t as i32,
                            alpha: 1.0f32,
                            lda: hd as i32,
                            ldb: t as i32,
                            beta: 0.0f32,
                            ldc: hd as i32,
                        },
                        batch_size: batch as i32,
                        stride_a: stride_v,
                        stride_b: (t * t) as i64,
                        stride_c: stride_v,
                    },
                    v,
                    attn,
                    out,
                )
                .map_err(|e| anyhow!("attn AV SGEMM: {e:?}"))?;
        }
        Ok(())
    }
}

// ─── GPU weights (f32) ───────────────────────────────────────────────────────

struct GpuLinear {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    out: usize,
    inn: usize,
}

struct GpuLayerNorm {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    eps: f32,
    dim: usize,
}

struct GpuConv1d {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: Option<GpuLayerNorm>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
}

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

struct GpuScratch {
    /// Max batch dimension this scratch was allocated for.
    b: usize,
    t: usize,
    h: usize,
    n_heads: usize,
    hd: usize,
    inter: usize,
    norm: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// Attention scores / weights (softmax in-place; no separate attn buffer).
    scores: CudaSlice<f32>,
    ctx_pack: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    o_gemm: CudaSlice<f32>,
    after_attn: CudaSlice<f32>,
    ff_norm: CudaSlice<f32>,
    ff_inter: CudaSlice<f32>,
    ff_gemm: CudaSlice<f32>,
    x0: CudaSlice<f32>,
    x1: CudaSlice<f32>,
}

impl GpuScratch {
    fn ensure(
        s: &CudaState,
        slot: &mut Option<Self>,
        b: usize,
        t: usize,
        h: usize,
        n_heads: usize,
        hd: usize,
        inter: usize,
    ) -> Result<()> {
        let need = match slot {
            Some(sc) => {
                sc.b < b
                    || sc.t != t
                    || sc.h != h
                    || sc.n_heads != n_heads
                    || sc.hd != hd
                    || sc.inter != inter
            }
            None => true,
        };
        if !need {
            return Ok(());
        }
        // Cap batch to keep scores buffer reasonable on 8GB Pascal.
        let b_alloc = b.max(1);
        let rows = b_alloc * t;
        let hs = t * hd;
        let nh = n_heads;
        *slot = Some(Self {
            b: b_alloc,
            t,
            h,
            n_heads,
            hd,
            inter,
            norm: s.alloc_uninit_f32(rows * h)?,
            qkv: s.alloc_uninit_f32(rows * 3 * h)?,
            q: s.alloc_uninit_f32(b_alloc * nh * hs)?,
            k: s.alloc_uninit_f32(b_alloc * nh * hs)?,
            v: s.alloc_uninit_f32(b_alloc * nh * hs)?,
            scores: s.alloc_uninit_f32(b_alloc * nh * t * t)?,
            ctx_pack: s.alloc_uninit_f32(b_alloc * nh * hs)?,
            ctx: s.alloc_uninit_f32(rows * h)?,
            o_gemm: s.alloc_uninit_f32(rows * h)?,
            after_attn: s.alloc_uninit_f32(rows * h)?,
            ff_norm: s.alloc_uninit_f32(rows * h)?,
            ff_inter: s.alloc_uninit_f32(rows * inter)?,
            ff_gemm: s.alloc_uninit_f32(rows * h)?,
            x0: s.alloc_uninit_f32(rows * h)?,
            x1: s.alloc_uninit_f32(rows * h)?,
        });
        Ok(())
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

pub struct CudaEngine {
    pub config: ModelConfig,
    pub state: Arc<CudaState>,
    conv_layers: Vec<GpuConv1d>,
    feat_proj_ln: GpuLayerNorm,
    feat_proj: GpuLinear,
    pos_w: CudaSlice<f32>,
    pos_b: CudaSlice<f32>,
    pos_k: usize,
    pos_groups: usize,
    layers: Vec<GpuEncoderLayer>,
    encoder_ln: GpuLayerNorm,
    lm_head: GpuLinear,
    scratch: Mutex<Option<GpuScratch>>,
}

impl CudaEngine {
    pub fn load(
        model_dir: &Path,
        config: ModelConfig,
        state: Arc<CudaState>,
    ) -> Result<Self> {
        let weights = crate::weights::load_weights(model_dir)?;
        log::info!(
            "CUDA engine: uploading {} tensors as f32 to device {}",
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
            let w = wt.to_f32_vec()?;
            let b = bt.to_f32_vec()?;
            Ok(GpuLinear {
                w: state.upload_f32(&w)?,
                b: state.upload_f32(&b)?,
                out,
                inn,
            })
        };

        let upload_ln = |prefix: &str| -> Result<GpuLayerNorm> {
            let w = weights
                .get(&format!("{prefix}.weight"))
                .ok_or_else(|| anyhow!("missing {prefix}.weight"))?
                .to_f32_vec()?;
            let b = weights
                .get(&format!("{prefix}.bias"))
                .ok_or_else(|| anyhow!("missing {prefix}.bias"))?
                .to_f32_vec()?;
            let dim = w.len();
            Ok(GpuLayerNorm {
                w: state.upload_f32(&w)?,
                b: state.upload_f32(&b)?,
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
            Ok(GpuLinear {
                w: state.upload_f32(&w)?,
                b: state.upload_f32(&b)?,
                out: 3 * out,
                inn,
            })
        };

        let mut conv_layers = Vec::with_capacity(config.conv_dim.len());
        for i in 0..config.conv_dim.len() {
            let prefix = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            let w_t = weights
                .get(&format!("{prefix}.conv.weight"))
                .ok_or_else(|| anyhow!("missing {prefix}.conv.weight"))?;
            let b_t = weights
                .get(&format!("{prefix}.conv.bias"))
                .ok_or_else(|| anyhow!("missing {prefix}.conv.bias"))?;
            if w_t.shape.len() != 3 {
                bail!("conv {i} weight shape {:?}", w_t.shape);
            }
            let out_ch = w_t.shape[0];
            let in_ch = w_t.shape[1];
            let k = w_t.shape[2];
            let w = w_t.to_f32_vec()?;
            let b = b_t.to_f32_vec()?;
            let ln = if weights.contains_key(&format!("{prefix}.layer_norm.weight")) {
                Some(upload_ln(&format!("{prefix}.layer_norm"))?)
            } else {
                None
            };
            conv_layers.push(GpuConv1d {
                w: state.upload_f32(&w)?,
                b: state.upload_f32(&b)?,
                ln,
                out_ch,
                in_ch,
                k,
                stride: config.conv_stride[i],
            });
        }

        let pos_k = config.num_conv_pos_embeddings;
        let pos_groups = config.num_conv_pos_embedding_groups;
        let g = weights
            .get("wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0")
            .ok_or_else(|| anyhow!("missing pos_conv weight_norm g"))?
            .to_f32_vec()?;
        let v = weights
            .get("wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1")
            .ok_or_else(|| anyhow!("missing pos_conv weight_norm v"))?
            .to_f32_vec()?;
        let pos_b_f = weights
            .get("wav2vec2.encoder.pos_conv_embed.conv.bias")
            .ok_or_else(|| anyhow!("missing pos_conv bias"))?
            .to_f32_vec()?;
        let in_g = h / pos_groups;
        if v.len() != h * in_g * pos_k {
            bail!(
                "pos_conv v len {} != {}*{}*{}",
                v.len(),
                h,
                in_g,
                pos_k
            );
        }
        let mut pos_w = vec![0.0f32; v.len()];
        for kk in 0..pos_k {
            let mut norm_sq = 0.0f32;
            for oi in 0..(h * in_g) {
                let val = v[oi * pos_k + kk];
                norm_sq += val * val;
            }
            let inv = norm_sq.sqrt().recip();
            let gk = if g.len() == pos_k {
                g[kk]
            } else if g.len() == 1 {
                g[0]
            } else {
                g[kk]
            };
            for oi in 0..(h * in_g) {
                let idx = oi * pos_k + kk;
                pos_w[idx] = v[idx] * gk * inv;
            }
        }

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
        let pos_w_gpu = state.upload_f32(&pos_w)?;
        let pos_b_gpu = state.upload_f32(&pos_b_f)?;
        let ordinal = state.ordinal;
        let n_layers = layers.len();
        log::info!(
            "CUDA engine ready: {n_layers} layers + {} FE convs (f32) on device {ordinal}",
            conv_layers.len()
        );
        Ok(Self {
            config,
            state,
            conv_layers,
            feat_proj_ln,
            feat_proj,
            pos_w: pos_w_gpu,
            pos_b: pos_b_gpu,
            pos_k,
            pos_groups,
            layers,
            encoder_ln,
            lm_head,
            scratch: Mutex::new(None),
        })
    }

    /// Full GPU forward: waveform → CTC logits `(T, vocab)` row-major f32.
    pub fn forward_logits(&self, waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let profile = std::env::var("CTC_PROFILE").ok().as_deref() == Some("1");
        let t0 = std::time::Instant::now();
        let hidden = self.frontend_hidden(waveform)?;
        let t_feat = hidden.len() / self.config.hidden_size;
        let t_front = t0.elapsed();
        let logits = self.encoder_and_lm_head(std::slice::from_ref(&hidden), t_feat)?;
        let vocab = self.lm_head.out;
        if profile {
            eprintln!(
                "[CTC_PROFILE cuda-f32] T={t_feat} frontend={:.3}s total={:.3}s",
                t_front.as_secs_f64(),
                t0.elapsed().as_secs_f64()
            );
        }
        Ok((logits.into_iter().next().unwrap(), t_feat, vocab))
    }

    /// Batched equal-length waveforms. Encoder micro-batch B≤2 (fat FFN + batched QK).
    pub fn forward_logits_batch(
        &self,
        waveforms: &[&[f32]],
    ) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        if waveforms.is_empty() {
            return Ok(Vec::new());
        }
        if waveforms.len() == 1 {
            let (l, t, c) = self.forward_logits(waveforms[0])?;
            return Ok(vec![(l, t, c)]);
        }
        let profile = std::env::var("CTC_PROFILE").ok().as_deref() == Some("1");
        let t0 = std::time::Instant::now();

        let mut hiddens = Vec::with_capacity(waveforms.len());
        let mut t_feat = 0usize;
        for w in waveforms {
            let h = self.frontend_hidden(w)?;
            if t_feat == 0 {
                t_feat = h.len() / self.config.hidden_size;
            } else if h.len() != t_feat * self.config.hidden_size {
                bail!("batch windows must share the same frame length");
            }
            hiddens.push(h);
        }

        // Full batch when possible (best on sm_61). Override: CTC_CUDA_BATCH=N.
        let micro: usize = std::env::var("CTC_CUDA_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8)
            .clamp(1, 8);
        let mut all_logits = Vec::with_capacity(hiddens.len());
        for chunk in hiddens.chunks(micro) {
            let mut outs = self.encoder_and_lm_head(chunk, t_feat)?;
            all_logits.append(&mut outs);
        }
        let vocab = self.lm_head.out;
        if profile {
            eprintln!(
                "[CTC_PROFILE cuda-f32 batch] B={} T={t_feat} total={:.3}s",
                waveforms.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(all_logits
            .into_iter()
            .map(|l| (l, t_feat, vocab))
            .collect())
    }

    fn frontend_hidden(&self, waveform: &[f32]) -> Result<CudaSlice<f32>> {
        let s = &*self.state;
        let mut x = s.upload_f32(waveform)?;
        let mut t_len = waveform.len();
        let mut channels = 1usize;

        for (li, conv) in self.conv_layers.iter().enumerate() {
            if channels != conv.in_ch {
                bail!(
                    "conv layer {li}: in_ch mismatch got {channels} expect {}",
                    conv.in_ch
                );
            }
            if t_len < conv.k {
                bail!("conv1d: sequence shorter than kernel ({t_len} < {})", conv.k);
            }
            let t_out = (t_len - conv.k) / conv.stride + 1;
            let kk = conv.in_ch * conv.k;
            let col = s.im2col_1d(&x, t_len, conv.in_ch, conv.k, conv.stride, t_out)?;
            let mut y = s.gemm_nt(&col, t_out, &conv.w, conv.out_ch, kk)?;
            let n = t_out * conv.out_ch;
            if let Some(ref ln) = conv.ln {
                s.add_bias_into(&mut y, &conv.b, n, conv.out_ch)?;
                let mut normed = s.layer_norm(&y, &ln.w, &ln.b, t_out, conv.out_ch, ln.eps)?;
                s.gelu_inplace(&mut normed, n)?;
                x = normed;
            } else {
                s.add_bias_gelu_inplace(&mut y, &conv.b, n, conv.out_ch)?;
                x = y;
            }
            t_len = t_out;
            channels = conv.out_ch;
        }

        let t_feat = t_len;
        let feat = s.layer_norm(
            &x,
            &self.feat_proj_ln.w,
            &self.feat_proj_ln.b,
            t_feat,
            self.feat_proj_ln.dim,
            self.feat_proj_ln.eps,
        )?;
        let mut hidden = s.linear_raw(
            &feat,
            t_feat,
            &self.feat_proj.w,
            self.feat_proj.out,
            self.feat_proj.inn,
            Some(&self.feat_proj.b),
        )?;
        let pos = self.pos_conv_embed(&hidden, t_feat)?;
        s.add_inplace(&mut hidden, &pos, t_feat * self.config.hidden_size)?;
        Ok(hidden)
    }

    /// Run encoder + lm_head on a batch of equal-length [T,H] GPU tensors.
    fn encoder_and_lm_head(
        &self,
        hiddens: &[CudaSlice<f32>],
        t: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let s = &*self.state;
        let b = hiddens.len();
        let h = self.config.hidden_size;
        let n_heads = self.config.num_attention_heads;
        let hd = h / n_heads;
        let inter = self.config.intermediate_size;
        {
            let mut slot = self.scratch.lock().map_err(|e| anyhow!("scratch lock: {e}"))?;
            GpuScratch::ensure(s, &mut slot, b, t, h, n_heads, hd, inter)?;
            let sc = slot.as_mut().unwrap();
            if b == 1 {
                s.stream.memcpy_dtod(&hiddens[0], &mut sc.x0)?;
            } else {
                // Device-side pack into [B,T,H] without host staging.
                for (bi, hid) in hiddens.iter().enumerate() {
                    let off = bi * t * h;
                    let mut dst = sc.x0.slice_mut(off..off + t * h);
                    s.stream
                        .memcpy_dtod(hid, &mut dst)
                        .map_err(|e| anyhow!("pack hidden bi={bi}: {e:?}"))?;
                }
            }
        }

        let mut src_is_x0 = true;
        for layer in &self.layers {
            self.encoder_layer_scratch(layer, b, t, src_is_x0)?;
            src_is_x0 = !src_is_x0;
        }

        let mut slot = self.scratch.lock().map_err(|e| anyhow!("scratch lock: {e}"))?;
        let sc = slot.as_mut().unwrap();
        let x_final = if src_is_x0 { &sc.x0 } else { &sc.x1 };
        let rows = b * t;
        s.layer_norm_into(
            &mut sc.norm,
            x_final,
            &self.encoder_ln.w,
            &self.encoder_ln.b,
            rows,
            self.encoder_ln.dim,
            self.encoder_ln.eps,
        )?;
        let vocab = self.lm_head.out;
        let mut logits_gpu = s.alloc_uninit_f32(rows * vocab)?;
        s.gemm_nt_into(
            &mut logits_gpu,
            &sc.norm,
            rows,
            &self.lm_head.w,
            self.lm_head.out,
            self.lm_head.inn,
        )?;
        s.add_bias_into(&mut logits_gpu, &self.lm_head.b, rows * vocab, vocab)?;
        drop(slot);
        let flat = s.download_f32(&logits_gpu)?;
        let mut outs = Vec::with_capacity(b);
        for bi in 0..b {
            outs.push(flat[bi * t * vocab..(bi + 1) * t * vocab].to_vec());
        }
        Ok(outs)
    }

    fn pos_conv_embed(&self, hidden: &CudaSlice<f32>, t: usize) -> Result<CudaSlice<f32>> {
        let s = &*self.state;
        let h = self.config.hidden_size;
        let k = self.pos_k;
        let groups = self.pos_groups;
        let in_g = h / groups;
        let pad = k / 2;
        let t_pad = t + 2 * pad;
        let remove = if k % 2 == 0 { 1 } else { 0 };
        let out_len_full = t_pad - k + 1;
        let use_len = (out_len_full - remove).min(t);
        let kk = in_g * k;

        let x_pad = s.pad_time(hidden, t, h, pad)?;
        let mut out = s.alloc_zeros_f32(t * h)?;

        for g in 0..groups {
            let c_start = g * in_g;
            let col = s.im2col_1d_ch(&x_pad, t_pad, h, c_start, in_g, k, 1, use_len)?;
            let w_offset = g * in_g * kk;
            let w_g = self.pos_w.slice(w_offset..w_offset + in_g * kk);
            let y_g = s.gemm_nt(&col, use_len, &w_g, in_g, kk)?;
            s.scatter_group_bias_gelu(&mut out, &y_g, &self.pos_b, use_len, in_g, h, c_start)?;
        }
        Ok(out)
    }

    /// Encoder layer over `b` sequences of length `t` packed as [B*T, H].
    fn encoder_layer_scratch(
        &self,
        layer: &GpuEncoderLayer,
        b: usize,
        t: usize,
        src_is_x0: bool,
    ) -> Result<()> {
        let s = &*self.state;
        let h = layer.hidden;
        let n_heads = layer.n_heads;
        let hd = layer.head_dim;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let rows = b * t;
        let batch_heads = b * n_heads;
        let mut sc = self.scratch.lock().map_err(|e| anyhow!("scratch lock: {e}"))?;
        let sc = sc.as_mut().ok_or_else(|| anyhow!("encoder scratch not initialized"))?;

        if src_is_x0 {
            s.layer_norm_into(
                &mut sc.norm,
                &sc.x0,
                &layer.ln_attn.w,
                &layer.ln_attn.b,
                rows,
                layer.ln_attn.dim,
                layer.ln_attn.eps,
            )?;
        } else {
            s.layer_norm_into(
                &mut sc.norm,
                &sc.x1,
                &layer.ln_attn.w,
                &layer.ln_attn.b,
                rows,
                layer.ln_attn.dim,
                layer.ln_attn.eps,
            )?;
        }

        s.gemm_nt_into(
            &mut sc.qkv,
            &sc.norm,
            rows,
            &layer.qkv.w,
            layer.qkv.out,
            layer.qkv.inn,
        )?;
        s.add_bias_into(&mut sc.qkv, &layer.qkv.b, rows * layer.qkv.out, layer.qkv.out)?;

        // Pack [B*T, 3H] → [B*n_heads, T, hd], full strided-batched MHA, merge.
        if b == 1 {
            s.split_qkv_into(&mut sc.q, &mut sc.k, &mut sc.v, &sc.qkv, t, n_heads, hd)?;
        } else {
            s.split_qkv_batched_into(&mut sc.q, &mut sc.k, &mut sc.v, &sc.qkv, b, t, n_heads, hd)?;
        }
        s.attention_qk_into(&mut sc.scores, &sc.q, &sc.k, batch_heads, t, hd, scale)?;
        // Softmax in-place on scores (reuse as attn weights) — skip second T² buffer write.
        s.softmax_inplace(&mut sc.scores, batch_heads * t, t)?;
        s.attention_av_strided(
            &mut sc.ctx_pack,
            &sc.scores,
            &sc.v,
            batch_heads,
            t,
            hd,
            (t * hd) as i64,
        )?;
        if b == 1 {
            s.merge_heads_into(&mut sc.ctx, &sc.ctx_pack, t, n_heads, hd)?;
        } else {
            s.merge_heads_batched_into(&mut sc.ctx, &sc.ctx_pack, b, t, n_heads, hd)?;
        }

        s.gemm_nt_into(
            &mut sc.o_gemm,
            &sc.ctx,
            rows,
            &layer.o_proj.w,
            layer.o_proj.out,
            layer.o_proj.inn,
        )?;
        if src_is_x0 {
            s.bias_residual_into(
                &mut sc.after_attn,
                &sc.o_gemm,
                &layer.o_proj.b,
                &sc.x0,
                rows,
                h,
            )?;
        } else {
            s.bias_residual_into(
                &mut sc.after_attn,
                &sc.o_gemm,
                &layer.o_proj.b,
                &sc.x1,
                rows,
                h,
            )?;
        }

        s.layer_norm_into(
            &mut sc.ff_norm,
            &sc.after_attn,
            &layer.ln_ff.w,
            &layer.ln_ff.b,
            rows,
            layer.ln_ff.dim,
            layer.ln_ff.eps,
        )?;
        s.gemm_nt_into(
            &mut sc.ff_inter,
            &sc.ff_norm,
            rows,
            &layer.ff_inter.w,
            layer.ff_inter.out,
            layer.ff_inter.inn,
        )?;
        s.add_bias_into(
            &mut sc.ff_inter,
            &layer.ff_inter.b,
            rows * layer.ff_inter.out,
            layer.ff_inter.out,
        )?;
        s.gelu_inplace(&mut sc.ff_inter, rows * layer.ff_inter.out)?;
        s.gemm_nt_into(
            &mut sc.ff_gemm,
            &sc.ff_inter,
            rows,
            &layer.ff_out.w,
            layer.ff_out.out,
            layer.ff_out.inn,
        )?;
        if src_is_x0 {
            s.bias_residual_into(
                &mut sc.x1,
                &sc.ff_gemm,
                &layer.ff_out.b,
                &sc.after_attn,
                rows,
                h,
            )?;
        } else {
            s.bias_residual_into(
                &mut sc.x0,
                &sc.ff_gemm,
                &layer.ff_out.b,
                &sc.after_attn,
                rows,
                h,
            )?;
        }
        Ok(())
    }
}
