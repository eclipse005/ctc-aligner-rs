//! CPU Wav2Vec2ForCTC engine (gemm + rayon).
//!
//! Inference-only port of HuggingFace `Wav2Vec2ForCTC` with
//! `do_stable_layer_norm=true` and `feat_extract_norm=layer` (MMS-300m).
//! Golden: Python transformers + ctc_forced_aligner.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use gemm::{gemm, Parallelism};
use rayon::prelude::*;

use crate::config::ModelConfig;
use crate::raw_tensor::RawTensor;

// ─── Weight containers ───────────────────────────────────────────────────────

#[derive(Clone)]
struct Linear {
    /// Row-major [out, in]
    w: Vec<f32>,
    b: Vec<f32>,
    out: usize,
    inn: usize,
}

impl Linear {
    fn from_map(weights: &HashMap<String, RawTensor>, prefix: &str) -> Result<Self> {
        let w_t = weights
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| anyhow!("missing {prefix}.weight"))?;
        let b_t = weights
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| anyhow!("missing {prefix}.bias"))?;
        if w_t.shape.len() != 2 {
            bail!("{prefix}.weight expected rank-2, got {:?}", w_t.shape);
        }
        let out = w_t.shape[0];
        let inn = w_t.shape[1];
        let w = w_t.to_f32_vec()?;
        let b = b_t.to_f32_vec()?;
        if w.len() != out * inn || b.len() != out {
            bail!("{prefix}: weight/bias size mismatch");
        }
        Ok(Self { w, b, out, inn })
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        self.forward_par(x, rows, Parallelism::Rayon(0))
    }

    fn forward_par(&self, x: &[f32], rows: usize, par: Parallelism) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * self.out];
        self.forward_into(x, rows, &mut out, par);
        out
    }

    fn forward_into(&self, x: &[f32], rows: usize, out: &mut [f32], par: Parallelism) {
        debug_assert_eq!(x.len(), rows * self.inn);
        debug_assert_eq!(out.len(), rows * self.out);
        // out = x @ W^T  (W is [out, in] row-major)
        unsafe {
            gemm(
                rows,
                self.out,
                self.inn,
                out.as_mut_ptr(),
                1,
                self.out as isize,
                false,
                x.as_ptr(),
                1,
                self.inn as isize,
                self.w.as_ptr(),
                self.inn as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                par,
            );
        }
        let n = self.out;
        let bias = &self.b;
        match par {
            Parallelism::None => {
                for r in 0..rows {
                    let row = &mut out[r * n..(r + 1) * n];
                    for j in 0..n {
                        row[j] += bias[j];
                    }
                }
            }
            _ => {
                out.par_chunks_mut(n).for_each(|row| {
                    for j in 0..n {
                        row[j] += bias[j];
                    }
                });
            }
        }
    }
}

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    fn from_map(weights: &HashMap<String, RawTensor>, prefix: &str, eps: f32) -> Result<Self> {
        let w = weights
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| anyhow!("missing {prefix}.weight"))?
            .to_f32_vec()?;
        let b = weights
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| anyhow!("missing {prefix}.bias"))?
            .to_f32_vec()?;
        Ok(Self { w, b, eps })
    }

    fn forward_inplace(&self, x: &mut [f32], rows: usize) {
        let dim = self.w.len();
        debug_assert_eq!(x.len(), rows * dim);
        let w = &self.w;
        let b = &self.b;
        let eps = self.eps;
        // Parallel over rows (frames).
        x.par_chunks_mut(dim).for_each(|row| {
            let mean = row.iter().sum::<f32>() / dim as f32;
            let mut var = 0.0f32;
            for v in row.iter() {
                let d = *v - mean;
                var += d * d;
            }
            var /= dim as f32;
            let inv = (var + eps).sqrt().recip();
            for j in 0..dim {
                row[j] = (row[j] - mean) * inv * w[j] + b[j];
            }
        });
    }

    fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut y = x.to_vec();
        self.forward_inplace(&mut y, rows);
        y
    }
}

struct Conv1dLayer {
    /// [out_ch, in_ch, k]
    w: Vec<f32>,
    b: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    stride: usize,
    ln: Option<LayerNorm>,
}

impl Conv1dLayer {
    fn forward(&self, x: &[f32], in_len: usize) -> Result<(Vec<f32>, usize)> {
        // x: [in_ch, L]
        if x.len() != self.in_ch * in_len {
            bail!(
                "conv1d input size {} != {}*{}",
                x.len(),
                self.in_ch,
                in_len
            );
        }
        if in_len < self.k {
            bail!("conv1d: sequence shorter than kernel ({in_len} < {})", self.k);
        }
        let out_len = (in_len - self.k) / self.stride + 1;
        let mut y = vec![0.0f32; self.out_ch * out_len];
        // Parallel over output channels.
        y.par_chunks_mut(out_len)
            .enumerate()
            .for_each(|(oc, y_ch)| {
                let w_base = oc * self.in_ch * self.k;
                for t in 0..out_len {
                    let mut acc = self.b[oc];
                    let t0 = t * self.stride;
                    for ic in 0..self.in_ch {
                        let x_base = ic * in_len + t0;
                        let w_row = w_base + ic * self.k;
                        for kk in 0..self.k {
                            acc += x[x_base + kk] * self.w[w_row + kk];
                        }
                    }
                    y_ch[t] = acc;
                }
            });

        // LayerNorm over channel dim at each time: HF does transpose → LN(last=out_ch) → transpose
        // y is [out_ch, T]; we need LN across out_ch for each t.
        if let Some(ln) = &self.ln {
            // Convert to [T, out_ch], LN, back.
            let mut tmp = vec![0.0f32; out_len * self.out_ch];
            for t in 0..out_len {
                for c in 0..self.out_ch {
                    tmp[t * self.out_ch + c] = y[c * out_len + t];
                }
            }
            ln.forward_inplace(&mut tmp, out_len);
            for t in 0..out_len {
                for c in 0..self.out_ch {
                    y[c * out_len + t] = tmp[t * self.out_ch + c];
                }
            }
        }

        y.par_iter_mut().for_each(|v| *v = gelu(*v));
        Ok((y, out_len))
    }
}

struct EncoderLayer {
    ln_attn: LayerNorm,
    /// Fused QKV projection: W is [3H, H] row-major (Q then K then V blocks).
    /// One gemm replaces three — halves weight packing traffic on the hot path.
    qkv: Linear,
    out: Linear,
    ln_ff: LayerNorm,
    ff_inter: Linear,
    ff_out: Linear,
    n_heads: usize,
    head_dim: usize,
    hidden: usize,
}

/// Grow-only scratch for one attention call (thread-local → zero cross-thread contention).
struct AttnScratch {
    qkv: Vec<f32>,
    q_pack: Vec<f32>,
    k_pack: Vec<f32>,
    v_pack: Vec<f32>,
    ctx_pack: Vec<f32>,
    ctx: Vec<f32>,
    scores: Vec<f32>,
    ctx_h: Vec<f32>,
}

impl AttnScratch {
    fn with_capacity(t: usize, h: usize, n_heads: usize, head_dim: usize) -> Self {
        let hs = t * head_dim;
        Self {
            qkv: Vec::with_capacity(t * 3 * h),
            q_pack: Vec::with_capacity(n_heads * hs),
            k_pack: Vec::with_capacity(n_heads * hs),
            v_pack: Vec::with_capacity(n_heads * hs),
            ctx_pack: Vec::with_capacity(n_heads * hs),
            ctx: Vec::with_capacity(t * h),
            scores: Vec::with_capacity(t * t),
            ctx_h: Vec::with_capacity(hs),
        }
    }

    fn ensure(&mut self, t: usize, h: usize, n_heads: usize, head_dim: usize) {
        let hs = t * head_dim;
        resize_uninit(&mut self.qkv, t * 3 * h);
        resize_uninit(&mut self.q_pack, n_heads * hs);
        resize_uninit(&mut self.k_pack, n_heads * hs);
        resize_uninit(&mut self.v_pack, n_heads * hs);
        resize_uninit(&mut self.ctx_pack, n_heads * hs);
        resize_uninit(&mut self.ctx, t * h);
        resize_uninit(&mut self.scores, t * t);
        resize_uninit(&mut self.ctx_h, hs);
    }
}

fn resize_uninit(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

fn with_attn_scratch<R>(t: usize, h: usize, n_heads: usize, head_dim: usize, f: impl FnOnce(&mut AttnScratch) -> R) -> R {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Option<AttnScratch>> = const { std::cell::RefCell::new(None) };
    }
    SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(AttnScratch::with_capacity(t, h, n_heads, head_dim));
        }
        let sc = slot.as_mut().unwrap();
        sc.ensure(t, h, n_heads, head_dim);
        f(sc)
    })
}

pub struct CpuEngine {
    pub config: ModelConfig,
    conv_layers: Vec<Conv1dLayer>,
    feat_proj_ln: LayerNorm,
    feat_proj: Linear,
    /// Materialized pos conv weight [hidden, hidden/groups, k] + bias
    pos_w: Vec<f32>,
    pos_b: Vec<f32>,
    pos_k: usize,
    pos_groups: usize,
    layers: Vec<EncoderLayer>,
    encoder_ln: LayerNorm,
    lm_head: Linear,
}

impl CpuEngine {
    pub fn load(model_dir: &Path, config: ModelConfig) -> Result<Self> {
        let weights = crate::weights::load_weights(model_dir)?;
        log::info!(
            "CPU engine: loaded {} tensors from {}",
            weights.len(),
            model_dir.display()
        );
        let eps = config.layer_norm_eps as f32;

        let mut conv_layers = Vec::with_capacity(config.conv_dim.len());
        for i in 0..config.conv_dim.len() {
            let prefix = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            let w_t = weights
                .get(&format!("{prefix}.conv.weight"))
                .ok_or_else(|| anyhow!("missing {prefix}.conv.weight"))?;
            let b_t = weights
                .get(&format!("{prefix}.conv.bias"))
                .ok_or_else(|| anyhow!("missing {prefix}.conv.bias"))?;
            // weight [out, in, k]
            if w_t.shape.len() != 3 {
                bail!("conv {i} weight shape {:?}", w_t.shape);
            }
            let out_ch = w_t.shape[0];
            let in_ch = w_t.shape[1];
            let k = w_t.shape[2];
            let w = w_t.to_f32_vec()?;
            let b = b_t.to_f32_vec()?;
            let ln = if weights.contains_key(&format!("{prefix}.layer_norm.weight")) {
                Some(LayerNorm::from_map(
                    &weights,
                    &format!("{prefix}.layer_norm"),
                    eps,
                )?)
            } else {
                None
            };
            conv_layers.push(Conv1dLayer {
                w,
                b,
                out_ch,
                in_ch,
                k,
                stride: config.conv_stride[i],
                ln,
            });
        }

        let feat_proj_ln =
            LayerNorm::from_map(&weights, "wav2vec2.feature_projection.layer_norm", eps)?;
        let feat_proj = Linear::from_map(&weights, "wav2vec2.feature_projection.projection")?;

        // Positional conv with weight_norm dim=2
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
        let pos_b = weights
            .get("wav2vec2.encoder.pos_conv_embed.conv.bias")
            .ok_or_else(|| anyhow!("missing pos_conv bias"))?
            .to_f32_vec()?;
        // v: [hidden, hidden/groups, k]
        let hidden = config.hidden_size;
        let in_g = hidden / pos_groups;
        if v.len() != hidden * in_g * pos_k {
            bail!(
                "pos_conv v len {} != {}*{}*{}",
                v.len(),
                hidden,
                in_g,
                pos_k
            );
        }
        // g: [1,1,k] — weight[o,i,k] = v * g[k] / ||v[:,:,k]|| over o,i
        let mut pos_w = vec![0.0f32; v.len()];
        for kk in 0..pos_k {
            let mut norm_sq = 0.0f32;
            for oi in 0..(hidden * in_g) {
                let val = v[oi * pos_k + kk];
                norm_sq += val * val;
            }
            let inv = norm_sq.sqrt().recip();
            let gk = if g.len() == pos_k {
                g[kk]
            } else if g.len() == 1 {
                g[0]
            } else {
                // (1,1,k)
                g[kk]
            };
            for oi in 0..(hidden * in_g) {
                let idx = oi * pos_k + kk;
                pos_w[idx] = v[idx] * gk * inv;
            }
        }

        let n_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / n_heads;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let base = format!("wav2vec2.encoder.layers.{i}");
            let q = Linear::from_map(&weights, &format!("{base}.attention.q_proj"))?;
            let k = Linear::from_map(&weights, &format!("{base}.attention.k_proj"))?;
            let v = Linear::from_map(&weights, &format!("{base}.attention.v_proj"))?;
            let qkv = fuse_qkv(q, k, v)?;
            layers.push(EncoderLayer {
                ln_attn: LayerNorm::from_map(&weights, &format!("{base}.layer_norm"), eps)?,
                qkv,
                out: Linear::from_map(&weights, &format!("{base}.attention.out_proj"))?,
                ln_ff: LayerNorm::from_map(&weights, &format!("{base}.final_layer_norm"), eps)?,
                ff_inter: Linear::from_map(
                    &weights,
                    &format!("{base}.feed_forward.intermediate_dense"),
                )?,
                ff_out: Linear::from_map(&weights, &format!("{base}.feed_forward.output_dense"))?,
                n_heads,
                head_dim,
                hidden: config.hidden_size,
            });
        }

        let encoder_ln = LayerNorm::from_map(&weights, "wav2vec2.encoder.layer_norm", eps)?;
        let lm_head = Linear::from_map(&weights, "lm_head")?;

        Ok(Self {
            config,
            conv_layers,
            feat_proj_ln,
            feat_proj,
            pos_w,
            pos_b,
            pos_k,
            pos_groups,
            layers,
            encoder_ln,
            lm_head,
        })
    }

    /// Forward waveform (mono f32, 16 kHz) → CTC logits `(T, vocab)` row-major.
    pub fn forward_logits(&self, waveform: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let outs = self.forward_logits_batch(&[waveform])?;
        outs.into_iter().next().ok_or_else(|| anyhow!("empty batch"))
    }

    /// Batched forward for equal-length waveforms (windowed emissions).
    ///
    /// FFN/LN/lm_head run as one fat gemm over `B*T` rows (amortises packing).
    /// Attention stays per-sequence (no cross-window mixing), parallel over batch.
    pub fn forward_logits_batch(
        &self,
        waveforms: &[&[f32]],
    ) -> Result<Vec<(Vec<f32>, usize, usize)>> {
        if waveforms.is_empty() {
            return Ok(Vec::new());
        }
        let b = waveforms.len();
        // Frontend (conv + proj + pos) — parallel over batch.
        let frontends: Result<Vec<Vec<f32>>> = waveforms
            .par_iter()
            .map(|w| self.frontend_hidden(w))
            .collect();
        let hiddens = frontends?;
        let h = self.config.hidden_size;
        let t = hiddens[0].len() / h;
        for hid in &hiddens {
            if hid.len() != t * h {
                bail!("batch windows must share the same frame length");
            }
        }

        // Stack [B, T, H] → [B*T, H] for fat FFN gemms.
        let mut stacked = vec![0.0f32; b * t * h];
        for (bi, hid) in hiddens.iter().enumerate() {
            stacked[bi * t * h..(bi + 1) * t * h].copy_from_slice(hid);
        }

        let profile = std::env::var("CTC_PROFILE").ok().as_deref() == Some("1");
        let t_enc0 = std::time::Instant::now();
        let mut t_attn = std::time::Duration::ZERO;
        let mut t_ffn = std::time::Duration::ZERO;

        for layer in &self.layers {
            // For typical window counts (B≈7 on 3m), sequential sequences +
            // multi-thread gemm inside attention beats parallel-B + ST gemm.
            let outer_par = b >= 12;
            let ta = std::time::Instant::now();
            let after_parts: Result<Vec<Vec<f32>>> = if outer_par {
                (0..b)
                    .into_par_iter()
                    .map(|bi| {
                        let x = &stacked[bi * t * h..(bi + 1) * t * h];
                        self.attn_residual(layer, x, t, true)
                    })
                    .collect()
            } else {
                let mut parts = Vec::with_capacity(b);
                for bi in 0..b {
                    let x = &stacked[bi * t * h..(bi + 1) * t * h];
                    parts.push(self.attn_residual(layer, x, t, false)?);
                }
                Ok(parts)
            };
            let after_parts = after_parts?;
            t_attn += ta.elapsed();
            let mut after_attn = vec![0.0f32; b * t * h];
            for (bi, part) in after_parts.into_iter().enumerate() {
                after_attn[bi * t * h..(bi + 1) * t * h].copy_from_slice(&part);
            }
            let tf = std::time::Instant::now();
            // FFN residual on full B*T (one fat gemm path).
            stacked = self.ffn_residual(layer, &after_attn, b * t)?;
            t_ffn += tf.elapsed();
        }

        self.encoder_ln.forward_inplace(&mut stacked, b * t);
        let logits_all = self.lm_head.forward(&stacked, b * t);
        let vocab = self.lm_head.out;
        if profile {
            eprintln!(
                "[CTC_PROFILE] B={b} T={t} attn={:.3}s ffn={:.3}s enc_total={:.3}s",
                t_attn.as_secs_f64(),
                t_ffn.as_secs_f64(),
                t_enc0.elapsed().as_secs_f64()
            );
        }

        let mut outs = Vec::with_capacity(b);
        for bi in 0..b {
            let logits = logits_all[bi * t * vocab..(bi + 1) * t * vocab].to_vec();
            outs.push((logits, t, vocab));
        }
        Ok(outs)
    }

    /// Conv feature extractor + projection + positional conv → [T, H].
    fn frontend_hidden(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let mut x = waveform.to_vec();
        let mut len = x.len();
        let mut channels = 1usize;
        for (li, conv) in self.conv_layers.iter().enumerate() {
            if channels != conv.in_ch {
                bail!(
                    "conv layer {li}: in_ch mismatch got {channels} expect {}",
                    conv.in_ch
                );
            }
            let (y, out_len) = conv.forward(&x, len)?;
            x = y;
            len = out_len;
            channels = conv.out_ch;
        }
        let t_feat = len;
        let c_feat = channels;
        let mut hidden = vec![0.0f32; t_feat * c_feat];
        for ti in 0..t_feat {
            for c in 0..c_feat {
                hidden[ti * c_feat + c] = x[c * t_feat + ti];
            }
        }
        self.feat_proj_ln.forward_inplace(&mut hidden, t_feat);
        hidden = self.feat_proj.forward(&hidden, t_feat);
        let h = self.config.hidden_size;
        let pos = self.pos_conv_embed(&hidden, t_feat)?;
        hidden.par_iter_mut().zip(pos.par_iter()).for_each(|(a, &b)| {
            *a += b;
        });
        debug_assert_eq!(hidden.len(), t_feat * h);
        Ok(hidden)
    }

    /// y = x + MHA(LN(x))
    ///
    /// Parallelism policy (single layer only — never nest B × heads × gemm):
    /// - `outer_parallel=false` (default for B≲12): multi-thread gemm, sequential heads
    /// - `outer_parallel=true`: parallel heads, single-thread gemm
    fn attn_residual(
        &self,
        layer: &EncoderLayer,
        x: &[f32],
        t: usize,
        outer_parallel: bool,
    ) -> Result<Vec<f32>> {
        let h_norm = layer.ln_attn.forward(x, t);
        let mut y = multihead_attention_fused(
            &h_norm,
            t,
            layer,
            outer_parallel,
        )?;
        // residual in-place
        y.par_iter_mut().zip(x.par_iter()).for_each(|(yo, &xi)| {
            *yo += xi;
        });
        Ok(y)
    }

    /// y = x + FFN(LN(x))  — x is [rows, H]
    fn ffn_residual(&self, layer: &EncoderLayer, x: &[f32], rows: usize) -> Result<Vec<f32>> {
        let h_norm = layer.ln_ff.forward(x, rows);
        let mut inter = layer.ff_inter.forward(&h_norm, rows);
        inter.par_iter_mut().for_each(|v| *v = gelu(*v));
        let mut y = layer.ff_out.forward(&inter, rows);
        y.par_iter_mut().zip(x.par_iter()).for_each(|(yo, &xi)| {
            *yo += xi;
        });
        Ok(y)
    }

    fn pos_conv_embed(&self, hidden: &[f32], t: usize) -> Result<Vec<f32>> {
        // hidden [T, H] → conv over time with groups, same padding then remove 1 if even k
        let h = self.config.hidden_size;
        let k = self.pos_k;
        let groups = self.pos_groups;
        let in_g = h / groups; // channels per group
        // pad both sides by k//2
        let pad = k / 2;
        let t_pad = t + 2 * pad;
        // x_ch: [H, T_pad]
        let mut x = vec![0.0f32; h * t_pad];
        for ti in 0..t {
            for c in 0..h {
                x[c * t_pad + (ti + pad)] = hidden[ti * h + c];
            }
        }
        let out_len_full = t_pad - k + 1; // stride 1
        // after SamePad with even k, remove last frame → should equal t
        let remove = if k % 2 == 0 { 1 } else { 0 };
        let out_len = out_len_full - remove;
        if out_len != t {
            // HF SamePad: if even kernel, remove 1 from the end
            log::debug!("pos_conv out_len={out_len} t={t} full={out_len_full}");
        }
        let use_len = out_len.min(t);
        let mut y = vec![0.0f32; h * use_len];

        // Grouped conv1d: groups independent, each group has in_g in and out_g=in_g out channels
        // weight shape [H, in_g, k] where H = groups * out_per_group, out_per_group = in_g
        y.par_chunks_mut(use_len)
            .enumerate()
            .for_each(|(oc, y_ch)| {
                let g = oc / in_g;
                let local_oc = oc % in_g;
                let _ = local_oc;
                // For grouped conv, output channel oc only sees input channels [g*in_g, (g+1)*in_g)
                // weight row oc: [in_g, k]
                let w_base = oc * in_g * k;
                for ti in 0..use_len {
                    let mut acc = self.pos_b[oc];
                    for ic_local in 0..in_g {
                        let ic = g * in_g + ic_local;
                        let x_base = ic * t_pad + ti;
                        let w_row = w_base + ic_local * k;
                        for kk in 0..k {
                            acc += x[x_base + kk] * self.pos_w[w_row + kk];
                        }
                    }
                    y_ch[ti] = gelu(acc);
                }
            });

        // transpose [H, T] → [T, H], pad/truncate to t
        let mut out = vec![0.0f32; t * h];
        let copy_t = use_len.min(t);
        for ti in 0..copy_t {
            for c in 0..h {
                out[ti * h + c] = y[c * use_len + ti];
            }
        }
        Ok(out)
    }

}

/// Stack Q/K/V into one Linear `[3H, H]` (row-major blocks).
fn fuse_qkv(q: Linear, k: Linear, v: Linear) -> Result<Linear> {
    if q.inn != k.inn || q.inn != v.inn || q.out != k.out || q.out != v.out {
        bail!("Q/K/V shape mismatch for fuse");
    }
    let h = q.out;
    let inn = q.inn;
    let mut w = vec![0.0f32; 3 * h * inn];
    w[0..h * inn].copy_from_slice(&q.w);
    w[h * inn..2 * h * inn].copy_from_slice(&k.w);
    w[2 * h * inn..].copy_from_slice(&v.w);
    let mut b = vec![0.0f32; 3 * h];
    b[0..h].copy_from_slice(&q.b);
    b[h..2 * h].copy_from_slice(&k.b);
    b[2 * h..].copy_from_slice(&v.b);
    Ok(Linear {
        w,
        b,
        out: 3 * h,
        inn,
    })
}

/// Fused-QKV multi-head attention with thread-local scratch (no per-layer malloc storm).
fn multihead_attention_fused(
    x: &[f32],
    t: usize,
    layer: &EncoderLayer,
    outer_parallel: bool,
) -> Result<Vec<f32>> {
    let h = layer.hidden;
    let n_heads = layer.n_heads;
    let head_dim = layer.head_dim;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let gemm_par = if outer_parallel {
        Parallelism::None
    } else {
        Parallelism::Rayon(0)
    };
    let par_heads = outer_parallel || t < 512;
    let head_stride = t * head_dim;

    with_attn_scratch(t, h, n_heads, head_dim, |sc| {
        // 1) Fused QKV: [T,H] → [T,3H]
        layer.qkv.forward_into(x, t, &mut sc.qkv[..t * 3 * h], gemm_par);

        // 2) Pack heads from Q/K/V slabs of qkv
        // qkv layout per row: [Q(0..H) | K(H..2H) | V(2H..3H)]
        pack_heads_from_qkv(&sc.qkv[..t * 3 * h], &mut sc.q_pack, &mut sc.k_pack, &mut sc.v_pack, t, h, n_heads, head_dim);

        // 3) Per-head attention into ctx_pack
        // Parallel heads need independent scores buffers — copy pack views via raw
        // pointers after packing completes (packs are read-only during this phase).
        if par_heads {
            // Encode pointers as usize so the closure is Sync (disjoint head writes).
            let q_addr = sc.q_pack.as_ptr() as usize;
            let k_addr = sc.k_pack.as_ptr() as usize;
            let v_addr = sc.v_pack.as_ptr() as usize;
            let ctx_addr = sc.ctx_pack.as_mut_ptr() as usize;
            (0..n_heads).into_par_iter().for_each(|head| {
                let base = head * head_stride;
                // SAFETY: packs read-only; each head writes a unique ctx range.
                let (qh, kh, vh) = unsafe {
                    (
                        std::slice::from_raw_parts((q_addr as *const f32).add(base), head_stride),
                        std::slice::from_raw_parts((k_addr as *const f32).add(base), head_stride),
                        std::slice::from_raw_parts((v_addr as *const f32).add(base), head_stride),
                    )
                };
                let out = attention_materialised(
                    qh,
                    kh,
                    vh,
                    t,
                    head_dim,
                    scale,
                    Parallelism::None,
                );
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        out.as_ptr(),
                        (ctx_addr as *mut f32).add(base),
                        head_stride,
                    );
                }
            });
        } else {
            for head in 0..n_heads {
                let base = head * head_stride;
                attention_materialised_into(
                    &sc.q_pack[base..base + head_stride],
                    &sc.k_pack[base..base + head_stride],
                    &sc.v_pack[base..base + head_stride],
                    t,
                    head_dim,
                    scale,
                    gemm_par,
                    &mut sc.scores[..t * t],
                    &mut sc.ctx_h[..head_stride],
                );
                sc.ctx_pack[base..base + head_stride]
                    .copy_from_slice(&sc.ctx_h[..head_stride]);
            }
        }

        // 4) Unpack [heads,T,hd] → [T,H]
        for head in 0..n_heads {
            let base = head * head_stride;
            for ti in 0..t {
                let src = base + ti * head_dim;
                let dst = ti * h + head * head_dim;
                sc.ctx[dst..dst + head_dim]
                    .copy_from_slice(&sc.ctx_pack[src..src + head_dim]);
            }
        }

        // 5) Output projection
        Ok(layer.out.forward_par(&sc.ctx[..t * h], t, gemm_par))
    })
}

/// Pack Q/K/V from fused [T, 3H] into three [n_heads, T, hd] buffers.
fn pack_heads_from_qkv(
    qkv: &[f32],
    q_pack: &mut [f32],
    k_pack: &mut [f32],
    v_pack: &mut [f32],
    t: usize,
    h: usize,
    _n_heads: usize,
    head_dim: usize,
) {
    let row = 3 * h;
    q_pack
        .par_chunks_mut(t * head_dim)
        .zip(k_pack.par_chunks_mut(t * head_dim))
        .zip(v_pack.par_chunks_mut(t * head_dim))
        .enumerate()
        .for_each(|(head, ((q_out, k_out), v_out))| {
            for ti in 0..t {
                let base = ti * row;
                let q_s = base + head * head_dim;
                let k_s = base + h + head * head_dim;
                let v_s = base + 2 * h + head * head_dim;
                let d = ti * head_dim;
                q_out[d..d + head_dim].copy_from_slice(&qkv[q_s..q_s + head_dim]);
                k_out[d..d + head_dim].copy_from_slice(&qkv[k_s..k_s + head_dim]);
                v_out[d..d + head_dim].copy_from_slice(&qkv[v_s..v_s + head_dim]);
            }
        });
}

fn attention_materialised(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    t: usize,
    head_dim: usize,
    scale: f32,
    par: Parallelism,
) -> Vec<f32> {
    let mut scores = vec![0.0f32; t * t];
    let mut ctx_h = vec![0.0f32; t * head_dim];
    attention_materialised_into(qh, kh, vh, t, head_dim, scale, par, &mut scores, &mut ctx_h);
    ctx_h
}

fn attention_materialised_into(
    qh: &[f32],
    kh: &[f32],
    vh: &[f32],
    t: usize,
    head_dim: usize,
    scale: f32,
    par: Parallelism,
    scores: &mut [f32],
    ctx_h: &mut [f32],
) {
    debug_assert!(scores.len() >= t * t);
    debug_assert!(ctx_h.len() >= t * head_dim);
    unsafe {
        gemm(
            t,
            t,
            head_dim,
            scores.as_mut_ptr(),
            1,
            t as isize,
            false,
            qh.as_ptr(),
            1,
            head_dim as isize,
            kh.as_ptr(),
            head_dim as isize,
            1,
            0.0,
            scale,
            false,
            false,
            false,
            par,
        );
    }
    scores[..t * t].par_chunks_mut(t).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = sum.recip();
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
    unsafe {
        gemm(
            t,
            head_dim,
            t,
            ctx_h.as_mut_ptr(),
            1,
            head_dim as isize,
            false,
            scores.as_ptr(),
            1,
            t as isize,
            vh.as_ptr(),
            1,
            head_dim as isize,
            0.0,
            1.0,
            false,
            false,
            false,
            par,
        );
    }
}

/// Exact GELU (erf), matching `torch.nn.functional.gelu` / HF `gelu`.
#[inline(always)]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}
