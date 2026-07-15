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
        debug_assert_eq!(x.len(), rows * self.inn);
        let mut out = vec![0.0f32; rows * self.out];
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
                Parallelism::Rayon(0),
            );
        }
        for r in 0..rows {
            let base = r * self.out;
            for j in 0..self.out {
                out[base + j] += self.b[j];
            }
        }
        out
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
        for r in 0..rows {
            let row = &mut x[r * dim..(r + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let mut var = 0.0f32;
            for v in row.iter() {
                let d = *v - mean;
                var += d * d;
            }
            var /= dim as f32;
            let inv = (var + self.eps).sqrt().recip();
            for j in 0..dim {
                row[j] = (row[j] - mean) * inv * self.w[j] + self.b[j];
            }
        }
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

        // GELU
        for v in y.iter_mut() {
            *v = gelu(*v);
        }
        Ok((y, out_len))
    }
}

struct EncoderLayer {
    ln_attn: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ln_ff: LayerNorm,
    ff_inter: Linear,
    ff_out: Linear,
    n_heads: usize,
    head_dim: usize,
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
            layers.push(EncoderLayer {
                ln_attn: LayerNorm::from_map(&weights, &format!("{base}.layer_norm"), eps)?,
                q: Linear::from_map(&weights, &format!("{base}.attention.q_proj"))?,
                k: Linear::from_map(&weights, &format!("{base}.attention.k_proj"))?,
                v: Linear::from_map(&weights, &format!("{base}.attention.v_proj"))?,
                out: Linear::from_map(&weights, &format!("{base}.attention.out_proj"))?,
                ln_ff: LayerNorm::from_map(&weights, &format!("{base}.final_layer_norm"), eps)?,
                ff_inter: Linear::from_map(
                    &weights,
                    &format!("{base}.feed_forward.intermediate_dense"),
                )?,
                ff_out: Linear::from_map(&weights, &format!("{base}.feed_forward.output_dense"))?,
                n_heads,
                head_dim,
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
        // Feature extractor expects [B=1, C=1, T]
        let mut x = waveform.to_vec();
        let mut len = x.len();
        let mut channels = 1usize;
        // first layer in_ch=1, store as [C, L]
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
        // x: [512, T] → transpose to [T, 512]
        let t_feat = len;
        let c_feat = channels;
        let mut hidden = vec![0.0f32; t_feat * c_feat];
        for t in 0..t_feat {
            for c in 0..c_feat {
                hidden[t * c_feat + c] = x[c * t_feat + t];
            }
        }
        // feature projection
        self.feat_proj_ln.forward_inplace(&mut hidden, t_feat);
        hidden = self.feat_proj.forward(&hidden, t_feat);
        let h = self.config.hidden_size;
        debug_assert_eq!(hidden.len(), t_feat * h);

        // pos conv embed + residual
        let pos = self.pos_conv_embed(&hidden, t_feat)?;
        for i in 0..hidden.len() {
            hidden[i] += pos[i];
        }

        // encoder layers (stable LN)
        for layer in &self.layers {
            hidden = self.encoder_layer_forward(layer, &hidden, t_feat)?;
        }
        self.encoder_ln.forward_inplace(&mut hidden, t_feat);

        // lm head
        let logits = self.lm_head.forward(&hidden, t_feat);
        let vocab = self.lm_head.out;
        Ok((logits, t_feat, vocab))
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

    fn encoder_layer_forward(
        &self,
        layer: &EncoderLayer,
        x: &[f32],
        t: usize,
    ) -> Result<Vec<f32>> {
        let h = self.config.hidden_size;
        // Attn residual
        let mut h_norm = layer.ln_attn.forward(x, t);
        let attn = multihead_attention(
            &h_norm,
            t,
            h,
            layer.n_heads,
            layer.head_dim,
            &layer.q,
            &layer.k,
            &layer.v,
            &layer.out,
        )?;
        let mut y = vec![0.0f32; t * h];
        for i in 0..y.len() {
            y[i] = x[i] + attn[i];
        }
        // FFN residual
        h_norm = layer.ln_ff.forward(&y, t);
        let mut inter = layer.ff_inter.forward(&h_norm, t);
        for v in inter.iter_mut() {
            *v = gelu(*v);
        }
        let ff = layer.ff_out.forward(&inter, t);
        for i in 0..y.len() {
            y[i] += ff[i];
        }
        Ok(y)
    }
}

fn multihead_attention(
    x: &[f32],
    t: usize,
    h: usize,
    n_heads: usize,
    head_dim: usize,
    q_lin: &Linear,
    k_lin: &Linear,
    v_lin: &Linear,
    o_lin: &Linear,
) -> Result<Vec<f32>> {
    let q = q_lin.forward(x, t);
    let k = k_lin.forward(x, t);
    let v = v_lin.forward(x, t);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // out_heads: [T, H] accumulated before out_proj — actually we compute context [T, H] then out_proj
    let mut ctx = vec![0.0f32; t * h];

    // Parallel over heads
    let head_outs: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .map(|head| {
            let mut qh = vec![0.0f32; t * head_dim];
            let mut kh = vec![0.0f32; t * head_dim];
            let mut vh = vec![0.0f32; t * head_dim];
            for ti in 0..t {
                let src = ti * h + head * head_dim;
                let dst = ti * head_dim;
                qh[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
                kh[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
                vh[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
            }
            // scores [T, T] = q @ k^T * scale
            let mut scores = vec![0.0f32; t * t];
            for i in 0..t {
                for j in 0..t {
                    let mut dot = 0.0f32;
                    let qi = i * head_dim;
                    let kj = j * head_dim;
                    for d in 0..head_dim {
                        dot += qh[qi + d] * kh[kj + d];
                    }
                    scores[i * t + j] = dot * scale;
                }
            }
            // softmax rows
            for i in 0..t {
                let row = &mut scores[i * t..(i + 1) * t];
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
            }
            // ctx_h [T, hd] = scores @ v
            let mut ctx_h = vec![0.0f32; t * head_dim];
            for i in 0..t {
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..t {
                        acc += scores[i * t + j] * vh[j * head_dim + d];
                    }
                    ctx_h[i * head_dim + d] = acc;
                }
            }
            ctx_h
        })
        .collect();

    for head in 0..n_heads {
        let ctx_h = &head_outs[head];
        for ti in 0..t {
            let src = ti * head_dim;
            let dst = ti * h + head * head_dim;
            ctx[dst..dst + head_dim].copy_from_slice(&ctx_h[src..src + head_dim]);
        }
    }

    Ok(o_lin.forward(&ctx, t))
}

/// Exact GELU (erf), matching `torch.nn.functional.gelu` / HF `gelu`.
fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}
