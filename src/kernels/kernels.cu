// Hand-written CUDA kernels for CTC Wav2Vec2ForCTC forced aligner.
//
// Conventions (match cohere-transcribe-native / qwen-aligner-rs):
//   - `extern "C"` entry points
//   - storage = __half, accumulation = float (sm_61+; no TC requirement)
//   - output pointer first where there is a distinct `out`
//
// Used by: layer_norm, gelu, residual, bias, non-causal softmax, head pack/unpack.

#include <cuda_fp16.h>

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ============================================================================
// LayerNorm: y[row, :] = LN(x[row, :]) * w + b
// One block per row; shared mem = 2 * blockDim.x * sizeof(float)
// ============================================================================
extern "C" __global__ void layer_norm_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ b,
    int rows,
    int dim,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    const __half* xr = x + (size_t)row * dim;
    __half* yr = y + (size_t)row * dim;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < dim; i += bs) {
        float v = __half2float(xr[i]);
        local_sum += v;
        local_sq += v * v;
    }
    smem[tid] = local_sum;
    smem[tid + bs] = local_sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[tid + bs] += smem[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = smem[0] / (float)dim;
    float var = smem[bs] / (float)dim - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < dim; i += bs) {
        float v = (__half2float(xr[i]) - mean) * inv_std;
        yr[i] = __float2half(v * __half2float(w[i]) + __half2float(b[i]));
    }
}

// ============================================================================
// GELU (erf form, matches torch.nn.functional.gelu / HF "gelu")
// ============================================================================
__device__ __forceinline__ float gelu_erf(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

extern "C" __global__ void gelu_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    y[i] = __float2half(gelu_erf(__half2float(x[i])));
}

extern "C" __global__ void gelu_inplace_f16(__half* __restrict__ x, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    x[i] = __float2half(gelu_erf(__half2float(x[i])));
}

// ============================================================================
// Elementwise add / residual
// ============================================================================
extern "C" __global__ void add_f16(
    __half* __restrict__ y,
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    y[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

extern "C" __global__ void add_inplace_f16(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

extern "C" __global__ void add_bias_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    int c = i % cols;
    x[i] = __float2half(__half2float(x[i]) + __half2float(bias[c]));
}

// y = x + residual  (after linear); optional bias on cols
extern "C" __global__ void bias_residual_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    int c = i % cols;
    y[i] = __float2half(
        __half2float(x[i]) + __half2float(bias[c]) + __half2float(residual[i])
    );
}

// ============================================================================
// Softmax over last dim (non-causal) — encoder attention
// One block per row; shared mem = blockDim.x * sizeof(float)
// ============================================================================
extern "C" __global__ void softmax_last_dim_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    int rows,
    int dim
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    const __half* xr = x + (size_t)row * dim;
    __half* yr = y + (size_t)row * dim;

    float local_max = -__int_as_float(0x7f800000); // -inf
    for (int i = tid; i < dim; i += bs) {
        local_max = fmaxf(local_max, __half2float(xr[i]));
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += bs) {
        local_sum += expf(__half2float(xr[i]) - mx);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / smem[0];

    for (int i = tid; i < dim; i += bs) {
        float e = expf(__half2float(xr[i]) - mx);
        yr[i] = __float2half(e * inv);
    }
}

// scale scores in-place: x *= scale
extern "C" __global__ void scale_inplace_f16(__half* __restrict__ x, int numel, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    x[i] = __float2half(__half2float(x[i]) * scale);
}

// ============================================================================
// Head pack/unpack for multi-head attention
// qkv: [T, 3H] → q/k/v packs [n_heads, T, hd]  OR split from [T, H]
// ============================================================================

// Split interleaved [T, H] into [n_heads, T, hd]
extern "C" __global__ void split_to_heads_f16(
    __half* __restrict__ out,       // [n_heads, T, hd]
    const __half* __restrict__ x,   // [T, H]
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = t * n_heads * hd;
    if (idx >= numel) return;
    int h = n_heads * hd;
    int head = idx / (t * hd);
    int rem = idx % (t * hd);
    int ti = rem / hd;
    int d = rem % hd;
    out[idx] = x[ti * h + head * hd + d];
}

// Merge [n_heads, T, hd] → [T, H]
extern "C" __global__ void merge_heads_f16(
    __half* __restrict__ out,       // [T, H]
    const __half* __restrict__ x,   // [n_heads, T, hd]
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = n_heads * hd;
    int numel = t * h;
    if (idx >= numel) return;
    int ti = idx / h;
    int c = idx % h;
    int head = c / hd;
    int d = c % hd;
    out[idx] = x[head * t * hd + ti * hd + d];
}

// Split fused QKV [T, 3H] into three [n_heads, T, hd]
extern "C" __global__ void split_qkv_to_heads_f16(
    __half* __restrict__ q,         // [n_heads, T, hd]
    __half* __restrict__ k,
    __half* __restrict__ v,
    const __half* __restrict__ qkv, // [T, 3H]
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = t * n_heads * hd;
    if (idx >= numel) return;
    int h = n_heads * hd;
    int head = idx / (t * hd);
    int rem = idx % (t * hd);
    int ti = rem / hd;
    int d = rem % hd;
    int base = ti * 3 * h + head * hd + d;
    q[idx] = qkv[base];
    k[idx] = qkv[base + h];
    v[idx] = qkv[base + 2 * h];
}

// f32 → f16 cast (upload helper / waveform)
extern "C" __global__ void cast_f32_to_f16(
    __half* __restrict__ y,
    const float* __restrict__ x,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    y[i] = __float2half(x[i]);
}

extern "C" __global__ void cast_f16_to_f32(
    float* __restrict__ y,
    const __half* __restrict__ x,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    y[i] = __half2float(x[i]);
}

// ============================================================================
// 1D im2col for FE conv: x [T_in, C_in] -> col [T_out, C_in*K]
// ============================================================================
extern "C" __global__ void im2col_1d_f16(
    __half* __restrict__ col,
    const __half* __restrict__ x,
    int t_in,
    int c_in,
    int k,
    int stride,
    int t_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = c_in * k;
    int numel = t_out * kk;
    if (idx >= numel) return;
    int t = idx / kk;
    int j = idx % kk;
    int ic = j / k;
    int ki = j % k;
    int t0 = t * stride + ki;
    // t_in only for bounds documentation; callers guarantee t0 < t_in
    (void)t_in;
    col[idx] = x[(size_t)t0 * c_in + ic];
}

// im2col with non-contiguous channel plane: x [T_in, c_full], take
// channels [c_start, c_start+c_in). Used for grouped pos_conv.
extern "C" __global__ void im2col_1d_ch_f16(
    __half* __restrict__ col,
    const __half* __restrict__ x,
    int t_in,
    int c_full,
    int c_start,
    int c_in,
    int k,
    int stride,
    int t_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = c_in * k;
    int numel = t_out * kk;
    if (idx >= numel) return;
    int t = idx / kk;
    int j = idx % kk;
    int ic = j / k;
    int ki = j % k;
    int t0 = t * stride + ki;
    (void)t_in;
    col[idx] = x[(size_t)t0 * c_full + c_start + ic];
}

// Pad sequence along time for pos conv: out [T+2*pad, H], zero borders
extern "C" __global__ void pad_time_f16(
    __half* __restrict__ out,
    const __half* __restrict__ x,
    int t,
    int h,
    int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_pad = t + 2 * pad;
    int numel = t_pad * h;
    if (idx >= numel) return;
    int ti = idx / h;
    int c = idx % h;
    if (ti < pad || ti >= pad + t) {
        out[idx] = __float2half(0.0f);
    } else {
        out[idx] = x[(size_t)(ti - pad) * h + c];
    }
}

// y[i] += bias[i % cols]; then GELU(y)  — FE path without LN
extern "C" __global__ void add_bias_gelu_inplace_f16(
    __half* __restrict__ x,
    const __half* __restrict__ bias,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    float v = __half2float(x[i]) + __half2float(bias[i % cols]);
    x[i] = __float2half(gelu_erf(v));
}

// Scatter gemm group output into [T, H] at channel offset, with bias+GELU.
// y_g: [t_out, c_g], out: [t, h]  (t_out <= t; remaining rows left unchanged/zero)
extern "C" __global__ void scatter_group_bias_gelu_f16(
    __half* __restrict__ out,
    const __half* __restrict__ y_g,
    const __half* __restrict__ bias,
    int t_out,
    int c_g,
    int h,
    int c_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = t_out * c_g;
    if (idx >= numel) return;
    int ti = idx / c_g;
    int j = idx % c_g;
    float v = __half2float(y_g[idx]) + __half2float(bias[c_start + j]);
    out[(size_t)ti * h + c_start + j] = __float2half(gelu_erf(v));
}

// ============================================================================
// f32 kernels — match Python CPU golden (float32) and favor Pascal (sm_61)
// where SGEMM is the native high-throughput path (no Tensor Core f16).
// ============================================================================

extern "C" __global__ void layer_norm_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int rows,
    int dim,
    float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    float local_sum = 0.0f, local_sq = 0.0f;
    for (int i = tid; i < dim; i += bs) {
        float v = xr[i];
        local_sum += v;
        local_sq += v * v;
    }
    smem[tid] = local_sum;
    smem[tid + bs] = local_sq;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
            smem[tid + bs] += smem[tid + bs + s];
        }
        __syncthreads();
    }
    float mean = smem[0] / (float)dim;
    float var = smem[bs] / (float)dim - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < dim; i += bs) {
        float v = (xr[i] - mean) * inv_std;
        yr[i] = v * w[i] + b[i];
    }
}

extern "C" __global__ void gelu_inplace_f32(float* __restrict__ x, int numel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    x[i] = gelu_erf(x[i]);
}

extern "C" __global__ void add_inplace_f32(
    float* __restrict__ a,
    const float* __restrict__ b,
    int numel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    a[i] += b[i];
}

extern "C" __global__ void add_bias_inplace_f32(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    x[i] += bias[i % cols];
}

extern "C" __global__ void bias_residual_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    int c = i % cols;
    y[i] = x[i] + bias[c] + residual[i];
}

extern "C" __global__ void softmax_last_dim_f32(
    float* __restrict__ y,
    const float* __restrict__ x,
    int rows,
    int dim
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    float local_max = -__int_as_float(0x7f800000); // -inf
    for (int i = tid; i < dim; i += bs) {
        local_max = fmaxf(local_max, xr[i]);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += bs) {
        local_sum += expf(xr[i] - mx);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / smem[0];

    for (int i = tid; i < dim; i += bs) {
        yr[i] = expf(xr[i] - mx) * inv;
    }
}

// In-place softmax over last dim. Passes: max → exp+sum (stash exp) → normalize.
extern "C" __global__ void softmax_inplace_last_dim_f32(
    float* __restrict__ x,
    int rows,
    int dim
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int bs = blockDim.x;
    float* xr = x + (size_t)row * dim;

    float local_max = -__int_as_float(0x7f800000);
    for (int i = tid; i < dim; i += bs) {
        local_max = fmaxf(local_max, xr[i]);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float mx = smem[0];

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += bs) {
        float e = expf(xr[i] - mx);
        xr[i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv = 1.0f / smem[0];
    for (int i = tid; i < dim; i += bs) {
        xr[i] *= inv;
    }
}

extern "C" __global__ void split_qkv_to_heads_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ qkv,
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = t * n_heads * hd;
    if (idx >= numel) return;
    int h = n_heads * hd;
    int head = idx / (t * hd);
    int rem = idx % (t * hd);
    int ti = rem / hd;
    int d = rem % hd;
    int base = ti * 3 * h + head * hd + d;
    q[idx] = qkv[base];
    k[idx] = qkv[base + h];
    v[idx] = qkv[base + 2 * h];
}

extern "C" __global__ void merge_heads_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = n_heads * hd;
    int numel = t * h;
    if (idx >= numel) return;
    int ti = idx / h;
    int c = idx % h;
    int head = c / hd;
    int d = c % hd;
    out[idx] = x[head * t * hd + ti * hd + d];
}

extern "C" __global__ void im2col_1d_f32(
    float* __restrict__ col,
    const float* __restrict__ x,
    int t_in,
    int c_in,
    int k,
    int stride,
    int t_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = c_in * k;
    int numel = t_out * kk;
    if (idx >= numel) return;
    int t = idx / kk;
    int j = idx % kk;
    int ic = j / k;
    int ki = j % k;
    int t0 = t * stride + ki;
    (void)t_in;
    col[idx] = x[(size_t)t0 * c_in + ic];
}

extern "C" __global__ void im2col_1d_ch_f32(
    float* __restrict__ col,
    const float* __restrict__ x,
    int t_in,
    int c_full,
    int c_start,
    int c_in,
    int k,
    int stride,
    int t_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kk = c_in * k;
    int numel = t_out * kk;
    if (idx >= numel) return;
    int t = idx / kk;
    int j = idx % kk;
    int ic = j / k;
    int ki = j % k;
    int t0 = t * stride + ki;
    (void)t_in;
    col[idx] = x[(size_t)t0 * c_full + c_start + ic];
}

extern "C" __global__ void pad_time_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int t,
    int h,
    int pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int t_pad = t + 2 * pad;
    int numel = t_pad * h;
    if (idx >= numel) return;
    int ti = idx / h;
    int c = idx % h;
    if (ti < pad || ti >= pad + t) {
        out[idx] = 0.0f;
    } else {
        out[idx] = x[(size_t)(ti - pad) * h + c];
    }
}

extern "C" __global__ void add_bias_gelu_inplace_f32(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int numel,
    int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) return;
    x[i] = gelu_erf(x[i] + bias[i % cols]);
}

extern "C" __global__ void scatter_group_bias_gelu_f32(
    float* __restrict__ out,
    const float* __restrict__ y_g,
    const float* __restrict__ bias,
    int t_out,
    int c_g,
    int h,
    int c_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numel = t_out * c_g;
    if (idx >= numel) return;
    int ti = idx / c_g;
    int j = idx % c_g;
    out[(size_t)ti * h + c_start + j] = gelu_erf(y_g[idx] + bias[c_start + j]);
}

// Batched QKV split: qkv [B*T, 3H] → q/k/v [B*n_heads, T, hd]
extern "C" __global__ void split_qkv_to_heads_batched_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ qkv,
    int b,
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = n_heads * hd;
    int numel = b * t * n_heads * hd;
    if (idx >= numel) return;
    int per_batch = t * n_heads * hd;
    int bi = idx / per_batch;
    int rem = idx % per_batch;
    int head = rem / (t * hd);
    int rem2 = rem % (t * hd);
    int ti = rem2 / hd;
    int d = rem2 % hd;
    int base = ((bi * t + ti) * 3 * h) + head * hd + d;
    int out_i = bi * per_batch + head * t * hd + ti * hd + d;
    q[out_i] = qkv[base];
    k[out_i] = qkv[base + h];
    v[out_i] = qkv[base + 2 * h];
}

// Batched merge: x [B*n_heads, T, hd] → out [B*T, H]
extern "C" __global__ void merge_heads_batched_f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    int b,
    int t,
    int n_heads,
    int hd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = n_heads * hd;
    int numel = b * t * h;
    if (idx >= numel) return;
    int bi = idx / (t * h);
    int rem = idx % (t * h);
    int ti = rem / h;
    int c = rem % h;
    int head = c / hd;
    int d = c % hd;
    out[idx] = x[bi * n_heads * t * hd + head * t * hd + ti * hd + d];
}
