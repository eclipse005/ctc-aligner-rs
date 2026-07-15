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
