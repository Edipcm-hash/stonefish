// kernels/stonefish.cu
// StonefishAI - CUDA kernels for GPT training.
//
// Compiled AT RUNTIME via NVRTC (this machine has no host C++ compiler;
// nvcc requires cl.exe, NVRTC does not). PTX is produced once and loaded
// as a module (src/gpu_kernels.rs).
//
// Layout: all kernels are extern "C" with plain C signatures; the Rust
// side launches them through `Kargs` (see src/sys/dev.rs).
// All matrices are row-major. "row" = token (B*S rows), "col" = feature.

// NVRTC does not include math headers by default; there is no INFINITY macro.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

// ------------------------------------------------------------
// Elementwise
// ------------------------------------------------------------

extern "C" __global__ void add_scalar_broadcast(float* out, const float* a,
                                               const float* bias, int rows,
                                               int cols) {
    // out[r,c] = a[r,c] + bias[c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        out[idx] = a[idx] + bias[idx % cols];
    }
}

// out = x + bias (bias (1,cols)), yerinde: x += bias
extern "C" __global__ void add_bias_inplace(float* x, const float* bias, int rows,
                                            int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) x[idx] += bias[idx % cols];
}

extern "C" __global__ void add_inplace(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

extern "C" __global__ void zero_fill(float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = 0.0f;
}

// GELU forward (tanh approximation): y = 0.5x(1+tanh(c(x+0.044715x^3)))
extern "C" __global__ void gelu_forward(float* out, const float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float k = 0.7978845608028654f; // sqrt(2/pi)
        float v = x[idx];
        float inner = k * (v + 0.044715f * v * v * v);
        out[idx] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// GELU geri: dx = dy * d/dx[0.5x(1+tanh(inner))]
extern "C" __global__ void gelu_backward(float* dx, const float* dy,
                                         const float* x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float k = 0.7978845608028654f;
        float v = x[idx];
        float inner = k * (v + 0.044715f * v * v * v);
        float t = tanhf(inner);
        float dinner = k * (1.0f + 3.0f * 0.044715f * v * v);
        float deriv = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * dinner;
        dx[idx] = dy[idx] * deriv;
    }
}

// ------------------------------------------------------------
// LayerNorm
// ------------------------------------------------------------
// One block per row. Assumes cols <= 1024 (d_model=128).
// cache: xhat (normalized input), inv_std (per row).

extern "C" __global__ void layernorm_forward(float* out, float* xhat_out,
                                             float* inv_out, const float* x,
                                             const float* gamma,
                                             const float* beta, int rows,
                                             int cols, float eps) {
    extern __shared__ float sh[]; // 2 * cols
    float* s_mean = sh;
    float* s_var = sh + blockDim.x;

    int r = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) sum += x[r * cols + j];
    // blok indirgemesi
    s_mean[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_mean[tid] += s_mean[tid + s];
        __syncthreads();
    }
    float mean = s_mean[0] / (float)cols;
    __syncthreads();

    float vsum = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float d = x[r * cols + j] - mean;
        vsum += d * d;
    }
    s_var[tid] = vsum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_var[tid] += s_var[tid + s];
        __syncthreads();
    }
    float var = s_var[0] / (float)cols;
    float inv = rsqrtf(var + eps);
    if (tid == 0) inv_out[r] = inv;
    __syncthreads();

    for (int j = tid; j < cols; j += blockDim.x) {
        float xh = (x[r * cols + j] - mean) * inv;
        xhat_out[r * cols + j] = xh;
        out[r * cols + j] = xh * gamma[j] + beta[j];
    }
}

// LayerNorm backward: produces dx; dgamma/dbeta are accumulated without
// atomics by a separate column-reduction kernel (layernorm_param_grad).
extern "C" __global__ void layernorm_backward(float* dx, const float* dy,
                                              const float* xhat,
                                              const float* inv,
                                              const float* gamma, int rows,
                                              int cols) {
    extern __shared__ float sh[]; // 2 * blockDim.x
    float* s_a = sh;
    float* s_b = sh + blockDim.x;

    int r = blockIdx.x;
    int tid = threadIdx.x;
    int base = r * cols;
    float inv_r = inv[r];
    float n = (float)cols;

    float sa = 0.0f, sb = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        float dxh = dy[base + j] * gamma[j];
        sa += dxh;
        sb += dxh * xhat[base + j];
    }
    s_a[tid] = sa;
    s_b[tid] = sb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_a[tid] += s_a[tid + s];
            s_b[tid] += s_b[tid + s];
        }
        __syncthreads();
    }
    float sum_dxh = s_a[0];
    float sum_dxh_xh = s_b[0];

    for (int j = tid; j < cols; j += blockDim.x) {
        float dxh = dy[base + j] * gamma[j];
        float xh = xhat[base + j];
        dx[base + j] =
            (1.0f / n) * inv_r * (n * dxh - sum_dxh - xh * sum_dxh_xh);
    }
}

// dgamma[j] = Σ_r dy[r,j]*xhat[r,j];  dbeta[j] = Σ_r dy[r,j]
// One block = one column (cols blocks), looping over rows.
extern "C" __global__ void layernorm_param_grad(float* dgamma, float* dbeta,
                                                const float* dy,
                                                const float* xhat, int rows,
                                                int cols) {
    int j = blockIdx.x;
    float ga = 0.0f, be = 0.0f;
    for (int r = 0; r < rows; r++) {
        float v = dy[r * cols + j];
        be += v;
        ga += v * xhat[r * cols + j];
    }
    dgamma[j] = ga;
    dbeta[j] = be;
}

extern "C" __global__ void add_param_grad(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] += src[idx];
}

// ------------------------------------------------------------
// Nedensel softmax (attention)
// ------------------------------------------------------------
// scores: (B*H, S, S) - a separate S x S block per (b,h), ld = S.
// One warp per row; on row i only j <= i is valid, the rest is 0.

extern "C" __global__ void masked_softmax_forward(float* w, const float* scores,
                                                  int seq, int n_heads,
                                                  int batch, float scale) {
    // One block = one (h, b, i) row; threads iterate over j.
    // Buffer layout [head][batch][seq][seq] (head is the slowest varying) -
    // because the GEMMs write strided over batch per head.
    int row = blockIdx.x;              // 0 .. heads*batch*seq-1
    int i = row % seq;
    int rest = row / seq;
    int b = rest % batch;
    int h = rest / batch;
    int bh = h * batch + b;

    const float* s = scores + (size_t)bh * seq * seq + (size_t)i * seq;
    float* o = w + (size_t)bh * seq * seq + (size_t)i * seq;

    __shared__ float red[256];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float mx = -INFINITY;
    for (int j = tid; j <= i; j += nthreads) {
        float v = s[j] * scale;
        if (v > mx) mx = v;
    }
    red[tid] = mx;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
        if (tid < st && red[tid + st] > red[tid]) red[tid] = red[tid + st];
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j <= i; j += nthreads) {
        sum += expf(s[j] * scale - maxv);
    }
    red[tid] = sum;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float inv = 1.0f / red[0];

    for (int j = tid; j < seq; j += nthreads) {
        o[j] = (j <= i) ? expf(s[j] * scale - maxv) * inv : 0.0f;
    }
}

// softmax geri (nedensel): ds = w * (dw - Σ_j' w·dw)
extern "C" __global__ void masked_softmax_backward(
    float* ds, const float* w, const float* dw, int seq, int n_heads, int batch,
    float scale) {
    // Same [head][batch][seq][seq] layout.
    int row = blockIdx.x;
    int i = row % seq;
    int rest = row / seq;
    int b = rest % batch;
    int h = rest / batch;
    int bh = h * batch + b;
    (void)n_heads;
    size_t base = (size_t)bh * seq * seq + (size_t)i * seq;

    __shared__ float red[256];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float dot = 0.0f;
    for (int j = tid; j <= i; j += nthreads) dot += w[base + j] * dw[base + j];
    red[tid] = dot;
    __syncthreads();
    for (int st = nthreads / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float d = red[0];

    for (int j = tid; j <= i; j += nthreads) {
        ds[base + j] = w[base + j] * (dw[base + j] - d) * scale;
    }
    for (int j = tid; j < seq; j += nthreads) {
        if (j > i) ds[base + j] = 0.0f;
    }
}

// ------------------------------------------------------------
// Embedding
// ------------------------------------------------------------

extern "C" __global__ void embedding_forward(float* out, const int* ids,
                                             const float* tok_emb,
                                             const float* pos_emb, int rows,
                                             int cols, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int d = idx % cols;
        int id = ids[r];
        int pos = r % seq_len;
        out[idx] = tok_emb[id * cols + d] + pos_emb[pos * cols + d];
    }
}

// Backward: atomic scatter (the same token may appear on several rows)
extern "C" __global__ void embedding_backward(float* dtok, float* dpos,
                                              const float* dout,
                                              const int* ids, int rows,
                                              int cols, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int r = idx / cols;
        int d = idx % cols;
        int id = ids[r];
        int pos = r % seq_len;
        float g = dout[idx];
        atomicAdd(&dtok[id * cols + d], g);
        atomicAdd(&dpos[pos * cols + d], g);
    }
}

// ------------------------------------------------------------
// Cross-entropy (logits: rows × classes, hedefler: rows)
// ------------------------------------------------------------

// the loss is written to a one-element array (mean over rows).
// probs (rows x classes) are kept for the backward pass.
extern "C" __global__ void ce_forward(float* loss_out, float* probs,
                                      const float* logits, const int* targets,
                                      int rows, int classes) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float red[256];

    float mx = -INFINITY;
    for (int j = tid; j < classes; j += blockDim.x) {
        float v = logits[r * classes + j];
        if (v > mx) mx = v;
    }
    red[tid] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st && red[tid + st] > red[tid]) red[tid] = red[tid + st];
        __syncthreads();
    }
    float maxv = red[0];
    __syncthreads();

    float sum = 0.0f;
    for (int j = tid; j < classes; j += blockDim.x) {
        sum += expf(logits[r * classes + j] - maxv);
    }
    red[tid] = sum;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) red[tid] += red[tid + st];
        __syncthreads();
    }
    float lse = maxv + logf(red[0]);
    float inv = 1.0f / red[0];

    for (int j = tid; j < classes; j += blockDim.x) {
        probs[r * classes + j] = expf(logits[r * classes + j] - maxv) * inv;
    }
    if (tid == 0) {
        float t = logits[r * classes + targets[r]];
        atomicAdd(loss_out, (lse - t) / (float)rows);
    }
}

// dlogits = (probs - onehot) / rows * upstream
extern "C" __global__ void ce_backward(float* dlogits, const float* probs,
                                       const int* targets, int rows, int classes,
                                       float upstream) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * classes;
    if (idx < total) {
        int r = idx / classes;
        int j = idx % classes;
        float p = probs[idx];
        if (j == targets[r]) p -= 1.0f;
        dlogits[idx] = p * upstream / (float)rows;
    }
}

// ------------------------------------------------------------
// Adam
// ------------------------------------------------------------

extern "C" __global__ void adam_step(float* param, const float* grad, float* m,
                                     float* v, float lr, float beta1,
                                     float beta2, float eps, float bc1,
                                     float bc2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad[idx];
        float mi = beta1 * m[idx] + (1.0f - beta1) * g;
        float vi = beta2 * v[idx] + (1.0f - beta2) * g * g;
        m[idx] = mi;
        v[idx] = vi;
        float mhat = mi / bc1;
        float vhat = vi / bc2;
        param[idx] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

// ------------------------------------------------------------
// Gradient clipping (global L2 norm) - two passes
// ------------------------------------------------------------

// Each block atomically accumulates its partial sum of squares.
extern "C" __global__ void sum_squares_atomic(float* acc, const float* x,
                                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sh[256];
    float s = 0.0f;
    if (idx < n) s = x[idx] * x[idx];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(acc, sh[0]);
}

// Apply the clipping scale.
extern "C" __global__ void scale_inplace(float* x, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= s;
}

// ============================================================
// RMSNorm - LayerNorm without the mean (LLaMA/PaLM).
//   y = x / sqrt(mean(x^2) + eps) * gamma
// No mean subtraction and no beta: less work, same quality.
// ============================================================

extern "C" __global__ void rmsnorm_forward(float* out, float* xhat, float* inv,
                                          const float* x, const float* gamma,
                                          int rows, int cols, float eps) {
    extern __shared__ float sh[];
    int r = blockIdx.x;
    int t = threadIdx.x;
    const float* xr = x + (long long)r * cols;
    float ss = 0.0f;
    for (int i = t; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    sh[t] = ss;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) sh[t] += sh[t + st];
        __syncthreads();
    }
    float ms = sh[0] / (float)cols;
    float r_inv = rsqrtf(ms + eps);
    if (t == 0) inv[r] = r_inv;
    float* orow = out + (long long)r * cols;
    float* xh = xhat + (long long)r * cols;
    for (int i = t; i < cols; i += blockDim.x) {
        float h = xr[i] * r_inv;
        xh[i] = h;
        orow[i] = h * gamma[i];
    }
}

// dx = r_inv * (dy*gamma - xhat * mean(dy*gamma*xhat))
extern "C" __global__ void rmsnorm_backward(float* dx, const float* dy,
                                           const float* xhat, const float* inv,
                                           const float* gamma,
                                           int rows, int cols) {
    extern __shared__ float sh[];
    int r = blockIdx.x;
    int t = threadIdx.x;
    const float* dyr = dy + (long long)r * cols;
    const float* xh = xhat + (long long)r * cols;
    float s = 0.0f;
    for (int i = t; i < cols; i += blockDim.x) {
        float g = dyr[i] * gamma[i];
        s += g * xh[i];
    }
    sh[t] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) sh[t] += sh[t + st];
        __syncthreads();
    }
    float mean = sh[0] / (float)cols;
    float r_inv = inv[r];
    float* dxr = dx + (long long)r * cols;
    for (int i = t; i < cols; i += blockDim.x) {
        float g = dyr[i] * gamma[i];
        dxr[i] = r_inv * (g - xh[i] * mean);
    }
}

// dgamma = sum_r dy*xhat  (per column)
//
// Rows are split across blocks (blockIdx.y): a single thread scanning all
// rows leaves 1024 columns x 2048 rows serial and slows the 250M model by ~1.8x.
extern "C" __global__ void rmsnorm_param_grad(float* dgamma, const float* dy,
                                             const float* xhat, int rows, int cols) {
    int c = blockIdx.x;
    if (c >= cols) return;
    int chunk = blockIdx.y;
    int nchunks = gridDim.y;
    float acc = 0.0f;
    for (int r = chunk; r < rows; r += nchunks) {
        long long off = (long long)r * cols + c;
        acc += dy[off] * xhat[off];
    }
    atomicAdd(dgamma + c, acc);
}

// ============================================================
// SwiGLU - gated FFN: silu(x*W1) * (x*W3)
// ============================================================

extern "C" __global__ void silu_mul_forward(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    out[i] = s * up[i];
}

// dgate = dy*up*silu'(gate) ; dup = dy*silu(gate)
extern "C" __global__ void silu_mul_backward(float* dgate, float* dup,
                                            const float* dy, const float* gate,
                                            const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float silu = g * sig;
    // silu'(g) = sig*(1 + g*(1-sig))
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    float d = dy[i];
    dgate[i] = d * up[i] * dsilu;
    dup[i] = d * silu;
}

// ============================================================
// RoPE - rotary position encoding. Rotates Q/K vectors by position.
// Gives relative position; removes the absolute position drift problem.
// ============================================================

// RoPE cos/sin table (seq x d_head/2), computed once.
// Without the table powf/cos/sin were called per element: measured, it
// slowed training by ~1.8x (transcendental functions are expensive).
__device__ void rope_apply(float* a, float* b, const float* cos_t, const float* sin_t,
                           int s, int j, int half, int inverse) {
    float c = cos_t[s * half + j];
    float sn = sin_t[s * half + j];
    if (inverse) sn = -sn;
    float x0 = a[0], x1 = b[0];
    a[0] = x0 * c - x1 * sn;
    b[0] = x0 * sn + x1 * c;
}

extern "C" __global__ void rope_forward(float* out, const float* x,
                                       const float* cos_t, const float* sin_t,
                                       int batch, int seq, int n_heads, int d_head) {
    int half = d_head / 2;
    long long total = (long long)batch * seq * n_heads * half;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = (int)(idx % half);
    long long tmp = idx / half;
    int h = (int)(tmp % n_heads);
    long long r = tmp / n_heads;
    int s = (int)(r % seq);
    long long base = (r * n_heads + h) * d_head + 2 * j;
    out[base] = x[base];
    out[base + 1] = x[base + 1];
    rope_apply(&out[base], &out[base + 1], cos_t, sin_t, s, j, half, 0);
}

extern "C" __global__ void rope_backward(float* dx, const float* dy,
                                        const float* cos_t, const float* sin_t,
                                        int batch, int seq, int n_heads, int d_head) {
    int half = d_head / 2;
    long long total = (long long)batch * seq * n_heads * half;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = (int)(idx % half);
    long long tmp = idx / half;
    int h = (int)(tmp % n_heads);
    long long r = tmp / n_heads;
    int s = (int)(r % seq);
    long long base = (r * n_heads + h) * d_head + 2 * j;
    dx[base] = dy[base];
    dx[base + 1] = dy[base + 1];
    rope_apply(&dx[base], &dx[base + 1], cos_t, sin_t, s, j, half, 1);
}

// ============================================================
// Dropout - randomly zero during training, scale by 1/(1-p).
// Each thread derives its own xorshift state from the seed.
// ============================================================

__device__ unsigned int hash_u32(unsigned int x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

extern "C" __global__ void dropout_inplace(float* x, int n, float p, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int h = hash_u32((unsigned int)i ^ seed);
    float u = (float)(h >> 8) / 16777216.0f;
    if (u < p) x[i] = 0.0f;
    else x[i] *= 1.0f / (1.0f - p);
}

// Apply the dropout mask in the backward pass (same seed -> same mask).
extern "C" __global__ void dropout_backward(float* dx, int n, float p, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int h = hash_u32((unsigned int)i ^ seed);
    float u = (float)(h >> 8) / 16777216.0f;
    if (u < p) dx[i] = 0.0f;
    else dx[i] *= 1.0f / (1.0f - p);
}

// RoPE architecture: token embedding only (no positional embedding).
extern "C" __global__ void embedding_forward_tok(float* out, const int* ids,
                                                const float* tok_emb,
                                                int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    out[idx] = tok_emb[(long long)ids[r] * cols + c];
}

// Gradient of the token embedding only (no positional gradient).
extern "C" __global__ void embedding_backward_tok(float* dtok, const float* dout,
                                                 const int* ids, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    atomicAdd(&dtok[(long long)ids[r] * cols + c], dout[idx]);
}

// ============================================================
// Helpers for KV-cache generation
// ============================================================

// Single-row softmax (no mask - every cached position is in the past).
extern "C" __global__ void softmax_vec(float* out, const float* x, int n) {
    extern __shared__ float sh[];
    int t = threadIdx.x;
    float mx = -INFINITY;
    for (int i = t; i < n; i += blockDim.x) mx = fmaxf(mx, x[i]);
    sh[t] = mx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) sh[t] = fmaxf(sh[t], sh[t + st]);
        __syncthreads();
    }
    float m = sh[0];
    __syncthreads();
    float s = 0.0f;
    for (int i = t; i < n; i += blockDim.x) s += expf(x[i] - m);
    sh[t] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (t < st) sh[t] += sh[t + st];
        __syncthreads();
    }
    float denom = 1.0f / sh[0];
    for (int i = t; i < n; i += blockDim.x) out[i] = expf(x[i] - m) * denom;
}

// RoPE for a single position (position = cache length during generation).
extern "C" __global__ void rope_forward_pos(float* out, const float* x,
                                           const float* cos_t, const float* sin_t,
                                           int n_heads, int d_head, int pos) {
    int half = d_head / 2;
    int total = n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int j = idx % half;
    int h = idx / half;
    long long base = (long long)h * d_head + 2 * j;
    float c = cos_t[pos * half + j];
    float sn = sin_t[pos * half + j];
    float x0 = x[base], x1 = x[base + 1];
    out[base] = x0 * c - x1 * sn;
    out[base + 1] = x0 * sn + x1 * c;
}

// Write into the cache. Layout: [head][capacity][d_head]
// k/v: (n_heads x d_head) - the head slices of a single token.
extern "C" __global__ void kv_store(float* cache_k, float* cache_v,
                                   const float* k, const float* v,
                                   int n_heads, int d_head, int pos, int capacity) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * d_head) return;
    int h = idx / d_head;
    int c = idx % d_head;
    long long off = ((long long)h * capacity + pos) * d_head + c;
    cache_k[off] = k[idx];
    cache_v[off] = v[idx];
}
