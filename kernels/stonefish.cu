// kernels/stonefish.cu
// StonefishAI — GPT eğitimi için CUDA çekirdekleri.
//
// NVRTC ile ÇALIŞMA ZAMANINDA derlenir (makinede host C++ derleyicisi yok;
// nvcc cl.exe istiyor, NVRTC istemiyor). PTX bir kez üretilip modül olarak
// yükleniyor (src/gpu_kernels.rs).
//
// Düzen: tüm çekirdekler `extern "C"` ve düz C imzasıyla; Rust tarafı
// `stream.launch_builder(...).arg(&ptr).arg(&n)` ile çağırır.
// Tüm matrisler row-major. "satır" = token (B*S satır), "kolon" = özellik.

// NVRTC varsayılan olarak math başlıklarını getirmez; INFINITY makrosu yok.
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

// GELU (tanh yaklaşımı) ileri: y = 0.5x(1+tanh(c(x+0.044715x³)))
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
// Satır başına bir blok. cols <= 1024 varsayımı (d_model=128).
// cache: xhat (normalize edilmiş giriş), inv_std (satır başına).

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

// LayerNorm geri: dx üretir; dgamma/dbeta'yı atomik olmadan biriktirmek için
// ayrı bir kolon-indirgeme çekirdeği kullanılıyor (layernorm_param_grad).
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
// Bir blok = bir kolon (cols kadar blok), satırlar üzerinde döner.
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
// scores: (B*H, S, S) — her (b,h) için ayrı S×S blok, ld = S.
// Satır başına bir warp; i. satırda yalnız j <= i geçerli, gerisi 0.

extern "C" __global__ void masked_softmax_forward(float* w, const float* scores,
                                                  int seq, int n_heads,
                                                  int batch, float scale) {
    // Bir blok = bir (h, b, i) satırı; thread'ler j üzerinde.
    // Buffer yerleşimi [head][batch][seq][seq] (head en yavaş değişen) —
    // GEMM'ler head başına batch üzerinde strided yazdığı için.
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
    // Aynı [head][batch][seq][seq] yerleşimi.
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

// Geri: atomik dağıtım (aynı token birden çok satırda geçebilir)
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

// loss tek elemanlık diziye yazılır (rows'a bölünmüş ortalama).
// probs (rows × classes) geri yayılım için saklanır.
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
// Gradyan kırpma (global L2 norm) — iki aşama
// ------------------------------------------------------------

// Her blok kısmi kareler toplamını atomik olarak biriktirir.
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

// Kırpma ölçeğini uygula.
extern "C" __global__ void scale_inplace(float* x, int n, float s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= s;
}

// ============================================================
// RMSNorm — LayerNorm'un ortalamasız hali (LLaMA/PaLM).
//   y = x / sqrt(mean(x^2) + eps) * gamma
// Ortalama çıkarma ve beta yok: daha az iş, eşit kalite.
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

extern "C" __global__ void rmsnorm_param_grad(float* dgamma, const float* dy,
                                             const float* xhat, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++) {
        acc += dy[(long long)r * cols + i] * xhat[(long long)r * cols + i];
    }
    atomicAdd(dgamma + i, acc);
}

// ============================================================
// SwiGLU — kapılı FFN: silu(x·W1) * (x·W3)
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
// RoPE — döner konum kodlaması. Q/K vektörlerini konuma göre döndürür.
// Göreli konum bilgisi verir; mutlak konum kayması sorununu bitirir.
// ============================================================

__device__ void rope_apply(float* a, float* b, int s, int j, int d_head, int inverse) {
    float freq = powf(10000.0f, -2.0f * (float)j / (float)d_head);
    float ang = (float)s * freq;
    float c = cosf(ang);
    float sn = sinf(ang);
    if (inverse) sn = -sn;
    float x0 = a[0], x1 = b[0];
    a[0] = x0 * c - x1 * sn;
    b[0] = x0 * sn + x1 * c;
}

extern "C" __global__ void rope_forward(float* out, const float* x,
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
    rope_apply(&out[base], &out[base + 1], s, j, d_head, 0);
}

extern "C" __global__ void rope_backward(float* dx, const float* dy,
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
    // ters dönüş
    rope_apply(&dx[base], &dx[base + 1], s, j, d_head, 1);
}

// ============================================================
// Dropout — eğitimde rastgele sıfırla, 1/(1-p) ile ölçekle.
// Her iş parçacığı kendi xorshift durumunu tohumdan türetir.
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

// Dropout maskesini geri yayılımda uygula (aynı tohum → aynı maske).
extern "C" __global__ void dropout_backward(float* dx, int n, float p, unsigned int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int h = hash_u32((unsigned int)i ^ seed);
    float u = (float)(h >> 8) / 16777216.0f;
    if (u < p) dx[i] = 0.0f;
    else dx[i] *= 1.0f / (1.0f - p);
}
