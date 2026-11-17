// src/gpu_train.rs
// Device-resident GPT eğiticisi.
//
// Tüm ağırlıklar, ara sonuçlar ve gradyanlar GPU'da kalır; adım başına tek
// CPU↔GPU trafiği loss değerini okumaktır. GEMM'ler cuBLAS, elementwise /
// normalizasyon / softmax / attention işleri `gpu_kernels` çekirdekleri.
//
// Autograd grafiği YOK: GPT mimarisine özel, elle yazılmış ileri/geri geçiş.
// Genel motor (autograd) referans yol olarak duruyor;
// `tests/gpu_train_equiv.rs` iki yolu hem gradyan hem loss eğrisi düzeyinde
// karşılaştırıyor.
//
// Mimarî (CPU ile birebir):
//   x = emb(ids) + pos
//   blok:  h1 = x + attn(LN1(x)) ;  out = h1 + ff2(gelu(ff1(LN2(h1))))
//   son:   logits = head(LNf(h))
//
// Satır paketleme: (batch, seq, d) → (batch*seq, d). Head dilimleri ve batch,
// GEMM view'ları + strided-batched çağrılarla ele alınır.

#![cfg(feature = "cuda")]

use crate::gpu::{self, GemmMode};
use crate::gpu_kernels as k;
use crate::transformer::{GptConfig, GPT};
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;

const LN_EPS: f32 = 1e-5;

fn stream() -> Arc<CudaStream> {
    gpu::get_ctx().default_stream()
}

fn alloc(n: usize) -> CudaSlice<f32> {
    stream().alloc_zeros::<f32>(n.max(1)).unwrap()
}

#[allow(dead_code)]
fn alloc_i32(n: usize) -> CudaSlice<i32> {
    stream().alloc_zeros::<i32>(n.max(1)).unwrap()
}

fn upload(v: &[f32]) -> CudaSlice<f32> {
    stream().clone_htod(v).unwrap()
}

fn upload_i32(v: &[i32]) -> CudaSlice<i32> {
    stream().clone_htod(v).unwrap()
}

fn dtod(dst: &mut CudaSlice<f32>, src: &CudaSlice<f32>) {
    stream().memcpy_dtod(src, dst).unwrap();
}

// ============================================================
// Parametre
// ============================================================

pub struct Param {
    pub data: CudaSlice<f32>,
    pub grad: CudaSlice<f32>,
    m: CudaSlice<f32>,
    v: CudaSlice<f32>,
    pub rows: usize,
    pub cols: usize,
    pub name: &'static str,
}

impl Param {
    fn from_host(rows: usize, cols: usize, name: &'static str, host: &[f32]) -> Self {
        assert_eq!(host.len(), rows * cols, "parametre boyutu: {name}");
        let n = rows * cols;
        Self {
            data: upload(host),
            grad: alloc(n),
            m: alloc(n),
            v: alloc(n),
            rows,
            cols,
            name,
        }
    }

    fn n(&self) -> usize {
        self.rows * self.cols
    }

    fn zero_grad(&mut self) {
        let n = self.n();
        k::zero_fill(&mut self.grad, n);
    }

    fn adam(&mut self, lr: f32, t: usize) {
        let n = self.n();
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-6f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        k::adam_step(
            &mut self.data,
            &self.grad,
            &mut self.m,
            &mut self.v,
            lr,
            b1,
            b2,
            eps,
            bc1,
            bc2,
            n,
        );
    }

    pub fn download(&self) -> Vec<f32> {
        stream().clone_dtoh(&self.data).unwrap()
    }

    pub fn download_grad(&self) -> Vec<f32> {
        stream().clone_dtoh(&self.grad).unwrap()
    }
}

// ============================================================
// Blok / aktivasyon
// ============================================================

struct GpuBlock {
    ln1_g: Param,
    ln1_b: Param,
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    ln2_g: Param,
    ln2_b: Param,
    ff1_w: Param,
    ff1_b: Param,
    ff2_w: Param,
    ff2_b: Param,
}

/// Katman başına geri yayılım için saklanan ara sonuçlar.
struct Acts {
    ln1: CudaSlice<f32>,
    xhat1: CudaSlice<f32>,
    inv1: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    w: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    res1: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    xhat2: CudaSlice<f32>,
    inv2: CudaSlice<f32>,
    ff1: CudaSlice<f32>,
    gelu: CudaSlice<f32>,
    dq: CudaSlice<f32>,
    dk: CudaSlice<f32>,
    dv: CudaSlice<f32>,
}

#[derive(Clone, Copy)]
pub struct Dims {
    pub v: usize,
    pub d: usize,
    pub f: usize,
    pub h: usize,
    pub b: usize,
    pub s: usize,
}

impl Dims {
    pub fn bs(&self) -> usize {
        self.b * self.s
    }
    pub fn dh(&self) -> usize {
        self.d / self.h
    }
    pub fn bh_seq(&self) -> usize {
        self.b * self.h * self.s
    }
    pub fn scale(&self) -> f32 {
        1.0 / (self.dh() as f32).sqrt()
    }
}

// ============================================================
// Eğitici
// ============================================================

pub struct GpuGpt {
    pub cfg: GptConfig,
    pub batch: usize,
    pub seq: usize,

    tok_emb: Param,
    pos_emb: Param,
    blocks: Vec<GpuBlock>,
    lnf_g: Param,
    lnf_b: Param,
    head_w: Param,
    head_b: Param,

    acts: Vec<Acts>,
    h: Vec<CudaSlice<f32>>,
    dh: Vec<CudaSlice<f32>>,

    lnf_out: CudaSlice<f32>,
    xhat_f: CudaSlice<f32>,
    inv_f: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    dlogits: CudaSlice<f32>,
    dlnf: CudaSlice<f32>,
    loss: CudaSlice<f32>,

    tmp_a: CudaSlice<f32>,
    tmp_b: CudaSlice<f32>,
    tmp_c: CudaSlice<f32>,
    tmp_fa: CudaSlice<f32>,
    tmp_fb: CudaSlice<f32>,
    /// İleri geçişte attention skorları, geri geçişte dW için yeniden kullanılır.
    scores: CudaSlice<f32>,
    dscores: CudaSlice<f32>,
    ones: CudaSlice<f32>,

    pub adam_t: usize,
}

fn mk_param(t: &crate::autograd::Tensor, name: &'static str) -> Param {
    let data = t.data.borrow();
    Param::from_host(data.rows, data.cols, name, &data.data)
}

impl GpuGpt {
    /// CPU modelindeki ağırlıklarla kurar — iki eğitici aynı noktadan başlar.
    pub fn from_cpu(model: &GPT, batch: usize, seq: usize) -> Self {
        let cfg = model.config;
        assert!(seq <= cfg.max_seq_len, "seq > max_seq_len");
        let (v, d, f) = (cfg.vocab_size, cfg.d_model, 4 * cfg.d_model);
        let bs = batch * seq;
        let bh_seq = batch * cfg.n_heads * seq;

        let blocks: Vec<GpuBlock> = model
            .blocks
            .iter()
            .map(|b| GpuBlock {
                ln1_g: mk_param(&b.ln1_gamma, "ln1_g"),
                ln1_b: mk_param(&b.ln1_beta, "ln1_b"),
                wq: mk_param(&b.attn.w_q, "wq"),
                wk: mk_param(&b.attn.w_k, "wk"),
                wv: mk_param(&b.attn.w_v, "wv"),
                wo: mk_param(&b.attn.w_o, "wo"),
                ln2_g: mk_param(&b.ln2_gamma, "ln2_g"),
                ln2_b: mk_param(&b.ln2_beta, "ln2_b"),
                ff1_w: mk_param(&b.ff1.weight, "ff1_w"),
                ff1_b: mk_param(&b.ff1.bias, "ff1_b"),
                ff2_w: mk_param(&b.ff2.weight, "ff2_w"),
                ff2_b: mk_param(&b.ff2.bias, "ff2_b"),
            })
            .collect();

        let h: Vec<CudaSlice<f32>> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let dh: Vec<CudaSlice<f32>> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let acts: Vec<Acts> = (0..cfg.n_layers)
            .map(|_| Acts {
                ln1: alloc(bs * d),
                xhat1: alloc(bs * d),
                inv1: alloc(bs),
                q: alloc(bs * d),
                k: alloc(bs * d),
                v: alloc(bs * d),
                w: alloc(bh_seq * seq),
                attn: alloc(bs * d),
                res1: alloc(bs * d),
                ln2: alloc(bs * d),
                xhat2: alloc(bs * d),
                inv2: alloc(bs),
                ff1: alloc(bs * f),
                gelu: alloc(bs * f),
                dq: alloc(bs * d),
                dk: alloc(bs * d),
                dv: alloc(bs * d),
            })
            .collect();

        Self {
            cfg,
            batch,
            seq,
            tok_emb: mk_param(&model.embedding.token_emb, "tok_emb"),
            pos_emb: mk_param(&model.embedding.pos_emb, "pos_emb"),
            blocks,
            lnf_g: mk_param(&model.ln_f_gamma, "lnf_g"),
            lnf_b: mk_param(&model.ln_f_beta, "lnf_b"),
            head_w: mk_param(&model.head.weight, "head_w"),
            head_b: mk_param(&model.head.bias, "head_b"),
            acts,
            h,
            dh,
            lnf_out: alloc(bs * d),
            xhat_f: alloc(bs * d),
            inv_f: alloc(bs),
            logits: alloc(bs * v),
            probs: alloc(bs * v),
            dlogits: alloc(bs * v),
            dlnf: alloc(bs * d),
            loss: alloc(1),
            tmp_a: alloc(bs * d),
            tmp_b: alloc(bs * d),
            tmp_c: alloc(bs * d),
            tmp_fa: alloc(bs * f),
            tmp_fb: alloc(bs * f),
            scores: alloc(bh_seq * seq),
            dscores: alloc(bh_seq * seq),
            ones: upload(&vec![1.0f32; bs]),
            adam_t: 0,
        }
    }

    pub fn dims(&self) -> Dims {
        Dims {
            v: self.cfg.vocab_size,
            d: self.cfg.d_model,
            f: 4 * self.cfg.d_model,
            h: self.cfg.n_heads,
            b: self.batch,
            s: self.seq,
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut out: Vec<&mut Param> = vec![&mut self.tok_emb, &mut self.pos_emb];
        for b in &mut self.blocks {
            out.push(&mut b.ln1_g);
            out.push(&mut b.ln1_b);
            out.push(&mut b.wq);
            out.push(&mut b.wk);
            out.push(&mut b.wv);
            out.push(&mut b.wo);
            out.push(&mut b.ln2_g);
            out.push(&mut b.ln2_b);
            out.push(&mut b.ff1_w);
            out.push(&mut b.ff1_b);
            out.push(&mut b.ff2_w);
            out.push(&mut b.ff2_b);
        }
        out.push(&mut self.lnf_g);
        out.push(&mut self.lnf_b);
        out.push(&mut self.head_w);
        out.push(&mut self.head_b);
        out
    }

    pub fn zero_grads(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }

    /// Adam + global gradyan kırpma. Kırpma öncesi normu döndürür.
    pub fn step(&mut self, lr: f32, clip: f32) -> f32 {
        self.adam_t += 1;
        let t = self.adam_t;

        let mut acc = alloc(1);
        for p in self.params_mut() {
            let n = p.n();
            k::sum_squares(&mut acc, &p.grad, n);
        }
        let norm = stream().clone_dtoh(&acc).unwrap()[0].sqrt();
        if clip > 0.0 && norm > clip {
            let s = clip / norm;
            for p in self.params_mut() {
                let n = p.n();
                k::scale_inplace(&mut p.grad, n, s);
            }
        }
        for p in self.params_mut() {
            p.adam(lr, t);
        }
        norm
    }

    // ========================================================
    // İleri + geri
    // ========================================================

    pub fn forward_backward(&mut self, ids: &[i32], targets: &[i32]) -> f32 {
        let dm = self.dims();
        assert_eq!(ids.len(), dm.bs());
        assert_eq!(targets.len(), dm.bs());

        let ids_dev = upload_i32(ids);
        let tgt_dev = upload_i32(targets);
        self.zero_grads();

        // ---- ileri ----
        k::embedding_forward(
            &mut self.h[0],
            &ids_dev,
            &self.tok_emb.data,
            &self.pos_emb.data,
            dm.bs(),
            dm.d,
            dm.s,
        );
        for l in 0..self.cfg.n_layers {
            self.block_forward(l, &dm);
        }

        k::layernorm_forward(
            &mut self.lnf_out,
            &mut self.xhat_f,
            &mut self.inv_f,
            &self.h[self.cfg.n_layers],
            &self.lnf_g.data,
            &self.lnf_b.data,
            dm.bs(),
            dm.d,
            LN_EPS,
        );
        gpu::dev_gemm(
            &mut self.logits,
            &self.lnf_out,
            &self.head_w.data,
            dm.bs(),
            dm.v,
            dm.d,
            dm.d,
            dm.v,
            dm.v,
            GemmMode::Nn,
        );
        k::add_bias_inplace(&mut self.logits, &self.head_b.data, dm.bs(), dm.v);
        k::zero_fill(&mut self.loss, 1);
        k::ce_forward(
            &mut self.loss,
            &mut self.probs,
            &self.logits,
            &tgt_dev,
            dm.bs(),
            dm.v,
        );
        let loss = stream().clone_dtoh(&self.loss).unwrap()[0];

        // ---- geri ----
        k::ce_backward(&mut self.dlogits, &self.probs, &tgt_dev, dm.bs(), dm.v, 1.0);

        gpu::dev_gemm(
            &mut self.head_w.grad,
            &self.lnf_out,
            &self.dlogits,
            dm.d,
            dm.v,
            dm.bs(),
            dm.d,
            dm.v,
            dm.v,
            GemmMode::Tn,
        );
        gpu::dev_gemm(
            &mut self.head_b.grad,
            &self.ones,
            &self.dlogits,
            1,
            dm.v,
            dm.bs(),
            dm.bs(),
            dm.v,
            dm.v,
            GemmMode::Nn,
        );
        // dlnf = dlogits·head_wᵀ
        gpu::dev_gemm(
            &mut self.dlnf,
            &self.dlogits,
            &self.head_w.data,
            dm.bs(),
            dm.d,
            dm.v,
            dm.v,
            dm.v,
            dm.d,
            GemmMode::Nt,
        );

        {
            let mut g = alloc(dm.d);
            let mut b = alloc(dm.d);
            k::layernorm_backward(
                &mut self.dh[self.cfg.n_layers],
                &self.dlnf,
                &self.xhat_f,
                &self.inv_f,
                &self.lnf_g.data,
                dm.bs(),
                dm.d,
            );
            k::layernorm_param_grad(&mut g, &mut b, &self.dlnf, &self.xhat_f, dm.bs(), dm.d);
            k::add_inplace(&mut self.lnf_g.grad, &g, dm.d);
            k::add_inplace(&mut self.lnf_b.grad, &b, dm.d);
        }

        for l in (0..self.cfg.n_layers).rev() {
            self.block_backward(l, &dm);
        }

        k::embedding_backward(
            &mut self.tok_emb.grad,
            &mut self.pos_emb.grad,
            &self.dh[0],
            &ids_dev,
            dm.bs(),
            dm.d,
            dm.s,
        );

        loss
    }

    fn block_forward(&mut self, l: usize, dm: &Dims) {
        // Disjoint alan ödünçleri: tek `&mut self` yerine alanları ayır.
        let Self {
            acts,
            blocks,
            h,
            scores,
            tmp_a,
            tmp_fa,
            tmp_fb,
            ..
        } = self;
        let (bs, d, f, s, nh, dh, scale) = (
            dm.bs(),
            dm.d,
            dm.f,
            dm.s,
            dm.h,
            dm.dh(),
            dm.scale(),
        );
        let a = &mut acts[l];
        let blk = &blocks[l];
        let (h_lo, h_hi) = h.split_at_mut(l + 1);
        let x_in = &h_lo[l];
        let out = &mut h_hi[0];

        // LN1
        k::layernorm_forward(
            &mut a.ln1,
            &mut a.xhat1,
            &mut a.inv1,
            x_in,
            &blk.ln1_g.data,
            &blk.ln1_b.data,
            bs,
            d,
            LN_EPS,
        );

        // q, k, v (bias yok)
        gpu::dev_gemm(&mut a.q, &a.ln1, &blk.wq.data, bs, d, d, d, d, d, GemmMode::Nn);
        gpu::dev_gemm(&mut a.k, &a.ln1, &blk.wk.data, bs, d, d, d, d, d, GemmMode::Nn);
        gpu::dev_gemm(&mut a.v, &a.ln1, &blk.wv.data, bs, d, d, d, d, d, GemmMode::Nn);

        // skorlar (her head, batch üzerinde strided)
        for hh in 0..nh {
            let off = hh * dh;
            let q_view = a.q.slice(off..);
            let k_view = a.k.slice(off..);
            let mut sc = scores.slice_mut(hh * dm.b * s * s..);
            gpu::dev_gemm_batched(
                &mut sc,
                &q_view,
                &k_view,
                s,
                s,
                dh,
                d,
                d,
                s,
                s * d,
                s * d,
                s * s,
                dm.b,
                0.0,
                GemmMode::Nt,
            );
        }
        k::masked_softmax_forward(&mut a.w, scores, dm.b, nh, s, scale);

        // attn = W·V
        for hh in 0..nh {
            let off = hh * dh;
            let w_view = a.w.slice(hh * dm.b * s * s..);
            let v_view = a.v.slice(off..);
            let mut av = a.attn.slice_mut(off..);
            gpu::dev_gemm_batched(
                &mut av,
                &w_view,
                &v_view,
                s,
                dh,
                s,
                s,
                d,
                d,
                s * s,
                s * d,
                s * d,
                dm.b,
                0.0,
                GemmMode::Nn,
            );
        }

        // proj = attn·wo
        gpu::dev_gemm(&mut *tmp_a, &a.attn, &blk.wo.data, bs, d, d, d, d, d, GemmMode::Nn);

        // res1 = x + proj
        dtod(&mut a.res1, x_in);
        k::add_inplace(&mut a.res1, tmp_a, bs * d);

        // LN2
        k::layernorm_forward(
            &mut a.ln2,
            &mut a.xhat2,
            &mut a.inv2,
            &a.res1,
            &blk.ln2_g.data,
            &blk.ln2_b.data,
            bs,
            d,
            LN_EPS,
        );

        // ff1 + bias → gelu → ff2 + bias
        gpu::dev_gemm(&mut *tmp_fa, &a.ln2, &blk.ff1_w.data, bs, f, d, d, f, f, GemmMode::Nn);
        k::add_bias_inplace(tmp_fa, &blk.ff1_b.data, bs, f);
        dtod(&mut a.ff1, tmp_fa);
        k::gelu_forward(&mut a.gelu, &a.ff1, bs * f);

        gpu::dev_gemm(&mut *tmp_fb, &a.gelu, &blk.ff2_w.data, bs, d, f, f, d, d, GemmMode::Nn);
        k::add_bias_inplace(tmp_fb, &blk.ff2_b.data, bs, d);

        // out = res1 + ff2_out
        dtod(out, &a.res1);
        k::add_inplace(out, tmp_fb, bs * d);
    }

    fn block_backward(&mut self, l: usize, dm: &Dims) {
        let Self {
            acts,
            blocks,
            dh: dh_vec,
            scores,
            dscores,
            ones,
            tmp_a,
            tmp_b,
            tmp_c,
            tmp_fa,
            tmp_fb,
            ..
        } = self;
        let (bs, d, f, s, nh, dhh, scale) = (
            dm.bs(),
            dm.d,
            dm.f,
            dm.s,
            dm.h,
            dm.dh(),
            dm.scale(),
        );
        let a = &mut acts[l];
        let blk = &mut blocks[l];
        let (dh_lo, dh_hi) = dh_vec.split_at_mut(l + 1);
        let dout = dh_hi[0].clone();

        // --- ff2 ---
        // dgelu = dOut·ff2_wᵀ  (BS,F)
        gpu::dev_gemm(
            tmp_fa,
            &dout,
            &blk.ff2_w.data,
            bs,
            f,
            d,
            d,
            d,
            f,
            GemmMode::Nt,
        );
        // dff2_w = geluᵀ·dOut (F,D)
        gpu::dev_gemm(
            &mut blk.ff2_w.grad,
            &a.gelu,
            &dout,
            f,
            d,
            bs,
            f,
            d,
            d,
            GemmMode::Tn,
        );
        // dff2_b = ones·dOut (1,D)
        gpu::dev_gemm(
            &mut blk.ff2_b.grad,
            ones,
            &dout,
            1,
            d,
            bs,
            bs,
            d,
            d,
            GemmMode::Nn,
        );

        // gelu geri
        k::gelu_backward(tmp_fb, tmp_fa, &a.ff1, bs * f);

        // --- ff1 ---
        // dln2 = dff1_out·ff1_wᵀ (BS,D)
        gpu::dev_gemm(
            tmp_b,
            tmp_fb,
            &blk.ff1_w.data,
            bs,
            d,
            f,
            f,
            f,
            d,
            GemmMode::Nt,
        );
        // dff1_w = ln2ᵀ·dff1_out (D,F)
        gpu::dev_gemm(
            &mut blk.ff1_w.grad,
            &a.ln2,
            tmp_fb,
            d,
            f,
            bs,
            d,
            f,
            f,
            GemmMode::Tn,
        );
        // dff1_b = ones·dff1_out (1,F)
        gpu::dev_gemm(
            &mut blk.ff1_b.grad,
            ones,
            tmp_fb,
            1,
            f,
            bs,
            bs,
            f,
            f,
            GemmMode::Nn,
        );

        // --- LN2 geri ---
        k::layernorm_backward(tmp_c, tmp_b, &a.xhat2, &a.inv2, &blk.ln2_g.data, bs, d);
        {
            let mut g = alloc(d);
            let mut b = alloc(d);
            k::layernorm_param_grad(&mut g, &mut b, tmp_b, &a.xhat2, bs, d);
            k::add_inplace(&mut blk.ln2_g.grad, &g, d);
            k::add_inplace(&mut blk.ln2_b.grad, &b, d);
        }

        // dh[l] = dOut (residual) + dx_ln2
        dtod(&mut dh_lo[l], &dout);
        k::add_inplace(&mut dh_lo[l], tmp_c, bs * d);

        // TEŞHİS
        if std::env::var("SFISH_DBG").is_ok() && l == 1 {
            let da = stream().clone_dtoh(&dout).unwrap();
            let at = stream().clone_dtoh(&a.attn).unwrap();
            println!("  [dbg] dOut[0..4]={:?}", &da[0..4]);
            println!("  [dbg] attn[0..4]={:?}", &at[0..4]);
        }

        // --- attention çıkış projeksiyonu ---
        // ÖNEMLİ: bu noktada dh[l] = dOut + dx_ln2'dir; attention dalının
        // yukarı akışı budur (res1 = x + proj olduğu için). Blok çıkışının
        // gradyanı (dOut) tek başına kullanılamaz.
        // dattn = dh[l]·woᵀ (BS,D) → tmp_a
        gpu::dev_gemm(
            tmp_a,
            &dh_lo[l],
            &blk.wo.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nt,
        );
        // dwo = attnᵀ·dh[l] (D,D)
        gpu::dev_gemm(
            &mut blk.wo.grad,
            &a.attn,
            &dh_lo[l],
            d,
            d,
            bs,
            d,
            d,
            d,
            GemmMode::Tn,
        );

        // TEŞHİS
        if std::env::var("SFISH_DBG").is_ok() && l == 1 {
            let g = stream().clone_dtoh(&blk.wo.grad).unwrap();
            println!("  [dbg] dWo[0..4]={:?}", &g[0..4]);
        }

        // --- attention geri ---
        // dV = Wᵀ·dAttn
        for hh in 0..nh {
            let off = hh * dhh;
            let w_view = a.w.slice(hh * dm.b * s * s..);
            let da_view = tmp_a.slice(off..);
            let mut dv_view = a.dv.slice_mut(off..);
            gpu::dev_gemm_batched(
                &mut dv_view,
                &w_view,
                &da_view,
                s,
                dhh,
                s,
                s,
                d,
                d,
                s * s,
                s * d,
                s * d,
                dm.b,
                0.0,
                GemmMode::Tn,
            );
        }
        // dW = dAttn·Vᵀ → scores (yeniden kullanım)
        for hh in 0..nh {
            let off = hh * dhh;
            let da_view = tmp_a.slice(off..);
            let v_view = a.v.slice(off..);
            let mut dw_view = scores.slice_mut(hh * dm.b * s * s..);
            gpu::dev_gemm_batched(
                &mut dw_view,
                &da_view,
                &v_view,
                s,
                s,
                dhh,
                d,
                d,
                s,
                s * d,
                s * d,
                s * s,
                dm.b,
                0.0,
                GemmMode::Nt,
            );
        }
        k::masked_softmax_backward(dscores, &a.w, scores, dm.b, nh, s, scale);

        // dQ = dS·K ; dK = dSᵀ·Q
        for hh in 0..nh {
            let off = hh * dhh;
            let ds_view = dscores.slice(hh * dm.b * s * s..);
            let k_view = a.k.slice(off..);
            let q_view = a.q.slice(off..);
            {
                let mut dq_view = a.dq.slice_mut(off..);
                gpu::dev_gemm_batched(
                    &mut dq_view,
                    &ds_view,
                    &k_view,
                    s,
                    dhh,
                    s,
                    s,
                    d,
                    d,
                    s * s,
                    s * d,
                    s * d,
                    dm.b,
                    0.0,
                    GemmMode::Nn,
                );
            }
            {
                let mut dk_view = a.dk.slice_mut(off..);
                gpu::dev_gemm_batched(
                    &mut dk_view,
                    &ds_view,
                    &q_view,
                    s,
                    dhh,
                    s,
                    s,
                    d,
                    d,
                    s * s,
                    s * d,
                    s * d,
                    dm.b,
                    0.0,
                    GemmMode::Tn,
                );
            }
        }

        // --- q/k/v projeksiyon gradyanları ---
        // dln1 = dQ·wqᵀ + dK·wkᵀ + dV·wvᵀ  (beta ile biriktir)
        gpu::dev_gemm(
            tmp_b,
            &a.dq,
            &blk.wq.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nt,
        );
        gpu::dev_gemm_batched(
            tmp_b,
            &a.dk,
            &blk.wk.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            0,
            0,
            0,
            1,
            1.0,
            GemmMode::Nt,
        );
        gpu::dev_gemm_batched(
            tmp_b,
            &a.dv,
            &blk.wv.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            0,
            0,
            0,
            1,
            1.0,
            GemmMode::Nt,
        );

        // dWq/dWk/dWv = ln1ᵀ·d{Q,K,V}
        gpu::dev_gemm(
            &mut blk.wq.grad,
            &a.ln1,
            &a.dq,
            d,
            d,
            bs,
            d,
            d,
            d,
            GemmMode::Tn,
        );
        gpu::dev_gemm(
            &mut blk.wk.grad,
            &a.ln1,
            &a.dk,
            d,
            d,
            bs,
            d,
            d,
            d,
            GemmMode::Tn,
        );
        gpu::dev_gemm(
            &mut blk.wv.grad,
            &a.ln1,
            &a.dv,
            d,
            d,
            bs,
            d,
            d,
            d,
            GemmMode::Tn,
        );

        // --- LN1 geri ---
        k::layernorm_backward(tmp_c, tmp_b, &a.xhat1, &a.inv1, &blk.ln1_g.data, bs, d);
        {
            let mut g = alloc(d);
            let mut b = alloc(d);
            k::layernorm_param_grad(&mut g, &mut b, tmp_b, &a.xhat1, bs, d);
            k::add_inplace(&mut blk.ln1_g.grad, &g, d);
            k::add_inplace(&mut blk.ln1_b.grad, &b, d);
        }
        k::add_inplace(&mut dh_lo[l], tmp_c, bs * d);
    }
}

impl GpuGpt {
    /// Test/teşhis için: isimle parametre gradyanı.
    pub fn grad_of(&mut self, name: &str, block: usize) -> Option<Vec<f32>> {
        for p in self.params_mut() {
            if p.name == name && block == usize::MAX {
                return Some(p.download_grad());
            }
        }
        let blk = &self.blocks[block];
        let p = match name {
            "ln1_g" => &blk.ln1_g,
            "ln1_b" => &blk.ln1_b,
            "wq" => &blk.wq,
            "wk" => &blk.wk,
            "wv" => &blk.wv,
            "wo" => &blk.wo,
            "ln2_g" => &blk.ln2_g,
            "ln2_b" => &blk.ln2_b,
            "ff1_w" => &blk.ff1_w,
            "ff1_b" => &blk.ff1_b,
            "ff2_w" => &blk.ff2_w,
            "ff2_b" => &blk.ff2_b,
            _ => return None,
        };
        Some(p.download_grad())
    }




    /// Cihazdaki ağırlıkları CPU modeline geri yazar (checkpoint için —
    /// checkpoint formatı CPU modeli üzerinden tanımlı).
    pub fn copy_to_cpu(&self, model: &mut GPT) {
        let put = |t: &crate::autograd::Tensor, p: &Param| {
            let host = p.download();
            let mut d = t.data.borrow_mut();
            assert_eq!(d.data.len(), host.len());
            d.data.copy_from_slice(&host);
        };
        put(&model.embedding.token_emb, &self.tok_emb);
        put(&model.embedding.pos_emb, &self.pos_emb);
        for (bi, blk) in model.blocks.iter().enumerate() {
            let gb = &self.blocks[bi];
            put(&blk.ln1_gamma, &gb.ln1_g);
            put(&blk.ln1_beta, &gb.ln1_b);
            put(&blk.attn.w_q, &gb.wq);
            put(&blk.attn.w_k, &gb.wk);
            put(&blk.attn.w_v, &gb.wv);
            put(&blk.attn.w_o, &gb.wo);
            put(&blk.ln2_gamma, &gb.ln2_g);
            put(&blk.ln2_beta, &gb.ln2_b);
            put(&blk.ff1.weight, &gb.ff1_w);
            put(&blk.ff1.bias, &gb.ff1_b);
            put(&blk.ff2.weight, &gb.ff2_w);
            put(&blk.ff2.bias, &gb.ff2_b);
        }
        put(&model.ln_f_gamma, &self.lnf_g);
        put(&model.ln_f_beta, &self.lnf_b);
        put(&model.head.weight, &self.head_w);
        put(&model.head.bias, &self.head_b);
    }
}
