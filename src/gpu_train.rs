// src/gpu_train.rs
// Device-resident GPT trainer.
//
// All weights, intermediates and gradients stay on the GPU; the only
// CPU<->GPU traffic per step is reading the loss value. GEMMs go to cuBLAS;
// elementwise / normalization / softmax / attention go to `gpu_kernels`.
//
// NO autograd graph: a hand-written forward/backward pass specific to GPT.
// The general engine (autograd) remains as the reference path;
// `tests/gpu_train_equiv.rs` compares the two paths at both the gradient
// and the loss-curve level.
//
// Architecture (identical to the CPU path):
//   x = emb(ids) + pos
//   blok:  h1 = x + attn(LN1(x)) ;  out = h1 + ff2(gelu(ff1(LN2(h1))))
//   son:   logits = head(LNf(h))
//
// Row packing: (batch, seq, d) -> (batch*seq, d). Head slices and batch are
// handled with GEMM views + strided-batched calls.

#![cfg(feature = "cuda")]

use crate::gpu::{self, GemmMode};
use crate::gpu_kernels as k;
use crate::sys::dev::{DevBuf, DevBufI32, DevPtr};
use crate::transformer::{swiglu_hidden, GptConfig, GPT, RMS_EPS};

fn alloc(n: usize) -> DevBuf {
    DevBuf::zeros(n)
}

fn upload(v: &[f32]) -> DevBuf {
    DevBuf::from_host(v)
}

fn upload_i32(v: &[i32]) -> DevBufI32 {
    DevBufI32::from_host(v)
}

fn dtod(dst: &mut DevBuf, src: &impl DevPtr) {
    dst.copy_from_ptr(src);
}

// ============================================================
// Parameters
// ============================================================

pub struct Param {
    pub data: DevBuf,
    pub grad: DevBuf,
    m: DevBuf,
    v: DevBuf,
    pub rows: usize,
    pub cols: usize,
    pub name: &'static str,
}

impl Param {
    fn from_host(rows: usize, cols: usize, name: &'static str, host: &[f32]) -> Self {
        assert_eq!(host.len(), rows * cols, "parameter size: {name}");
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

    pub fn n(&self) -> usize {
        self.rows * self.cols
    }

    fn zero_grad(&mut self) {
        let n = self.n();
        k::zero_fill(&self.grad, n);
    }

    fn adam(&mut self, lr: f32, t: usize) {
        let n = self.n();
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-6f32);
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        k::adam_step(
            &self.data, &self.grad, &self.m, &self.v, lr, b1, b2, eps, bc1, bc2, n,
        );
    }

    pub fn download(&self) -> Vec<f32> {
        self.data.to_host()
    }

    pub fn download_grad(&self) -> Vec<f32> {
        self.grad.to_host()
    }
}

// ============================================================
// Blok / aktivasyon
// ============================================================

pub struct GpuBlock {
    /// RMSNorm gamma (no beta)
    norm1_g: Param,
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    norm2_g: Param,
    /// SwiGLU gate (d -> hidden)
    ff_gate: Param,
    /// SwiGLU up projection (d -> hidden)
    ff_up: Param,
    /// SwiGLU down projection (hidden -> d)
    ff_down: Param,
}

impl GpuBlock {
    // Read-only accessors for the KV-cache generation path.
    pub fn gamma1(&self) -> &DevBuf {
        &self.norm1_g.data
    }
    pub fn gamma2(&self) -> &DevBuf {
        &self.norm2_g.data
    }
    pub fn wq(&self) -> &DevBuf {
        &self.wq.data
    }
    pub fn wk(&self) -> &DevBuf {
        &self.wk.data
    }
    pub fn wv(&self) -> &DevBuf {
        &self.wv.data
    }
    pub fn wo(&self) -> &DevBuf {
        &self.wo.data
    }
    pub fn ff_gate(&self) -> &DevBuf {
        &self.ff_gate.data
    }
    pub fn ff_up(&self) -> &DevBuf {
        &self.ff_up.data
    }
    pub fn ff_down(&self) -> &DevBuf {
        &self.ff_down.data
    }
}
/// Intermediate results kept for the per-layer backward pass.
struct Acts {
    x1: DevBuf,    // RMSNorm1 output
    xhat1: DevBuf, // normalized (before gamma)
    inv1: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    w: DevBuf,    // attention weights (B*H*S, S)
    attn: DevBuf, // attention output before the projection
    res1: DevBuf, // x + attn_proj
    x2: DevBuf,   // RMSNorm2 output
    xhat2: DevBuf,
    inv2: DevBuf,
    gate: DevBuf, // x2·W1
    up: DevBuf,   // x2·W3
    act: DevBuf,  // silu(gate) ⊙ up
    dq: DevBuf,
    dk: DevBuf,
    dv: DevBuf,
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
// Trainer
// ============================================================

pub struct GpuGpt {
    pub cfg: GptConfig,
    pub batch: usize,
    pub seq: usize,

    /// Token embedding - this is ALSO THE OUTPUT MATRIX (weight tying).
    tok_emb: Param,
    blocks: Vec<GpuBlock>,
    lnf_g: Param,
    /// Dropout probability during training (0 for evaluation/generation).
    pub dropout_p: f32,
    /// Seed for the dropout masks (updated every step).
    drop_seed: u32,
    /// Dropout is applied when true.
    pub training: bool,
    /// RoPE cos/sin tablosu: (seq × d_head/2)
    rope_cos: DevBuf,
    rope_sin: DevBuf,

    acts: Vec<Acts>,
    h: Vec<DevBuf>,
    dh: Vec<DevBuf>,

    lnf_out: DevBuf,
    xhat_f: DevBuf,
    inv_f: DevBuf,
    logits: DevBuf,
    probs: DevBuf,
    dlogits: DevBuf,
    dlnf: DevBuf,
    loss: DevBuf,

    tmp_a: DevBuf,
    tmp_b: DevBuf,
    tmp_c: DevBuf,
    tmp_fa: DevBuf,
    tmp_fb: DevBuf,
    tmp_fc: DevBuf,
    /// Attention scores in the forward pass, reused for dW in the backward pass.
    scores: DevBuf,
    dscores: DevBuf,
    #[allow(dead_code)]
    pub adam_t: usize,
}

fn mk_param(t: &crate::autograd::Tensor, name: &'static str) -> Param {
    let data = t.data.borrow();
    Param::from_host(data.rows, data.cols, name, &data.data)
}

/// Random initialization directly on the device, without a CPU weight twin.
/// (For a 250M model a CPU twin means 1 GB of RAM and upload time.)
fn seed_uniform(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let r = ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5;
        v.push(r * scale);
    }
    v
}

impl GpuGpt {
    /// Random initialization from scratch (no CPU twin). Same distribution as
    /// GPT::new: embeddings 0.02, projections at a 1/sqrt(d) scale.
    pub fn new_random(cfg: GptConfig, batch: usize, seq: usize, seed: u64) -> Self {
        let (v, d) = (cfg.vocab_size, cfg.d_model);
        let f = swiglu_hidden(d);
        let bs = batch * seq;
        let bh_seq = batch * cfg.n_heads * seq;
        let mut s = seed;

        let emb_scale = 0.02;
        let proj_scale = (1.0 / d as f32).sqrt();
        let ff_scale = (2.0 / d as f32).sqrt();

        let mut mk = |rows: usize, cols: usize, scale: f32, name: &'static str| {
            Param::from_host(rows, cols, name, &seed_uniform(rows * cols, scale, &mut s))
        };
        let _zeros = |rows: usize, cols: usize, name: &'static str| {
            Param::from_host(rows, cols, name, &vec![0.0; rows * cols])
        };
        let orig = Param::from_host;
        let _ = orig;

        let blocks: Vec<GpuBlock> = (0..cfg.n_layers)
            .map(|_| GpuBlock {
                norm1_g: Param::from_host(1, d, "norm1_g", &vec![1.0; d]),
                wq: mk(d, d, proj_scale, "wq"),
                wk: mk(d, d, proj_scale, "wk"),
                wv: mk(d, d, proj_scale, "wv"),
                wo: mk(d, d, proj_scale, "wo"),
                norm2_g: Param::from_host(1, d, "norm2_g", &vec![1.0; d]),
                ff_gate: mk(d, f, ff_scale, "ff_gate"),
                ff_up: mk(d, f, ff_scale, "ff_up"),
                ff_down: mk(f, d, proj_scale, "ff_down"),
            })
            .collect();

        let h: Vec<DevBuf> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let dh: Vec<DevBuf> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let acts: Vec<Acts> = (0..cfg.n_layers)
            .map(|_| Acts {
                x1: alloc(bs * d),
                xhat1: alloc(bs * d),
                inv1: alloc(bs),
                q: alloc(bs * d),
                k: alloc(bs * d),
                v: alloc(bs * d),
                w: alloc(bh_seq * seq),
                attn: alloc(bs * d),
                res1: alloc(bs * d),
                x2: alloc(bs * d),
                xhat2: alloc(bs * d),
                inv2: alloc(bs),
                gate: alloc(bs * f),
                up: alloc(bs * f),
                act: alloc(bs * f),
                dq: alloc(bs * d),
                dk: alloc(bs * d),
                dv: alloc(bs * d),
            })
            .collect();

        Self {
            cfg,
            batch,
            seq,
            tok_emb: mk(v, d, emb_scale, "tok_emb"),
            blocks,
            lnf_g: Param::from_host(1, d, "lnf_g", &vec![1.0; d]),
            dropout_p: 0.0,
            drop_seed: 1,
            training: false,
            rope_cos: {
                let (cc, _) = crate::gpu_kernels::rope_tables(seq, cfg.d_model / cfg.n_heads);
                upload(&cc)
            },
            rope_sin: {
                let (_, ss) = crate::gpu_kernels::rope_tables(seq, cfg.d_model / cfg.n_heads);
                upload(&ss)
            },
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
            tmp_fc: alloc(bs * f),
            scores: alloc(bh_seq * seq),
            dscores: alloc(bh_seq * seq),
            adam_t: 0,
        }
    }

    /// Builds from the CPU model's weights - both trainers start from the same point.
    pub fn from_cpu(model: &GPT, batch: usize, seq: usize) -> Self {
        let cfg = model.config;
        assert!(seq <= cfg.max_seq_len, "seq > max_seq_len");
        let (v, d) = (cfg.vocab_size, cfg.d_model);
        let f = swiglu_hidden(d);
        let bs = batch * seq;
        let bh_seq = batch * cfg.n_heads * seq;

        let blocks: Vec<GpuBlock> = model
            .blocks
            .iter()
            .map(|b| GpuBlock {
                norm1_g: mk_param(&b.norm1_gamma, "norm1_g"),
                wq: mk_param(&b.attn.w_q, "wq"),
                wk: mk_param(&b.attn.w_k, "wk"),
                wv: mk_param(&b.attn.w_v, "wv"),
                wo: mk_param(&b.attn.w_o, "wo"),
                norm2_g: mk_param(&b.norm2_gamma, "norm2_g"),
                ff_gate: mk_param(&b.ff_gate, "ff_gate"),
                ff_up: mk_param(&b.ff_up, "ff_up"),
                ff_down: mk_param(&b.ff_down, "ff_down"),
            })
            .collect();

        let h: Vec<DevBuf> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let dh: Vec<DevBuf> = (0..=cfg.n_layers).map(|_| alloc(bs * d)).collect();
        let acts: Vec<Acts> = (0..cfg.n_layers)
            .map(|_| Acts {
                x1: alloc(bs * d),
                xhat1: alloc(bs * d),
                inv1: alloc(bs),
                q: alloc(bs * d),
                k: alloc(bs * d),
                v: alloc(bs * d),
                w: alloc(bh_seq * seq),
                attn: alloc(bs * d),
                res1: alloc(bs * d),
                x2: alloc(bs * d),
                xhat2: alloc(bs * d),
                inv2: alloc(bs),
                gate: alloc(bs * f),
                up: alloc(bs * f),
                act: alloc(bs * f),
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
            blocks,
            lnf_g: mk_param(&model.ln_f_gamma, "lnf_g"),
            dropout_p: 0.0,
            drop_seed: 1,
            training: false,
            rope_cos: {
                let (cc, _) = crate::gpu_kernels::rope_tables(seq, cfg.d_model / cfg.n_heads);
                upload(&cc)
            },
            rope_sin: {
                let (_, ss) = crate::gpu_kernels::rope_tables(seq, cfg.d_model / cfg.n_heads);
                upload(&ss)
            },
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
            tmp_fc: alloc(bs * f),
            scores: alloc(bh_seq * seq),
            dscores: alloc(bh_seq * seq),
            adam_t: 0,
        }
    }

    /// Read-only accessors for the KV-cache generation path.
    pub fn tok_emb_ref(&self) -> &Param {
        &self.tok_emb
    }
    pub fn lnf_g_ref(&self) -> &DevBuf {
        &self.lnf_g.data
    }
    pub fn block_ref(&self, l: usize) -> &GpuBlock {
        &self.blocks[l]
    }
    pub fn rope_cos_ref(&self) -> &DevBuf {
        &self.rope_cos
    }
    pub fn rope_sin_ref(&self) -> &DevBuf {
        &self.rope_sin
    }

    pub fn dims(&self) -> Dims {
        Dims {
            v: self.cfg.vocab_size,
            d: self.cfg.d_model,
            f: swiglu_hidden(self.cfg.d_model),
            h: self.cfg.n_heads,
            b: self.batch,
            s: self.seq,
        }
    }

    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        let mut out: Vec<&mut Param> = vec![&mut self.tok_emb];
        for b in &mut self.blocks {
            out.push(&mut b.norm1_g);
            out.push(&mut b.wq);
            out.push(&mut b.wk);
            out.push(&mut b.wv);
            out.push(&mut b.wo);
            out.push(&mut b.norm2_g);
            out.push(&mut b.ff_gate);
            out.push(&mut b.ff_up);
            out.push(&mut b.ff_down);
        }
        out.push(&mut self.lnf_g);
        out
    }

    pub fn zero_grads(&mut self) {
        for p in self.params_mut() {
            p.zero_grad();
        }
    }

    /// Adam + global gradient clipping. Returns the norm before clipping.
    pub fn step(&mut self, lr: f32, clip: f32) -> f32 {
        self.adam_t += 1;
        let t = self.adam_t;

        let acc = alloc(1);
        for p in self.params_mut() {
            let n = p.n();
            k::sum_squares(&acc, &p.grad, n);
        }
        let norm = acc.to_host()[0].sqrt();
        if clip > 0.0 && norm > clip {
            let s = clip / norm;
            for p in self.params_mut() {
                let n = p.n();
                k::scale_inplace(&p.grad, n, s);
            }
        }
        for p in self.params_mut() {
            p.adam(lr, t);
        }
        norm
    }

    /// Forward + cross-entropy only: no gradients (the evaluation path).
    pub fn forward_loss(&mut self, ids: &[i32], targets: &[i32]) -> f32 {
        self.training = false; // evaluation: no dropout
        let dm = self.dims();
        assert_eq!(ids.len(), dm.bs());
        assert_eq!(targets.len(), dm.bs());
        let ids_dev = upload_i32(ids);
        let tgt_dev = upload_i32(targets);

        self.head_forward(&ids_dev, &dm);
        k::zero_fill(&self.loss, 1);
        k::ce_forward(
            &self.loss,
            &self.probs,
            &self.logits,
            &tgt_dev,
            dm.bs(),
            dm.v,
        );
        self.loss.to_host()[0]
    }

    /// Returns the logits of the given row. During generation the context is
    /// right-padded (causal attention keeps the positions unchanged), so this
    /// reads the row of the last real token.
    pub fn forward_logits_at(&mut self, ids: &[i32], row: usize) -> Vec<f32> {
        self.training = false; // generation: no dropout
        let dm = self.dims();
        assert_eq!(ids.len(), dm.bs(), "ids.len() must equal batch*seq");
        let ids_dev = upload_i32(ids);

        self.head_forward(&ids_dev, &dm);

        // the requested row's logits
        let all = self.logits.to_host();
        let last = row * dm.v;
        all[last..last + dm.v].to_vec()
    }

    /// Body forward pass: embedding -> layers -> final RMSNorm -> logits (tied).
    fn head_forward(&mut self, ids_dev: &DevBufI32, dm: &Dims) {
        k::embedding_forward_tok(&self.h[0], ids_dev, &self.tok_emb.data, dm.bs(), dm.d);
        for l in 0..self.cfg.n_layers {
            self.block_forward(l, dm);
        }
        k::rmsnorm_forward(
            &self.lnf_out,
            &self.xhat_f,
            &self.inv_f,
            &self.h[self.cfg.n_layers],
            &self.lnf_g.data,
            dm.bs(),
            dm.d,
            RMS_EPS,
        );
        // weight tying: logits = lnf_out · tok_embᵀ
        gpu::dev_gemm(
            &mut self.logits,
            &self.lnf_out,
            &self.tok_emb.data,
            dm.bs(),
            dm.v,
            dm.d,
            dm.d,
            dm.d,
            dm.v,
            GemmMode::Nt,
        );
    }

    /// `seq` must be 1 to compute the logits of a single token.
    pub fn seq_len(&self) -> usize {
        self.seq
    }

    /// Write the weights to disk (for resume). No CPU twin needed.
    /// Format: magic, version, config (5 x u32), adam_t (u64), tensor count,
    ///         then rows/cols + f32 data for every tensor.
    ///
    /// **Atomic write:** data is first written to `path.tmp`, synced to disk,
    /// and only then moved over `path`. If the power fails during the write
    /// the previous checkpoint stays intact (otherwise a half-written 848 MB
    /// file would be unrecoverable).
    pub fn save_state(&mut self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let cfg = self.cfg;
        let tmp = format!("{path}.tmp");
        let f = std::fs::File::create(&tmp)?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(b"SFEN001\0")?;
        w.write_all(&1u32.to_le_bytes())?;
        for v in [
            cfg.vocab_size,
            cfg.max_seq_len,
            cfg.d_model,
            cfg.n_layers,
            cfg.n_heads,
        ] {
            w.write_all(&(v as u32).to_le_bytes())?;
        }
        w.write_all(&(self.adam_t as u64).to_le_bytes())?;
        let params = self.params_mut();
        w.write_all(&(params.len() as u32).to_le_bytes())?;
        for p in params {
            w.write_all(&(p.rows as u32).to_le_bytes())?;
            w.write_all(&(p.cols as u32).to_le_bytes())?;
            let host = p.download();
            let mut bytes = Vec::with_capacity(host.len() * 4);
            for v in host {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            w.write_all(&bytes)?;
        }
        w.flush()?;
        // Flush to the physical disk, then move atomically into place.
        w.get_ref().sync_all()?;
        drop(w);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Loads weights written by `save_state`; returns `adam_t`.
    /// Errors out on a config mismatch (no silent corruption).
    pub fn load_state(&mut self, path: &str) -> std::io::Result<usize> {
        use std::io::Read;
        let f = std::fs::File::open(path)?;
        let mut r = std::io::BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"SFEN001\0" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "wrong magic - this is not an SFEN weight file",
            ));
        }
        let mut u4 = [0u8; 4];
        r.read_exact(&mut u4)?;
        let ver = u32::from_le_bytes(u4);
        if ver != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported version: {ver}"),
            ));
        }
        let mut cfg = [0usize; 5];
        for c in cfg.iter_mut() {
            r.read_exact(&mut u4)?;
            *c = u32::from_le_bytes(u4) as usize;
        }
        let mine = self.cfg;
        let want = [
            mine.vocab_size,
            mine.max_seq_len,
            mine.d_model,
            mine.n_layers,
            mine.n_heads,
        ];
        if cfg != want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("config mismatch: file {cfg:?}, model {want:?}"),
            ));
        }
        let mut u8b = [0u8; 8];
        r.read_exact(&mut u8b)?;
        let adam_t = u64::from_le_bytes(u8b) as usize;
        r.read_exact(&mut u4)?;
        let n = u32::from_le_bytes(u4) as usize;
        let params = self.params_mut();
        if n != params.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("tensor count mismatch: file {n}, model {}", params.len()),
            ));
        }
        for p in params {
            r.read_exact(&mut u4)?;
            let rows = u32::from_le_bytes(u4) as usize;
            r.read_exact(&mut u4)?;
            let cols = u32::from_le_bytes(u4) as usize;
            if (rows, cols) != (p.rows, p.cols) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "{}: shape mismatch (file {rows}x{cols}, model {}x{})",
                        p.name, p.rows, p.cols
                    ),
                ));
            }
            let mut bytes = vec![0u8; rows * cols * 4];
            r.read_exact(&mut bytes)?;
            let host: Vec<f32> = bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            p.data = upload(&host);
            let n = p.n();
            k::zero_fill(&p.grad, n);
        }
        self.adam_t = adam_t;
        Ok(adam_t)
    }

    // ========================================================
    // Forward + backward
    // ========================================================

    pub fn forward_backward(&mut self, ids: &[i32], targets: &[i32]) -> f32 {
        self.forward_backward_with(ids, targets, true, 1.0)
    }

    /// Gradient accumulation support: with `zero_grads=false` consecutive calls
    /// sum the gradients; `loss_scale` scales each micro-batch's contribution
    /// (use 1/accum for the mean over the effective batch).
    pub fn forward_backward_with(
        &mut self,
        ids: &[i32],
        targets: &[i32],
        zero_grads: bool,
        loss_scale: f32,
    ) -> f32 {
        let dm = self.dims();
        assert_eq!(ids.len(), dm.bs());
        assert_eq!(targets.len(), dm.bs());
        // fresh dropout masks every step
        self.drop_seed = self.drop_seed.wrapping_add(0x9E37_79B9);
        self.training = true;

        let ids_dev = upload_i32(ids);
        let tgt_dev = upload_i32(targets);
        if zero_grads {
            self.zero_grads();
        }

        // ---- forward ----
        self.head_forward(&ids_dev, &dm);
        k::zero_fill(&self.loss, 1);
        k::ce_forward(
            &self.loss,
            &self.probs,
            &self.logits,
            &tgt_dev,
            dm.bs(),
            dm.v,
        );
        let loss = self.loss.to_host()[0];

        // ---- backward ----
        k::ce_backward(
            &self.dlogits,
            &self.probs,
            &tgt_dev,
            dm.bs(),
            dm.v,
            loss_scale,
        );

        // --- weight tying: dtok_emb += dlogitsᵀ·lnf_out (V,D) ---
        gpu::dev_gemm(
            &mut self.tok_emb.grad,
            &self.dlogits,
            &self.lnf_out,
            dm.v,
            dm.d,
            dm.bs(),
            dm.v,
            dm.d,
            dm.d,
            GemmMode::Tn,
        );
        // dlnf = dlogits·tok_emb (BS,D)
        gpu::dev_gemm(
            &mut self.dlnf,
            &self.dlogits,
            &self.tok_emb.data,
            dm.bs(),
            dm.d,
            dm.v,
            dm.v,
            dm.d,
            dm.d,
            GemmMode::Nn,
        );

        {
            let g = alloc(dm.d);
            k::rmsnorm_param_grad(&g, &self.dlnf, &self.xhat_f, dm.bs(), dm.d);
            k::add_inplace(&self.lnf_g.grad, &g, dm.d);
            k::rmsnorm_backward(
                &self.dh[self.cfg.n_layers],
                &self.dlnf,
                &self.xhat_f,
                &self.inv_f,
                &self.lnf_g.data,
                dm.bs(),
                dm.d,
            );
        }

        for l in (0..self.cfg.n_layers).rev() {
            self.block_backward(l, &dm);
        }

        k::embedding_backward_tok(&self.tok_emb.grad, &self.dh[0], &ids_dev, dm.bs(), dm.d);

        loss
    }

    fn block_forward(&mut self, l: usize, dm: &Dims) {
        // Disjoint field borrows: split the fields instead of one `&mut self`.
        let Self {
            acts,
            blocks,
            h,
            scores,
            tmp_a,
            tmp_b,
            dropout_p,
            drop_seed,
            training,
            rope_cos,
            rope_sin,
            ..
        } = self;
        let (bs, d, f, s, nh, dhh, scale) = (dm.bs(), dm.d, dm.f, dm.s, dm.h, dm.dh(), dm.scale());
        let a = &mut acts[l];
        let blk = &blocks[l];
        let (h_lo, h_hi) = h.split_at_mut(l + 1);
        let x_in = &h_lo[l];
        let out = &mut h_hi[0];

        // A different dropout mask per layer (the same one is regenerated in backward).
        let base = *drop_seed ^ (l as u32).wrapping_mul(0x9E37_79B1);
        let (seed_proj, seed_ffn) = (base, base ^ 0x5BF0_3635);
        let drop = if *training && *dropout_p > 0.0 {
            *dropout_p
        } else {
            0.0
        };

        // --- RMSNorm 1 ---
        k::rmsnorm_forward(
            &a.x1,
            &a.xhat1,
            &a.inv1,
            x_in,
            &blk.norm1_g.data,
            bs,
            d,
            RMS_EPS,
        );

        // --- q, k, v (no bias) ---
        gpu::dev_gemm(
            &mut a.q,
            &a.x1,
            &blk.wq.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nn,
        );
        gpu::dev_gemm(
            &mut a.k,
            &a.x1,
            &blk.wk.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nn,
        );
        gpu::dev_gemm(
            &mut a.v,
            &a.x1,
            &blk.wv.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nn,
        );

        // --- RoPE: only Q and K are rotated (in-place is safe) ---
        k::rope_forward(&a.q, &a.q, rope_cos, rope_sin, dm.b, s, nh, dhh);
        k::rope_forward(&a.k, &a.k, rope_cos, rope_sin, dm.b, s, nh, dhh);

        // --- scores (per head, strided over batch) ---
        for hh in 0..nh {
            let off = hh * dhh;
            let q_view = a.q.slice(off..);
            let k_view = a.k.slice(off..);
            let mut sc = scores.slice_mut(hh * dm.b * s * s..);
            gpu::dev_gemm_batched(
                &mut sc,
                &q_view,
                &k_view,
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
        k::masked_softmax_forward(&a.w, scores, dm.b, nh, s, scale);

        // --- attn = W·V ---
        for hh in 0..nh {
            let off = hh * dhh;
            let w_view = a.w.slice(hh * dm.b * s * s..);
            let v_view = a.v.slice(off..);
            let mut av = a.attn.slice_mut(off..);
            gpu::dev_gemm_batched(
                &mut av,
                &w_view,
                &v_view,
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

        // --- proj = attn·wo  (+ dropout) ---
        gpu::dev_gemm(
            tmp_a,
            &a.attn,
            &blk.wo.data,
            bs,
            d,
            d,
            d,
            d,
            d,
            GemmMode::Nn,
        );
        if drop > 0.0 {
            k::dropout_inplace(tmp_a, bs * d, drop, seed_proj);
        }

        // --- res1 = x + proj ---
        dtod(&mut a.res1, x_in);
        k::add_inplace(&a.res1, tmp_a, bs * d);

        // --- RMSNorm 2 ---
        k::rmsnorm_forward(
            &a.x2,
            &a.xhat2,
            &a.inv2,
            &a.res1,
            &blk.norm2_g.data,
            bs,
            d,
            RMS_EPS,
        );

        // --- SwiGLU FFN: silu(x·W1) ⊙ (x·W3) ·W2 ---
        gpu::dev_gemm(
            &mut a.gate,
            &a.x2,
            &blk.ff_gate.data,
            bs,
            f,
            d,
            d,
            f,
            f,
            GemmMode::Nn,
        );
        gpu::dev_gemm(
            &mut a.up,
            &a.x2,
            &blk.ff_up.data,
            bs,
            f,
            d,
            d,
            f,
            f,
            GemmMode::Nn,
        );
        k::silu_mul_forward(&a.act, &a.gate, &a.up, bs * f);

        gpu::dev_gemm(
            tmp_b,
            &a.act,
            &blk.ff_down.data,
            bs,
            d,
            f,
            f,
            d,
            d,
            GemmMode::Nn,
        );
        if drop > 0.0 {
            k::dropout_inplace(tmp_b, bs * d, drop, seed_ffn);
        }

        // --- out = res1 + ffn ---
        dtod(out, &a.res1);
        k::add_inplace(out, tmp_b, bs * d);
    }

    fn block_backward(&mut self, l: usize, dm: &Dims) {
        let Self {
            acts,
            blocks,
            dh: dh_vec,
            scores,
            dscores,
            tmp_a,
            tmp_b,
            tmp_c,
            tmp_fa,
            tmp_fb,
            tmp_fc,
            dropout_p,
            drop_seed,
            training,
            rope_cos,
            rope_sin,
            ..
        } = self;
        let (bs, d, f, s, nh, dhh, scale) = (dm.bs(), dm.d, dm.f, dm.s, dm.h, dm.dh(), dm.scale());
        let a = &mut acts[l];
        let blk = &mut blocks[l];
        let (dh_lo, dh_hi) = dh_vec.split_at_mut(l + 1);
        let dout = dh_hi[0].full();

        let base = *drop_seed ^ (l as u32).wrapping_mul(0x9E37_79B1);
        let (seed_proj, seed_ffn) = (base, base ^ 0x5BF0_3635);
        let drop = if *training && *dropout_p > 0.0 {
            *dropout_p
        } else {
            0.0
        };

        // ================= FFN branch =================
        // dfn = dOut (same dropout mask) -> tmp_b
        dtod(tmp_b, &dout);
        if drop > 0.0 {
            k::dropout_backward(tmp_b, bs * d, drop, seed_ffn);
        }
        // dact = dfn·ff_downᵀ (BS,F) → tmp_fa
        gpu::dev_gemm(
            tmp_fa,
            tmp_b,
            &blk.ff_down.data,
            bs,
            f,
            d,
            d,
            d,
            f,
            GemmMode::Nt,
        );
        // dff_down = actᵀ·dfn (F,D)
        gpu::dev_gemm(
            &mut blk.ff_down.grad,
            &a.act,
            tmp_b,
            f,
            d,
            bs,
            f,
            d,
            d,
            GemmMode::Tn,
        );
        // SwiGLU backward: dgate -> tmp_fb, dup -> tmp_fc
        k::silu_mul_backward(tmp_fb, tmp_fc, tmp_fa, &a.gate, &a.up, bs * f);

        // dff_gate = x2ᵀ·dgate (D,F) ; dff_up = x2ᵀ·dup
        gpu::dev_gemm(
            &mut blk.ff_gate.grad,
            &a.x2,
            tmp_fb,
            d,
            f,
            bs,
            d,
            f,
            f,
            GemmMode::Tn,
        );
        gpu::dev_gemm(
            &mut blk.ff_up.grad,
            &a.x2,
            tmp_fc,
            d,
            f,
            bs,
            d,
            f,
            f,
            GemmMode::Tn,
        );

        // dx2 = dgate·ff_gateᵀ + dup·ff_upᵀ (BS,D) → tmp_b
        gpu::dev_gemm(
            tmp_b,
            tmp_fb,
            &blk.ff_gate.data,
            bs,
            d,
            f,
            f,
            f,
            d,
            GemmMode::Nt,
        );
        gpu::dev_gemm_batched(
            tmp_b,
            tmp_fc,
            &blk.ff_up.data,
            bs,
            d,
            f,
            f,
            f,
            d,
            0,
            0,
            0,
            1,
            1.0,
            GemmMode::Nt,
        );

        // ---- RMSNorm 2 backward ----
        {
            let g = alloc(d);
            k::rmsnorm_param_grad(&g, tmp_b, &a.xhat2, bs, d);
            k::add_inplace(&blk.norm2_g.grad, &g, d);
        }
        k::rmsnorm_backward(tmp_c, tmp_b, &a.xhat2, &a.inv2, &blk.norm2_g.data, bs, d);

        // dh[l] = dOut (residual) + dx_rmsnorm2 -> upstream of the attention branch
        dtod(&mut dh_lo[l], &dout);
        k::add_inplace(&dh_lo[l], tmp_c, bs * d);

        // ================= attention branch =================
        // dattn = dh[l]*wo^T (BS,D) -> tmp_a   (+ the same dropout mask)
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
        if drop > 0.0 {
            k::dropout_backward(tmp_a, bs * d, drop, seed_proj);
        }
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
        // dW = dAttn*V^T -> scores (reused buffer)
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

        // RoPE backward: dQ/dK are rotated back (Q, K were rotated in the forward pass)
        k::rope_backward(&a.dq, &a.dq, rope_cos, rope_sin, dm.b, s, nh, dhh);
        k::rope_backward(&a.dk, &a.dk, rope_cos, rope_sin, dm.b, s, nh, dhh);

        // dln1 = dQ*wq^T + dK*wk^T + dV*wv^T (accumulated with beta) -> tmp_b
        gpu::dev_gemm(tmp_b, &a.dq, &blk.wq.data, bs, d, d, d, d, d, GemmMode::Nt);
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

        // dWq/dWk/dWv = x1ᵀ·d{Q,K,V}
        gpu::dev_gemm(
            &mut blk.wq.grad,
            &a.x1,
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
            &a.x1,
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
            &a.x1,
            &a.dv,
            d,
            d,
            bs,
            d,
            d,
            d,
            GemmMode::Tn,
        );

        // ---- RMSNorm 1 backward ----
        {
            let g = alloc(d);
            k::rmsnorm_param_grad(&g, tmp_b, &a.xhat1, bs, d);
            k::add_inplace(&blk.norm1_g.grad, &g, d);
        }
        k::rmsnorm_backward(tmp_c, tmp_b, &a.xhat1, &a.inv1, &blk.norm1_g.data, bs, d);
        k::add_inplace(&dh_lo[l], tmp_c, bs * d);
    }
}

impl GpuGpt {
    /// For tests/diagnostics: a parameter gradient by name.
    pub fn grad_of(&mut self, name: &str, block: usize) -> Option<Vec<f32>> {
        for p in self.params_mut() {
            if p.name == name && block == usize::MAX {
                return Some(p.download_grad());
            }
        }
        let blk = &self.blocks[block];
        let p = match name {
            "norm1_g" => &blk.norm1_g,
            "wq" => &blk.wq,
            "wk" => &blk.wk,
            "wv" => &blk.wv,
            "wo" => &blk.wo,
            "norm2_g" => &blk.norm2_g,
            "ff_gate" => &blk.ff_gate,
            "ff_up" => &blk.ff_up,
            "ff_down" => &blk.ff_down,
            _ => return None,
        };
        Some(p.download_grad())
    }

    /// Writes the device weights back into the CPU model (for checkpoints -
    /// the checkpoint format is defined in terms of the CPU model).
    pub fn copy_to_cpu(&self, model: &mut GPT) {
        let put = |t: &crate::autograd::Tensor, p: &Param| {
            let host = p.download();
            let mut d = t.data.borrow_mut();
            assert_eq!(d.data.len(), host.len());
            d.data.copy_from_slice(&host);
        };
        put(&model.embedding.token_emb, &self.tok_emb);
        for (bi, blk) in model.blocks.iter().enumerate() {
            let gb = &self.blocks[bi];
            put(&blk.norm1_gamma, &gb.norm1_g);
            put(&blk.attn.w_q, &gb.wq);
            put(&blk.attn.w_k, &gb.wk);
            put(&blk.attn.w_v, &gb.wv);
            put(&blk.attn.w_o, &gb.wo);
            put(&blk.norm2_gamma, &gb.norm2_g);
            put(&blk.ff_gate, &gb.ff_gate);
            put(&blk.ff_up, &gb.ff_up);
            put(&blk.ff_down, &gb.ff_down);
        }
        // weight tying: no separate output matrix (tok_emb is shared)
        put(&model.ln_f_gamma, &self.lnf_g);
    }
}
