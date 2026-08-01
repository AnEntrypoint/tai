use rayon::prelude::*;
use std::time::Instant;

use crate::kernels;
use crate::model::{Int8Head, Model, Qt};

#[derive(Default)]
pub struct Profile {
    pub input_ns: u128,
    pub attn_ns: u128,
    pub ffn_ns: u128,
    pub ple_ns: u128,
    pub head_ns: u128,
    pub calls: u64,
}

impl Profile {
    pub fn reset(&mut self) {
        *self = Profile::default();
    }

    pub fn report(&self) -> String {
        if self.calls == 0 {
            return "profile: no forward calls".to_string();
        }
        let ms = 1e6 * self.calls as f64;
        format!(
            "profile ms/token: input {:.1} | attn {:.1} | ffn {:.1} | ple {:.1} | head {:.1}",
            self.input_ns as f64 / ms,
            self.attn_ns as f64 / ms,
            self.ffn_ns as f64 / ms,
            self.ple_ns as f64 / ms,
            self.head_ns as f64 / ms,
        )
    }
}

pub struct Runtime {
    x: Vec<f32>,
    h: Vec<f32>,
    qkv: Vec<f32>,
    att: Vec<f32>,
    g1: Vec<f32>,
    g2: Vec<f32>,
    ple: Vec<f32>,
    tmp_p: Vec<f32>,
    trow: Vec<f32>,
    logits: Vec<f32>,
    scores: Vec<f32>,
    kcache: Vec<f32>,
    vcache: Vec<f32>,
    rope_c: Vec<f32>,
    rope_s: Vec<f32>,
    head8: Option<Int8Head>,
    xq: Vec<i8>,
    scaled: Vec<f32>,
    order: Vec<f32>,
    last_argmax: usize,
    pos: usize,
    head_rows: usize,
    pub profiling: bool,
    pub profile: Profile,
}

impl Runtime {
    pub fn new(m: &Model, head_rows: usize, int8_head: bool) -> Runtime {
        let c = &m.cfg;
        let dh = c.dim / c.n_heads;
        let half = dh / 2;
        let mut rope_c = vec![0.0; c.seq_len * half];
        let mut rope_s = vec![0.0; c.seq_len * half];
        for pos in 0..c.seq_len {
            for i in 0..half {
                let freq = c.rope_theta.powf(-2.0 * i as f32 / dh as f32);
                rope_c[pos * half + i] = (pos as f32 * freq).cos();
                rope_s[pos * half + i] = (pos as f32 * freq).sin();
            }
        }
        let head8 = if int8_head {
            Int8Head::stage(&m.tok_emb).ok()
        } else {
            None
        };
        Runtime {
            x: vec![0.0; c.dim],
            h: vec![0.0; c.ffn.max(c.dim)],
            qkv: vec![0.0; 3 * c.dim],
            att: vec![0.0; c.dim],
            g1: vec![0.0; c.ffn],
            g2: vec![0.0; c.ple_dim.max(c.ffn)],
            ple: vec![0.0; c.n_layers * c.ple_dim],
            tmp_p: vec![0.0; c.n_layers * c.ple_dim],
            trow: vec![0.0; c.n_layers * c.ple_dim],
            logits: vec![0.0; head_rows],
            scores: vec![0.0; c.seq_len],
            kcache: vec![0.0; c.n_layers * c.seq_len * c.dim],
            vcache: vec![0.0; c.n_layers * c.seq_len * c.dim],
            rope_c,
            rope_s,
            head8,
            xq: vec![0; c.dim],
            scaled: Vec::with_capacity(head_rows),
            order: Vec::with_capacity(head_rows),
            last_argmax: 0,
            pos: 0,
            head_rows,
            profiling: false,
            profile: Profile::default(),
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn argmax(&self) -> usize {
        self.last_argmax
    }

    pub fn int8_head(&self) -> bool {
        self.head8.is_some()
    }

    pub fn logits_and_scratch(&mut self) -> (&[f32], &mut Vec<f32>, &mut Vec<f32>) {
        (&self.logits, &mut self.scaled, &mut self.order)
    }

    pub fn forward(&mut self, m: &Model, token: usize, pool: &rayon::ThreadPool, avx2: bool) {
        let c = &m.cfg;
        let (d, l, p, f) = (c.dim, c.n_layers, c.ple_dim, c.ffn);
        let (nh, dh, s) = (c.n_heads, c.dim / c.n_heads, c.seq_len);
        let half = dh / 2;
        let pos = self.pos;
        let profiling = self.profiling;
        let mut t0 = Instant::now();

        let Self {
            x, h, qkv, att, g1, g2, ple, tmp_p, trow, logits, scores, kcache, vcache,
            rope_c, rope_s, head8, xq, last_argmax, pos: rpos, head_rows,
            profiling: _, profile, ..
        } = self;

        kernels::deq_row(&m.tok_emb, token, x);
        matvec(&m.ple_model_proj, x, tmp_p, avx2);
        let dscale = 1.0 / (d as f32).sqrt();
        for v in tmp_p.iter_mut() {
            *v *= dscale;
        }
        for li in 0..l {
            kernels::rmsnorm_ip(&mut tmp_p[li * p..(li + 1) * p], m.ple_proj_norm);
        }
        kernels::deq_row(&m.ple_table, token, trow);
        let sp = (p as f32).sqrt();
        for i in 0..l * p {
            ple[i] = (tmp_p[i] + trow[i] * sp) * std::f32::consts::FRAC_1_SQRT_2;
        }
        let rc = &rope_c[pos * half..(pos + 1) * half];
        let rs = &rope_s[pos * half..(pos + 1) * half];
        if profiling {
            profile.input_ns += t0.elapsed().as_nanos();
            t0 = Instant::now();
        }

        for li in 0..l {
            let lw = &m.layers[li];
            kernels::rmsnorm_into(x, lw.attn_norm, h);
            matvec(&lw.qkv, h, qkv, avx2);
            let (qv, rest) = qkv.split_at_mut(d);
            let (kv, vv) = rest.split_at_mut(d);
            for hh in 0..nh {
                for i in 0..half {
                    let (c0, s0) = (rc[i], rs[i]);
                    let (a, b) = (hh * dh + i, hh * dh + i + half);
                    let (q1, q2) = (qv[a], qv[b]);
                    qv[a] = q1 * c0 - q2 * s0;
                    qv[b] = q2 * c0 + q1 * s0;
                    let (k1, k2) = (kv[a], kv[b]);
                    kv[a] = k1 * c0 - k2 * s0;
                    kv[b] = k2 * c0 + k1 * s0;
                }
            }
            let kbase = li * s * d;
            kcache[kbase + pos * d..kbase + (pos + 1) * d].copy_from_slice(kv);
            vcache[kbase + pos * d..kbase + (pos + 1) * d].copy_from_slice(vv);
            let kc = &kcache[kbase..kbase + s * d];
            let vc = &vcache[kbase..kbase + s * d];
            let scale = 1.0 / (dh as f32).sqrt();
            for hh in 0..nh {
                let qh = &qv[hh * dh..(hh + 1) * dh];
                let ao = &mut att[hh * dh..(hh + 1) * dh];
                for a in ao.iter_mut() {
                    *a = 0.0;
                }
                let mut maxs = -1e30f32;
                for t in 0..=pos {
                    let kt = &kc[t * d + hh * dh..t * d + hh * dh + dh];
                    let mut dot = 0.0f32;
                    for i in 0..dh {
                        dot += qh[i] * kt[i];
                    }
                    dot *= scale;
                    scores[t] = dot;
                    if dot > maxs {
                        maxs = dot;
                    }
                }
                let mut denom = 0.0f32;
                for t in 0..=pos {
                    let w = (scores[t] - maxs).exp();
                    denom += w;
                    let vt = &vc[t * d + hh * dh..t * d + hh * dh + dh];
                    for i in 0..dh {
                        ao[i] += w * vt[i];
                    }
                }
                for a in ao.iter_mut() {
                    *a /= denom;
                }
            }
            matvec(&lw.attn_proj, att, h, avx2);
            for i in 0..d {
                x[i] += h[i];
            }
            if profiling {
                profile.attn_ns += t0.elapsed().as_nanos();
                t0 = Instant::now();
            }

            kernels::rmsnorm_into(x, lw.ffn_norm, h);
            matvec(&lw.gate, h, g1, avx2);
            matvec(&lw.up, h, g2, avx2);
            for i in 0..f {
                g1[i] = kernels::silu(g1[i]) * g2[i];
            }
            matvec(&lw.down, g1, h, avx2);
            for i in 0..d {
                x[i] += h[i];
            }
            if profiling {
                profile.ffn_ns += t0.elapsed().as_nanos();
                t0 = Instant::now();
            }

            matvec(&lw.ple_gate, x, g2, avx2);
            for i in 0..p {
                g2[i] = kernels::gelu(g2[i]) * ple[li * p + i];
            }
            matvec(&lw.ple_proj, g2, h, avx2);
            kernels::rmsnorm_ip(&mut h[..d], lw.ple_norm);
            for i in 0..d {
                x[i] += h[i];
            }
            if profiling {
                profile.ple_ns += t0.elapsed().as_nanos();
                t0 = Instant::now();
            }
        }

        kernels::rmsnorm_ip(x, m.out_norm);
        match head8 {
            Some(h8) => {
                let xs = kernels::quantize_act(x, xq);
                let asum = kernels::act_sum(xq);
                *last_argmax =
                    matvec_head_int8(h8, xq, asum, xs, logits, *head_rows, pool, avx2);
            }
            None => {
                *last_argmax = matvec_head(&m.tok_emb, x, logits, *head_rows, pool, avx2);
            }
        }
        if profiling {
            profile.head_ns += t0.elapsed().as_nanos();
            profile.calls += 1;
        }
        *rpos += 1;
    }
}

fn matvec(q: &Qt, x: &[f32], y: &mut [f32], avx2: bool) {
    for r in 0..q.rows {
        y[r] = kernels::matvec_row(q, x, r, avx2);
    }
}

#[inline]
fn head8_row(h: &Int8Head, xq: &[i8], asum: i32, xs: f32, r: usize, avx2: bool) -> f32 {
    let w = &h.w[r * h.cols..(r + 1) * h.cols];
    let d = kernels::dot_u8_i8(w, xq, avx2);
    (d - 8 * asum) as f32 * h.scale[r] * xs
}

fn matvec_head_int8(
    h: &Int8Head,
    xq: &[i8],
    asum: i32,
    xs: f32,
    y: &mut [f32],
    rows: usize,
    pool: &rayon::ThreadPool,
    avx2: bool,
) -> usize {
    let threads = pool.current_num_threads();
    if threads < 2 || rows * h.cols < (1 << 18) {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0;
        for r in 0..rows {
            let v = head8_row(h, xq, asum, xs, r, avx2);
            y[r] = v;
            if v > bv {
                bv = v;
                bi = r;
            }
        }
        return bi;
    }
    let chunk = rows.div_ceil(threads * 4).max(1);
    pool.install(|| {
        y[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, ch)| {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0;
                for (k, slot) in ch.iter_mut().enumerate() {
                    let r = ci * chunk + k;
                    let v = head8_row(h, xq, asum, xs, r, avx2);
                    *slot = v;
                    if v > bv {
                        bv = v;
                        bi = r;
                    }
                }
                (bv, bi)
            })
            .reduce(|| (f32::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a })
            .1
    })
}

fn matvec_head(
    q: &Qt,
    x: &[f32],
    y: &mut [f32],
    rows: usize,
    pool: &rayon::ThreadPool,
    avx2: bool,
) -> usize {
    let threads = pool.current_num_threads();
    if threads < 2 || rows * q.cols < (1 << 18) {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0;
        for r in 0..rows {
            let v = kernels::matvec_row(q, x, r, avx2);
            y[r] = v;
            if v > bv {
                bv = v;
                bi = r;
            }
        }
        return bi;
    }
    let chunk = rows.div_ceil(threads * 4).max(1);
    pool.install(|| {
        y[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, ch)| {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0;
                for (k, slot) in ch.iter_mut().enumerate() {
                    let r = ci * chunk + k;
                    let v = kernels::matvec_row(q, x, r, avx2);
                    *slot = v;
                    if v > bv {
                        bv = v;
                        bi = r;
                    }
                }
                (bv, bi)
            })
            .reduce(|| (f32::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a })
            .1
    })
}
