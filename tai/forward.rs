use rayon::prelude::*;
use std::time::Instant;

use crate::kernels;
use crate::model::{Int8Mat, Model, Qt};

#[derive(Default)]
pub struct Profile {
    pub input_ns: u128,
    pub attn_ns: u128,
    pub ffn_ns: u128,
    pub ple_ns: u128,
    pub head_ns: u128,
    pub matvec_ns: u128,
    pub deq_ns: u128,
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
            "profile ms/token: input {:.2} | attn {:.2} | ffn {:.2} | ple {:.2} | head {:.2} | matvec {:.2} | deq {:.2}",
            self.input_ns as f64 / ms,
            self.attn_ns as f64 / ms,
            self.ffn_ns as f64 / ms,
            self.ple_ns as f64 / ms,
            self.head_ns as f64 / ms,
            self.matvec_ns as f64 / ms,
            self.deq_ns as f64 / ms,
        )
    }
}


pub struct Staged {
    ple_model_proj: Int8Mat,
    qkv: Vec<Int8Mat>,
    attn_proj: Vec<Int8Mat>,
    gate: Vec<Int8Mat>,
    up: Vec<Int8Mat>,
    down: Vec<Int8Mat>,
    ple_gate: Vec<Int8Mat>,
    ple_proj: Vec<Int8Mat>,
}

impl Staged {
    fn build(m: &Model) -> Result<Staged, String> {
        let l = m.layers.len();
        let mut qkv = Vec::with_capacity(l);
        let mut attn_proj = Vec::with_capacity(l);
        let mut gate = Vec::with_capacity(l);
        let mut up = Vec::with_capacity(l);
        let mut down = Vec::with_capacity(l);
        let mut ple_gate = Vec::with_capacity(l);
        let mut ple_proj = Vec::with_capacity(l);
        for lw in &m.layers {
            qkv.push(Int8Mat::stage(&lw.qkv)?);
            attn_proj.push(Int8Mat::stage(&lw.attn_proj)?);
            gate.push(Int8Mat::stage(&lw.gate)?);
            up.push(Int8Mat::stage(&lw.up)?);
            down.push(Int8Mat::stage(&lw.down)?);
            ple_gate.push(Int8Mat::stage(&lw.ple_gate)?);
            ple_proj.push(Int8Mat::stage(&lw.ple_proj)?);
        }
        Ok(Staged {
            ple_model_proj: Int8Mat::stage(&m.ple_model_proj)?,
            qkv,
            attn_proj,
            gate,
            up,
            down,
            ple_gate,
            ple_proj,
        })
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
    denoms: Vec<f32>,
    kcache: Vec<f32>,
    vcache: Vec<f32>,
    rope_c: Vec<f32>,
    rope_s: Vec<f32>,
    head8: Option<Int8Mat>,
    pub i4_head: bool,
    staged: Option<Staged>,
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
        let (head8, staged) = if int8_head {
            (Int8Mat::stage(&m.tok_emb).ok(), Staged::build(m).ok())
        } else {
            (None, None)
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
            scores: vec![0.0; c.n_heads * c.seq_len],
            denoms: vec![0.0; c.n_heads],
            kcache: vec![0.0; c.n_layers * c.seq_len * c.dim],
            vcache: vec![0.0; c.n_layers * c.seq_len * c.dim],
            rope_c,
            rope_s,
            head8,
            i4_head: false,
            staged,
            xq: vec![0; c.dim.max(c.ffn).max(c.ple_dim)],
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
            rope_c, rope_s, head8, i4_head, staged, xq, denoms, last_argmax,
            pos: rpos, head_rows, profiling: _, profile, ..
        } = self;

        deq_timed(&m.tok_emb, token, x, &mut profile.deq_ns, profiling);
        matvec_any(staged.as_ref().map(|s| &s.ple_model_proj), &m.ple_model_proj, x, xq, tmp_p, avx2, &mut profile.matvec_ns, profiling);
        let dscale = 1.0 / (d as f32).sqrt();
        for v in tmp_p.iter_mut() {
            *v *= dscale;
        }
        for li in 0..l {
            kernels::rmsnorm_ip(&mut tmp_p[li * p..(li + 1) * p], m.ple_proj_norm);
        }
        deq_timed(&m.ple_table, token, trow, &mut profile.deq_ns, profiling);
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
            matvec_any(staged.as_ref().map(|s| &s.qkv[li]), &lw.qkv, h, xq, qkv, avx2, &mut profile.matvec_ns, profiling);
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
            for t in 0..=pos {
                let kt = &kc[t * d..t * d + d];
                for hh in 0..nh {
                    scores[hh * s + t] = kernels::dot_f32(
                        &qv[hh * dh..(hh + 1) * dh],
                        &kt[hh * dh..(hh + 1) * dh],
                        avx2,
                    ) * scale;
                }
            }
            for hh in 0..nh {
                let sc = &mut scores[hh * s..(hh + 1) * s];
                let mut maxs = -1e30f32;
                for t in 0..=pos {
                    if sc[t] > maxs {
                        maxs = sc[t];
                    }
                }
                let mut denom = 0.0f32;
                for t in 0..=pos {
                    let w = (sc[t] - maxs).exp();
                    sc[t] = w;
                    denom += w;
                }
                denoms[hh] = denom;
            }
            for a in att.iter_mut() {
                *a = 0.0;
            }
            for t in 0..=pos {
                let vt = &vc[t * d..t * d + d];
                for hh in 0..nh {
                    kernels::fma_broadcast(
                        &mut att[hh * dh..(hh + 1) * dh],
                        scores[hh * s + t],
                        &vt[hh * dh..(hh + 1) * dh],
                        avx2,
                    );
                }
            }
            for hh in 0..nh {
                let denom = denoms[hh];
                for a in &mut att[hh * dh..(hh + 1) * dh] {
                    *a /= denom;
                }
            }
            matvec_any(staged.as_ref().map(|s| &s.attn_proj[li]), &lw.attn_proj, att, xq, h, avx2, &mut profile.matvec_ns, profiling);
            for i in 0..d {
                x[i] += h[i];
            }
            if profiling {
                profile.attn_ns += t0.elapsed().as_nanos();
                t0 = Instant::now();
            }

            kernels::rmsnorm_into(x, lw.ffn_norm, h);
            matvec_any(staged.as_ref().map(|s| &s.gate[li]), &lw.gate, h, xq, g1, avx2, &mut profile.matvec_ns, profiling);
            matvec_any(staged.as_ref().map(|s| &s.up[li]), &lw.up, h, xq, g2, avx2, &mut profile.matvec_ns, profiling);
            for i in 0..f {
                g1[i] = kernels::silu(g1[i]) * g2[i];
            }
            matvec_any(staged.as_ref().map(|s| &s.down[li]), &lw.down, g1, xq, h, avx2, &mut profile.matvec_ns, profiling);
            for i in 0..d {
                x[i] += h[i];
            }
            if profiling {
                profile.ffn_ns += t0.elapsed().as_nanos();
                t0 = Instant::now();
            }

            matvec_any(staged.as_ref().map(|s| &s.ple_gate[li]), &lw.ple_gate, x, xq, g2, avx2, &mut profile.matvec_ns, profiling);
            for i in 0..p {
                g2[i] = kernels::gelu(g2[i]) * ple[li * p + i];
            }
            matvec_any(staged.as_ref().map(|s| &s.ple_proj[li]), &lw.ple_proj, g2, xq, h, avx2, &mut profile.matvec_ns, profiling);
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
                let xq = &mut xq[..x.len()];
                let xs = kernels::quantize_act(x, xq);
                let asum = kernels::act_sum(xq);
                *last_argmax = if *i4_head {
                    matvec_head_i4(&m.tok_emb, &h8.scale, xq, asum, xs, logits, *head_rows, pool, avx2)
                } else {
                    matvec_head_int8(h8, xq, asum, xs, logits, *head_rows, pool, avx2)
                };
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

fn matvec_any(
    t8: Option<&Int8Mat>,
    q: &Qt,
    x: &[f32],
    xq: &mut [i8],
    y: &mut [f32],
    avx2: bool,
    acc: &mut u128,
    on: bool,
) {
    let t = if on { Some(Instant::now()) } else { None };
    match t8 {
        Some(t8) => {
            let xq = &mut xq[..x.len()];
            let xs = kernels::quantize_act(x, xq);
            let asum = kernels::act_sum(xq);
            int8_matvec_rows(t8, xq, asum, xs, 0, t8.rows, avx2, y, &mut |_, _| {});
        }
        None => {
            for r in 0..q.rows {
                y[r] = kernels::matvec_row(q, x, r, avx2);
            }
        }
    }
    if let Some(t) = t {
        *acc += t.elapsed().as_nanos();
    }
}

fn deq_timed(q: &Qt, r: usize, out: &mut [f32], acc: &mut u128, on: bool) {
    let t = if on { Some(Instant::now()) } else { None };
    kernels::deq_row(q, r, out);
    if let Some(t) = t {
        *acc += t.elapsed().as_nanos();
    }
}

#[inline]
fn head8_row(h: &Int8Mat, xq: &[i8], asum: i32, xs: f32, r: usize, avx2: bool) -> f32 {
    let w = &h.w[r * h.cols..(r + 1) * h.cols];
    let d = kernels::dot_u8_i8(w, xq, avx2);
    (d - 8 * asum) as f32 * h.scale[r] * xs
}

#[inline]
fn int8_matvec_rows(t8: &Int8Mat, xq: &[i8], asum: i32, xs: f32, r0: usize, r1: usize, avx2: bool, y: &mut [f32], on_max: &mut dyn FnMut(usize, f32)) {
    let mut r = r0;
    while r + 4 <= r1 {
        let mut d = [0i32; 4];
        kernels::dot_u8_i8_x4(&t8.w[r * t8.cols..(r + 4) * t8.cols], xq, t8.cols, avx2, &mut d);
        for k in 0..4 {
            let v = (d[k] - 8 * asum) as f32 * t8.scale[r + k] * xs;
            y[r + k - r0] = v;
            on_max(r + k, v);
        }
        r += 4;
    }
    while r < r1 {
        let v = head8_row(t8, xq, asum, xs, r, avx2);
        y[r - r0] = v;
        on_max(r, v);
        r += 1;
    }
}

#[inline]
fn i4_head_rows(
    q: &Qt,
    scale: &[f32],
    xq: &[i8],
    asum: i32,
    xs: f32,
    r0: usize,
    r1: usize,
    avx2: bool,
    y: &mut [f32],
    on_max: &mut dyn FnMut(usize, f32),
) {
    let mut r = r0;
    while r + 4 <= r1 {
        let mut d = [0i32; 4];
        kernels::dot_i4_u8_i8_x4(&q.codes[r * q.row_bytes..], q.row_bytes, xq, q.cols, avx2, &mut d);
        for k in 0..4 {
            let v = (d[k] - 8 * asum) as f32 * scale[r + k] * xs;
            y[r + k - r0] = v;
            on_max(r + k, v);
        }
        r += 4;
    }
    while r < r1 {
        let row = &q.codes[r * q.row_bytes..r * q.row_bytes + q.row_bytes];
        let mut total = 0i32;
        for j in 0..q.cols {
            let b = row[j >> 1];
            let c = if j & 1 == 1 { b >> 4 } else { b & 0xF };
            total += c as i32 * xq[j] as i32;
        }
        let v = (total - 8 * asum) as f32 * scale[r] * xs;
        y[r - r0] = v;
        on_max(r, v);
        r += 1;
    }
}

fn matvec_head_i4(
    q: &Qt,
    scale: &[f32],
    xq: &[i8],
    asum: i32,
    xs: f32,
    y: &mut [f32],
    rows: usize,
    pool: &rayon::ThreadPool,
    avx2: bool,
) -> usize {
    let threads = pool.current_num_threads();
    if threads < 2 || rows * q.cols < (1 << 18) {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = 0;
        i4_head_rows(q, scale, xq, asum, xs, 0, rows, avx2, y, &mut |r, v| {
            if v > bv {
                bv = v;
                bi = r;
            }
        });
        return bi;
    }
    let chunk = rows.div_ceil(threads * 4).max(1);
    pool.install(|| {
        y[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, ch)| {
                let r0 = ci * chunk;
                let r1 = (r0 + ch.len()).min(rows);
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0;
                i4_head_rows(q, scale, xq, asum, xs, r0, r1, avx2, ch, &mut |r, v| {
                    if v > bv {
                        bv = v;
                        bi = r;
                    }
                });
                (bv, bi)
            })
            .reduce(|| (f32::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a })
            .1
    })
}

fn matvec_head_int8(
    h: &Int8Mat,
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
        int8_matvec_rows(h, xq, asum, xs, 0, rows, avx2, y, &mut |r, v| {
            if v > bv {
                bv = v;
                bi = r;
            }
        });
        return bi;
    }
    let chunk = rows.div_ceil(threads * 4).max(1);
    pool.install(|| {
        y[..rows]
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, ch)| {
                let r0 = ci * chunk;
                let r1 = (r0 + ch.len()).min(rows);
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0;
                int8_matvec_rows(h, xq, asum, xs, r0, r1, avx2, ch, &mut |r, v| {
                    if v > bv {
                        bv = v;
                        bi = r;
                    }
                });
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
