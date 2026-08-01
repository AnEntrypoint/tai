use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

pub const MAGIC: u32 = 0x504C4531;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn: usize,
    pub ple_dim: usize,
    pub seq_len: usize,
    pub group: usize,
    pub rope_theta: f32,
}

#[derive(Clone, Copy)]
pub struct Qt<'a> {
    pub codes: &'a [u8],
    pub scales: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
    pub n_groups: usize,
    pub row_bytes: usize,
}

#[derive(Clone, Copy)]
pub struct Layer<'a> {
    pub attn_norm: &'a [u8],
    pub qkv: Qt<'a>,
    pub attn_proj: Qt<'a>,
    pub ffn_norm: &'a [u8],
    pub gate: Qt<'a>,
    pub up: Qt<'a>,
    pub down: Qt<'a>,
    pub ple_gate: Qt<'a>,
    pub ple_proj: Qt<'a>,
    pub ple_norm: &'a [u8],
}

pub struct Model<'a> {
    pub cfg: Config,
    pub tok_emb: Qt<'a>,
    pub ple_model_proj: Qt<'a>,
    pub ple_proj_norm: &'a [u8],
    pub ple_table: Qt<'a>,
    pub layers: Vec<Layer<'a>>,
    pub out_norm: &'a [u8],
}

#[inline]
pub fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]])
}

struct Cur<'a> {
    b: &'a [u8],
    off: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .off
            .checked_add(n)
            .ok_or_else(|| "model size overflow while binding tensors".to_string())?;
        if end > self.b.len() {
            return Err(format!(
                "truncated model: tensor needs {} bytes at offset {}, file has {}",
                n,
                self.off,
                self.b.len()
            ));
        }
        let s = &self.b[self.off..end];
        self.off = end;
        Ok(s)
    }

    fn i32(&mut self) -> Result<i32, String> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32(&mut self) -> Result<f32, String> {
        let b = self.take(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn bind_q<'a>(cur: &mut Cur<'a>, rows: usize, cols: usize) -> Result<Qt<'a>, String> {
    let group = cur.i32()?;
    if group <= 0 {
        return Err(format!("quant tensor has non-positive group {group}"));
    }
    let group = group as usize;
    let n_groups = cols.div_ceil(group);
    let row_bytes = cols.div_ceil(2);
    let codes_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| "quant tensor size overflow".to_string())?;
    let scales_len = rows
        .checked_mul(n_groups)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| "quant tensor size overflow".to_string())?;
    let codes = cur.take(codes_len)?;
    let scales = cur.take(scales_len)?;
    Ok(Qt { codes, scales, rows, cols, group, n_groups, row_bytes })
}

fn bind_f<'a>(cur: &mut Cur<'a>, n: usize) -> Result<&'a [u8], String> {
    let len = n
        .checked_mul(4)
        .ok_or_else(|| "fp32 tensor size overflow".to_string())?;
    cur.take(len)
}

fn dim_field(name: &str, v: i32) -> Result<usize, String> {
    if v <= 0 {
        return Err(format!("model header has non-positive {name} ({v})"));
    }
    Ok(v as usize)
}

impl<'a> Model<'a> {
    pub fn bind(b: &'a [u8]) -> Result<Model<'a>, String> {
        let mut cur = Cur { b, off: 0 };
        let magic = cur.i32()? as u32;
        if magic != MAGIC {
            return Err(format!(
                "bad model magic 0x{magic:08X}, expected 0x{MAGIC:08X} (not a PLE1 model.bin)"
            ));
        }
        let vocab = dim_field("vocab", cur.i32()?)?;
        let dim = dim_field("dim", cur.i32()?)?;
        let n_layers = dim_field("n_layers", cur.i32()?)?;
        let n_heads = dim_field("n_heads", cur.i32()?)?;
        let ffn = dim_field("ffn", cur.i32()?)?;
        let ple_dim = dim_field("ple_dim", cur.i32()?)?;
        let seq_len = dim_field("seq_len", cur.i32()?)?;
        let group = dim_field("group", cur.i32()?)?;
        let rope_theta = cur.f32()?;
        if dim % n_heads != 0 {
            return Err(format!(
                "model header dim {dim} is not divisible by n_heads {n_heads}"
            ));
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(format!("model header has invalid rope_theta {rope_theta}"));
        }
        let cfg = Config {
            vocab, dim, n_layers, n_heads, ffn, ple_dim, seq_len, group, rope_theta,
        };

        let tok_emb = bind_q(&mut cur, vocab, dim)?;
        let ple_model_proj = bind_q(&mut cur, n_layers * ple_dim, dim)?;
        let ple_proj_norm = bind_f(&mut cur, ple_dim)?;
        let ple_table = bind_q(&mut cur, vocab, n_layers * ple_dim)?;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(Layer {
                attn_norm: bind_f(&mut cur, dim)?,
                qkv: bind_q(&mut cur, 3 * dim, dim)?,
                attn_proj: bind_q(&mut cur, dim, dim)?,
                ffn_norm: bind_f(&mut cur, dim)?,
                gate: bind_q(&mut cur, ffn, dim)?,
                up: bind_q(&mut cur, ffn, dim)?,
                down: bind_q(&mut cur, dim, ffn)?,
                ple_gate: bind_q(&mut cur, ple_dim, dim)?,
                ple_proj: bind_q(&mut cur, dim, ple_dim)?,
                ple_norm: bind_f(&mut cur, dim)?,
            });
        }
        let out_norm = bind_f(&mut cur, dim)?;
        if cur.off != b.len() {
            return Err(format!(
                "model file has {} trailing bytes after the last tensor",
                b.len() - cur.off
            ));
        }
        Ok(Model {
            cfg, tok_emb, ple_model_proj, ple_proj_norm, ple_table, layers, out_norm,
        })
    }
}

pub struct ModelFile {
    mmap: Mmap,
}

impl ModelFile {
    pub fn open(path: &Path) -> Result<ModelFile, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        Ok(ModelFile { mmap })
    }

    pub fn model(&self) -> Result<Model<'_>, String> {
        Model::bind(&self.mmap)
    }
}
