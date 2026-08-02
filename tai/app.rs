use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::forward::Runtime;
use crate::kernels;
use crate::model::ModelFile;
use crate::sample;

#[derive(Parser)]
#[command(name = "tai", version, about = "Desktop inference for the PLE TinyLM")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Generate(GenArgs),
    Verify(VerifyArgs),
    Bench(BenchArgs),
    Ppl(PplArgs),
}

#[derive(Args)]
struct GenArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long, value_name = "IDS")]
    prompt_ids: Option<String>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long)]
    stop_string: Vec<String>,
    #[arg(long, default_value_t = 200)]
    tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long)]
    vocab_cap: Option<usize>,
    #[arg(long)]
    scalar: bool,
    #[arg(long)]
    fp32_head: bool,
    #[arg(long)]
    i4_head: bool,
}

#[derive(Args)]
struct VerifyArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    golden: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long)]
    scalar: bool,
}

#[derive(Args)]
struct BenchArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 200)]
    tokens: usize,
    #[arg(long, default_value = "1,2,4,8,16")]
    threads: String,
    #[arg(long, value_name = "IDS")]
    prompt_ids: Option<String>,
    #[arg(long)]
    vocab_cap: Option<usize>,
    #[arg(long)]
    scalar: bool,
    #[arg(long)]
    fp32_head: bool,
    #[arg(long)]
    i4_head: bool,
}

#[derive(Args)]
struct PplArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    val: PathBuf,
    #[arg(long, default_value_t = 8)]
    windows: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long)]
    scalar: bool,
    #[arg(long)]
    fp32_head: bool,
}

pub fn run() -> i32 {
    let cli = Cli::parse();
    let result = match &cli.cmd {
        Cmd::Generate(a) => generate(a),
        Cmd::Verify(a) => verify(a),
        Cmd::Bench(a) => bench(a),
        Cmd::Ppl(a) => ppl(a),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

fn parse_ids(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        out.push(
            part.parse::<usize>()
                .map_err(|_| format!("bad token id {part:?} (expected comma-separated integers)"))?,
        );
    }
    if out.is_empty() {
        return Err("prompt id list is empty".to_string());
    }
    Ok(out)
}

fn parse_thread_list(s: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let n = part
            .parse::<usize>()
            .map_err(|_| format!("bad thread count {part:?}"))?;
        if n == 0 {
            return Err("thread count must be at least 1".to_string());
        }
        out.push(n);
    }
    if out.is_empty() {
        return Err("thread list is empty".to_string());
    }
    Ok(out)
}

fn build_pool(threads: usize) -> Result<rayon::ThreadPool, String> {
    let n = if threads == 0 {
        num_cpus::get_physical().max(1)
    } else {
        threads
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .map_err(|e| format!("failed to build thread pool: {e}"))
}

fn head_cap(cfg_vocab: usize, cap: Option<usize>) -> Result<usize, String> {
    let v = cap.unwrap_or(cfg_vocab).min(cfg_vocab);
    if v == 0 {
        return Err("vocab cap is 0; nothing to sample from".to_string());
    }
    Ok(v)
}

fn load_tokenizer(path: &Option<PathBuf>) -> Result<Option<tokenizers::Tokenizer>, String> {
    match path {
        Some(p) => tokenizers::Tokenizer::from_file(p)
            .map(Some)
            .map_err(|e| format!("{}: {e}", p.display())),
        None => Ok(None),
    }
}

fn emit_token(
    out: &mut impl std::io::Write,
    tok: &Option<tokenizers::Tokenizer>,
    id: usize,
) -> Result<(), String> {
    match tok {
        Some(t) => {
            let piece = t
                .decode(&[id as u32], true)
                .map_err(|e| format!("decode token {id}: {e}"))?;
            out.write_all(piece.as_bytes())
                .and_then(|()| out.flush())
                .map_err(|e| format!("stdout: {e}"))
        }
        None => out
            .write_all(format!("{id} ").as_bytes())
            .and_then(|()| out.flush())
            .map_err(|e| format!("stdout: {e}")),
    }
}

fn generate(a: &GenArgs) -> Result<i32, String> {
    let mf = ModelFile::open(&a.model)?;
    let m = mf.model()?;
    let c = &m.cfg;
    eprintln!(
        "loaded: V={} D={} L={} H={} F={} P={} group={}",
        c.vocab, c.dim, c.n_layers, c.n_heads, c.ffn, c.ple_dim, c.group
    );
    let tok = load_tokenizer(&a.tokenizer)?;
    let ids = match (&a.prompt, &a.prompt_ids) {
        (Some(_), Some(_)) => {
            return Err("pass only one of --prompt and --prompt-ids".to_string());
        }
        (Some(text), None) => {
            let t = tok.as_ref().ok_or_else(|| {
                "a text prompt needs --tokenizer <bpe.json>; otherwise use --prompt-ids"
                    .to_string()
            })?;
            t.encode(text.as_str(), false)
                .map_err(|e| format!("tokenize prompt: {e}"))?
                .get_ids()
                .iter()
                .map(|&v| v as usize)
                .collect()
        }
        (None, Some(raw)) => parse_ids(raw)?,
        (None, None) => {
            return Err("no prompt: pass --prompt <text> or --prompt-ids <ids>".to_string());
        }
    };
    if ids.len() > c.seq_len {
        return Err(format!(
            "prompt length {} exceeds model seq_len {}",
            ids.len(),
            c.seq_len
        ));
    }
    for &id in &ids {
        if id >= c.vocab {
            return Err(format!("prompt token id {id} >= vocab {}", c.vocab));
        }
    }
    let rows = head_cap(c.vocab, a.vocab_cap)?;
    let pool = build_pool(a.threads)?;
    let avx2 = kernels::have_avx2() && !a.scalar;
    eprintln!(
        "threads={} avx2={} head_rows={}",
        pool.current_num_threads(),
        avx2,
        rows
    );
    if !a.stop_string.is_empty() && tok.is_none() {
        return Err("--stop-string needs --tokenizer (text mode)".to_string());
    }
    let eot_id = tok
        .as_ref()
        .and_then(|t| t.token_to_id("<|endoftext|>"))
        .map(|v| v as usize);
    let mut rt = Runtime::new(&m, rows, avx2 && !a.fp32_head);
    rt.i4_head = a.i4_head;
    let mut rng = StdRng::seed_from_u64(a.seed);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for &id in &ids {
        rt.forward(&m, id, &pool, avx2);
    }

    let mut decoded = 0usize;
    let mut emitted = String::new();
    let t_start = Instant::now();
    for _ in 0..a.tokens {
        if rt.pos() >= c.seq_len {
            eprintln!("\nreached seq_len {}; stopping", c.seq_len);
            break;
        }
        let next = {
            let am = rt.argmax();
            let (logits, scaled, order) = rt.logits_and_scratch();
            sample::sample(logits, a.temperature, a.top_k, &mut rng, scaled, order, am)
        };
        if eot_id == Some(next) {
            break;
        }
        if a.stop_string.is_empty() {
            emit_token(&mut out, &tok, next)?;
        } else {
            let piece = tok
                .as_ref()
                .map(|t| t.decode(&[next as u32], true))
                .transpose()
                .map_err(|e| format!("decode token {next}: {e}"))?
                .unwrap_or_default();
            emitted.push_str(&piece);
            let cut = a
                .stop_string
                .iter()
                .filter_map(|m| emitted.find(m.as_str()))
                .min();
            if let Some(idx) = cut {
                emitted.truncate(idx);
                break;
            }
        }
        rt.forward(&m, next, &pool, avx2);
        decoded += 1;
    }
    if !a.stop_string.is_empty() {
        out.write_all(emitted.as_bytes())
            .and_then(|()| out.flush())
            .map_err(|e| format!("stdout: {e}"))?;
    }
    let elapsed = t_start.elapsed().as_secs_f64();
    writeln!(out).map_err(|e| format!("stdout: {e}"))?;
    if decoded > 0 {
        eprintln!(
            "--- {} tokens in {:.2} s ---  {:.2} tok/s ({:.1} ms/token)",
            decoded,
            elapsed,
            decoded as f64 / elapsed,
            elapsed * 1000.0 / decoded as f64
        );
    }
    Ok(0)
}

fn verify(a: &VerifyArgs) -> Result<i32, String> {
    let mf = ModelFile::open(&a.model)?;
    let m = mf.model()?;
    let c = &m.cfg;
    println!(
        "loaded: V={} D={} L={} H={} F={} P={} group={}",
        c.vocab, c.dim, c.n_layers, c.n_heads, c.ffn, c.ple_dim, c.group
    );
    let text = std::fs::read_to_string(&a.golden)
        .map_err(|e| format!("{}: {e}", a.golden.display()))?;
    let mut it = text.split_whitespace();
    let plen: usize = it
        .next()
        .ok_or_else(|| "golden: missing prompt length".to_string())?
        .parse()
        .map_err(|_| "golden: bad prompt length".to_string())?;
    let mut ids = Vec::with_capacity(plen);
    for _ in 0..plen {
        ids.push(
            it.next()
                .ok_or_else(|| "golden: prompt id list truncated".to_string())?
                .parse::<usize>()
                .map_err(|_| "golden: bad token id".to_string())?,
        );
    }
    let v = c.vocab;
    let mut ref_logits = Vec::with_capacity(v);
    for _ in 0..v {
        ref_logits.push(
            it.next()
                .ok_or_else(|| format!("golden: expected {v} logits, file ended early"))?
                .parse::<f32>()
                .map_err(|_| "golden: bad logit value".to_string())?,
        );
    }
    if plen == 0 || plen > c.seq_len {
        return Err(format!(
            "golden prompt length {plen} outside 1..=seq_len {}",
            c.seq_len
        ));
    }
    for &id in &ids {
        if id >= v {
            return Err(format!("golden prompt id {id} >= vocab {v}"));
        }
    }
    let pool = build_pool(a.threads)?;
    let avx2 = kernels::have_avx2() && !a.scalar;
    let mut rt = Runtime::new(&m, v, false);
    for &id in &ids {
        rt.forward(&m, id, &pool, avx2);
    }
    let logits = rt.logits();
    let mut maxabs = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut c_top = 0usize;
    let mut r_top = 0usize;
    for i in 0..v {
        let d = logits[i] as f64 - ref_logits[i] as f64;
        if d.abs() > maxabs {
            maxabs = d.abs();
        }
        sum2 += d * d;
        if logits[i] > logits[c_top] {
            c_top = i;
        }
        if ref_logits[i] > ref_logits[r_top] {
            r_top = i;
        }
    }
    println!("sample logits (idx: rust vs ref):");
    for probe in [265usize, 14, 1, 12, 13, 100, 5000, 20000] {
        if probe < v {
            println!(
                "  [{:5}]  rust={:8.4}  ref={:8.4}",
                probe, logits[probe], ref_logits[probe]
            );
        }
    }
    println!("logits: rust top={}  pytorch top={}", c_top, r_top);
    println!(
        "max abs diff = {:.5}   rms diff = {:.6}",
        maxabs,
        (sum2 / v as f64).sqrt()
    );
    if maxabs < 0.02 {
        println!("PASS: rust matches PyTorch golden");
        Ok(0)
    } else {
        println!("FAIL: numerics diverge");
        Ok(2)
    }
}

fn bench(a: &BenchArgs) -> Result<i32, String> {
    let mf = ModelFile::open(&a.model)?;
    let m = mf.model()?;
    let c = &m.cfg;
    eprintln!(
        "loaded: V={} D={} L={} H={} F={} P={} group={}",
        c.vocab, c.dim, c.n_layers, c.n_heads, c.ffn, c.ple_dim, c.group
    );
    let ids = match &a.prompt_ids {
        Some(raw) => parse_ids(raw)?,
        None => vec![1],
    };
    if ids.len() > c.seq_len {
        return Err(format!(
            "prompt length {} exceeds model seq_len {}",
            ids.len(),
            c.seq_len
        ));
    }
    for &id in &ids {
        if id >= c.vocab {
            return Err(format!("prompt token id {id} >= vocab {}", c.vocab));
        }
    }
    let rows = head_cap(c.vocab, a.vocab_cap)?;
    let avx2 = kernels::have_avx2() && !a.scalar;
    eprintln!("avx2={avx2} head_rows={rows} tokens={}", a.tokens);
    println!(
        "{:>8} {:>12} {:>12}   profile (ms/token)",
        "threads", "tok/s", "ms/token"
    );
    for tc in parse_thread_list(&a.threads)? {
        let pool = build_pool(tc)?;
        let mut rt = Runtime::new(&m, rows, avx2 && !a.fp32_head);
        rt.i4_head = a.i4_head;
        for &id in &ids {
            rt.forward(&m, id, &pool, avx2);
        }
        for _ in 0..4.min(a.tokens) {
            let next = rt.argmax();
            rt.forward(&m, next, &pool, avx2);
        }
        rt.reset();
        for &id in &ids {
            rt.forward(&m, id, &pool, avx2);
        }
        rt.profiling = true;
        rt.profile.reset();
        let t0 = Instant::now();
        let mut fwd = 0.0f64;
        let mut decoded = 0usize;
        for _ in 0..a.tokens {
            if rt.pos() >= c.seq_len {
                break;
            }
            let next = rt.argmax();
            let f0 = Instant::now();
            rt.forward(&m, next, &pool, avx2);
            fwd += f0.elapsed().as_secs_f64();
            decoded += 1;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        if decoded == 0 {
            println!("{tc:>8} {:>12} {:>12}   (no tokens decoded)", "-", "-");
            continue;
        }
        let n = decoded as f64;
        println!(
            "{:>8} {:>12.2} {:>12.2}   {} | fwd {:.2} other {:.2}",
            tc,
            n / elapsed,
            elapsed * 1000.0 / n,
            rt.profile.report().replace("profile ms/token: ", ""),
            fwd * 1000.0 / n,
            (elapsed - fwd) * 1000.0 / n,
        );
    }
    Ok(0)
}

fn ppl(a: &PplArgs) -> Result<i32, String> {
    let mf = ModelFile::open(&a.model)?;
    let m = mf.model()?;
    let c = &m.cfg;
    let raw =
        std::fs::read(&a.val).map_err(|e| format!("{}: {e}", a.val.display()))?;
    if raw.len() % 2 != 0 {
        return Err(format!(
            "{}: odd byte count {}, expected uint16 tokens",
            a.val.display(),
            raw.len()
        ));
    }
    let n_tok = raw.len() / 2;
    let s = c.seq_len;
    let v = c.vocab;
    let pool = build_pool(a.threads)?;
    let avx2 = kernels::have_avx2() && !a.scalar;
    let int8 = avx2 && !a.fp32_head;
    let mut rt = Runtime::new(&m, v, int8);
    let mut nll = 0.0f64;
    let mut count = 0u64;
    for w in 0..a.windows {
        let base = w * s;
        if base + s + 1 > n_tok {
            break;
        }
        rt.reset();
        for pos in 0..s {
            let tok = u16::from_le_bytes([raw[2 * (base + pos)], raw[2 * (base + pos) + 1]]) as usize;
            if tok >= v {
                return Err(format!("val token id {tok} >= vocab {v}"));
            }
            rt.forward(&m, tok, &pool, avx2);
            let target =
                u16::from_le_bytes([raw[2 * (base + pos + 1)], raw[2 * (base + pos + 1) + 1]])
                    as usize;
            let logits = rt.logits();
            let mut mx = f32::NEG_INFINITY;
            for &lv in logits {
                if lv > mx {
                    mx = lv;
                }
            }
            let mut sum = 0.0f64;
            for &lv in logits {
                sum += ((lv - mx) as f64).exp();
            }
            nll += sum.ln() - (logits[target] - mx) as f64;
            count += 1;
        }
    }
    if count == 0 {
        return Err(format!(
            "val file {} has too few tokens for one window of seq_len {s}",
            a.val.display()
        ));
    }
    let mean = nll / count as f64;
    let mode = if int8 { "int8-head" } else { "fp32-head" };
    println!(
        "{:<18}  val CE {:.4}  ppl {:.2}   ({} predictions)",
        mode,
        mean,
        mean.exp(),
        count
    );
    Ok(0)
}
