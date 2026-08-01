use rand::rngs::StdRng;
use rand::Rng;

pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

pub fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut StdRng) -> usize {
    if temperature <= 0.0 || top_k == 1 {
        return argmax(logits);
    }
    let n = logits.len();
    let mut scaled: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    let k = top_k.min(n);
    let threshold = if k > 0 && k < n {
        let mut order = scaled.clone();
        let (_, &mut t, _) = order
            .select_nth_unstable_by(k - 1, |a, b| {
                b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
            });
        t
    } else {
        f32::NEG_INFINITY
    };
    let mut maxs = f32::NEG_INFINITY;
    for &v in scaled.iter() {
        if v >= threshold && v > maxs {
            maxs = v;
        }
    }
    let mut total = 0.0f32;
    for v in scaled.iter_mut() {
        *v = if *v >= threshold { (*v - maxs).exp() } else { 0.0 };
        total += *v;
    }
    if total <= 0.0 || !total.is_finite() {
        return argmax(logits);
    }
    let mut r = rng.random::<f32>() * total;
    for (i, &v) in scaled.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return i;
        }
    }
    argmax(logits)
}
