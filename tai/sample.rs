use rand::rngs::StdRng;
use rand::Rng;

pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut StdRng,
    scaled: &mut Vec<f32>,
    order: &mut Vec<f32>,
    argmax: usize,
) -> usize {
    if temperature <= 0.0 || top_k == 1 {
        return argmax;
    }
    let n = logits.len();
    scaled.clear();
    scaled.extend_from_slice(logits);
    for v in scaled.iter_mut() {
        *v /= temperature;
    }
    let k = top_k.min(n);
    let threshold = if k > 0 && k < n {
        order.clear();
        order.extend_from_slice(scaled);
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
        return argmax;
    }
    let mut r = rng.random::<f32>() * total;
    for (i, &v) in scaled.iter().enumerate() {
        r -= v;
        if r <= 0.0 {
            return i;
        }
    }
    argmax
}
