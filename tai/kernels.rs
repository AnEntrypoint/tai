use crate::model::{f32_at, Qt};

pub const RMS_EPS: f32 = 1e-6;

#[inline]
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn scale_at(q: &Qt, r: usize, gi: usize) -> f32 {
    let i = (r * q.n_groups + gi) * 2;
    half_to_f32(u16::from_le_bytes([q.scales[i], q.scales[i + 1]]))
}

#[inline]
fn code_at(row: &[u8], j: usize) -> i32 {
    let b = row[j >> 1];
    (if j & 1 == 1 { b >> 4 } else { b & 0xF }) as i32 - 8
}

pub fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ONCE.get_or_init(|| {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        })
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub fn deq_row(q: &Qt, r: usize, out: &mut [f32]) {
    let row = &q.codes[r * q.row_bytes..r * q.row_bytes + q.row_bytes];
    for gi in 0..q.n_groups {
        let begin = gi * q.group;
        let end = (begin + q.group).min(q.cols);
        let scale = scale_at(q, r, gi);
        let mut j = begin;
        if j & 1 == 1 && j < end {
            out[j] = code_at(row, j) as f32 * scale;
            j += 1;
        }
        while j + 1 < end {
            let b = row[j >> 1];
            out[j] = ((b & 0xF) as i32 - 8) as f32 * scale;
            out[j + 1] = ((b >> 4) as i32 - 8) as f32 * scale;
            j += 2;
        }
        if j < end {
            out[j] = code_at(row, j) as f32 * scale;
        }
    }
}

pub fn matvec_row(q: &Qt, x: &[f32], r: usize, avx2: bool) -> f32 {
    let row = &q.codes[r * q.row_bytes..r * q.row_bytes + q.row_bytes];
    let mut acc = 0.0f32;
    for gi in 0..q.n_groups {
        let begin = gi * q.group;
        let end = (begin + q.group).min(q.cols);
        let mut gacc = 0.0f32;
        let mut j = begin;
        if j & 1 == 1 && j < end {
            gacc += code_at(row, j) as f32 * x[j];
            j += 1;
        }
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            gacc += unsafe { avx2_kernel::dot_i4_f32(row, x, j, end) };
            acc += gacc * scale_at(q, r, gi);
            continue;
        }
        while j + 1 < end {
            let b = row[j >> 1];
            gacc += ((b & 0xF) as i32 - 8) as f32 * x[j];
            gacc += ((b >> 4) as i32 - 8) as f32 * x[j + 1];
            j += 2;
        }
        if j < end {
            gacc += code_at(row, j) as f32 * x[j];
        }
        acc += gacc * scale_at(q, r, gi);
    }
    acc
}

#[cfg(target_arch = "x86_64")]
mod avx2_kernel {
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_i4_f32(row: &[u8], x: &[f32], begin: usize, end: usize) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut j = begin;
        while j + 32 <= end {
            let bytes = _mm_loadu_si128(row.as_ptr().add(j >> 1) as *const __m128i);
            let lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
            let hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
            let v = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
            let v = _mm256_sub_epi8(v, _mm256_set1_epi8(8));
            let w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_castsi256_si128(v)));
            let w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(
                _mm256_castsi256_si128(v),
            )));
            let w2 =
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_extracti128_si256::<1>(v)));
            let w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(
                _mm256_extracti128_si256::<1>(v),
            )));
            acc = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.as_ptr().add(j)), acc);
            acc = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.as_ptr().add(j + 8)), acc);
            acc = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x.as_ptr().add(j + 16)), acc);
            acc = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x.as_ptr().add(j + 24)), acc);
            j += 32;
        }
        let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        let mut total = _mm_cvtss_f32(s);
        while j < end {
            let b = *row.get_unchecked(j >> 1);
            let c = (if j & 1 == 1 { b >> 4 } else { b & 0xF }) as i32 - 8;
            total += c as f32 * x.get_unchecked(j);
            j += 1;
        }
        total
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

pub fn rmsnorm_into(x: &[f32], w: &[u8], out: &mut [f32]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + RMS_EPS).sqrt();
    for i in 0..n {
        out[i] = f32_at(w, i) * x[i] * inv;
    }
}

pub fn rmsnorm_ip(x: &mut [f32], w: &[u8]) {
    let n = x.len();
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + RMS_EPS).sqrt();
    for i in 0..n {
        x[i] *= f32_at(w, i) * inv;
    }
}

pub fn quantize_act(x: &[f32], xq: &mut [i8]) -> f32 {
    let mut xmax = 1e-8f32;
    for &v in x {
        let a = v.abs();
        if a > xmax {
            xmax = a;
        }
    }
    let inv = 127.0 / xmax;
    for (o, &v) in xq.iter_mut().zip(x.iter()) {
        let q = libm::rintf(v * inv) as i32;
        *o = q.clamp(-127, 127) as i8;
    }
    xmax / 127.0
}

pub fn act_sum(xq: &[i8]) -> i32 {
    let mut s = 0i32;
    for &v in xq {
        s += v as i32;
    }
    s
}

pub fn dot_u8_i8(w: &[u8], a: &[i8], avx2: bool) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { avx2_int8::dot_u8_i8_avx2(w, a) };
    }
    let mut total = 0i32;
    for i in 0..w.len() {
        total += w[i] as i32 * a[i] as i32;
    }
    total
}

pub fn dot_u8_i8_x4(w: &[u8], a: &[i8], cols: usize, avx2: bool, out: &mut [i32; 4]) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            avx2_int8::dot_u8_i8_x4_avx2(w, a, cols, out);
        }
        return;
    }
    for r in 0..4 {
        let row = &w[r * cols..(r + 1) * cols];
        let mut total = 0i32;
        for i in 0..cols {
            total += row[i] as i32 * a[i] as i32;
        }
        out[r] = total;
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_int8 {
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_u8_i8_x4_avx2(w: &[u8], a: &[i8], cols: usize, out: &mut [i32; 4]) {
        let n = cols & !31;
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        let mut acc2 = _mm256_setzero_si256();
        let mut acc3 = _mm256_setzero_si256();
        let one = _mm256_set1_epi16(1);
        let mut i = 0;
        while i < n {
            let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let w0 = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let w1 = _mm256_loadu_si256(w.as_ptr().add(cols + i) as *const __m256i);
            let w2 = _mm256_loadu_si256(w.as_ptr().add(2 * cols + i) as *const __m256i);
            let w3 = _mm256_loadu_si256(w.as_ptr().add(3 * cols + i) as *const __m256i);
            acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(_mm256_maddubs_epi16(w0, av), one));
            acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(_mm256_maddubs_epi16(w1, av), one));
            acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(_mm256_maddubs_epi16(w2, av), one));
            acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(_mm256_maddubs_epi16(w3, av), one));
            i += 32;
        }
        let mut totals = [0i32; 4];
        for (k, acc) in [acc0, acc1, acc2, acc3].iter().enumerate() {
            let s = _mm_add_epi32(
                _mm256_castsi256_si128(*acc),
                _mm256_extracti128_si256::<1>(*acc),
            );
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32::<1>(s));
            totals[k] = _mm_cvtsi128_si32(s);
        }
        while i < cols {
            let a0 = *a.get_unchecked(i) as i32;
            totals[0] += *w.get_unchecked(i) as i32 * a0;
            totals[1] += *w.get_unchecked(cols + i) as i32 * a0;
            totals[2] += *w.get_unchecked(2 * cols + i) as i32 * a0;
            totals[3] += *w.get_unchecked(3 * cols + i) as i32 * a0;
            i += 1;
        }
        *out = totals;
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_u8_i8_avx2(w: &[u8], a: &[i8]) -> i32 {
        let mut acc = _mm256_setzero_si256();
        let n = w.len() & !31;
        let mut i = 0;
        while i < n {
            let wv = _mm256_loadu_si256(w.as_ptr().add(i) as *const __m256i);
            let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
            let p = _mm256_maddubs_epi16(wv, av);
            let q = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
            acc = _mm256_add_epi32(acc, q);
            i += 32;
        }
        let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32::<1>(s));
        let mut total = _mm_cvtsi128_si32(s);
        while i < w.len() {
            total += *w.get_unchecked(i) as i32 * *a.get_unchecked(i) as i32;
            i += 1;
        }
        total
    }
}

pub fn dot_f32(a: &[f32], b: &[f32], avx2: bool) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { avx2_attn::dot_f32_avx2(a, b) };
    }
    let mut total = 0.0f32;
    for i in 0..a.len() {
        total += a[i] * b[i];
    }
    total
}

pub fn fma_broadcast(dst: &mut [f32], w: f32, src: &[f32], avx2: bool) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        unsafe {
            avx2_attn::fma_broadcast_avx2(dst, w, src);
        }
        return;
    }
    for i in 0..dst.len() {
        dst[i] += w * src[i];
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2_attn {
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len() & !7;
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < n {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                acc,
            );
            i += 8;
        }
        let s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        let mut total = _mm_cvtss_f32(s);
        while i < a.len() {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        total
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn fma_broadcast_avx2(dst: &mut [f32], w: f32, src: &[f32]) {
        let wv = _mm256_set1_ps(w);
        let n = dst.len() & !7;
        let mut i = 0;
        while i < n {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let s = _mm256_loadu_ps(src.as_ptr().add(i));
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_fmadd_ps(wv, s, d));
            i += 8;
        }
        while i < dst.len() {
            *dst.get_unchecked_mut(i) += w * src.get_unchecked(i);
            i += 1;
        }
    }
}
