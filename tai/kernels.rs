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
fn scale_at(q: &Qt, r: usize, gi: usize) -> f32 {
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
