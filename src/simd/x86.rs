use crate::simd::SimdValueExt;

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use core::ops::{BitXor, BitXorAssign};

#[derive(Copy, Clone)]
pub struct SimdValue(pub arch::__m128i);

impl SimdValueExt for SimdValue {
    unsafe fn new(value: [u64; 2]) -> Self {
        Self(arch::_mm_set_epi64x(value[1] as i64, value[0] as i64))
    }

    unsafe fn fold_16(self, x_mod_p: Self) -> Self {
        Self(arch::_mm_clmulepi64_si128(self.0, x_mod_p.0, 0x00))
            ^ Self(arch::_mm_clmulepi64_si128(self.0, x_mod_p.0, 0x11))
    }

    unsafe fn fold_8(self, x_mod_p: Self) -> Self {
        Self(arch::_mm_clmulepi64_si128(self.0, x_mod_p.0, 0x00))
            ^ Self(arch::_mm_srli_si128(self.0, 8))
    }

    unsafe fn fold_4(self, x_mod_p: Self) -> Self {
        Self(arch::_mm_clmulepi64_si128(
            arch::_mm_and_si128(self.0, arch::_mm_cvtsi32_si128(!0)),
            x_mod_p.0,
            0x10,
        )) ^ Self(arch::_mm_srli_si128(self.0, 4))
    }

    unsafe fn barret_reduction_32(self, px_u: Self) -> u32 {
        let t1 = Self(arch::_mm_clmulepi64_si128(
            arch::_mm_and_si128(self.0, arch::_mm_cvtsi32_si128(!0)),
            px_u.0,
            0x10,
        ));
        let t2 = Self(arch::_mm_clmulepi64_si128(
            arch::_mm_and_si128(t1.0, arch::_mm_cvtsi32_si128(!0)),
            px_u.0,
            0x00,
        ));
        arch::_mm_extract_epi32((self ^ t2).0, 1) as u32
    }

    unsafe fn barret_reduction_64(self, px_u: Self) -> u64 {
        let t1 = Self(arch::_mm_clmulepi64_si128(self.0, px_u.0, 0x10));
        let t2 = Self(arch::_mm_clmulepi64_si128(t1.0, px_u.0, 0x00));
        let t2hi = Self(arch::_mm_slli_si128(t1.0, 8));
        arch::_mm_extract_epi64((self ^ t2 ^ t2hi).0, 1) as u64
    }
}

impl BitXor for SimdValue {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(unsafe { arch::_mm_xor_si128(self.0, rhs.0) })
    }
}

impl BitXorAssign for SimdValue {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs;
    }
}
