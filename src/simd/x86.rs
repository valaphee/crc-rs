use crate::simd::SimdValueExt;

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;
use core::ops::{BitXor, BitXorAssign};
use std::{arch::x86_64::_mm_shuffle_epi8, fmt::Debug, ptr::read_unaligned};

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

    unsafe fn fold_4n(self, x_mod_p: Self) -> Self {
        let tmp = Self(arch::_mm_clmulepi64_si128(self.0, x_mod_p.0, 0x01)) ^ self;
        let data128 = Self(arch::_mm_clmulepi64_si128(tmp.0, x_mod_p.0, 0x01)) ^ self;
        Self(arch::_mm_srli_si128(arch::_mm_slli_si128(data128.0, 8), 8))
    }

    unsafe fn barret_reduction_32n(self, px_u: Self) -> u32 {
        let t1 = Self(arch::_mm_clmulepi64_si128(
            arch::_mm_srli_si128(self.0, 4),
            px_u.0,
            0x10,
        ));
        let t2 = Self(arch::_mm_clmulepi64_si128(
            arch::_mm_srli_si128((t1 ^ self).0, 4),
            px_u.0,
            0x00,
        ));
        arch::_mm_extract_epi32((self ^ t2).0, 0) as u32
    }

    unsafe fn swap_bytes(self) -> Self {
        Self(arch::_mm_shuffle_epi8(self.0, std::mem::transmute([0x0Fu8, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00])))
    }

    unsafe fn shift_right(self, num: u8) -> Self {
        let p = read_unaligned(CRC_XMM_SHIFT_TAB.value.as_ptr().offset(16 + num as isize) as *const arch::__m128i);
        Self(_mm_shuffle_epi8(self.0, p))
    }

    unsafe fn shift_left(self, num: u8) -> Self {
        let p = read_unaligned(CRC_XMM_SHIFT_TAB.value.as_ptr().offset(16 - num as isize) as *const arch::__m128i);
        Self(_mm_shuffle_epi8(self.0, p))
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

impl Debug for SimdValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe {
            let b: [u8; 16] = std::mem::transmute(self.0);
            for b in b {
                write!(f, "{:02X} ", b)?;
            }
            Ok(())
        }
    }
}


#[repr(C)]
struct CrcXmmShiftTab {
    _alignment: [arch::__m128i; 0],
    value: [u8; 48],
}

static CRC_XMM_SHIFT_TAB: CrcXmmShiftTab = CrcXmmShiftTab {
    _alignment: [],
    value: [
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    ]
};
