mod x86;

use core::ops::{BitXor, BitXorAssign};
use crc_catalog::Algorithm;

#[derive(Debug)]
pub struct SimdConstants {
    pub k1: u64,
    pub k2: u64,
    pub k3: u64,
    pub k4: u64,
    pub k5: u64,
    pub k6: u64,
    pub px: u64,
    pub u: u64,
}

impl SimdConstants {
    pub const fn new(algorithm: &Algorithm<u32>) -> Self {
        const fn xt_mod_px(mut t: u32, px: u64) -> u64 {
            if t < 32 {
                return 0;
            }
            t -= 31;

            let mut n = 0x080000000;
            let mut i = 0;
            while i < t {
                n <<= 1;
                if n & 0x100000000 != 0 {
                    n ^= px;
                }
                i += 1;
            }
            n << 32
        }

        const fn u(px: u64) -> u64 {
            let mut q = 0;
            let mut n = 0x100000000;
            let mut i = 0;
            while i < 33 {
                q <<= 1;
                if n & 0x100000000 != 0 {
                    q |= 1;
                    n ^= px;
                }
                n <<= 1;
                i += 1;
            }
            q
        }

        let px = (algorithm.poly as u64) << (u32::BITS as u8 - algorithm.width);
        if algorithm.refin {
            Self {
                k1: xt_mod_px(4 * 128 + 32, px).reverse_bits() << 1,
                k2: xt_mod_px(4 * 128 - 32, px).reverse_bits() << 1,
                k3: xt_mod_px(128 + 32, px).reverse_bits() << 1,
                k4: xt_mod_px(128 - 32, px).reverse_bits() << 1,
                k5: xt_mod_px(64, px).reverse_bits() << 1,
                k6: xt_mod_px(32, px).reverse_bits() << 1,
                px: px.reverse_bits() >> 31,
                u: u(px).reverse_bits() >> 31,
            }
        } else {
            Self {
                k1: xt_mod_px(4 * 128 + 64, px),
                k2: xt_mod_px(4 * 128, px),
                k3: xt_mod_px(128 + 64, px),
                k4: xt_mod_px(128, px),
                k5: xt_mod_px(96, px),
                k6: xt_mod_px(64, px),
                px,
                u: u(px),
            }
        }
    }
}

pub(crate) trait SimdValueExt: BitXor + BitXorAssign + Sized {
    unsafe fn new(value: [u64; 2]) -> Self;

    unsafe fn fold_16(self, x_mod_p: Self) -> Self;

    unsafe fn fold_8(self, x_mod_p: Self) -> Self;

    unsafe fn fold_4(self, x_mod_p: Self) -> Self;

    unsafe fn barret_reduction(self, px_u: Self) -> u32;
}

pub(crate) use x86::SimdValue;
