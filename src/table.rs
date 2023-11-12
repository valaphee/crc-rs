use crate::util::*;
use crc_catalog::Algorithm;

pub(crate) const fn crc8_table(width: u8, poly: u8, reflect: bool) -> [u8; 256] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (8u8 - width)
    } else {
        poly << (8u8 - width)
    };

    let mut table = [0u8; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc8(poly, reflect, i as u8);
        i += 1;
    }
    table
}

pub(crate) const fn crc8_table_slice_16(width: u8, poly: u8, reflect: bool) -> [[u8; 256]; 16] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (8u8 - width)
    } else {
        poly << (8u8 - width)
    };

    let mut table = [[0u8; 256]; 16];
    let mut i = 0;
    while i < 256 {
        table[0][i] = crc8(poly, reflect, i as u8);
        i += 1;
    }

    let mut i = 0;
    while i < 256 {
        let mut e = 1;
        while e < 16 {
            let one_lower = table[e - 1][i];
            table[e][i] = table[0][one_lower as usize];
            e += 1;
        }
        i += 1;
    }
    table
}

pub(crate) const fn crc16_table(width: u8, poly: u16, reflect: bool) -> [u16; 256] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (16u8 - width)
    } else {
        poly << (16u8 - width)
    };

    let mut table = [0u16; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc16(poly, reflect, i as u16);
        i += 1;
    }
    table
}

pub(crate) const fn crc16_table_slice_16(width: u8, poly: u16, reflect: bool) -> [[u16; 256]; 16] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (16u8 - width)
    } else {
        poly << (16u8 - width)
    };

    let mut table = [[0u16; 256]; 16];
    let mut i = 0;
    while i < 256 {
        table[0][i] = crc16(poly, reflect, i as u16);
        i += 1;
    }

    let mut i = 0;
    while i < 256 {
        let mut e = 1;
        while e < 16 {
            let one_lower = table[e - 1][i];
            if reflect {
                table[e][i] = (one_lower >> 8) ^ table[0][(one_lower & 0xFF) as usize];
            } else {
                table[e][i] = (one_lower << 8) ^ table[0][((one_lower >> 8) & 0xFF) as usize];
            }
            e += 1;
        }
        i += 1;
    }
    table
}

pub(crate) const fn crc32_table(width: u8, poly: u32, reflect: bool) -> [u32; 256] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (32u8 - width)
    } else {
        poly << (32u8 - width)
    };

    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = crc32(poly, reflect, i as u32);
        i += 1;
    }

    table
}

pub(crate) const fn crc32_table_slice_16(width: u8, poly: u32, reflect: bool) -> [[u32; 256]; 16] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (32u8 - width)
    } else {
        poly << (32u8 - width)
    };

    let mut table = [[0u32; 256]; 16];
    let mut i = 0;
    while i < 256 {
        table[0][i] = crc32(poly, reflect, i as u32);
        i += 1;
    }

    let mut i = 0;
    while i < 256 {
        let mut e = 1;
        while e < 16 {
            let one_lower = table[e - 1][i];
            if reflect {
                table[e][i] = (one_lower >> 8) ^ table[0][(one_lower & 0xFF) as usize];
            } else {
                table[e][i] = (one_lower << 8) ^ table[0][((one_lower >> 24) & 0xFF) as usize];
            }
            e += 1;
        }
        i += 1;
    }
    table
}

pub(crate) const fn crc64_table(width: u8, poly: u64, reflect: bool) -> [u64; 256] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (64u8 - width)
    } else {
        poly << (64u8 - width)
    };

    let mut table = [0u64; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc64(poly, reflect, i as u64);
        i += 1;
    }
    table
}

pub(crate) const fn crc64_table_slice_16(width: u8, poly: u64, reflect: bool) -> [[u64; 256]; 16] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (64u8 - width)
    } else {
        poly << (64u8 - width)
    };

    let mut table = [[0u64; 256]; 16];
    let mut i = 0;
    while i < 256 {
        table[0][i] = crc64(poly, reflect, i as u64);
        i += 1;
    }

    let mut i = 0;
    while i < 256 {
        let mut e = 1;
        while e < 16 {
            let one_lower = table[e - 1][i];
            if reflect {
                table[e][i] = (one_lower >> 8) ^ table[0][(one_lower & 0xFF) as usize];
            } else {
                table[e][i] = (one_lower << 8) ^ table[0][((one_lower >> 56) & 0xFF) as usize];
            }
            e += 1;
        }
        i += 1;
    }
    table
}

pub(crate) const fn crc128_table(width: u8, poly: u128, reflect: bool) -> [u128; 256] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (128u8 - width)
    } else {
        poly << (128u8 - width)
    };

    let mut table = [0u128; 256];
    let mut i = 0;
    while i < table.len() {
        table[i] = crc128(poly, reflect, i as u128);
        i += 1;
    }
    table
}

pub(crate) const fn crc128_table_slice_16(
    width: u8,
    poly: u128,
    reflect: bool,
) -> [[u128; 256]; 16] {
    let poly = if reflect {
        let poly = poly.reverse_bits();
        poly >> (128u8 - width)
    } else {
        poly << (128u8 - width)
    };

    let mut table = [[0u128; 256]; 16];
    let mut i = 0;
    while i < 256 {
        table[0][i] = crc128(poly, reflect, i as u128);
        i += 1;
    }

    let mut i = 0;
    while i < 256 {
        let mut e = 1;
        while e < 16 {
            let one_lower = table[e - 1][i];
            if reflect {
                table[e][i] = (one_lower >> 8) ^ table[0][(one_lower & 0xFF) as usize];
            } else {
                table[e][i] = (one_lower << 8) ^ table[0][((one_lower >> 120) & 0xFF) as usize];
            }
            e += 1;
        }
        i += 1;
    }
    table
}

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
    pub const fn new_32(algorithm: &Algorithm<u32>) -> Self {
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

        let px =
            1 << algorithm.width | (algorithm.poly as u64) << (u32::BITS as u8 - algorithm.width);
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

    pub const fn new_64(algorithm: &Algorithm<u64>) -> Self {
        const fn xt_mod_px(mut t: u32, px: u64) -> u64 {
            if t < 64 {
                return 0;
            }
            t -= 63;

            let mut n = 0x08000000000000000;
            let mut i = 0;
            while i < t {
                n = (n << 1) ^ ((0 - (n >> 63)) & px);
                i += 1;
            }
            n
        }

        const fn u(px: u64) -> u64 {
            let mut q = 0;
            let mut n = 0x10000000000000000;
            let mut i = 0;
            while i < 65 {
                q <<= 1;
                if n & 0x10000000000000000 != 0 {
                    q |= 1;
                    n ^= px as u128;
                }
                n <<= 1;
                i += 1;
            }
            q
        }

        let px = 1u64 << algorithm.width | algorithm.poly << (u64::BITS as u8 - algorithm.width);
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
