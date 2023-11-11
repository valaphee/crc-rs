use crate::util::*;

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
pub struct PclmulqdqCoefficients {
    pub k1: i64,
    pub k2: i64,
    pub k3: i64,
    pub k4: i64,
    pub k5: i64,
    pub k6: i64,
    pub px: i64,
    pub u: i64,
}

impl PclmulqdqCoefficients {
    pub const fn new(rk08: u64, width: u8) -> Self {
        let rk08 = 1u64 << width | rk08;
        const fn grk07(rk08: u64) -> u64 {
            let mut q = 0;
            let mut n = 0x100000000;
            let mut i = 0;
            while i < 33 {
                q <<= 1;
                if n & 0x100000000 != 0 {
                    q |= 1;
                    n ^= rk08;
                }
                n <<= 1;
                i += 1;
            }
            q
        }

        const fn grk(rk08: u64, mut e: u64) -> u64 {
            if e < 32 {
                return 0;
            }
            e -= 31;

            let mut n = 0x080000000;
            let mut i = 0;
            while i < e {
                n <<= 1;
                if n & 0x100000000 != 0 {
                    n ^= rk08;
                }
                i += 1;
            }
            n << 32
        }

        Self {
            k1: (grk(rk08, 4 * 128 + 32).reverse_bits() << 1) as i64,
            k2: (grk(rk08, 4 * 128 - 32).reverse_bits() << 1) as i64,
            k3: (grk(rk08, 128 + 32).reverse_bits() << 1) as i64,
            k4: (grk(rk08, 128 - 32).reverse_bits() << 1) as i64,
            k5: (grk(rk08, 64).reverse_bits() << 1) as i64,
            k6: (grk(rk08, 32).reverse_bits() << 1) as i64,
            px: (rk08.reverse_bits() >> 31) as i64,
            u: (grk07(rk08).reverse_bits() >> 31) as i64,
        }
    }
}
