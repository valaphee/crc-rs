use crate::util::crc64;
use crate::SimdConstants;
use core::arch::x86_64 as arch;
use crc_catalog::Algorithm;

mod bytewise;
mod default;
mod nolookup;
mod simd;
mod slice16;

const fn init(algorithm: &Algorithm<u64>, initial: u64) -> u64 {
    if algorithm.refin {
        initial.reverse_bits() >> (64u8 - algorithm.width)
    } else {
        initial << (64u8 - algorithm.width)
    }
}

const fn finalize(algorithm: &Algorithm<u64>, mut crc: u64) -> u64 {
    if algorithm.refin ^ algorithm.refout {
        crc = crc.reverse_bits();
    }
    if !algorithm.refout {
        crc >>= 64u8 - algorithm.width;
    }
    crc ^ algorithm.xorout
}

const fn update_nolookup(mut crc: u64, algorithm: &Algorithm<u64>, bytes: &[u8]) -> u64 {
    let poly = if algorithm.refin {
        let poly = algorithm.poly.reverse_bits();
        poly >> (64u8 - algorithm.width)
    } else {
        algorithm.poly << (64u8 - algorithm.width)
    };

    let mut i = 0;
    if algorithm.refin {
        while i < bytes.len() {
            let to_crc = (crc ^ bytes[i] as u64) & 0xFF;
            crc = crc64(poly, algorithm.refin, to_crc) ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i < bytes.len() {
            let to_crc = ((crc >> 56) ^ bytes[i] as u64) & 0xFF;
            crc = crc64(poly, algorithm.refin, to_crc) ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

const fn update_bytewise(mut crc: u64, reflect: bool, table: &[u64; 256], bytes: &[u8]) -> u64 {
    let mut i = 0;
    let len = bytes.len();
    if reflect {
        while i < len {
            let table_index = ((crc ^ bytes[i] as u64) & 0xFF) as usize;
            crc = table[table_index] ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i < len {
            let table_index = (((crc >> 56) ^ bytes[i] as u64) & 0xFF) as usize;
            crc = table[table_index] ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

const fn update_slice16(
    mut crc: u64,
    reflect: bool,
    table: &[[u64; 256]; 16],
    bytes: &[u8],
) -> u64 {
    let mut i = 0;
    let len = bytes.len();
    if reflect {
        while i + 16 <= len {
            let current0 = bytes[i] ^ crc as u8;
            let current1 = bytes[i + 1] ^ (crc >> 8) as u8;
            let current2 = bytes[i + 2] ^ (crc >> 16) as u8;
            let current3 = bytes[i + 3] ^ (crc >> 24) as u8;
            let current4 = bytes[i + 4] ^ (crc >> 32) as u8;
            let current5 = bytes[i + 5] ^ (crc >> 40) as u8;
            let current6 = bytes[i + 6] ^ (crc >> 48) as u8;
            let current7 = bytes[i + 7] ^ (crc >> 56) as u8;

            crc = table[0][bytes[i + 15] as usize]
                ^ table[1][bytes[i + 14] as usize]
                ^ table[2][bytes[i + 13] as usize]
                ^ table[3][bytes[i + 12] as usize]
                ^ table[4][bytes[i + 11] as usize]
                ^ table[5][bytes[i + 10] as usize]
                ^ table[6][bytes[i + 9] as usize]
                ^ table[7][bytes[i + 8] as usize]
                ^ table[8][current7 as usize]
                ^ table[9][current6 as usize]
                ^ table[10][current5 as usize]
                ^ table[11][current4 as usize]
                ^ table[12][current3 as usize]
                ^ table[13][current2 as usize]
                ^ table[14][current1 as usize]
                ^ table[15][current0 as usize];

            i += 16;
        }

        while i < len {
            let table_index = ((crc ^ bytes[i] as u64) & 0xFF) as usize;
            crc = table[0][table_index] ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i + 16 <= len {
            let current0 = bytes[i] ^ (crc >> 56) as u8;
            let current1 = bytes[i + 1] ^ (crc >> 48) as u8;
            let current2 = bytes[i + 2] ^ (crc >> 40) as u8;
            let current3 = bytes[i + 3] ^ (crc >> 32) as u8;
            let current4 = bytes[i + 4] ^ (crc >> 24) as u8;
            let current5 = bytes[i + 5] ^ (crc >> 16) as u8;
            let current6 = bytes[i + 6] ^ (crc >> 8) as u8;
            let current7 = bytes[i + 7] ^ crc as u8;

            crc = table[0][bytes[i + 15] as usize]
                ^ table[1][bytes[i + 14] as usize]
                ^ table[2][bytes[i + 13] as usize]
                ^ table[3][bytes[i + 12] as usize]
                ^ table[4][bytes[i + 11] as usize]
                ^ table[5][bytes[i + 10] as usize]
                ^ table[6][bytes[i + 9] as usize]
                ^ table[7][bytes[i + 8] as usize]
                ^ table[8][current7 as usize]
                ^ table[9][current6 as usize]
                ^ table[10][current5 as usize]
                ^ table[11][current4 as usize]
                ^ table[12][current3 as usize]
                ^ table[13][current2 as usize]
                ^ table[14][current1 as usize]
                ^ table[15][current0 as usize];

            i += 16;
        }

        while i < len {
            let table_index = (((crc >> 56) ^ bytes[i] as u64) & 0xFF) as usize;
            crc = table[0][table_index] ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

#[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
pub(crate) unsafe fn update_simd(
    crc: u64,
    algorithm: &Algorithm<u64>,
    constants: &SimdConstants,
    mut bytes: &[u8],
) -> u64 {
    if bytes.len() < 128 {
        return update_nolookup(crc, algorithm, bytes);
    }

    unsafe fn next(bytes: &mut &[u8]) -> arch::__m128i {
        debug_assert!(bytes.len() >= 16);
        let value = arch::_mm_loadu_si128(bytes.as_ptr() as *const arch::__m128i);
        *bytes = &bytes[16..];
        return value;
    }

    // M(x) mod P(x) = {H(x) • [x^(T+64) % P(x)]} ⊕ {L(x) • [x^T % P(x)]} ⊕ [G(x) % P(x)]
    unsafe fn mx_mod_px(
        gx_mod_px: arch::__m128i,
        hx_lx: arch::__m128i,
        x_mod_p: arch::__m128i,
    ) -> arch::__m128i {
        arch::_mm_xor_si128(
            arch::_mm_xor_si128(hx_lx, arch::_mm_clmulepi64_si128(gx_mod_px, x_mod_p, 0x00)),
            arch::_mm_clmulepi64_si128(gx_mod_px, x_mod_p, 0x11),
        )
    }

    // Step 1 - Iteratively Fold by 4:
    let mut x0 = next(&mut bytes);
    let mut x1 = next(&mut bytes);
    let mut x2 = next(&mut bytes);
    let mut x3 = next(&mut bytes);
    x0 = arch::_mm_xor_si128(x0, arch::_mm_cvtsi64_si128(crc as i64));
    let k1_k2 = arch::_mm_set_epi64x(constants.k2 as i64, constants.k1 as i64);
    while bytes.len() >= 64 {
        x0 = mx_mod_px(x0, next(&mut bytes), k1_k2);
        x1 = mx_mod_px(x1, next(&mut bytes), k1_k2);
        x2 = mx_mod_px(x2, next(&mut bytes), k1_k2);
        x3 = mx_mod_px(x3, next(&mut bytes), k1_k2);
    }

    // Step 2 - Iteratively Fold by 1:
    let k3_k4 = arch::_mm_set_epi64x(constants.k4 as i64, constants.k3 as i64);
    let mut x = mx_mod_px(x0, x1, k3_k4);
    x = mx_mod_px(x, x2, k3_k4);
    x = mx_mod_px(x, x3, k3_k4);
    while bytes.len() >= 16 {
        x = mx_mod_px(x, next(&mut bytes), k3_k4);
    }

    // Step 3 - Final Reduction of 128-bits
    let x = arch::_mm_xor_si128(
        arch::_mm_clmulepi64_si128(x, k3_k4, 0x10),
        arch::_mm_srli_si128(x, 8),
    );
    let x = arch::_mm_xor_si128(
        arch::_mm_clmulepi64_si128(
            x,
            arch::_mm_set_epi64x(0, constants.k5 as i64),
            0x00,
        ),
        arch::_mm_srli_si128(x, 4),
    );

    // Algorithm 1. Barrett Reduction Algorithm for a degree-32 polynomial modulus (polynomials defined over GF(2))
    let px_u = arch::_mm_set_epi64x(constants.u as i64, constants.px as i64);

    // Step 1: T1(x) = ⌊(R(x) % x^32)⌋ • μ
    let t1 = arch::_mm_clmulepi64_si128(
        x,
        px_u,
        0x10,
    );

    // Step 2: T2(x) = ⌊(T1(x) % x^32)⌋ • P(x)
    let t2 = arch::_mm_clmulepi64_si128(
        t1,
        px_u,
        0x00,
    );

    // Step 3: C(x) = R(x) ⊕ T2(x) % x^32
    let cx = arch::_mm_extract_epi64(arch::_mm_xor_si128(x, t2), 1) as u64;

    if !bytes.is_empty() {
        update_nolookup(cx, algorithm, bytes)
    } else {
        cx
    }
}

#[cfg(test)]
mod test {
    use crate::{Bytewise, Crc, Implementation, NoTable, Slice16};
    use crc_catalog::{Algorithm, CRC_64_ECMA_182};

    #[test]
    fn default_table_size() {
        const TABLE_SIZE: usize = core::mem::size_of::<<u64 as Implementation>::Table>();
        const BYTES_PER_ENTRY: usize = 8;
        #[cfg(all(
            feature = "no-table-mem-limit",
            feature = "bytewise-mem-limit",
            feature = "slice16-mem-limit"
        ))]
        {
            const EXPECTED: usize = 0;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }
        #[cfg(all(
            feature = "no-table-mem-limit",
            feature = "bytewise-mem-limit",
            not(feature = "slice16-mem-limit")
        ))]
        {
            const EXPECTED: usize = 0;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }
        #[cfg(all(
            feature = "no-table-mem-limit",
            not(feature = "bytewise-mem-limit"),
            feature = "slice16-mem-limit"
        ))]
        {
            const EXPECTED: usize = 0;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }
        #[cfg(all(
            feature = "no-table-mem-limit",
            not(feature = "bytewise-mem-limit"),
            not(feature = "slice16-mem-limit")
        ))]
        {
            const EXPECTED: usize = 0;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }

        #[cfg(all(
            not(feature = "no-table-mem-limit"),
            feature = "bytewise-mem-limit",
            feature = "slice16-mem-limit"
        ))]
        {
            const EXPECTED: usize = 256 * BYTES_PER_ENTRY;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }
        #[cfg(all(
            not(feature = "no-table-mem-limit"),
            feature = "bytewise-mem-limit",
            not(feature = "slice16-mem-limit")
        ))]
        {
            const EXPECTED: usize = 256 * BYTES_PER_ENTRY;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }

        #[cfg(all(
            not(feature = "no-table-mem-limit"),
            not(feature = "bytewise-mem-limit"),
            feature = "slice16-mem-limit"
        ))]
        {
            const EXPECTED: usize = 256 * 16 * BYTES_PER_ENTRY;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }

        #[cfg(all(
            not(feature = "no-table-mem-limit"),
            not(feature = "bytewise-mem-limit"),
            not(feature = "slice16-mem-limit")
        ))]
        {
            const EXPECTED: usize = 256 * BYTES_PER_ENTRY;
            let _ = EXPECTED;
            const _: () = assert!(EXPECTED == TABLE_SIZE);
        }
        let _ = TABLE_SIZE;
        let _ = BYTES_PER_ENTRY;
    }

    /// Test this optimized version against the well known implementation to ensure correctness
    #[test]
    fn correctness() {
        let data: &[&str] = &[
        "",
        "1",
        "1234",
        "123456789",
        "0123456789ABCDE",
        "01234567890ABCDEFGHIJK",
        "01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK01234567890ABCDEFGHIJK",
    ];

        pub const CRC_64_ECMA_182_REFLEX: Algorithm<u64> = Algorithm {
            width: 64,
            poly: 0x42f0e1eba9ea3693,
            init: 0x0000000000000000,
            refin: true,
            refout: false,
            xorout: 0x0000000000000000,
            check: 0x6c40df5f0b497347,
            residue: 0x0000000000000000,
        };

        let algs_to_test = [&CRC_64_ECMA_182, &CRC_64_ECMA_182_REFLEX];

        for alg in algs_to_test {
            for data in data {
                let crc_slice16 = Crc::<Slice16<u64>>::new(alg);
                let crc_nolookup = Crc::<NoTable<u64>>::new(alg);
                let expected = Crc::<Bytewise<u64>>::new(alg).checksum(data.as_bytes());

                // Check that doing all at once works as expected
                assert_eq!(crc_slice16.checksum(data.as_bytes()), expected);
                assert_eq!(crc_nolookup.checksum(data.as_bytes()), expected);

                let mut digest = crc_slice16.digest();
                digest.update(data.as_bytes());
                assert_eq!(digest.finalize(), expected);

                let mut digest = crc_nolookup.digest();
                digest.update(data.as_bytes());
                assert_eq!(digest.finalize(), expected);

                // Check that we didn't break updating from multiple sources
                if data.len() > 2 {
                    let data = data.as_bytes();
                    let data1 = &data[..data.len() / 2];
                    let data2 = &data[data.len() / 2..];
                    let mut digest = crc_slice16.digest();
                    digest.update(data1);
                    digest.update(data2);
                    assert_eq!(digest.finalize(), expected);
                    let mut digest = crc_nolookup.digest();
                    digest.update(data1);
                    digest.update(data2);
                    assert_eq!(digest.finalize(), expected);
                }
            }
        }
    }
}
