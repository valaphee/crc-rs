use crate::util::crc32;
use crate::PclmulqdqCoefficients;
use core::arch::x86_64 as arch;
use crc_catalog::Algorithm;

mod bytewise;
mod default;
mod nolookup;
mod pclmulqdq;
mod slice16;

// init is shared between all impls
const fn init(algorithm: &Algorithm<u32>, initial: u32) -> u32 {
    if algorithm.refin {
        initial.reverse_bits() >> (32u8 - algorithm.width)
    } else {
        initial << (32u8 - algorithm.width)
    }
}

// finalize is shared between all impls
const fn finalize(algorithm: &Algorithm<u32>, mut crc: u32) -> u32 {
    if algorithm.refin ^ algorithm.refout {
        crc = crc.reverse_bits();
    }
    if !algorithm.refout {
        crc >>= 32u8 - algorithm.width;
    }
    crc ^ algorithm.xorout
}

const fn update_nolookup(mut crc: u32, algorithm: &Algorithm<u32>, bytes: &[u8]) -> u32 {
    let poly = if algorithm.refin {
        let poly = algorithm.poly.reverse_bits();
        poly >> (32u8 - algorithm.width)
    } else {
        algorithm.poly << (32u8 - algorithm.width)
    };

    let mut i = 0;
    if algorithm.refin {
        while i < bytes.len() {
            let to_crc = (crc ^ bytes[i] as u32) & 0xFF;
            crc = crc32(poly, algorithm.refin, to_crc) ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i < bytes.len() {
            let to_crc = ((crc >> 24) ^ bytes[i] as u32) & 0xFF;
            crc = crc32(poly, algorithm.refin, to_crc) ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

const fn update_bytewise(mut crc: u32, reflect: bool, table: &[u32; 256], bytes: &[u8]) -> u32 {
    let mut i = 0;
    if reflect {
        while i < bytes.len() {
            let table_index = ((crc ^ bytes[i] as u32) & 0xFF) as usize;
            crc = table[table_index] ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i < bytes.len() {
            let table_index = (((crc >> 24) ^ bytes[i] as u32) & 0xFF) as usize;
            crc = table[table_index] ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

const fn update_slice16(
    mut crc: u32,
    reflect: bool,
    table: &[[u32; 256]; 16],
    bytes: &[u8],
) -> u32 {
    let mut i = 0;
    if reflect {
        while i + 16 <= bytes.len() {
            let mut current_slice = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];

            current_slice[0] ^= crc as u8;
            current_slice[1] ^= (crc >> 8) as u8;
            current_slice[2] ^= (crc >> 16) as u8;
            current_slice[3] ^= (crc >> 24) as u8;

            crc = table[0][bytes[i + 15] as usize]
                ^ table[1][bytes[i + 14] as usize]
                ^ table[2][bytes[i + 13] as usize]
                ^ table[3][bytes[i + 12] as usize]
                ^ table[4][bytes[i + 11] as usize]
                ^ table[5][bytes[i + 10] as usize]
                ^ table[6][bytes[i + 9] as usize]
                ^ table[7][bytes[i + 8] as usize]
                ^ table[8][bytes[i + 7] as usize]
                ^ table[9][bytes[i + 6] as usize]
                ^ table[10][bytes[i + 5] as usize]
                ^ table[11][bytes[i + 4] as usize]
                ^ table[12][current_slice[3] as usize]
                ^ table[13][current_slice[2] as usize]
                ^ table[14][current_slice[1] as usize]
                ^ table[15][current_slice[0] as usize];

            i += 16;
        }

        // Last few bytes
        while i < bytes.len() {
            let table_index = ((crc ^ bytes[i] as u32) & 0xFF) as usize;
            crc = table[0][table_index] ^ (crc >> 8);
            i += 1;
        }
    } else {
        while i + 16 <= bytes.len() {
            let mut current_slice = [bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]];

            current_slice[0] ^= (crc >> 24) as u8;
            current_slice[1] ^= (crc >> 16) as u8;
            current_slice[2] ^= (crc >> 8) as u8;
            current_slice[3] ^= crc as u8;

            crc = table[0][bytes[i + 15] as usize]
                ^ table[1][bytes[i + 14] as usize]
                ^ table[2][bytes[i + 13] as usize]
                ^ table[3][bytes[i + 12] as usize]
                ^ table[4][bytes[i + 11] as usize]
                ^ table[5][bytes[i + 10] as usize]
                ^ table[6][bytes[i + 9] as usize]
                ^ table[7][bytes[i + 8] as usize]
                ^ table[8][bytes[i + 7] as usize]
                ^ table[9][bytes[i + 6] as usize]
                ^ table[10][bytes[i + 5] as usize]
                ^ table[11][bytes[i + 4] as usize]
                ^ table[12][current_slice[3] as usize]
                ^ table[13][current_slice[2] as usize]
                ^ table[14][current_slice[1] as usize]
                ^ table[15][current_slice[0] as usize];

            i += 16;
        }

        // Last few bytes
        while i < bytes.len() {
            let table_index = (((crc >> 24) ^ bytes[i] as u32) & 0xFF) as usize;
            crc = table[0][table_index] ^ (crc << 8);
            i += 1;
        }
    }
    crc
}

unsafe fn update_pclmulqdq(
    crc: u32,
    algorithm: &Algorithm<u32>,
    table: &PclmulqdqCoefficients,
    mut bytes: &[u8],
) -> u32 {
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
    unsafe fn mx_mod_px(gx_mod_px: arch::__m128i, hx_lx: arch::__m128i, x_mod_p: arch::__m128i) -> arch::__m128i {
        arch::_mm_xor_si128(arch::_mm_xor_si128(hx_lx, arch::_mm_clmulepi64_si128(gx_mod_px, x_mod_p, 0x00)), arch::_mm_clmulepi64_si128(gx_mod_px, x_mod_p, 0x11))
    }

    // Step 1 - Iteratively Fold by 4:
    let mut x3 = next(&mut bytes);
    let mut x2 = next(&mut bytes);
    let mut x1 = next(&mut bytes);
    let mut x0 = next(&mut bytes);
    x3 = arch::_mm_xor_si128(x3, arch::_mm_cvtsi32_si128(crc as i32));
    let k1_k2 = arch::_mm_set_epi64x(table.k2, table.k1);
    while bytes.len() >= 64 {
        x3 = mx_mod_px(x3, next(&mut bytes), k1_k2);
        x2 = mx_mod_px(x2, next(&mut bytes), k1_k2);
        x1 = mx_mod_px(x1, next(&mut bytes), k1_k2);
        x0 = mx_mod_px(x0, next(&mut bytes), k1_k2);
    }

    // Step 2 - Iteratively Fold by 1:
    let k3_k4 = arch::_mm_set_epi64x(table.k4, table.k3);
    let mut x = mx_mod_px(x3, x2, k3_k4);
    x = mx_mod_px(x, x1, k3_k4);
    x = mx_mod_px(x, x0, k3_k4);
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
            arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, 0, !0)),
            arch::_mm_set_epi64x(0, table.k5),
            0x00,
        ),
        arch::_mm_srli_si128(x, 4),
    );

    // Algorithm 1. Barrett Reduction Algorithm for a degree-32 polynomial modulus (polynomials defined over GF(2))
    let px_u = arch::_mm_set_epi64x(table.u, table.px);

    // Step 1: T1(x) = ⌊(R(x) % x^32)⌋ • μ
    let t1 = arch::_mm_clmulepi64_si128(
        arch::_mm_and_si128(x, arch::_mm_set_epi32(0, 0, 0, !0)),
        px_u,
        0x10,
    );

    // Step 2: T2(x) = ⌊(T1(x) % x^32)⌋ • P(x)
    let t2 = arch::_mm_clmulepi64_si128(
        arch::_mm_and_si128(t1, arch::_mm_set_epi32(0, 0, 0, !0)),
        px_u,
        0x00,
    );

    // Step 3: C(x) = R(x) ⊕ T2(x) % x^32
    let c = arch::_mm_extract_epi32(arch::_mm_xor_si128(x, t2), 1) as u32;

    if !bytes.is_empty() {
        update_nolookup(c, algorithm, bytes)
    } else {
        c
    }
}

#[cfg(test)]
mod test {
    use crate::{Bytewise, Crc, Implementation, NoTable, Slice16};
    use crc_catalog::{Algorithm, CRC_32_ISCSI};

    #[test]
    fn default_table_size() {
        const TABLE_SIZE: usize = core::mem::size_of::<<u32 as Implementation>::Table>();
        const BYTES_PER_ENTRY: usize = 4;
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

        pub const CRC_32_ISCSI_NONREFLEX: Algorithm<u32> = Algorithm {
            width: 32,
            poly: 0x1edc6f41,
            init: 0xffffffff,
            // This is the only flag that affects the optimized code path
            refin: false,
            refout: true,
            xorout: 0xffffffff,
            check: 0xe3069283,
            residue: 0xb798b438,
        };

        let algs_to_test = [&CRC_32_ISCSI, &CRC_32_ISCSI_NONREFLEX];

        for alg in algs_to_test {
            for data in data {
                let crc_slice16 = Crc::<Slice16<u32>>::new(alg);
                let crc_nolookup = Crc::<NoTable<u32>>::new(alg);
                let expected = Crc::<Bytewise<u32>>::new(alg).checksum(data.as_bytes());

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
