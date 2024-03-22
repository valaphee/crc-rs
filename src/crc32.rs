use crate::simd::{SimdValue, SimdValueOps};
use crate::util::crc32;
use crc_catalog::Algorithm;

mod bytewise;
mod default;
mod nolookup;
mod simd;
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

#[target_feature(enable = "pclmulqdq", enable = "sse2", enable = "sse4.1")]
pub(crate) unsafe fn update_simd(
    mut crc: u32,
    coefficients: &[SimdValue; 4],
    first_chunk: &[SimdValue; 4],
    chunks: &[[SimdValue; 4]],
) -> u32 {
    let mut x4 = *first_chunk;
    x4[0] = x4[0].xor(crc as u64);

    let k1_k2 = coefficients[0];
    for chunk in chunks {
        for (x, value) in x4.iter_mut().zip(chunk.iter()) {
            *x = x.fold_16(k1_k2, *value)
        }
    }

    let k3_k4 = coefficients[1];
    let mut x = x4[0].fold_16(k3_k4, x4[1]);
    x = x.fold_16(k3_k4, x4[2]);
    x = x.fold_16(k3_k4, x4[3]);

    let k5_k6 = coefficients[2];
    let x = x.fold_8(k3_k4);
    let x = x.fold_4(k5_k6);

    let px_u = coefficients[3];
    crc = x.barret_reduction_32(px_u);

    return crc;
}

#[cfg(test)]
mod test {
    use crate::{Bytewise, Crc, Implementation, NoTable, Simd, Slice16};
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
            //"",
            //"1",
            //"1234",
            //"123456789",
            //"0123456789A",
            //"01234567890ABCDEFGHIJK",
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
                let crc_simd = Crc::<Simd<u32>>::new(alg);
                let expected = Crc::<Bytewise<u32>>::new(alg).checksum(data.as_bytes());

                // Check that doing all at once works as expected
                assert_eq!(crc_slice16.checksum(data.as_bytes()), expected);
                assert_eq!(crc_nolookup.checksum(data.as_bytes()), expected);
                assert_eq!(crc_simd.checksum(data.as_bytes()), expected);

                let mut digest = crc_slice16.digest();
                digest.update(data.as_bytes());
                assert_eq!(digest.finalize(), expected);

                let mut digest = crc_nolookup.digest();
                digest.update(data.as_bytes());
                assert_eq!(digest.finalize(), expected);

                let mut digest = crc_simd.digest();
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
                    let mut digest = crc_simd.digest();
                    digest.update(data1);
                    digest.update(data2);
                    assert_eq!(digest.finalize(), expected);
                }
            }
        }
    }
}
