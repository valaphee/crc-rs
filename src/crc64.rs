use crate::{simd::{SimdConstants, SimdValue, SimdValueExt}, util::crc64};
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
    mut crc: u64,
    algorithm: &Algorithm<u64>,
    constants: &SimdConstants,
    bytes: &[u8],
) -> u64 {
    //println!("{:X?}", constants);
    let (bytes_before, values, bytes_after) = bytes.align_to::<[SimdValue; 4]>();
    //let (chunks, remaining_values) = values.as_chunks::<8>();

    // Less than bytes needed for 16-byte alignment + 128
    let Some(x4) = values.get(0) else {
        return update_nolookup(crc, algorithm, bytes);
    };

    crc = update_nolookup(crc, algorithm, bytes_before);

    // Step 1 - Iteratively Fold by 4:
    let mut x4 = *x4;
    x4[0] ^= SimdValue::new([crc as u64, 0]);
    let k1_k2 = SimdValue::new([constants.k1, constants.k2]);
    for chunk in &values[1..] {
        x4[0] = x4[0].fold_16(k1_k2) ^ chunk[0];
        x4[1] = x4[1].fold_16(k1_k2) ^ chunk[1];
        x4[2] = x4[2].fold_16(k1_k2) ^ chunk[2];
        x4[3] = x4[3].fold_16(k1_k2) ^ chunk[3];
        /*x4[4] = x4[4].fold_16(k1_k2) ^ chunk[4];
        x4[5] = x4[5].fold_16(k1_k2) ^ chunk[5];
        x4[6] = x4[6].fold_16(k1_k2) ^ chunk[6];
        x4[7] = x4[7].fold_16(k1_k2) ^ chunk[7];*/
    }

    // Step 2 - Iteratively Fold by 1:
    /*let coeffs = [
        SimdValue::new([constants.k3, constants.k4]), // fold by distance of 112 bytes
        SimdValue::new([constants.k3_1, constants.k4_1]), // fold by distance of 96 bytes
        SimdValue::new([constants.k3_2, constants.k4_2]), // fold by distance of 80 bytes
        SimdValue::new([constants.k3_3, constants.k4_3]), // fold by distance of 64 bytes
        SimdValue::new([constants.k3_4, constants.k4_4]), // fold by distance of 48 bytes
        SimdValue::new([constants.k3_5, constants.k4_5]), // fold by distance of 32 bytes
        SimdValue::new([constants.k3_6, constants.k4_6]), // fold by distance of 16 bytes
    ];*/
    let k3_k4 = SimdValue::new([constants.k3, constants.k4]);
    let mut x = x4[0].fold_16(k3_k4) ^ x4[1];
    x = x.fold_16(k3_k4) ^ x4[2];
    x = x.fold_16(k3_k4) ^ x4[3];
    //let x = x4.iter().zip(&coeffs).fold(x4[7], |acc, (m, c)| acc ^ m.fold_16(*c));
    /*for block in remaining_values {
        x = x.fold_16(k3_k4) ^ *block;
    }*/

    // Step 3 - Final Reduction of 128-bits
    let k5_k5 = SimdValue::new([constants.k5, constants.k5]);
    let x = x.fold_8(k5_k5);

    // Barrett Reduction
    let px_u = SimdValue::new([constants.px, constants.u]);
    let cx = x.barret_reduction_64(px_u);

    update_nolookup(cx, algorithm, bytes_after)
}

#[cfg(test)]
mod test {
    use crate::{Bytewise, Crc, Implementation, NoTable, Simd, Slice16};
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
        /*"",
        "1",
        "1234",
        "123456789",
        "0123456789ABCDE",
        "01234567890ABCDEFGHIJK",*/
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

        let algs_to_test = [/*&CRC_64_ECMA_182,*/ &CRC_64_ECMA_182_REFLEX];

        for alg in algs_to_test {
            for data in data {
                let crc_slice16 = Crc::<Slice16<u64>>::new(alg);
                let crc_nolookup: Crc<NoTable<u64>> = Crc::<NoTable<u64>>::new(alg);
                let crc_simd = Crc::<Simd<u64>>::new(alg);
                let expected = Crc::<Bytewise<u64>>::new(alg).checksum(data.as_bytes());

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
