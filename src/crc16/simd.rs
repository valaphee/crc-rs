use crate::crc32::update_simd;
use crate::simd::SimdConstants;
use crate::table::crc16_table_slice_16;
use crate::{Algorithm, Crc, Digest, Simd};

use super::{finalize, init};

impl Crc<Simd<u16>> {
    pub const fn new(algorithm: &'static Algorithm<u16>) -> Self {
        let table = crc16_table_slice_16(algorithm.width, algorithm.poly, algorithm.refin);
        Self {
            algorithm,
            table: (table, SimdConstants::new_32(&Algorithm {
                width: algorithm.width,
                poly: algorithm.poly as u32,
                init: algorithm.init as u32,
                refin: algorithm.refin,
                refout: algorithm.refout,
                xorout: algorithm.xorout as u32,
                check: algorithm.check as u32,
                residue: algorithm.residue as u32,
            })),
        }
    }

    pub fn checksum(&self, bytes: &[u8]) -> u16 {
        let mut crc = init(self.algorithm, self.algorithm.init);
        crc = self.update(crc, bytes);
        finalize(self.algorithm, crc)
    }

    fn update(&self, crc: u16, bytes: &[u8]) -> u16 {
        // TODO
        unsafe {
            update_simd(
                crc as u32,
                &Algorithm {
                    width: self.algorithm.width,
                    poly: self.algorithm.poly as u32,
                    init: self.algorithm.init as u32,
                    refin: self.algorithm.refin,
                    refout: self.algorithm.refout,
                    xorout: self.algorithm.xorout as u32,
                    check: self.algorithm.check as u32,
                    residue: self.algorithm.residue as u32,
                },
                &self.table.1,
                bytes,
            ) as u16
        }
    }

    pub fn digest(&self) -> Digest<Simd<u16>> {
        self.digest_with_initial(self.algorithm.init)
    }

    /// Construct a `Digest` with a given initial value.
    ///
    /// This overrides the initial value specified by the algorithm.
    /// The effects of the algorithm's properties `refin` and `width`
    /// are applied to the custom initial value.
    pub fn digest_with_initial(&self, initial: u16) -> Digest<Simd<u16>> {
        let value = init(self.algorithm, initial);
        Digest::new(self, value)
    }
}

impl<'a> Digest<'a, Simd<u16>> {
    const fn new(crc: &'a Crc<Simd<u16>>, value: u16) -> Self {
        Digest { crc, value }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.value = self.crc.update(self.value, bytes);
    }

    pub const fn finalize(self) -> u16 {
        finalize(self.crc.algorithm, self.value)
    }
}
