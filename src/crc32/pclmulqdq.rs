use crate::{Algorithm, Crc, Digest, Sse, SseCoefficients};

use super::{finalize, init, update_sse};

impl Crc<Sse<u32>> {
    pub const fn new(algorithm: &'static Algorithm<u32>) -> Self {
        Self {
            algorithm,
            table: SseCoefficients::new(algorithm),
        }
    }

    pub fn checksum(&self, bytes: &[u8]) -> u32 {
        let mut crc = init(self.algorithm, self.algorithm.init);
        crc = self.update(crc, bytes);
        finalize(self.algorithm, crc)
    }

    fn update(&self, crc: u32, bytes: &[u8]) -> u32 {
        unsafe { update_sse(crc, self.algorithm, &self.table, bytes) }
    }

    pub fn digest(&self) -> Digest<Sse<u32>> {
        self.digest_with_initial(self.algorithm.init)
    }

    /// Construct a `Digest` with a given initial value.
    ///
    /// This overrides the initial value specified by the algorithm.
    /// The effects of the algorithm's properties `refin` and `width`
    /// are applied to the custom initial value.
    pub fn digest_with_initial(&self, initial: u32) -> Digest<Sse<u32>> {
        let value = init(self.algorithm, initial);
        Digest::new(self, value)
    }
}

impl<'a> Digest<'a, Sse<u32>> {
    const fn new(crc: &'a Crc<Sse<u32>>, value: u32) -> Self {
        Digest { crc, value }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        self.value = self.crc.update(self.value, bytes);
    }

    pub const fn finalize(self) -> u32 {
        finalize(self.crc.algorithm, self.value)
    }
}
