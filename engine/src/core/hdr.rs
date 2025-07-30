#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct HDR {
    gamma: f32,
    exposure: f32,
}

impl Default for HDR {
    #[inline]
    fn default() -> Self {
        Self {
            gamma: 2.2,
            exposure: 0.1,
        }
    }
}

impl HDR {
    /*
    pub fn gamma(&self) -> f32 {
        unsafe { read_unaligned(&self.gamma as *const _) }
    }

    pub fn exposure(&self) -> f32 {
        self.exposure.to_owned()
    }
    */

    #[inline]
    pub fn new(gamma: f32, exposure: f32) -> Self {
        Self { gamma, exposure }
    }
}
