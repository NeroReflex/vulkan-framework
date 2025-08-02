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
        unsafe { std::ptr::read(&self.gamma as *const f32) }

        let gamma_ptr = &self.gamma as *const _ as *const std::ffi::c_void;
        unsafe { read_unaligned(gamma_ptr as *const f32) }
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
