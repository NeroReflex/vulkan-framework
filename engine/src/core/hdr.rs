#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct HDR {
    gamma: f32,
    exposure: f32,
}

impl Default for HDR {
    fn default() -> Self {
        Self {
            gamma: 2.2,
            exposure: 0.1,
        }
    }
}
