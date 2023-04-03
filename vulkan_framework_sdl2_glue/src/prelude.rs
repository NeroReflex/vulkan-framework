use std::ffi::c_char;

pub struct SDL2Error {
    error: Vec<c_char>
}

impl SDL2Error {
    pub(crate) fn new(err_str: *const c_char) -> Self {
        todo!()
    }
}