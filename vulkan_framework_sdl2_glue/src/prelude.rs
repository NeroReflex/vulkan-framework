use std::ffi::c_char;
use std::ffi::CStr;

pub struct SDL2Error {
    error: Vec<c_char>,
}

impl SDL2Error {
    pub(crate) fn new(err_str: *const c_char) -> Self {
        let mut raw = Vec::<c_char>::new();

        if !err_str.is_null() {
            unsafe {
                raw = CStr::from_ptr(err_str)
                    .to_bytes()
                    .iter()
                    .map(|ch| *ch as c_char)
                    .collect::<Vec<c_char>>();
            };
        } else {
            // raw will contain nothing
        }

        Self { error: raw }
    }
}

/*impl ToString for SDL2Error {
    fn to_string(&self) -> String {
        let mut str = String::from("");
        for ch in self.error.iter() {
            str.push((*ch as u8) as char);
        }

        str
    }
}*/

impl std::fmt::Display for SDL2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut str = String::from("");
        for ch in self.error.iter() {
            str.push((*ch as u8) as char);
        }

        write!(f, "{}", str)
    }
}
