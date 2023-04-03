use std::ffi::c_char;
use std::ffi::c_int;

use crate::prelude::*;

use sdl2_sys::*;
use sdl2_sys::SDL_WindowFlags::*;

pub struct Window {
    window: *mut SDL_Window,

}

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            SDL_DestroyWindow(self.window);
        }
    }
}

impl Window {
    

    pub fn new(title: &String, width: c_int, height: c_int, x_position: Option<c_int>, y_position: Option<c_int>) -> Result<Self, SDL2Error> {
        let mut title_bytes = title.as_bytes()
            .iter()
            .map(|c| *c as c_char)
            .collect::<Vec<c_char>>();

        // the title C-string nul-terminator
        title_bytes.push(0);

        let window: *mut SDL_Window;

        unsafe {
            
            window = SDL_CreateWindow(
                title_bytes.as_slice().as_ptr(),
                match x_position {
                    Some(x_pos) => { x_pos as c_int },
                    _ => { SDL_WINDOWPOS_UNDEFINED_MASK as c_int }
                },
                match y_position {
                    Some(y_pos) => { y_pos as c_int },
                    _ => { SDL_WINDOWPOS_UNDEFINED_MASK as c_int }
                },
                width,
                height,
                (SDL_WINDOW_SHOWN as u32) | (SDL_WINDOW_VULKAN as u32)
            );
        }

        if window.is_null() {
            unsafe {
                return Err(
                    SDL2Error::new(SDL_GetError())
                );
            }
        }

        Ok(
            Self {
                window: window
            }
        )
    }
}