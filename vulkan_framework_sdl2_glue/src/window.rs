use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_uint;
use std::ffi::CStr;

use crate::prelude::*;

use vulkan_framework;

//use sdl2_sys::SDL_WindowFlags::*;
use sdl2_sys::*;

use std::sync::Arc;

pub struct Window {
    window: *mut SDL_Window,
    surface: Option<VkSurfaceKHR>,
}

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            SDL_DestroyWindow(self.window);
        }
    }
}

impl Window {
    pub fn new(
        title: &String,
        width: c_int,
        height: c_int,
        x_position: Option<c_int>,
        y_position: Option<c_int>,
    ) -> Result<Self, SDL2Error> {
        let mut title_bytes = title
            .as_bytes()
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
                    Some(x_pos) => x_pos as c_int,
                    _ => SDL_WINDOWPOS_UNDEFINED_MASK as c_int,
                },
                match y_position {
                    Some(y_pos) => y_pos as c_int,
                    _ => SDL_WINDOWPOS_UNDEFINED_MASK as c_int,
                },
                width,
                height,
                (sdl2_sys::SDL_WindowFlags::SDL_WINDOW_SHOWN as u32)
                    | (sdl2_sys::SDL_WindowFlags::SDL_WINDOW_VULKAN as u32),
            );
        }

        if window.is_null() {
            unsafe {
                return Err(SDL2Error::new(SDL_GetError()));
            }
        }

        Ok(Self {
            window,
            surface: Option::<VkSurfaceKHR>::None,
        })
    }

    /**
     * Create a vulkan surface given out as an ash handle from the current window.
     *
     * This function can only be called once per window.
     */
    pub fn create_surface(
        &mut self,
        instance: Arc<vulkan_framework::instance::Instance>,
    ) -> Result<Arc<vulkan_framework::surface::Surface>, SDL2Error> {
        match self.surface {
            Option::Some(_) => Err(SDL2Error::new(std::ptr::null())),
            Option::None => unsafe {
                let mut surface: VkSurfaceKHR = 0;

                match SDL_Vulkan_CreateSurface(
                    self.window,
                    instance.native_handle() as VkInstance,
                    &mut surface,
                ) {
                    SDL_bool::SDL_TRUE => {
                        match vulkan_framework::surface::Surface::from_raw(instance, surface) {
                            Ok(sfc) => Ok(sfc),
                            Err(_err) => Err(SDL2Error::new(std::ptr::null())),
                        }
                    }
                    SDL_bool::SDL_FALSE => Err(SDL2Error::new(std::ptr::null())),
                }
            },
        }
    }

    pub fn get_vulkan_instance_extensions(&mut self) -> Result<Vec<String>, SDL2Error> {
        let mut names = Vec::<String>::new();

        unsafe {
            let mut ext_count: c_uint = 0;
            let count_result = SDL_Vulkan_GetInstanceExtensions(
                self.window,
                &mut ext_count as *mut c_uint,
                std::ptr::null_mut(),
            );
            assert_eq!(count_result, SDL_bool::SDL_TRUE);

            // fill the space with lots of nullpointers
            let mut ext_names = Vec::<*const c_char>::new();
            for _i in 0..ext_count {
                ext_names.push(std::ptr::null());
            }

            let names_result = SDL_Vulkan_GetInstanceExtensions(
                self.window,
                &mut ext_count as *mut c_uint,
                ext_names.as_mut_ptr(),
            );
            assert_eq!(names_result, SDL_bool::SDL_TRUE);

            for ext_name in ext_names.iter() {
                match CStr::from_ptr(*ext_name).to_str() {
                    Ok(raw_name) => {
                        names.push(String::from(raw_name));
                    }
                    Err(_err) => {
                        return Err(SDL2Error::new(std::ptr::null()));
                    }
                }
            }
        }

        Ok(names)
    }
}
