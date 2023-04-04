pub mod prelude;
pub mod window;

use sdl2_sys::*;

pub fn init() {
    unsafe {
        SDL_Init(SDL_INIT_VIDEO);

        SDL_Vulkan_LoadLibrary(std::ptr::null());
    }
}

pub fn deinit() {
    unsafe {
        SDL_Vulkan_UnloadLibrary();
        SDL_Quit();
    }
}
