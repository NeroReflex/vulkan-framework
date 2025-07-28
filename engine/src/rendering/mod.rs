pub mod pipeline;
pub mod resources;
pub mod surface;
pub mod system;

use sdl2::video::WindowBuildError;
use thiserror::Error;

use crate::rendering::resources::ResourceError;

#[derive(Error, Debug)]
pub enum RenderingError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vulkan_framework::prelude::VulkanError),

    #[error("SDL errored while creating the window")]
    Window(#[from] WindowBuildError),

    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Resource error: {0}")]
    ResourceError(#[from] ResourceError),

    #[error("An unexpected number of swapchain images has been returned")]
    NotEnoughSwapchainImages,

    #[error("An unexpected error has occurred: there is no swapchain currently available")]
    NoSwapchain,

    #[error("{0}")]
    Unknown(String),
}

pub type RenderingResult<T> = Result<T, RenderingError>;

pub(crate) const MAX_FRAMES_IN_FLIGHT_NO_MALLOC: usize = 8;
pub(crate) const MAX_TEXTURES: u32 = 256;
pub(crate) const MAX_MATERIALS: u32 = 128;
