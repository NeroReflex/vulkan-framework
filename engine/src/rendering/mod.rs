pub mod pipeline;
pub mod rendering_dimensions;
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

    #[error("SDL errored while creating the window: {0}")]
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

/// Maximum number of frames in flight that avoids having to allocate
/// memory on the heap for frame-specific resources
pub(crate) const MAX_FRAMES_IN_FLIGHT_NO_MALLOC: usize = 4;

pub(crate) const MAX_TEXTURES: u32 = 256;
pub(crate) const MAX_MATERIALS: u32 = 128;

/// Max number of meshes in a scene: 4095 because it will fit 12 bits
/// on the custom index field (that is 24 bits) leaving the others
/// 12 bits for the instance ID
pub(crate) const MAX_MESHES: u32 = 4095;
pub(crate) const MAX_DIRECTIONAL_LIGHTS: u32 = 8;
