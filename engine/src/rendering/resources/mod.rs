pub mod collection;
pub mod materials;
pub mod mesh;
pub mod object;
pub mod texture;

use thiserror::Error;

use crate::rendering::resources::object::MaterialGPU;

#[derive(Error, Debug)]
pub enum ResourceError {
    #[error("All texture slots are occupied, there is no room for a new one")]
    NoTextureSlotAvailable,

    #[error("All mesh slots are occupied, there is no room for a new one")]
    NoMeshSlotAvailable,

    #[error("Incomplete texture: {0}")]
    IncompleteTexture(String),

    #[error("Invalid object format")]
    InvalidObjectFormat,

    #[error("Resource is too large to fit in reserved GPU memory")]
    ResourceTooLarge,

    #[error("Cannot remove the empty texture")]
    AttemptedRemovalOfEmptyTexture,

    #[error("Missing vertex buffer data")]
    MissingVertexBuffer,
}

pub type ResourceResult<T> = Result<T, ResourceError>;

const SIZEOF_MATERIAL_DEFINITION: usize = std::mem::size_of::<MaterialGPU>();
