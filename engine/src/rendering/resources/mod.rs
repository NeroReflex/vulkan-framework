pub mod collection;
pub mod mesh;
pub mod object;
pub mod texture;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ResourceError {
    #[error("All texture slots are occupied, there is no room for a new one")]
    NoTextureSlotAvailable,

    #[error("Incomplete texture: {0}")]
    IncompleteTexture(String),

    #[error("Invalid object format")]
    InvalidObjectFormat,
}

pub type ResourceResult<T> = Result<T, ResourceError>;
